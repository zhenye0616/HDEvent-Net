#!/usr/bin/env python3
"""
Train a simple anomaly head on top of exported KG embeddings.

Example:
    python scripts/train_anomaly_head.py \
        --embeddings checkpoints/ucf_graphhd_transe_neg_22_10_2025_14:37:34/video_embeddings.pt \
        --output-dir checkpoints/ucf_graphhd_transe_neg_22_10_2025_14:37:34/anomaly_head_video
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Model.anomaly_head import AnomalyHead


@dataclass
class SplitData:
    features: torch.Tensor
    anomaly: torch.Tensor
    classes: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.features.size(0))


def load_embeddings(path: Path) -> Dict[str, torch.Tensor | list]:
    payload = torch.load(path, map_location="cpu")
    required = ["embeddings", "anomaly_labels", "class_labels", "splits"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return payload


def select_split(payload: Dict[str, torch.Tensor | list], split: str) -> SplitData:
    embeddings: torch.Tensor = payload["embeddings"]
    anomaly_labels = payload["anomaly_labels"]
    class_labels = payload["class_labels"]
    splits = payload["splits"]

    mask = torch.tensor([s == split for s in splits], dtype=torch.bool)
    if mask.sum() == 0:
        return SplitData(
            features=torch.empty(0, embeddings.size(1)),
            anomaly=torch.empty(0),
            classes=torch.empty(0),
        )

    feats = embeddings[mask]
    anomaly = torch.tensor([anomaly_labels[i] for i, keep in enumerate(mask.tolist()) if keep], dtype=torch.float32)
    cls = torch.tensor([class_labels[i] for i, keep in enumerate(mask.tolist()) if keep], dtype=torch.long)

    valid_mask = anomaly >= 0
    return SplitData(
        features=feats[valid_mask],
        anomaly=anomaly[valid_mask],
        classes=cls[valid_mask],
    )


def make_loader(split: SplitData, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(split.features, split.anomaly.unsqueeze(1), split.classes)
    effective_shuffle = shuffle and split.num_samples > 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=effective_shuffle)


def evaluate(
    model: AnomalyHead,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion_bin: nn.Module,
    criterion_cls: nn.Module,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_binary = 0
    binary_count = 0
    correct_cls = 0
    cls_count = 0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 1:
                continue
            feats, labels_bin, labels_cls = batch
            feats = feats.to(device)
            labels_bin = labels_bin.to(device)
            labels_cls = labels_cls.to(device)

            logits_bin, logits_cls = model(feats)
            loss_bin = criterion_bin(logits_bin, labels_bin)
            loss_cls = 0.0

            preds_bin = (torch.sigmoid(logits_bin) >= 0.5).float()
            correct_binary += (preds_bin == labels_bin).sum().item()
            binary_count += labels_bin.size(0)

            if num_classes > 0:
                mask = labels_cls >= 0
                if mask.any():
                    logits_masked = logits_cls[mask]
                    labels_masked = labels_cls[mask]
                    loss_cls = criterion_cls(logits_masked, labels_masked)
                    pred_cls = torch.argmax(logits_masked, dim=1)
                    correct_cls += (pred_cls == labels_masked).sum().item()
                    cls_count += labels_masked.size(0)

            loss = loss_bin + (loss_cls if isinstance(loss_cls, torch.Tensor) else loss_cls)
            total_loss += loss.item() * labels_bin.size(0)
            total_samples += labels_bin.size(0)

    if total_samples == 0:
        return 0.0, {"binary_accuracy": 0.0, "class_accuracy": 0.0}

    metrics = {
        "binary_accuracy": correct_binary / binary_count if binary_count else 0.0,
        "class_accuracy": correct_cls / cls_count if cls_count else 0.0,
    }
    return total_loss / total_samples, metrics


def train_anomaly_head(
    payload: Dict[str, torch.Tensor | list],
    anomaly_classes: list[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    num_classes = len(anomaly_classes)
    embed_dim = int(payload["embeddings"].size(1))  # type: ignore[arg-type]

    splits = {split: select_split(payload, split) for split in ("train", "val", "test")}

    model = AnomalyHead(embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_cls = nn.CrossEntropyLoss()

    train_loader = make_loader(splits["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(splits["val"], args.batch_size, shuffle=False)
    test_loader = make_loader(splits["test"], args.batch_size, shuffle=False)

    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            if len(batch) == 1:
                continue
            feats, labels_bin, labels_cls = batch
            feats = feats.to(device)
            labels_bin = labels_bin.to(device)
            labels_cls = labels_cls.to(device)

            optimizer.zero_grad()
            logits_bin, logits_cls = model(feats)
            loss_bin = criterion_bin(logits_bin, labels_bin)

            loss_cls = torch.tensor(0.0, device=device)
            if num_classes > 0:
                mask = labels_cls >= 0
                if mask.any():
                    loss_cls = criterion_cls(logits_cls[mask], labels_cls[mask])
            loss = loss_bin + args.class_weight * loss_cls
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels_bin.size(0)
            sample_count += labels_bin.size(0)

        train_loss = epoch_loss / sample_count if sample_count else 0.0
        val_loss, val_metrics = evaluate(model, val_loader, device, num_classes, criterion_bin, criterion_cls)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_binary_accuracy": val_metrics["binary_accuracy"],
                "val_class_accuracy": val_metrics["class_accuracy"],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_loss, test_metrics = evaluate(model, test_loader, device, num_classes, criterion_bin, criterion_cls)

    train_eval_loader = make_loader(splits["train"], args.batch_size, shuffle=False)
    train_eval_loss, train_eval_metrics = evaluate(
        model, train_eval_loader, device, num_classes, criterion_bin, criterion_cls
    )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_binary_accuracy": test_metrics["binary_accuracy"],
        "test_class_accuracy": test_metrics["class_accuracy"],
        "train_eval_loss": train_eval_loss,
        "train_binary_accuracy": train_eval_metrics["binary_accuracy"],
        "train_class_accuracy": train_eval_metrics["class_accuracy"],
        "num_train_samples": splits["train"].num_samples,
        "num_val_samples": splits["val"].num_samples,
        "num_test_samples": splits["test"].num_samples,
        "anomaly_classes": anomaly_classes,
        "head_state_dict": model.state_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple anomaly head on exported embeddings.")
    parser.add_argument("--embeddings", type=Path, required=True, help="Path to video/segment embeddings .pt file.")
    parser.add_argument("--anomaly-classes", type=Path, default=None, help="Optional metadata JSON containing anomaly class names.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write head checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer.")
    parser.add_argument("--class-weight", type=float, default=1.0, help="Weight multiplier for multi-class loss.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    payload = load_embeddings(args.embeddings)
    metadata_path = args.anomaly_classes
    anomaly_classes: list[str] = []
    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        anomaly_classes = metadata.get("anomaly_classes", [])
    else:
        # Fallback: infer from payload class names
        normal_tokens = {"class:normal", "class:normal_videos"}
        class_names = [
            name
            for name in payload.get("class_names", [])
            if isinstance(name, str) and name and name not in normal_tokens
        ]
        anomaly_classes = sorted(set(class_names))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = train_anomaly_head(payload, anomaly_classes, args)

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": results["history"],
                "best_val_loss": results["best_val_loss"],
                "test_loss": results["test_loss"],
                "test_binary_accuracy": results["test_binary_accuracy"],
                "test_class_accuracy": results["test_class_accuracy"],
                "train_eval_loss": results["train_eval_loss"],
                "train_binary_accuracy": results["train_binary_accuracy"],
                "train_class_accuracy": results["train_class_accuracy"],
                "num_train_samples": results["num_train_samples"],
                "num_val_samples": results["num_val_samples"],
                "num_test_samples": results["num_test_samples"],
                "anomaly_classes": results["anomaly_classes"],
            },
            handle,
            indent=2,
        )

    torch.save(
        {
            "model_state": results["head_state_dict"],
            "anomaly_classes": results["anomaly_classes"],
            "config": vars(args),
        },
        args.output_dir / "head.pt",
    )
    print(f"[train_anomaly_head] wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
