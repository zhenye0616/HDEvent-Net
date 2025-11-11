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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from Model.anomaly_head import AnomalyHead, MILAggregator

@dataclass
class SplitData:
    features: torch.Tensor
    anomaly: torch.Tensor
    classes: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.features.size(0))


@dataclass
class BagSample:
    features: torch.Tensor
    anomaly: float
    classes: int


class BagDataset(Dataset):
    def __init__(self, bags: List[BagSample]):
        self.bags = bags

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag = self.bags[idx]
        return bag.features, float(bag.anomaly), int(bag.classes)


def bag_collate(batch):
    if not batch:
        return [], torch.empty(0), torch.empty(0, dtype=torch.long)
    feats, anomaly, classes = zip(*batch)
    feats = [torch.as_tensor(f) for f in feats]
    anomaly = torch.tensor(anomaly, dtype=torch.float32)
    classes = torch.tensor(classes, dtype=torch.long)
    return feats, anomaly, classes


def load_embeddings(path: Path) -> Dict[str, torch.Tensor | list]:
    payload = torch.load(path, map_location="cpu")
    required = ["embeddings", "anomaly_labels", "class_labels", "splits", "names"]
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


def segment_to_video_name(segment_name: str) -> str:
    if not segment_name.startswith("seg:"):
        raise ValueError(f"Unexpected segment token: {segment_name}")
    payload = segment_name[len("seg:") :]
    if ":" not in payload:
        raise ValueError(f"Malformed segment token: {segment_name}")
    video_path, _ = payload.rsplit(":", 1)
    return f"video:{video_path}"


def build_segment_splits(
    payload: Dict[str, torch.Tensor | list],
) -> Dict[str, List[BagSample]]:
    embeddings: torch.Tensor = payload["embeddings"]
    anomaly_labels = payload["anomaly_labels"]
    class_labels = payload["class_labels"]
    splits = payload["splits"]
    names = payload["names"]

    grouped: Dict[str, Dict[str, List[int]]] = {split: {} for split in ("train", "val", "test")}
    for idx, name in enumerate(names):
        split = splits[idx]
        if split not in grouped:
            continue
        parent = segment_to_video_name(name)
        grouped[split].setdefault(parent, []).append(idx)

    result: Dict[str, List[BagSample]] = {}
    for split, bags in grouped.items():
        samples: List[BagSample] = []
        for indices in bags.values():
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long)
            feats = embeddings.index_select(0, index_tensor)
            anomaly_vals = torch.tensor([anomaly_labels[i] for i in indices], dtype=torch.float32)
            class_vals = torch.tensor([class_labels[i] for i in indices], dtype=torch.long)
            anomaly_label = float(anomaly_vals[0].item()) if anomaly_vals.numel() else 0.0
            cls_candidates = class_vals[class_vals >= 0]
            class_label = int(cls_candidates[0].item()) if cls_candidates.numel() else -1
            samples.append(BagSample(features=feats, anomaly=anomaly_label, classes=class_label))
        result[split] = samples
    return result


def compute_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
    if scores.numel() == 0 or labels.numel() == 0:
        return None
    scores = scores.reshape(-1).float().cpu()
    labels = labels.reshape(-1).float().cpu()
    uniques = torch.unique(labels)
    if uniques.numel() < 2:
        return None
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_total = pos_mask.sum().item()
    neg_total = neg_mask.sum().item()
    if pos_total == 0 or neg_total == 0:
        return None
    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]
    pos_cumsum = torch.cumsum((sorted_labels == 1).float(), dim=0)
    neg_cumsum = torch.cumsum((sorted_labels == 0).float(), dim=0)
    tpr = pos_cumsum / pos_total
    fpr = neg_cumsum / neg_total
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])
    auc = torch.trapz(tpr, fpr).item()
    return auc


def summarize_binary_predictions(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    default_threshold: float = 0.5,
    recall_target: float = 0.9,
) -> Dict[str, object]:
    """Compute richer binary classification metrics from raw scores."""
    summary: Dict[str, object] = {
        "roc_auc": None,
        "pr_auc": None,
        "support": {"positive": 0, "negative": 0},
        "best_f1": None,
        "thresholds": {},
    }
    if scores.numel() == 0 or labels.numel() == 0:
        return summary

    scores = scores.reshape(-1).float().cpu()
    labels = labels.reshape(-1).float().cpu()
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_total = int(pos_mask.sum().item())
    neg_total = int(neg_mask.sum().item())
    summary["support"] = {"positive": pos_total, "negative": neg_total}

    if pos_total == 0 or neg_total == 0:
        return summary

    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_labels = labels[indices]

    tp_cumsum = torch.cumsum((sorted_labels == 1).float(), dim=0)
    fp_cumsum = torch.cumsum((sorted_labels == 0).float(), dim=0)
    denom = tp_cumsum + fp_cumsum
    precision = torch.where(denom > 0, tp_cumsum / denom, torch.ones_like(denom))
    recall = tp_cumsum / pos_total

    # Precision-recall AUC (append sentinels for a proper curve)
    pr_precision = torch.cat(
        [torch.tensor([1.0], dtype=torch.float32), precision, torch.tensor([0.0], dtype=torch.float32)]
    )
    pr_recall = torch.cat(
        [torch.tensor([0.0], dtype=torch.float32), recall, torch.tensor([1.0], dtype=torch.float32)]
    )
    pr_auc = torch.trapz(pr_precision, pr_recall).item()
    summary["pr_auc"] = pr_auc

    roc_auc = compute_roc_auc(scores, labels)
    summary["roc_auc"] = roc_auc

    eps = 1e-8
    f1_vals = 2 * precision * recall / torch.clamp(precision + recall, min=eps)
    best_idx = int(torch.argmax(f1_vals).item())
    best_threshold = float(sorted_scores[best_idx].item())

    def threshold_stats(threshold: float) -> Dict[str, object]:
        preds = (scores >= threshold).float()
        tp = int(((preds == 1) & (labels == 1)).sum().item())
        fp = int(((preds == 1) & (labels == 0)).sum().item())
        fn = int(((preds == 0) & (labels == 1)).sum().item())
        tn = int(((preds == 0) & (labels == 0)).sum().item())
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / pos_total if pos_total > 0 else 0.0
        denom_pr = precision_val + recall_val
        f1_val = (2 * precision_val * recall_val / denom_pr) if denom_pr > 0 else 0.0
        return {
            "threshold": float(threshold),
            "precision": float(precision_val),
            "recall": float(recall_val),
            "f1": float(f1_val),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    summary["best_f1"] = {
        **threshold_stats(best_threshold),
        "curve_index": best_idx,
    }

    summary["thresholds"][f"{default_threshold:.2f}"] = threshold_stats(default_threshold)

    if recall_target is not None:
        mask = recall >= recall_target
        if mask.any():
            target_idx = int(torch.where(mask)[0][0].item())
            target_threshold = float(sorted_scores[target_idx].item())
            summary["thresholds"][f"recall@{recall_target:.2f}"] = {
                **threshold_stats(target_threshold),
                "target_recall": float(recall_target),
                "curve_index": target_idx,
            }

    return summary


def make_video_loader(split: SplitData, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(split.features, split.anomaly.unsqueeze(1), split.classes)
    effective_shuffle = shuffle and split.num_samples > 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=effective_shuffle)


def make_bag_loader(bags: List[BagSample], batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = BagDataset(bags)
    if len(dataset) == 0:
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=bag_collate)
    effective_batch = max(1, batch_size)
    effective_shuffle = shuffle and len(dataset) > 0
    return DataLoader(dataset, batch_size=effective_batch, shuffle=effective_shuffle, collate_fn=bag_collate)


def prepare_batch(
    batch,
    device: torch.device,
    granularity: str,
    aggregator: Optional[MILAggregator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if granularity == "video":
        feats, labels_bin, labels_cls = batch
        return feats.to(device), labels_bin.to(device), labels_cls.to(device)

    if aggregator is None:
        raise ValueError("Aggregator must be provided for segment-level training.")

    feats_list, labels_bin, labels_cls = batch
    pooled = []
    for feats in feats_list:
        feats = feats.to(device)
        pooled_vec, _ = aggregator(feats)
        pooled.append(pooled_vec)
    if not pooled:
        return (
            torch.empty(0, aggregator.input_dim, device=device),
            labels_bin.unsqueeze(1).to(device),
            labels_cls.to(device),
        )
    feats_tensor = torch.stack(pooled, dim=0)
    labels_bin = labels_bin.unsqueeze(1).to(device)
    labels_cls = labels_cls.to(device)
    return feats_tensor, labels_bin, labels_cls


def sample_balanced_indices(
    split: SplitData,
    num_classes: int,
    per_class: int,
    seed: int,
) -> Tuple[Optional[torch.Tensor], List[int], int]:
    if split.anomaly.numel() == 0:
        return None, list(range(num_classes)), 0

    anomalies = split.anomaly.reshape(-1)
    classes = split.classes.reshape(-1) if split.classes.numel() == anomalies.numel() else None

    generator = torch.Generator().manual_seed(seed)
    pos_chunks: List[torch.Tensor] = []
    missing: List[int] = []

    for cls_id in range(num_classes):
        mask = anomalies >= 0.5
        if classes is not None:
            mask = mask & (classes == cls_id)
        cls_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        if cls_indices.numel() == 0:
            missing.append(cls_id)
            continue
        if cls_indices.numel() >= per_class:
            perm = torch.randperm(cls_indices.numel(), generator=generator)[:per_class]
            chosen = cls_indices.index_select(0, perm)
        else:
            reps = torch.randint(0, cls_indices.numel(), (per_class,), generator=generator)
            chosen = cls_indices.index_select(0, reps)
        pos_chunks.append(chosen)

    if not pos_chunks:
        return None, missing, 0

    pos_indices = torch.cat(pos_chunks, dim=0)
    neg_indices = torch.nonzero(anomalies < 0.5, as_tuple=False).view(-1)
    if neg_indices.numel() == 0:
        return None, missing, 0

    if neg_indices.numel() >= pos_indices.numel():
        perm = torch.randperm(neg_indices.numel(), generator=generator)[:pos_indices.numel()]
        neg_selected = neg_indices.index_select(0, perm)
    else:
        reps = torch.randint(0, neg_indices.numel(), (pos_indices.numel(),), generator=generator)
        neg_selected = neg_indices.index_select(0, reps)

    combined = torch.cat([pos_indices, neg_selected], dim=0)
    combined = combined[torch.randperm(combined.numel(), generator=generator)]
    return combined, missing, int(pos_indices.numel())


def sample_balanced_bags(
    bags: List[BagSample],
    num_classes: int,
    per_class: int,
    seed: int,
) -> Tuple[Optional[List[BagSample]], List[int], int]:
    if not bags:
        return None, list(range(num_classes)), 0

    class_map: Dict[int, List[BagSample]] = {cls_id: [] for cls_id in range(num_classes)}
    negatives: List[BagSample] = []

    for bag in bags:
        if float(bag.anomaly) >= 0.5 and bag.classes >= 0 and bag.classes < num_classes:
            class_map.setdefault(bag.classes, []).append(bag)
        elif float(bag.anomaly) < 0.5:
            negatives.append(bag)

    rng = random.Random(seed)
    positives: List[BagSample] = []
    missing: List[int] = []

    for cls_id in range(num_classes):
        candidates = class_map.get(cls_id, [])
        if not candidates:
            missing.append(cls_id)
            continue
        if len(candidates) >= per_class:
            chosen = rng.sample(candidates, per_class)
        else:
            chosen = [rng.choice(candidates) for _ in range(per_class)]
        positives.extend(chosen)

    if not positives or not negatives:
        return None, missing, 0

    total_pos = len(positives)
    if len(negatives) >= total_pos:
        negative_choices = rng.sample(negatives, total_pos)
    else:
        negative_choices = [rng.choice(negatives) for _ in range(total_pos)]

    combined = positives + negative_choices
    rng.shuffle(combined)
    return combined, missing, total_pos


def evaluate(
	model: AnomalyHead,
	loader: DataLoader,
	device: torch.device,
	num_classes: int,
    class_names: Optional[List[str]],
    criterion_bin: nn.Module,
    criterion_cls: nn.Module,
    granularity: str,
    aggregator: Optional[MILAggregator] = None,
) -> Tuple[float, Dict[str, float]]:
	model.eval()
	if aggregator is not None:
		aggregator.eval()
	total_loss = 0.0
	total_samples = 0
	prob_buffer: List[torch.Tensor] = []
	label_buffer: List[torch.Tensor] = []
	per_class_scores: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}
	per_class_labels: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}

	with torch.no_grad():
		for batch in loader:
			if len(batch) == 0:
				continue
			feats, labels_bin, labels_cls = prepare_batch(batch, device, granularity, aggregator)
			if feats.numel() == 0:
				continue

			logits_bin, logits_cls = model(feats)
			loss_bin = criterion_bin(logits_bin, labels_bin)
			loss_cls = torch.tensor(0.0, device=device)

			probs = torch.sigmoid(logits_bin).detach().cpu().view(-1)
			prob_buffer.append(probs)
			label_buffer.append(labels_bin.detach().cpu().view(-1))

			if num_classes > 0:
				mask = labels_cls >= 0
				if mask.any():
					logits_masked = logits_cls[mask]
					labels_masked = labels_cls[mask]
					loss_cls = criterion_cls(logits_masked, labels_masked)
					probs_cls = torch.softmax(logits_masked, dim=1).detach().cpu()
					labels_masked_cpu = labels_masked.detach().cpu()
					for class_idx in range(num_classes):
						per_class_scores[class_idx].append(probs_cls[:, class_idx])
						per_class_labels[class_idx].append((labels_masked_cpu == class_idx).float())

			loss = loss_bin + loss_cls
			total_loss += loss.item() * labels_bin.size(0)
			total_samples += labels_bin.size(0)

	if total_samples == 0:
		return 0.0, {
            "mauc": None,
            "roc_auc": None,
            "pr_auc": None,
            "support": {"positive": 0, "negative": 0},
            "best_f1": None,
            "thresholds": {},
            "per_class_mauc": {str(i): None for i in range(num_classes)},
            "per_class_metrics": {},
        }

	all_probs = torch.cat(prob_buffer) if prob_buffer else torch.empty(0)
	all_labels = torch.cat(label_buffer) if label_buffer else torch.empty(0)
	summary = summarize_binary_predictions(all_probs, all_labels)

	per_class_mauc: Dict[str, Optional[float]] = {}
	per_class_metrics: Dict[str, Dict[str, Optional[float]]] = {}
	if num_classes > 0:
		for class_idx in range(num_classes):
			if per_class_scores[class_idx]:
				class_scores = torch.cat(per_class_scores[class_idx])
				class_labels = torch.cat(per_class_labels[class_idx])
				roc = compute_roc_auc(class_scores, class_labels)
			else:
				roc = None
			if class_names and class_idx < len(class_names):
				key = class_names[class_idx]
			else:
				key = str(class_idx)
			per_class_mauc[key] = roc
			per_class_metrics[key] = {"roc_auc": roc}

	metrics = {
		"mauc": summary["roc_auc"],
		"roc_auc": summary["roc_auc"],
		"pr_auc": summary["pr_auc"],
		"support": summary["support"],
		"best_f1": summary["best_f1"],
		"thresholds": summary["thresholds"],
		"per_class_mauc": per_class_mauc,
		"per_class_metrics": per_class_metrics,
	}
	if aggregator is not None:
		aggregator.train()
	return total_loss / total_samples, metrics


def train_anomaly_head(
    payload: Dict[str, torch.Tensor | list],
    anomaly_classes: list[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    num_classes = len(anomaly_classes)
    embed_dim = int(payload["embeddings"].size(1))  # type: ignore[arg-type]

    granularity = args.granularity

    video_splits: Dict[str, SplitData] | None = None
    segment_splits: Dict[str, List[BagSample]] | None = None

    if granularity == "video":
        video_splits = {split: select_split(payload, split) for split in ("train", "val", "test")}
        train_loader = make_video_loader(video_splits["train"], args.batch_size, shuffle=True)
        val_loader = make_video_loader(video_splits["val"], args.batch_size, shuffle=False)
        test_loader = make_video_loader(video_splits["test"], args.batch_size, shuffle=False)
        num_train_samples = video_splits["train"].num_samples
        num_val_samples = video_splits["val"].num_samples
        num_test_samples = video_splits["test"].num_samples
        aggregator: Optional[MILAggregator] = None
    else:
        segment_splits = build_segment_splits(payload)
        train_loader = make_bag_loader(segment_splits["train"], args.batch_size, shuffle=True)
        val_loader = make_bag_loader(segment_splits["val"], args.batch_size, shuffle=False)
        test_loader = make_bag_loader(segment_splits["test"], args.batch_size, shuffle=False)
        num_train_samples = len(segment_splits["train"])
        num_val_samples = len(segment_splits["val"])
        num_test_samples = len(segment_splits["test"])
        aggregator = MILAggregator(
            embed_dim,
            pooling=args.mil_pooling,
            hidden_dim=args.mil_hidden,
        ).to(device)

    model = AnomalyHead(
        embed_dim,
        num_classes,
        use_mlp=args.mlp_head,
        hidden_dim=args.mlp_hidden,
        dropout=args.mlp_dropout,
    ).to(device)
    trainable_params = list(model.parameters())
    if aggregator is not None:
        trainable_params += list(aggregator.parameters())
    trainable_params = [param for param in trainable_params if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_cls = nn.CrossEntropyLoss()

    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        if aggregator is not None:
            aggregator.train()
        epoch_loss = 0.0
        sample_count = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
            unit="batch",
        )
        for batch in progress:
            if len(batch) == 0:
                continue
            feats, labels_bin, labels_cls = prepare_batch(batch, device, granularity, aggregator)
            if feats.numel() == 0:
                continue

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
        progress.close()

        train_loss = epoch_loss / sample_count if sample_count else 0.0
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes,
            anomaly_classes,
            criterion_bin,
            criterion_cls,
            granularity,
            aggregator,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mauc": val_metrics["mauc"],
                "val_per_class_mauc": val_metrics["per_class_mauc"],
                "val_pr_auc": val_metrics.get("pr_auc"),
                "val_best_f1": val_metrics.get("best_f1"),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "aggregator_state": aggregator.state_dict() if aggregator is not None else None,
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        if aggregator is not None and best_state.get("aggregator_state") is not None:
            aggregator.load_state_dict(best_state["aggregator_state"])
        optimizer.load_state_dict(best_state["optimizer_state"])

    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        device,
        num_classes,
        anomaly_classes,
        criterion_bin,
        criterion_cls,
        granularity,
        aggregator,
    )

    if granularity == "video":
        assert video_splits is not None
        train_eval_loader = make_video_loader(video_splits["train"], args.batch_size, shuffle=False)
    else:
        assert segment_splits is not None
        train_eval_loader = make_bag_loader(segment_splits["train"], args.batch_size, shuffle=False)
    train_eval_loss, train_eval_metrics = evaluate(
        model,
        train_eval_loader,
        device,
        num_classes,
        anomaly_classes,
        criterion_bin,
        criterion_cls,
        granularity,
        aggregator,
    )

    balanced_eval: Dict[str, object] = {"enabled": bool(args.balanced_eval), "val": None, "test": None}
    if args.balanced_eval:
        eval_seed = args.balanced_eval_seed
        per_class = max(1, args.balanced_samples_per_class)
        if granularity == "video":
            assert video_splits is not None
            val_indices, val_missing, val_pos = sample_balanced_indices(
                video_splits["val"], num_classes, per_class, eval_seed
            )
            val_missing_names = [
                anomaly_classes[i] if i < len(anomaly_classes) else str(i) for i in val_missing
            ]
            if val_indices is not None:
                balanced_val_split = SplitData(
                    features=video_splits["val"].features.index_select(0, val_indices),
                    anomaly=video_splits["val"].anomaly.index_select(0, val_indices),
                    classes=video_splits["val"].classes.index_select(0, val_indices)
                    if video_splits["val"].classes.numel() > 0
                    else video_splits["val"].classes,
                )
                val_balanced_loader = make_video_loader(balanced_val_split, args.batch_size, shuffle=False)
                _, balanced_val_metrics = evaluate(
                    model,
                    val_balanced_loader,
                    device,
                    num_classes,
                    anomaly_classes,
                    criterion_bin,
                    criterion_cls,
                    granularity,
                    aggregator,
                )
                balanced_eval["val"] = {
                    "metrics": balanced_val_metrics,
                    "seed": eval_seed,
                    "anomaly_samples": val_pos,
                    "normal_samples": val_pos,
                    "missing_classes": val_missing_names,
                }
            else:
                balanced_eval["val"] = {
                    "metrics": None,
                    "seed": eval_seed,
                    "anomaly_samples": 0,
                    "normal_samples": 0,
                    "missing_classes": val_missing_names,
                }

            test_indices, test_missing, test_pos = sample_balanced_indices(
                video_splits["test"], num_classes, per_class, eval_seed
            )
            test_missing_names = [
                anomaly_classes[i] if i < len(anomaly_classes) else str(i) for i in test_missing
            ]
            if test_indices is not None:
                balanced_test_split = SplitData(
                    features=video_splits["test"].features.index_select(0, test_indices),
                    anomaly=video_splits["test"].anomaly.index_select(0, test_indices),
                    classes=video_splits["test"].classes.index_select(0, test_indices)
                    if video_splits["test"].classes.numel() > 0
                    else video_splits["test"].classes,
                )
                test_balanced_loader = make_video_loader(balanced_test_split, args.batch_size, shuffle=False)
                _, balanced_test_metrics = evaluate(
                    model,
                    test_balanced_loader,
                    device,
                    num_classes,
                    anomaly_classes,
                    criterion_bin,
                    criterion_cls,
                    granularity,
                    aggregator,
                )
                balanced_eval["test"] = {
                    "metrics": balanced_test_metrics,
                    "seed": eval_seed,
                    "anomaly_samples": test_pos,
                    "normal_samples": test_pos,
                    "missing_classes": test_missing_names,
                }
            else:
                balanced_eval["test"] = {
                    "metrics": None,
                    "seed": eval_seed,
                    "anomaly_samples": 0,
                    "normal_samples": 0,
                    "missing_classes": test_missing_names,
                }
        else:
            assert segment_splits is not None
            balanced_val_bags, val_missing, val_pos = sample_balanced_bags(
                segment_splits["val"], num_classes, per_class, eval_seed
            )
            val_missing_names = [
                anomaly_classes[i] if i < len(anomaly_classes) else str(i) for i in val_missing
            ]
            if balanced_val_bags is not None:
                val_balanced_loader = make_bag_loader(balanced_val_bags, args.batch_size, shuffle=False)
                _, balanced_val_metrics = evaluate(
                    model,
                    val_balanced_loader,
                    device,
                    num_classes,
                    anomaly_classes,
                    criterion_bin,
                    criterion_cls,
                    granularity,
                    aggregator,
                )
                balanced_eval["val"] = {
                    "metrics": balanced_val_metrics,
                    "seed": eval_seed,
                    "anomaly_samples": val_pos,
                    "normal_samples": val_pos,
                    "missing_classes": val_missing_names,
                }
            else:
                balanced_eval["val"] = {
                    "metrics": None,
                    "seed": eval_seed,
                    "anomaly_samples": 0,
                    "normal_samples": 0,
                    "missing_classes": val_missing_names,
                }

            balanced_test_bags, test_missing, test_pos = sample_balanced_bags(
                segment_splits["test"], num_classes, per_class, eval_seed
            )
            test_missing_names = [
                anomaly_classes[i] if i < len(anomaly_classes) else str(i) for i in test_missing
            ]
            if balanced_test_bags is not None:
                test_balanced_loader = make_bag_loader(balanced_test_bags, args.batch_size, shuffle=False)
                _, balanced_test_metrics = evaluate(
                    model,
                    test_balanced_loader,
                    device,
                    num_classes,
                    anomaly_classes,
                    criterion_bin,
                    criterion_cls,
                    granularity,
                    aggregator,
                )
                balanced_eval["test"] = {
                    "metrics": balanced_test_metrics,
                    "seed": eval_seed,
                    "anomaly_samples": test_pos,
                    "normal_samples": test_pos,
                    "missing_classes": test_missing_names,
                }
            else:
                balanced_eval["test"] = {
                    "metrics": None,
                    "seed": eval_seed,
                    "anomaly_samples": 0,
                    "normal_samples": 0,
                    "missing_classes": test_missing_names,
                }

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_mauc": test_metrics["mauc"],
        "test_pr_auc": test_metrics.get("pr_auc"),
        "test_per_class_mauc": test_metrics["per_class_mauc"],
        "train_eval_loss": train_eval_loss,
        "train_mauc": train_eval_metrics["mauc"],
        "train_pr_auc": train_eval_metrics.get("pr_auc"),
        "train_per_class_mauc": train_eval_metrics["per_class_mauc"],
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
        "num_test_samples": num_test_samples,
        "anomaly_classes": anomaly_classes,
        "test_metrics_full": test_metrics,
        "train_eval_metrics_full": train_eval_metrics,
        "balanced_eval": balanced_eval,
        "head_state_dict": model.state_dict(),
        "aggregator_state": aggregator.state_dict() if aggregator is not None else None,
        "aggregator_config": {
            "pooling": args.mil_pooling,
            "hidden_dim": args.mil_hidden,
        }
        if aggregator is not None
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple anomaly head on exported embeddings.")
    parser.add_argument("--embeddings", type=Path, required=True, help="Path to video/segment embeddings .pt file.")
    parser.add_argument("--anomaly-classes", type=Path, default=None, help="Optional metadata JSON containing anomaly class names.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write head checkpoints and metrics.")
    parser.add_argument("--granularity", choices=("video", "segment"), default="segment", help="Use video-level or segment-level training.")
    parser.add_argument("--mil-pooling", choices=("mean", "max", "attention"), default="attention", help="Pooling strategy for segment-level MIL (ignored for video mode).")
    parser.add_argument("--mil-hidden", type=int, default=128, help="Hidden size for attention pooling (segment mode).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer.")
    parser.add_argument("--class-weight", type=float, default=1.0, help="Weight multiplier for multi-class loss.")
    parser.add_argument("--mlp-head", action="store_true", help="Use a two-layer MLP anomaly head instead of a single linear layer.")
    parser.add_argument("--mlp-hidden", type=int, default=256, help="Hidden dimension when MLP head is enabled.")
    parser.add_argument("--mlp-dropout", type=float, default=0.3, help="Dropout rate for the MLP head.")
    parser.add_argument("--balanced-eval", action="store_true", help="Report additional metrics on balanced val/test subsets (equal anomaly vs normal).")
    parser.add_argument("--balanced-eval-seed", type=int, default=1234, help="Random seed used for balanced evaluation sampling.")
    parser.add_argument("--balanced-samples-per-class", type=int, default=1, help="Number of anomaly samples to draw per class when building balanced eval subsets.")
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
                "test_mauc": results["test_mauc"],
                "test_pr_auc": results["test_pr_auc"],
                "test_metrics": results["test_metrics_full"],
                "test_per_class_mauc": results["test_per_class_mauc"],
                "train_eval_loss": results["train_eval_loss"],
                "train_mauc": results["train_mauc"],
                "train_pr_auc": results["train_pr_auc"],
                "train_eval_metrics": results["train_eval_metrics_full"],
                "train_per_class_mauc": results["train_per_class_mauc"],
                "num_train_samples": results["num_train_samples"],
                "num_val_samples": results["num_val_samples"],
                "num_test_samples": results["num_test_samples"],
                "anomaly_classes": results["anomaly_classes"],
                "granularity": args.granularity,
                "mil_pooling": args.mil_pooling if args.granularity == "segment" else None,
                "balanced_eval": results["balanced_eval"],
            },
            handle,
            indent=2,
        )

    checkpoint_payload = {
        "model_state": results["head_state_dict"],
        "anomaly_classes": results["anomaly_classes"],
        "config": vars(args),
    }
    if results.get("aggregator_state") is not None:
        checkpoint_payload["aggregator_state"] = results["aggregator_state"]
        checkpoint_payload["aggregator_config"] = results.get("aggregator_config")
    torch.save(checkpoint_payload, args.output_dir / "head.pt")
    print(f"[train_anomaly_head] wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
