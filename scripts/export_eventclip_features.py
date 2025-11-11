#!/usr/bin/env python3
"""
Extract frozen Event-CLIP features (video + segment) for baseline anomaly heads.

The script reads precomputed Event-CLIP .npy tensors (e.g., vitb/event_thr_10),
filters a specific augmentation index, and serializes payloads that mirror the
KG export format consumed by scripts/train_anomaly_head.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Data.dataset import UCFCrimeEventDataset
from Utils.KG_builder import canonical_video_entity

NORMAL_CLASS_TOKENS = {"class:normal", "class:normal_videos"}


@dataclass
class Entry:
    name: str
    embedding: torch.Tensor
    anomaly_label: int
    class_name: str
    split: str


def slugify_class(name: str) -> str:
    token = name.strip().lower().replace(" ", "_")
    # Some manifests use "Normal_Videos" etc.; keep both tokens for compatibility.
    if token in {"normal_videos"}:
        return "class:normal_videos"
    return f"class:{token}"


def load_manifest_splits(manifest_root: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    mapping: Dict[str, str] = {}
    counts: Dict[str, int] = {split: 0 for split in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        manifest_path = manifest_root / f"{split}.txt"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                token = token.split()[0]
                if not token.startswith("video:"):
                    token = f"video:{token}"
                try:
                    canonical = canonical_video_entity(token)
                except ValueError:
                    continue
                if canonical in mapping and mapping[canonical] != split:
                    raise ValueError(f"Video {canonical} appears in multiple splits ({mapping[canonical]} vs {split}).")
                mapping[canonical] = split
        counts[split] = sum(1 for split_name in mapping.values() if split_name == split)
    return mapping, counts


def derive_video_entity(sample_path: Path, variant: str) -> str:
    relative = sample_path
    # Ensure the path includes the variant prefix (vitb/...) before canonicalization.
    if variant:
        relative = Path(variant) / relative
    no_ext = relative.with_suffix("")
    raw_token = f"video:{no_ext.as_posix()}"
    return canonical_video_entity(raw_token)


def build_segment_name(video_entity: str, segment_idx: int) -> str:
    payload = video_entity[len("video:") :]
    return f"seg:{payload}:{segment_idx:05d}"


def pack_entries(entries: List[Entry], class_to_idx: Dict[str, int]) -> Dict[str, object]:
    if not entries:
        return {}
    embeddings = torch.stack([entry.embedding for entry in entries])
    return {
        "names": [entry.name for entry in entries],
        "indices": list(range(len(entries))),
        "embeddings": embeddings,
        "class_names": [entry.class_name for entry in entries],
        "anomaly_labels": [int(entry.anomaly_label) for entry in entries],
        "class_labels": [
            int(class_to_idx.get(entry.class_name, -1)) for entry in entries
        ],
        "splits": [entry.split for entry in entries],
    }


def summarize_stats(stats: Dict[str, Dict[str, int]]) -> List[str]:
    lines: List[str] = []
    for split in ("train", "val", "test"):
        split_stats = stats.get(split, {})
        videos = split_stats.get("videos", 0)
        positives = split_stats.get("positive_videos", 0)
        segments = split_stats.get("segments", 0)
        lines.append(
            f"  {split:5s}: {videos:4d} videos ({positives:3d} anomaly) | {segments:6d} segments"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen Event-CLIP features for baseline anomaly heads.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/Data_1/UCFCrime_dataset"),
        help="Root directory containing Event-CLIP variants (e.g., vitb).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="vitb",
        help="Variant subdirectory under data-root (default: vitb).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="event_thr_10",
        help="Event-CLIP subfolder to read (e.g., event_thr_10).",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("Data/manifests"),
        help="Directory containing train/val/test manifest .txt files.",
    )
    parser.add_argument(
        "--augmentation-idx",
        type=int,
        default=5,
        help="Augmentation index to keep (matches filename suffix '__{idx}').",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of classes to load. Defaults to all available folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline_features"),
        help="Destination directory for serialized feature payloads.",
    )
    parser.add_argument(
        "--video-filename",
        type=str,
        default="video_eventclip.pt",
        help="Filename for video-level embeddings (within output-dir).",
    )
    parser.add_argument(
        "--segment-filename",
        type=str,
        default="segment_eventclip.pt",
        help="Filename for segment-level embeddings (within output-dir).",
    )
    args = parser.parse_args()

    variant_root = args.data_root / args.variant
    split_dir = variant_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    manifest_map, manifest_counts = load_manifest_splits(args.manifest_root)
    if not manifest_map:
        raise FileNotFoundError(f"No manifest entries found under {args.manifest_root}")

    dataset = UCFCrimeEventDataset(
        data_root=variant_root,
        split=args.split,
        classes=args.classes,
        augmentation_idx=args.augmentation_idx,
    )

    video_entries: List[Entry] = []
    segment_entries: List[Entry] = []
    stats: Dict[str, Dict[str, int]] = {
        split: {"videos": 0, "positive_videos": 0, "segments": 0}
        for split in ("train", "val", "test")
    }
    skipped_missing_manifest = 0
    sanitized_videos = 0
    sanitized_segments = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        metadata = sample["metadata"]
        sample_path = Path(metadata["file_path"]).relative_to(variant_root)
        video_entity = derive_video_entity(sample_path, args.variant)
        split = manifest_map.get(video_entity)
        if split is None:
            skipped_missing_manifest += 1
            continue

        temporal = sample["temporal_features"].detach().clone()
        invalid_mask = ~torch.isfinite(temporal)
        if invalid_mask.any():
            sanitized_segments += int(invalid_mask.any(dim=1).sum().item())
            temporal = torch.nan_to_num(
                temporal, nan=0.0, posinf=0.0, neginf=0.0
            )
            sanitized_videos += 1
        pooled = temporal.mean(dim=0)

        class_token = slugify_class(metadata["class_name"])
        anomaly_label = 0 if class_token in NORMAL_CLASS_TOKENS else 1
        video_entries.append(
            Entry(
                name=video_entity,
                embedding=pooled,
                anomaly_label=anomaly_label,
                class_name=class_token,
                split=split,
            )
        )
        stats.setdefault(split, {"videos": 0, "positive_videos": 0, "segments": 0})
        stats[split]["videos"] += 1
        if anomaly_label == 1:
            stats[split]["positive_videos"] += 1

        temporal = temporal.reshape(temporal.shape[0], -1)
        num_segments = temporal.shape[0]
        stats[split]["segments"] += int(num_segments)
        for seg_idx in range(num_segments):
            segment_entries.append(
                Entry(
                    name=build_segment_name(video_entity, seg_idx),
                    embedding=temporal[seg_idx],
                    anomaly_label=anomaly_label,
                    class_name=class_token,
                    split=split,
                )
            )

    if not video_entries:
        raise RuntimeError("No videos matched the provided manifests. Check augmentation index or manifests.")

    anomaly_class_tokens = sorted(
        {entry.class_name for entry in video_entries if entry.anomaly_label == 1}
    )
    class_to_idx = {token: idx for idx, token in enumerate(anomaly_class_tokens)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_payload = pack_entries(video_entries, class_to_idx)
    segment_payload = pack_entries(segment_entries, class_to_idx)

    video_out = args.output_dir / args.video_filename
    segment_out = args.output_dir / args.segment_filename
    torch.save(video_payload, video_out)
    torch.save(segment_payload, segment_out)

    metadata = {
        "data_root": str(args.data_root),
        "variant": args.variant,
        "split": args.split,
        "augmentation_idx": args.augmentation_idx,
        "manifest_root": str(args.manifest_root),
        "output_dir": str(args.output_dir),
        "video_file": str(video_out),
        "segment_file": str(segment_out),
        "num_videos": len(video_entries),
        "num_segments": len(segment_entries),
        "anomaly_classes": anomaly_class_tokens,
        "manifest_counts": manifest_counts,
        "split_stats": stats,
        "sanitized_videos": sanitized_videos,
        "sanitized_segments": sanitized_segments,
        "skipped_missing_manifest": skipped_missing_manifest,
    }
    metadata_path = args.output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[export_eventclip_features] Saved {len(video_entries)} videos to {video_out}")
    print(f"[export_eventclip_features] Saved {len(segment_entries)} segments to {segment_out}")
    for line in summarize_stats(stats):
        print(line)
    if skipped_missing_manifest:
        print(f"Skipped {skipped_missing_manifest} samples missing from manifests.")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
