#!/usr/bin/env python3
"""
Export learned entity embeddings (videos and segments) with anomaly labels.

Usage example:
    python scripts/export_embeddings.py \
        --checkpoint checkpoints/ucf_graphhd_transe_neg_22_10_2025_14:37:34/model.pt \
        --dataset UCF_Crime \
        --output-dir exports/ucf_graphhd_transe_neg_22_10_2025_14:37:34
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.KG_builder import canonical_video_entity


Split = str
EntityName = str


def load_entity_to_id(path: Path) -> Dict[EntityName, int]:
    mapping: Dict[EntityName, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            name, idx = line.split("\t")
            mapping[name] = int(idx)
    return mapping


def iter_triples(path: Path) -> Iterable[Tuple[str, str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            head, rel, tail = line.split("\t")
            yield head, rel, tail


def build_entity_map_from_triples(triple_root: Path, target_count: int) -> Dict[EntityName, int]:
    order: "OrderedDict[EntityName, int]" = OrderedDict()
    for split in ("train", "val", "test"):
        triple_path = triple_root / f"{split}.txt"
        if not triple_path.exists():
            continue
        for head, _, tail in iter_triples(triple_path):
            for name in (head.lower(), tail.lower()):
                if name not in order:
                    order[name] = len(order)
                    if len(order) == target_count:
                        return dict(order)
    if len(order) != target_count:
        raise ValueError(
            f"Unable to recover {target_count} entities from triples under {triple_root}. "
            f"Found only {len(order)} unique entities. Please supply --entity-map pointing to the training entity2id.txt."
        )
    return dict(order)


def load_video_lists(manifest_root: Path) -> Dict[Split, List[str]]:
    splits: Dict[Split, List[str]] = {}
    for split in ("train", "val", "test"):
        manifest_path = manifest_root / f"{split}.txt"
        videos: List[str] = []
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    token = line.strip()
                    if not token:
                        continue
                    try:
                        videos.append(canonical_video_entity(token))
                    except Exception:
                        normalized = token if token.startswith("video:") else f"video:{token}"
                        videos.append(canonical_video_entity(normalized))
        splits[split] = videos
    return splits


def choose_split(splits: Sequence[str]) -> str:
	if not splits:
		return "unknown"
	priority = ("val", "test", "train")
	for key in priority:
		if key in splits:
			return key
	return sorted(splits)[0]


NORMAL_CLASSES = {"class:normal", "class:normal_videos"}


def export_embeddings(
    checkpoint: Path,
    dataset_root: Path,
    manifest_root: Path,
    output_dir: Path,
    *,
    entity_map_path: Optional[Path] = None,
) -> None:
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    init_embed = state_dict["init_embed"]
    embed_count = init_embed.shape[0]

    if entity_map_path is not None:
        entity_map = load_entity_to_id(entity_map_path)
        max_idx = max(entity_map.values()) if entity_map else -1
        if len(entity_map) != embed_count or max_idx >= embed_count:
            filtered = {name: idx for name, idx in entity_map.items() if idx < embed_count}
            if len(filtered) != embed_count:
                raise ValueError(
                    f"Checkpoint entity count ({embed_count}) does not align with supplied entity map entries "
                    f"({len(entity_map)} total, max idx {max_idx}). Provide the entity2id.txt from training time."
                )
            print(
                f"[export_embeddings] entity map has {len(entity_map)} entries; "
                f"using the {embed_count} entries whose indices match the checkpoint."
            )
            entity_map = filtered
    else:
        entity_map = build_entity_map_from_triples(dataset_root, embed_count)

    # Aggregate metadata from triples
    triple_root = dataset_root
    video_to_class: Dict[str, str] = {}
    segment_to_video: Dict[str, str] = {}
    segment_to_class: Dict[str, str] = {}
    video_splits: Dict[str, set[str]] = defaultdict(set)
    segment_splits: Dict[str, set[str]] = defaultdict(set)

    for split in ("train", "val", "test"):
        triple_path = triple_root / f"{split}.txt"
        if not triple_path.exists():
            continue
        for head, rel, tail in iter_triples(triple_path):
            rel = rel.lower()
            head = head.lower()
            tail = tail.lower()

            if head.startswith("video:"):
                video_splits[head].add(split)
            if tail.startswith("video:"):
                video_splits[tail].add(split)
            if head.startswith("seg:"):
                segment_splits[head].add(split)
            if tail.startswith("seg:"):
                segment_splits[tail].add(split)

            if rel == "class_of":
                video_to_class[head] = tail
            elif rel == "class_of_reverse":
                video_to_class[tail] = head
            elif rel == "part_of":
                segment_to_video[head] = tail
            elif rel == "part_of_reverse":
                segment_to_video[tail] = head

    # Supplement segment splits using parent videos
    for segment, video in segment_to_video.items():
        if segment not in segment_splits and video in video_splits:
            segment_splits[segment] = set(video_splits[video])
        # Track class for segments
        if video in video_to_class:
            segment_to_class[segment] = video_to_class[video]

    # Build anomaly labels
    anomaly_classes = sorted(
        {cls for cls in video_to_class.values() if cls not in NORMAL_CLASSES}
    )
    class_to_idx = {name: idx for idx, name in enumerate(anomaly_classes)}

    manifest_videos = load_video_lists(manifest_root)
    for split, videos in manifest_videos.items():
        for video in videos:
            video_splits[video].add(split)

    # Prepare export directory
    output_dir.mkdir(parents=True, exist_ok=True)

    def entity_entries(prefix: str) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        for name, idx in entity_map.items():
            if not name.startswith(prefix):
                continue
            embedding = init_embed[idx].detach().cpu()
            if prefix == "video:":
                cls = video_to_class.get(name)
                splits = video_splits.get(name, set())
            else:
                cls = segment_to_class.get(name)
                splits = segment_splits.get(name, set())
            anomaly_label = None
            class_label = -1
            if cls is not None:
                anomaly_label = 0 if cls in NORMAL_CLASSES else 1
                if anomaly_label == 1:
                    class_label = class_to_idx.get(cls, -1)
            entry = {
                "name": name,
                "index": idx,
                "embedding": embedding,
                "class_name": cls,
                "anomaly_label": anomaly_label,
                "class_label": class_label,
                "split": choose_split(list(splits)),
            }
            entries.append(entry)
        return entries

    video_entries = entity_entries("video:")
    segment_entries = entity_entries("seg:")

    def pack(entries: List[Dict[str, object]]) -> Dict[str, object]:
        if not entries:
            return {}
        embeddings = torch.stack([e["embedding"] for e in entries])
        return {
            "names": [e["name"] for e in entries],
            "indices": [int(e["index"]) for e in entries],
            "embeddings": embeddings,
            "class_names": [e["class_name"] for e in entries],
            "anomaly_labels": [
                int(e["anomaly_label"]) if e["anomaly_label"] is not None else -1
                for e in entries
            ],
            "class_labels": [int(e["class_label"]) for e in entries],
            "splits": [e["split"] for e in entries],
        }

    video_payload = pack(video_entries)
    segment_payload = pack(segment_entries)

    if video_payload:
        torch.save(video_payload, output_dir / "video_embeddings.pt")
    if segment_payload:
        torch.save(segment_payload, output_dir / "segment_embeddings.pt")

    metadata = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "manifest_root": str(manifest_root),
        "num_entities": embed_count,
        "num_videos": len(video_entries),
        "num_segments": len(segment_entries),
        "anomaly_classes": anomaly_classes,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[export_embeddings] wrote metadata to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export KG embeddings for anomaly heads.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to saved model checkpoint (e.g., checkpoints/run/model.pt).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="UCF_Crime",
        help="Name of dataset directory under Data/.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional override for dataset root (defaults to Data/<dataset>).",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("Data/manifests"),
        help="Directory containing train/val/test manifest files.",
    )
    parser.add_argument(
        "--entity-map",
        type=Path,
        default=None,
        help="Optional path to entity2id.txt used during training. If omitted the script reconstructs the entity order from triples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write exported embeddings and metadata.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root or Path("Data") / args.dataset
    export_embeddings(
        checkpoint=args.checkpoint,
        dataset_root=dataset_root,
        manifest_root=args.manifest_root,
        output_dir=args.output_dir,
        entity_map_path=args.entity_map,
    )


if __name__ == "__main__":
    main()
