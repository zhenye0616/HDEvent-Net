#!/usr/bin/env python3
"""Evaluate how well shared anomaly attributes bridge class embeddings."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set

import torch


def load_entity_embeddings(checkpoint_path: Path) -> torch.Tensor:
    state = torch.load(checkpoint_path, map_location="cpu")
    init_embed = state["state_dict"]["init_embed"]
    return init_embed.detach().float()


def load_id_map(path: Path) -> Mapping[str, int]:
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name, idx = line.strip().split("\t")
            mapping[name] = int(idx)
    return mapping


def collect_triples(data_root: Path, splits: Sequence[str]) -> Iterable[Sequence[str]]:
    for split in splits:
        path = data_root / f"{split}.txt"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield line.strip().split("\t")


def build_graph_maps(data_root: Path) -> tuple[
    MutableMapping[str, str],
    MutableMapping[str, str],
    MutableMapping[str, Set[str]],
]:
    segment_to_video: Dict[str, str] = {}
    video_to_class: Dict[str, str] = {}
    attr_to_segments: Dict[str, Set[str]] = defaultdict(set)

    for sub, rel, obj in collect_triples(data_root, ("train", "val", "test")):
        if rel == "part_of":
            segment_to_video[sub] = obj
        elif rel == "class_of":
            video_to_class[sub] = obj
        elif rel == "has_attribute":
            attr_to_segments[obj].add(sub)

    return segment_to_video, video_to_class, attr_to_segments


def infer_segment_classes(
    segment_to_video: Mapping[str, str],
    video_to_class: Mapping[str, str],
) -> Mapping[str, str]:
    segment_to_class: Dict[str, str] = {}
    for segment, video in segment_to_video.items():
        cls = video_to_class.get(video)
        if cls is None:
            continue
        segment_to_class[segment] = cls
    return segment_to_class


def sample_vectors(vectors: torch.Tensor, max_samples: int = 200) -> torch.Tensor:
    n = vectors.size(0)
    if n <= max_samples:
        return vectors
    idx = torch.randperm(n)[:max_samples]
    return vectors[idx]


def mean_pairwise_distance(vectors: torch.Tensor) -> float:
    n = vectors.size(0)
    if n < 2:
        return float("nan")
    distances = torch.pdist(vectors, p=2)
    return distances.mean().item()


def mean_cross_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    distances = torch.cdist(a, b, p=2)
    return distances.mean().item()


def slug_to_readable(class_entity: str) -> str:
    if ":" in class_entity:
        class_entity = class_entity.split(":", 1)[1]
    return class_entity.replace("_", " ")


def evaluate_attributes(
    embed: torch.Tensor,
    entity2id: Mapping[str, int],
    attr_to_segments: Mapping[str, Set[str]],
    segment_to_class: Mapping[str, str],
    anomaly_classes: Set[str],
    attributes: Iterable[str],
    max_samples: int = 200,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    for attr_name in attributes:
        attr_entity = f"attribute:{attr_name}"
        segments = attr_to_segments.get(attr_entity)
        if not segments:
            results.append(
                {
                    "attribute": attr_name,
                    "num_segments": 0,
                    "num_classes": 0,
                    "details": [],
                    "intra_mean": float("nan"),
                    "cross_mean": float("nan"),
                }
            )
            continue

        class_vectors: Dict[str, torch.Tensor] = {}
        for segment in segments:
            cls = segment_to_class.get(segment)
            if cls is None:
                continue
            cls_slug = cls.split(":", 1)[1] if ":" in cls else cls
            if cls_slug not in anomaly_classes:
                continue
            ent_idx = entity2id.get(segment)
            if ent_idx is None:
                continue
            class_vectors.setdefault(cls_slug, []).append(embed[ent_idx])

        class_vectors = {
            cls: sample_vectors(torch.stack(vecs), max_samples=max_samples)
            for cls, vecs in class_vectors.items()
            if len(vecs) >= 2
        }

        if not class_vectors:
            results.append(
                {
                    "attribute": attr_name,
                    "num_segments": len(segments),
                    "num_classes": 0,
                    "details": [],
                    "intra_mean": float("nan"),
                    "cross_mean": float("nan"),
                }
            )
            continue

        intra_stats = []
        for cls, vecs in class_vectors.items():
            intra_stats.append(
                {"class": cls, "count": vecs.size(0), "intra_distance": mean_pairwise_distance(vecs)}
            )

        # Cross-class distances
        cross_distances: List[float] = []
        classes = sorted(class_vectors.keys())
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                da = class_vectors[classes[i]]
                db = class_vectors[classes[j]]
                cross_distances.append(mean_cross_distance(da, db))

        cross_mean = float("nan")
        if cross_distances:
            cross_mean = float(sum(cross_distances) / len(cross_distances))

        results.append(
            {
                "attribute": attr_name,
                "num_segments": sum(v.size(0) for v in class_vectors.values()),
                "num_classes": len(class_vectors),
                "details": intra_stats,
                "intra_mean": float(
                    sum(item["intra_distance"] for item in intra_stats if not math.isnan(item["intra_distance"]))
                    / max(
                        len([1 for item in intra_stats if not math.isnan(item["intra_distance"])]),
                        1,
                    )
                    if intra_stats
                    else float("nan")
                ),
                "cross_mean": cross_mean,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether shared anomaly attributes connect class embeddings."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("Data/UCF_Crime"))
    parser.add_argument(
        "--entity-map",
        type=Path,
        default=None,
        help="Optional path to entity2id mapping. If omitted, the mapping is reconstructed from triples (matching KG.py).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained KG checkpoint (model.pt).",
    )
    parser.add_argument(
        "--attributes",
        nargs="*",
        default=[
            "anomaly_violence",
            "anomaly_crowd_panic",
            "anomaly_escape_pursuit",
            "anomaly_property_damage",
            "anomaly_weapon",
            "anomaly_restraint_response",
            "anomaly_vulnerable_victim",
            "anomaly_fire_smoke",
        ],
        help="Attribute slugs (without the 'attribute:' prefix) to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of segment embeddings per class to include in distance calculations.",
    )
    parser.add_argument(
        "--anomaly-classes",
        nargs="*",
        default=[
            "abuse",
            "arrest",
            "arson",
            "assault",
            "burglary",
            "explosion",
            "fighting",
            "roadaccidents",
            "robbery",
            "stealing",
            "shoplifting",
            "vandalism",
            "shooting",
        ],
        help="Class slugs considered anomalous (class:<slug>).",
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    embed = load_entity_embeddings(args.checkpoint)
    if args.entity_map and args.entity_map.exists():
        entity2id = load_id_map(args.entity_map)
    else:
        entity2id = {}
        for sub, _, obj in collect_triples(args.dataset_root, ("train", "test", "val")):
            if sub not in entity2id:
                entity2id[sub] = len(entity2id)
            if obj not in entity2id:
                entity2id[obj] = len(entity2id)
    segment_to_video, video_to_class, attr_to_segments = build_graph_maps(args.dataset_root)
    segment_to_class = infer_segment_classes(segment_to_video, video_to_class)

    anomaly_classes = set(args.anomaly_classes)
    results = evaluate_attributes(
        embed,
        entity2id,
        attr_to_segments,
        segment_to_class,
        anomaly_classes,
        attributes=args.attributes,
        max_samples=args.max_samples,
    )

    for item in results:
        print(f"\nAttribute: {item['attribute']}")
        print(f"  Segment count: {item['num_segments']} | Classes: {item['num_classes']}")
        print(f"  Mean intra-class distance: {item['intra_mean']:.4f}")
        print(f"  Mean cross-class distance: {item['cross_mean']:.4f}")
        for detail in item["details"]:
            print(
                f"    - {detail['class']}: n={detail['count']}, intra={detail['intra_distance']:.4f}"
            )


if __name__ == "__main__":
    main()
