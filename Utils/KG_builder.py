import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower().replace("&", "and")).strip("_")


def canonical_video_entity(raw_name: str) -> str:
    token = raw_name.strip()
    if token.startswith("video:"):
        token = token[len("video:") :]
    token = token.replace("\\", "/")
    token = token.replace(":", "/")
    token = token.strip("/")
    parts = token.split("/")
    if parts and parts[0].lower() in {"train", "val", "test"}:
        token = "/".join(parts[1:])
    return f"video:{token.lower()}"


def _parse_segment_name(seg_name: str, *, require_split: bool = False) -> Tuple[str, str, Optional[str]]:
    """
    Canonicalize a raw segment token.

    Returns
    -------
    segment_entity: str
        Canonical segment identifier (no split prefix).
    video_entity: str
        Parent video entity identifier.
    split: str
        Split token (train/val/test/...).
    """
    token = seg_name.strip()
    if not token.startswith("seg:"):
        raise ValueError(f"Unexpected segment name: {seg_name}")

    payload = token[len("seg:") :].replace("\\", "/").strip("/")
    try:
        video_path_with_split, seg_index = payload.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(f"Segment name missing index: {seg_name}") from exc

    video_path_with_split = video_path_with_split.strip("/")
    split_token: Optional[str]
    if "/" not in video_path_with_split:
        if require_split:
            raise ValueError(f"Segment name missing split: {seg_name}")
        split_token = None
        video_path = video_path_with_split
    else:
        candidate_split, remainder = video_path_with_split.split("/", 1)
        candidate_lower = candidate_split.lower()
        if candidate_lower in {"train", "val", "test"}:
            split_token = candidate_lower
            video_path = remainder
        elif require_split:
            raise ValueError(f"Segment name missing split: {seg_name}")
        else:
            split_token = None
            video_path = video_path_with_split
    video_entity = canonical_video_entity(f"video:{video_path}")
    segment_suffix = video_entity[len("video:") :]
    seg_index = seg_index.strip().lower()
    segment_entity = f"seg:{segment_suffix}:{seg_index}"
    return segment_entity, video_entity, split_token


def canonical_segment_entity(seg_name: str) -> str:
    segment_entity, _, _ = _parse_segment_name(seg_name)
    return segment_entity


def _segment_to_video(seg_name: str) -> str:
    _, video_entity, _ = _parse_segment_name(seg_name)
    return video_entity


def _load_segments_for_split(segments_root: Path, split: str) -> Dict[str, List[str]]:
    segments_by_video: Dict[str, List[str]] = {}
    split_key = split.lower()
    split_dir = segments_root / split_key
    if not split_dir.exists():
        return segments_by_video

    for jsonl_path in sorted(split_dir.glob("*.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                seg_name = record.get("seg_name")
                if not seg_name:
                    continue
                segment_entity, video_entity, segment_split = _parse_segment_name(
                    seg_name, require_split=True
                )
                if segment_split != split_key:
                    raise ValueError(
                        f"Segment split mismatch for {seg_name}: expected {split_key}, found {segment_split}"
                    )
                segments_by_video.setdefault(video_entity, []).append(segment_entity)

    for video_entity, seg_list in segments_by_video.items():
        seg_list.sort()

    return segments_by_video


def load_segment_attributes(path: Path) -> Dict[str, List[Dict[str, float]]]:
    segment_to_attrs: Dict[str, List[Dict[str, float]]] = {}
    if not path.exists():
        return segment_to_attrs

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            seg_name = record.get("segment")
            attrs = record.get("top_attributes", [])
            if seg_name:
                try:
                    segment_entity, _, _ = _parse_segment_name(seg_name)
                except ValueError:
                    continue
                segment_to_attrs[segment_entity] = attrs
    return segment_to_attrs


def sanitize_video_filename(video_entity: str) -> str:
    return (
        video_entity.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def segment_class_from_name(segment_name: str) -> Optional[str]:
    if not segment_name.startswith("seg:"):
        return None
    try:
        _, video_entity, _ = _parse_segment_name(segment_name)
    except ValueError:
        return None
    video_payload = video_entity[len("video:") :]
    parts = video_payload.split("/")
    if len(parts) < 3:
        return None
    return parts[2]


def _load_manifest_videos(manifest_root: Path, split: str) -> List[str]:
    path = manifest_root / f"{split}.txt"
    if not path.exists():
        return []
    videos: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            if token.startswith("seg:"):
                continue
            normalized = token if token.startswith("video:") else f"video:{token}"
            videos.append(canonical_video_entity(normalized))
    return videos


def _infer_class_entity(video_entity: str) -> str:
    token = video_entity[len("video:") :]
    parts = token.split("/")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer class from video entity: {video_entity}")
    class_slug = _slugify(parts[2])
    return f"class:{class_slug}"


def _iter_structure_triples(
    video_entity: str,
    segments: List[str],
) -> Iterator[Tuple[str, str, str]]:
    class_entity = _infer_class_entity(video_entity)
    yield (video_entity, "class_of", class_entity)

    for seg in segments:
        yield (seg, "part_of", video_entity)

    for prev_seg, next_seg in zip(segments, segments[1:]):
        yield (prev_seg, "precedes", next_seg)


def _split_attribute_triples(
    triples: List[Tuple[str, str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Split attribute triples into train/val/test while ensuring every segment keeps at least one
    attribute edge in the training split to anchor its embedding.
    """
    total = len(triples)
    if total == 0:
        return [], [], []

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Attribute split ratios must sum to 1.0")

    rng = random.Random(seed)

    mandatory_train: Dict[str, Tuple[str, str, str]] = {}
    remaining: List[Tuple[str, str, str]] = []
    for triple in triples:
        seg = triple[0]
        if seg not in mandatory_train:
            mandatory_train[seg] = triple
        else:
            remaining.append(triple)

    train_triples: List[Tuple[str, str, str]] = list(mandatory_train.values())
    rng.shuffle(remaining)

    target_train = max(len(train_triples), int(round(train_ratio * total)))
    target_train = min(total, target_train)

    remaining_capacity = total - target_train
    target_val = int(round(val_ratio * total))
    target_val = min(remaining_capacity, target_val)

    extra_train_needed = max(0, target_train - len(train_triples))
    extra_train_take = min(len(remaining), extra_train_needed)
    if extra_train_take:
        train_triples.extend(remaining[:extra_train_take])
        remaining = remaining[extra_train_take:]

    val_triples = remaining[:target_val]
    remaining = remaining[target_val:]

    test_triples = remaining

    return train_triples, val_triples, test_triples


def build_triple_files(
    segments_root: Path,
    manifest_root: Path,
    output_root: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    segment_attrs_root: Optional[Path] = None,
    attr_train_ratio: float = 0.8,
    attr_val_ratio: float = 0.1,
    attr_test_ratio: float = 0.1,
    split_seed: int = 42,
) -> None:
    segments_root = Path(segments_root)
    manifest_root = Path(manifest_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    structural_triples: Set[Tuple[str, str, str]] = set()
    attribute_triples: List[Tuple[str, str, str]] = []
    video_splits: Dict[str, Set[str]] = {}
    segment_splits: Dict[str, Set[str]] = {}

    for split in splits:
        split_key = split.lower()
        segments_by_video = _load_segments_for_split(segments_root, split_key)
        manifest_videos = _load_manifest_videos(manifest_root, split_key)

        for video_entity in manifest_videos:
            video_splits.setdefault(video_entity, set()).add(split_key)

        all_videos = sorted(set(segments_by_video.keys()) | set(manifest_videos))
        if not all_videos:
            continue

        attrs_lookup: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        if segment_attrs_root is not None:
            attrs_dir = Path(segment_attrs_root) / split_key
            if attrs_dir.exists():
                for attr_path in attrs_dir.glob("*.jsonl"):
                    records = load_segment_attributes(attr_path)
                    if not records:
                        continue
                    with attr_path.open("r", encoding="utf-8") as handle:
                        first_line = handle.readline().strip()
                        if not first_line:
                            continue
                        try:
                            payload = json.loads(first_line)
                        except json.JSONDecodeError:
                            continue
                    video_name = payload.get("video")
                    if not video_name:
                        continue
                    try:
                        canonical_name = canonical_video_entity(video_name)
                    except Exception:
                        continue
                    attrs_lookup[canonical_name] = records
        for video_entity in all_videos:
            segments = segments_by_video.get(video_entity, [])
            video_splits.setdefault(video_entity, set()).add(split_key)
            if not segments:
                continue
            attr_records: Dict[str, List[Dict[str, float]]] = {}
            if segment_attrs_root is not None:
                attr_records = attrs_lookup.get(video_entity, {})
            attr_class_cache: Optional[str] = None
            for triple in _iter_structure_triples(video_entity, segments):
                sub, rel, obj = (triple[0].lower(), triple[1].lower(), triple[2].lower())
                structural_triples.add((sub, rel, obj))
            for segment in segments:
                segment_splits.setdefault(segment, set()).add(split_key)
                if attr_records and segment in attr_records:
                    if attr_class_cache is None:
                        class_name = segment_class_from_name(segment)
                        attr_class_cache = _slugify(class_name) if class_name else None
                    class_slug = attr_class_cache
                    if not class_slug:
                        continue
                    for attr in attr_records[segment]:
                        attr_name = attr.get("name")
                        if not attr_name:
                            continue
                        attr_slug = _slugify(attr_name)
                        attr_entity = f"attribute:{class_slug}:{attr_slug}"
                        attribute_triples.append(
                            (segment.lower(), "has_attribute", attr_entity.lower())
                        )

    for video_entity, splits_membership in video_splits.items():
        for split_name in sorted(splits_membership):
            structural_triples.add((video_entity, "in_split", f"split:{split_name}"))

    for segment_entity, splits_membership in segment_splits.items():
        for split_name in sorted(splits_membership):
            structural_triples.add((segment_entity, "in_split", f"split:{split_name}"))

    train_attr, val_attr, test_attr = _split_attribute_triples(
        attribute_triples,
        attr_train_ratio,
        attr_val_ratio,
        attr_test_ratio,
        split_seed,
    )

    structural_trip_list = sorted(structural_triples)
    train_attr_sorted = sorted(train_attr)
    val_attr_sorted = sorted(val_attr)
    test_attr_sorted = sorted(test_attr)

    train_lines = structural_trip_list + train_attr_sorted
    val_lines = val_attr_sorted
    test_lines = test_attr_sorted

    for split_name, split_lines in (("train", train_lines), ("val", val_lines), ("test", test_lines)):
        out_path = output_root / f"{split_name}.txt"
        with out_path.open("w", encoding="utf-8") as handle:
            for sub, rel, obj in split_lines:
                handle.write(f"{sub}\t{rel}\t{obj}\n")


def _build_triples_cli(args: argparse.Namespace) -> None:
    build_triple_files(
        segments_root=args.segments_root,
        manifest_root=args.manifest_root,
        output_root=args.output_root,
        splits=args.splits,
        segment_attrs_root=args.segment_attrs_root,
        attr_train_ratio=args.attr_split[0],
        attr_val_ratio=args.attr_split[1],
        attr_test_ratio=args.attr_split[2],
        split_seed=args.split_seed,
    )


def _parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KG builder utilities")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser(
        "build-triples", help="Generate structural triples from precomputed segments."
    )
    build_parser.add_argument(
        "--segments-root",
        type=Path,
        default=Path("HDEvent-Net/Data/segments"),
        help="Directory containing segment JSONL files.",
    )
    build_parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("HDEvent-Net/Data/UCF_Crime"),
        help="Directory with split manifests (.txt).",
    )
    build_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("HDEvent-Net/Data/UCF_Crime"),
        help="Directory to write split triple files.",
    )
    build_parser.add_argument(
        "--segment-attrs-root",
        type=Path,
        default=None,
        help="Optional directory with per-segment attribute JSONLs",
    )
    build_parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "val", "test"),
        help="Dataset splits to process.",
    )
    build_parser.add_argument(
        "--attr-split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Ratios for splitting has_attribute triples into train/val/test.",
    )
    build_parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for attribute triple splitting.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_arguments()
    if args.command == "build-triples":
        _build_triples_cli(args)
