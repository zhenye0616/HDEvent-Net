import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower().replace("&", "and")).strip("_")


def _to_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _split_video_token(raw: str) -> List[str]:
    token = raw.strip()
    if token.startswith("video:"):
        token = token[len("video:") :]
    token = token.replace("\\", "/").replace(":", "/").strip("/")
    return [part for part in token.split("/") if part]


def _split_segment_token(seg_name: str) -> Tuple[str, List[str], Optional[str]]:
    token = seg_name.strip()
    if not token.startswith("seg:"):
        raise ValueError(f"Unexpected segment name: {seg_name}")
    payload = token[len("seg:") :].replace("\\", "/").strip("/")
    if ":" in payload:
        path_part, index_part = payload.rsplit(":", 1)
    else:
        path_part, index_part = payload, None
    parts = [part for part in path_part.split("/") if part]
    if not parts:
        raise ValueError(f"Malformed segment token: {seg_name}")
    split_token = parts[0].lower()
    components = parts[1:]
    return split_token, components, index_part


def _canonicalize_components(components: List[str]) -> Tuple[str, Optional[str]]:
    if not components:
        raise ValueError("Cannot canonicalize empty components")
    cleaned = [comp.strip() for comp in components if comp.strip()]
    if not cleaned:
        raise ValueError("Cannot canonicalize empty components")

    variant = cleaned[0].lower()
    event = cleaned[1].lower() if len(cleaned) > 1 else None
    class_name = cleaned[2] if len(cleaned) > 2 else None
    remainder = cleaned[3:] if len(cleaned) > 3 else []

    base_parts = [variant]
    if event:
        base_parts.append(event)
    base_parts.extend(part.lower() for part in remainder if part)
    base_slug = "/".join(base_parts)
    if not base_slug:
        raise ValueError("Failed to derive canonical video slug")

    class_slug = _slugify(class_name) if class_name else None
    return base_slug, class_slug


def canonical_video_and_class(raw_name: str) -> Tuple[str, Optional[str]]:
    parts = _split_video_token(raw_name)
    base_slug, class_slug = _canonicalize_components(parts)
    return f"video:{base_slug}", class_slug


def canonical_video_entity(raw_name: str) -> str:
    return canonical_video_and_class(raw_name)[0]


def _canonical_segment_entity(
    base_slug: str, start_ms: Optional[int], end_ms: Optional[int], index: Optional[str]
) -> str:
    if start_ms is not None and end_ms is not None:
        return f"seg:{base_slug}:{start_ms:06d}-{end_ms:06d}"
    fallback = index.strip().lower() if index else "idx0"
    return f"seg:{base_slug}:{fallback}"


def canonical_segment_entity(seg_name: str, *, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> str:
    _, components, index = _split_segment_token(seg_name)
    base_slug, _ = _canonicalize_components(components)
    return _canonical_segment_entity(base_slug, start_ms, end_ms, index)


def _load_segments_for_split(
    segments_root: Path, split: str
) -> Tuple[Dict[str, List[str]], Dict[str, Optional[str]]]:
    segments_by_video: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    video_classes: Dict[str, Optional[str]] = {}
    split_key = split.lower()
    split_dir = segments_root / split_key
    if not split_dir.exists():
        return {}, {}

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
                segment_split, components, index_token = _split_segment_token(seg_name)
                if segment_split != split_key:
                    raise ValueError(
                        f"Segment split mismatch for {seg_name}: expected {split_key}, found {segment_split}"
                    )
                base_slug, class_slug = _canonicalize_components(components)
                video_entity = f"video:{base_slug}"
                start_ms = _to_int(record.get("start_ms"))
                end_ms = _to_int(record.get("end_ms"))
                segment_entity = _canonical_segment_entity(base_slug, start_ms, end_ms, index_token)

                segments_by_video.setdefault(video_entity, []).append((segment_entity, start_ms))
                if class_slug:
                    prev = video_classes.get(video_entity)
                    if prev and prev != class_slug:
                        raise ValueError(
                            f"Conflicting class assignments for {video_entity}: {prev} vs {class_slug}"
                        )
                    video_classes[video_entity] = class_slug

    ordered_segments: Dict[str, List[str]] = {}
    for video_entity, seg_list in segments_by_video.items():
        seg_list.sort(key=lambda item: (item[1] if item[1] is not None else float("inf"), item[0]))
        ordered_segments[video_entity] = [seg for seg, _ in seg_list]

    return ordered_segments, video_classes


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
            if not seg_name:
                continue
            try:
                _, components, index_token = _split_segment_token(seg_name)
            except ValueError:
                continue
            base_slug, _ = _canonicalize_components(components)
            start_ms = _to_int(record.get("start_ms"))
            end_ms = _to_int(record.get("end_ms"))
            segment_entity = _canonical_segment_entity(base_slug, start_ms, end_ms, index_token)
            segment_to_attrs[segment_entity] = attrs
    return segment_to_attrs


def sanitize_video_filename(video_entity: str) -> str:
    return (
        video_entity.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _load_manifest_videos(manifest_root: Path, split: str) -> Dict[str, Optional[str]]:
    path = manifest_root / f"{split}.txt"
    if not path.exists():
        return {}
    videos: Dict[str, Optional[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            if token.startswith("seg:"):
                continue
            normalized = token if token.startswith("video:") else f"video:{token}"
            try:
                video_entity, class_slug = canonical_video_and_class(normalized)
            except ValueError:
                continue
            if video_entity in videos:
                if videos[video_entity] and class_slug and videos[video_entity] != class_slug:
                    continue
            else:
                videos[video_entity] = class_slug
    return videos


def _iter_structure_triples(
    video_entity: str,
    class_slug: Optional[str],
    segments: List[str],
) -> Iterator[Tuple[str, str, str]]:
    if class_slug:
        yield (video_entity, "class_of", f"class:{class_slug}")

    for seg in segments:
        yield (seg, "part_of", video_entity)


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
    split_triples: Set[Tuple[str, str, str]] = set()
    attribute_triples: List[Tuple[str, str, str]] = []
    video_splits: Dict[str, Set[str]] = {}
    segment_splits: Dict[str, Set[str]] = {}
    video_classes: Dict[str, Optional[str]] = {}

    for split in splits:
        split_key = split.lower()
        split_segments, split_video_classes = _load_segments_for_split(segments_root, split_key)
        split_manifest_videos = _load_manifest_videos(manifest_root, split_key)

        for video_entity, class_slug in split_video_classes.items():
            prev = video_classes.get(video_entity)
            if prev and class_slug and prev != class_slug:
                raise ValueError(
                    f"Conflicting class assignments for {video_entity}: {prev} vs {class_slug} (segments)"
                )
            if class_slug:
                video_classes[video_entity] = class_slug

        for video_entity, class_slug in split_manifest_videos.items():
            if class_slug:
                prev = video_classes.get(video_entity)
                if prev and prev != class_slug:
                    continue
                video_classes.setdefault(video_entity, class_slug)

        for video_entity in split_segments:
            video_splits.setdefault(video_entity, set()).add(split_key)
            for segment in split_segments[video_entity]:
                segment_splits.setdefault(segment, set()).add(split_key)

        for video_entity in split_manifest_videos:
            video_splits.setdefault(video_entity, set()).add(split_key)

        all_videos = sorted(set(split_segments.keys()) | set(split_manifest_videos.keys()))
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
                        canonical_name, _ = canonical_video_and_class(video_name)
                    except Exception:
                        continue
                    attrs_lookup[canonical_name] = records

        for video_entity in all_videos:
            segments = split_segments.get(video_entity, [])
            if not segments:
                continue
            class_slug = video_classes.get(video_entity)
            attr_records: Dict[str, List[Dict[str, float]]] = {}
            if segment_attrs_root is not None:
                attr_records = attrs_lookup.get(video_entity, {})
            for triple in _iter_structure_triples(video_entity, class_slug, segments):
                sub, rel, obj = (triple[0].lower(), triple[1].lower(), triple[2].lower())
                structural_triples.add((sub, rel, obj))
            for segment in segments:
                if attr_records and segment in attr_records:
                    for attr in attr_records[segment]:
                        attr_name = attr.get("name")
                        if not attr_name:
                            continue
                        attr_slug = _slugify(attr_name)
                        attr_entity = f"attribute:{attr_slug}"
                        attribute_triples.append(
                            (segment.lower(), "has_attribute", attr_entity.lower())
                        )

    for video_entity, splits_membership in video_splits.items():
        for split_name in sorted(splits_membership):
            split_triples.add((video_entity, "in_split", f"split:{split_name}"))

    for segment_entity, splits_membership in segment_splits.items():
        for split_name in sorted(splits_membership):
            split_triples.add((segment_entity, "in_split", f"split:{split_name}"))

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

    if split_triples:
        provenance_path = output_root / "split_provenance.txt"
        with provenance_path.open("w", encoding="utf-8") as handle:
            for sub, rel, obj in sorted(split_triples):
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
