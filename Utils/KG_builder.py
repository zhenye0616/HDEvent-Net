import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower().replace("&", "and")).strip("_")


def canonical_video_entity(raw_name: str) -> str:
    token = raw_name.strip()
    if token.startswith("video:"):
        token = token[len("video:") :]
    token = token.replace("\\", "/")
    token = token.replace(":", "/")
    token = token.strip("/")
    return f"video:{token.lower()}"


def canonical_segment_entity(seg_name: str) -> str:
    return seg_name.strip().lower()


def _segment_to_video(seg_name: str) -> str:
    if not seg_name.startswith("seg:"):
        raise ValueError(f"Unexpected segment name: {seg_name}")
    payload = seg_name[len("seg:") :]
    try:
        video_path, _ = payload.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(f"Segment name missing index: {seg_name}") from exc
    try:
        _, rest = video_path.split("/", 1)
    except ValueError as exc:
        raise ValueError(f"Segment name missing split: {seg_name}") from exc
    return canonical_video_entity(f"video:{rest}")


def _load_segments_for_split(segments_root: Path, split: str) -> Dict[str, List[str]]:
    segments_by_video: Dict[str, List[str]] = {}
    split_dir = segments_root / split
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
                video_entity = _segment_to_video(seg_name)
                segments_by_video.setdefault(video_entity, []).append(
                    canonical_segment_entity(seg_name)
                )

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
                segment_to_attrs[seg_name.strip().lower()] = attrs
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
    payload = segment_name[len("seg:") :]
    parts = payload.split("/")
    if len(parts) < 4:
        return None
    return parts[3]


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
            videos.append(
                canonical_video_entity(line if line.startswith("video:") else f"video:{line}")
            )
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


def build_triple_files(
    segments_root: Path,
    manifest_root: Path,
    output_root: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    segment_attrs_root: Optional[Path] = None,
) -> None:
    segments_root = Path(segments_root)
    manifest_root = Path(manifest_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        segments_by_video = _load_segments_for_split(segments_root, split)
        manifest_videos = _load_manifest_videos(manifest_root, split)

        all_videos = sorted(set(segments_by_video.keys()) | set(manifest_videos))
        if not all_videos:
            continue

        lines: List[str] = []
        attrs_lookup: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        if segment_attrs_root is not None:
            attrs_dir = Path(segment_attrs_root) / split
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
            if not segments:
                continue
            attr_records: Dict[str, List[Dict[str, float]]] = {}
            if segment_attrs_root is not None:
                attr_records = attrs_lookup.get(video_entity, {})
            attr_class_cache: Optional[str] = None
            for triple in _iter_structure_triples(video_entity, segments):
                sub, rel, obj = (triple[0].lower(), triple[1].lower(), triple[2].lower())
                lines.append("\t".join((sub, rel, obj)))
            if attr_records:
                for segment in segments:
                    if segment in attr_records:
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
                            lines.append("\t".join((segment, "has_attribute", attr_entity)))

        out_path = output_root / f"{split}.txt"
        with out_path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")


def _build_triples_cli(args: argparse.Namespace) -> None:
    build_triple_files(
        segments_root=args.segments_root,
        manifest_root=args.manifest_root,
        output_root=args.output_root,
        splits=args.splits,
        segment_attrs_root=args.segment_attrs_root,
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

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_arguments()
    if args.command == "build-triples":
        _build_triples_cli(args)
