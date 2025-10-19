import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


###############################################################################
# Helpers
###############################################################################


def setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("build_kg_indices")


def slugify(text: str) -> str:
    slug = text.strip().lower()
    slug = slug.replace("&", "and")
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def load_attributes(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"attributes file must be a dict: {path}")
    return data


def iter_manifests(manifest_root: Path, splits: Iterable[str]) -> Iterable[str]:
    for split in splits:
        manifest = manifest_root / f"{split}.txt"
        if not manifest.exists():
            continue
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


@dataclass(frozen=True)
class SegmentRecord:
    seg_name: str
    split: Optional[str]
    video_name: str


def _manifest_to_video_token(raw_name: str) -> Optional[str]:
    """
    Extract a canonical video token from a manifest line.

    Manifest entries may come from older scripts that emit tab-delimited rows:
    class<TAB>video_id<TAB>aug_idx<TAB>/abs/path/to/file.npy
    We prefer to recover the on-disk path because it encodes the full hierarchy.
    """
    if not raw_name:
        return None

    if "\t" not in raw_name:
        return raw_name.strip()

    fields = [field.strip() for field in raw_name.split("\t")]
    # Prefer the file path column if it exists.
    if fields and fields[-1]:
        path_str = fields[-1]
        try:
            path = Path(path_str)
            stem = path.stem
            parts = path.parts
            # Expect .../<root>/<split>/<class>/<video>.npy; guard for short paths.
            if len(parts) >= 4:
                root, split, cls = parts[-4], parts[-3], parts[-2]
                return f"{root}/{split}/{cls}/{stem}"
            # Fallback: drop any leading slash and join remaining pieces.
            return "/".join(part for part in parts if part)
        except Exception:
            return path_str

    class_name = fields[0] if len(fields) > 0 else ""
    video_id = fields[1] if len(fields) > 1 else ""
    aug_idx = fields[2] if len(fields) > 2 else ""
    if aug_idx:
        video_id = f"{video_id}__{aug_idx}"
    tokens = [token for token in (class_name, video_id) if token]
    return "/".join(tokens) if tokens else None


def canonical_video_entity(raw: str) -> str:
    token = raw.strip()
    if token.startswith("video:"):
        token = token[len("video:") :]
    token = token.replace("\\", "/")
    token = token.replace(":", "/")
    token = token.strip("/")
    parts = token.split("/")
    if parts and parts[0].lower() in {"train", "val", "test"}:
        token = "/".join(parts[1:])
    return f"video:{token.lower()}"


def parse_segment_name(seg_name: str) -> SegmentRecord:
    """
    Parse a raw segment token into canonical identifiers.
    """
    if not seg_name.startswith("seg:"):
        raise ValueError(f"Unexpected segment name: {seg_name}")
    payload = seg_name[len("seg:") :].replace("\\", "/").strip("/")
    try:
        video_path_with_split, seg_index = payload.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(f"Segment name missing index: {seg_name}") from exc

    video_path_with_split = video_path_with_split.strip("/")
    split_token: Optional[str] = None
    video_path = video_path_with_split
    if "/" in video_path_with_split:
        candidate_split, remainder = video_path_with_split.split("/", 1)
        candidate_lower = candidate_split.lower()
        if candidate_lower in {"train", "val", "test"}:
            split_token = candidate_lower
            video_path = remainder
        else:
            video_path = video_path_with_split
    video_name = canonical_video_entity(f"video:{video_path}")
    segment_suffix = video_name[len("video:") :]
    seg_index = seg_index.strip().lower()
    canonical_segment = f"seg:{segment_suffix}:{seg_index}"

    return SegmentRecord(seg_name=canonical_segment, split=split_token, video_name=video_name)


def collect_segments(segments_root: Path) -> Tuple[Set[str], Set[str]]:
    """
    Returns (video_entities, segment_entities) discovered under segments_root.
    """
    video_entities: Set[str] = set()
    segment_entities: Set[str] = set()

    if not segments_root.exists():
        return video_entities, segment_entities

    for split_dir in sorted(p for p in segments_root.iterdir() if p.is_dir()):
        for jsonl_path in sorted(split_dir.glob("*.jsonl")):
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    seg_name = record.get("seg_name")
                    if not seg_name:
                        continue
                    parsed = parse_segment_name(seg_name)
                    video_entities.add(parsed.video_name)
                    segment_entities.add(parsed.seg_name)

    return video_entities, segment_entities


###############################################################################
# Main logic
###############################################################################


def build_entity_order(
    attributes_path: Path,
    segments_root: Path,
    manifest_root: Path,
    splits: Iterable[str],
) -> List[str]:
    attributes = load_attributes(attributes_path)

    class_entities: List[str] = []
    attribute_entities: List[str] = []

    for class_name in sorted(attributes.keys(), key=str.lower):
        class_slug = slugify(class_name)
        class_entities.append(f"class:{class_slug}")
        for attr in attributes[class_name]:
            attr_slug = slugify(attr)
            attribute_entities.append(f"attribute:{class_slug}:{attr_slug}")

    # Video entities: prefer those referenced in manifests, fall back to segment discovery.
    manifest_videos: Set[str] = set()
    for raw_name in iter_manifests(manifest_root, splits):
        token = _manifest_to_video_token(raw_name)
        if not token:
            continue
        if token.startswith("video:"):
            manifest_videos.add(canonical_video_entity(token))
        else:
            manifest_videos.add(canonical_video_entity(f"video:{token}"))

    segment_videos, segment_entities = collect_segments(segments_root)

    all_videos = sorted(manifest_videos | segment_videos)
    all_segments = sorted(segment_entities)

    entity_order: List[str] = []
    entity_order.extend(class_entities)
    entity_order.extend(attribute_entities)
    entity_order.extend(all_videos)
    entity_order.extend(all_segments)

    return entity_order


def build_relation_order(relations: Iterable[str]) -> List[str]:
    base = list(relations)
    reverse = [f"{rel}_reverse" for rel in base]
    return base + reverse


def write_mapping(entries: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, name in enumerate(entries):
            f.write(f"{name}\t{idx}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic KG ID files.")
    parser.add_argument(
        "--attributes",
        type=Path,
        default=Path("Data/attributes.json"),
        help="Path to attributes.json file.",
    )
    parser.add_argument(
        "--segments-root",
        type=Path,
        default=Path("Data/segments"),
        help="Root directory containing segment JSONLs.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("Data/UCF_Crime"),
        help="Directory with train/val/test manifests.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=("train", "val", "test"),
        help="Splits to include when collecting videos.",
    )
    parser.add_argument(
        "--entity-output",
        type=Path,
        default=Path("Data/UCF_Crime/entity2id.txt"),
        help="Output path for entity2id mapping.",
    )
    parser.add_argument(
        "--relation-output",
        type=Path,
        default=Path("Data/UCF_Crime/relation2id.txt"),
        help="Output path for relation2id mapping.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.verbose)

    logger.info("Collecting entities from attributes, manifests, and segments...")
    entity_entries = build_entity_order(
        attributes_path=args.attributes,
        segments_root=args.segments_root,
        manifest_root=args.manifest_root,
        splits=args.splits,
    )
    logger.info("Total entities collected: %d", len(entity_entries))
    write_mapping(entity_entries, args.entity_output)
    logger.info("Wrote entity mapping to %s", args.entity_output)

    relations = ["has_attribute", "part_of", "precedes", "class_of", "in_split"]
    relation_entries = build_relation_order(relations)
    write_mapping(relation_entries, args.relation_output)
    logger.info("Wrote relation mapping to %s", args.relation_output)


if __name__ == "__main__":
    main()
