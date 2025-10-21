import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import numpy as np


@dataclass
class Segment:
    """Container for precomputed segment metadata."""

    seg_name: str
    start_ms: int
    end_ms: int
    n_events: int
    start_idx: int
    end_idx: int
    mid_ms: int
    duration_ms: int
    seg_idx: int
    flags: List[str] = field(default_factory=list)
    video_id: str = ""
    feature_mean: Optional[float] = None
    feature_std: Optional[float] = None
    feature_min: Optional[float] = None
    feature_max: Optional[float] = None


def setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("precompute_segments")


class EventStream(Dict[str, np.ndarray]):
    """Thin wrapper for loaded event features."""


def _parse_video_reference(video_id: str) -> Dict[str, Optional[str]]:
    """
    Break the manifest video identifier into components if possible.
    Supports strings like:
      - "video:vitb:event_thr_10:abuse001_x264"
      - "vitb:event_thr_10:Abuse/Abuse001_x264"
      - "Abuse/Abuse001_x264__0"
    """
    raw = video_id.strip()
    if not raw:
        raise ValueError("Empty video identifier received.")

    result: Dict[str, Optional[str]] = {
        "raw": raw,
        "variant": None,
        "feature_split": None,
        "path_hint": None,
        "base_name": None,
    }

    token = raw
    if token.startswith("video:"):
        token = token[len("video:") :]

    parts: List[str] = token.split(":")
    if len(parts) >= 3:
        result["variant"] = parts[0]
        if parts[1].startswith("event_") or parts[1] in {"rgb"}:
            result["feature_split"] = parts[1]
            remainder = parts[2:]
        else:
            remainder = parts[1:]
    else:
        remainder = parts

    if remainder:
        result["path_hint"] = "/".join(remainder)
        result["base_name"] = remainder[-1]
    return result


_PATH_INDEX: Dict[Path, Dict[str, List[Path]]] = {}


def _build_path_index(root: Path) -> Dict[str, List[Path]]:
    """
    Build (and cache) a map from lowercase file stem to candidate paths under root.
    """
    root = root.resolve()
    if root in _PATH_INDEX:
        return _PATH_INDEX[root]

    index: Dict[str, List[Path]] = {}
    if not root.exists():
        _PATH_INDEX[root] = index
        return index

    for suffix in ("*.npy", "*.npz"):
        for path in root.rglob(suffix):
            stem = path.stem.lower()
            index.setdefault(stem, []).append(path)
    _PATH_INDEX[root] = index
    return index


def _resolve_feature_path(
    video_id: str,
    data_root: Path,
    feature_split: Optional[str],
    logger: logging.Logger,
) -> Path:
    """
    Resolve the .npy/.npz feature path corresponding to the manifest identifier.
    Tries exact matches first, then falls back to indexed search.
    """
    info = _parse_video_reference(video_id)
    search_root = data_root
    if feature_split is None and info["feature_split"]:
        feature_split = info["feature_split"]
    if feature_split:
        search_root = search_root / feature_split

    path_hint = info["path_hint"]
    base_name = info["base_name"]

    candidates: List[Path] = []

    # Direct path hint (with optional extension).
    if path_hint:
        hint_path = search_root / path_hint
        if hint_path.exists():
            candidates.append(hint_path)
        else:
            for ext in (".npy", ".npz"):
                candidate = hint_path.with_suffix(ext)
                if candidate.exists():
                    candidates.append(candidate)

    # Indexed search by base name.
    if not candidates and base_name:
        index = _build_path_index(search_root)
        base_lower = base_name.lower()
        exact = index.get(base_lower, [])
        if exact:
            candidates.extend(exact)
        else:
            # Allow augmentation suffixes (e.g., "__0")
            for stem, paths in index.items():
                if stem.startswith(base_lower + "__"):
                    candidates.extend(paths)

    if not candidates:
        msg = (
            f"Could not locate feature file for '{video_id}'. "
            f"Searched under {search_root}."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Deterministic order: choose the lexicographically smallest resolved path.
    resolved = sorted({c.resolve() for c in candidates})[0]
    return resolved


def load_event_stream(
    video_id: str,
    split: str,
    data_root: Path,
    logger: logging.Logger,
    *,
    feature_split: Optional[str] = None,
    sample_period_ms: float = 1.0,
) -> EventStream:
    """
    Load the temporal event features for a given video.

    Returns a dictionary with:
      - 'features': np.ndarray [T, d]
      - 'timestamps_ms': np.ndarray [T] (monotonic, int64)
      - 'path': str (resolved feature file path)
    """
    path = _resolve_feature_path(video_id, data_root, feature_split, logger)

    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "features" in data:
            features = data["features"]
        else:
            raise KeyError(f"'features' array missing in {path}")
        timestamps = data.get("timestamps")
        if timestamps is None:
            num_events = features.shape[0]
            timestamps = np.arange(num_events, dtype=np.int64) * sample_period_ms
    else:
        features = data
        num_events = features.shape[0] if features.ndim >= 1 else 0
        timestamps = np.arange(num_events, dtype=np.int64) * sample_period_ms

    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[None, :]

    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) != features.shape[0]:
        raise ValueError(
            f"Mismatched timestamps/features in {path}: "
            f"{len(timestamps)} timestamps vs {features.shape[0]} features"
        )

    timestamps_ms = np.round(timestamps).astype(np.int64)
    return EventStream(
        features=features,
        timestamps_ms=timestamps_ms,
        path=str(path),
        split=split,
        video_id=video_id,
    )


def adaptive_segment(
    events: EventStream,
    target_events: int,
    dt_min_ms: int,
    dt_max_ms: int,
    overlap: float,
    name_factory: Callable[[int, int, int], str],
) -> Iterator[Segment]:
    'Adaptive segmentation generator. Returns Segment objects.'
    timestamps = np.asarray(events["timestamps_ms"], dtype=np.int64)
    if timestamps.size == 0:
        return

    overlap = max(0.0, min(0.99, overlap))  # prevent zero stride
    total_events = int(timestamps.size)

    seg_idx = 0
    start_idx = 0

    while start_idx < total_events:
        seg_flags: List[str] = []
        end_idx = start_idx
        start_ts = int(timestamps[start_idx])

        while end_idx + 1 < total_events:
            candidate_idx = end_idx + 1
            dt = int(timestamps[candidate_idx] - start_ts)
            count = candidate_idx - start_idx + 1

            if ((count >= target_events and dt >= dt_min_ms) or dt >= dt_max_ms):
                end_idx = candidate_idx
                break
            end_idx = candidate_idx

        end_ts = int(timestamps[end_idx])
        count = end_idx - start_idx + 1
        dt = end_ts - start_ts

        if count < target_events:
            seg_flags.append("short_events")
        if dt < dt_min_ms:
            seg_flags.append("short_duration")
        if dt > dt_max_ms:
            seg_flags.append("overflow_duration")
        if end_idx == total_events - 1 and seg_flags:
            seg_flags = [flag for flag in seg_flags if flag != "overflow_duration"]
            if "short_tail" not in seg_flags:
                seg_flags.append("short_tail")

        seg_name = name_factory(seg_idx, start_idx, end_idx)
        duration_ms = max(0, dt)
        mid_ts = start_ts + (duration_ms // 2)

        feature_mean: Optional[float] = None
        feature_std: Optional[float] = None
        feature_min: Optional[float] = None
        feature_max: Optional[float] = None
        features = events.get("features")
        if features is not None and end_idx >= start_idx:
            window = features[start_idx : end_idx + 1]
            if getattr(window, "size", 0):
                flat = window.reshape(-1)
                feature_mean = float(flat.mean())
                feature_std = float(flat.std())
                feature_min = float(flat.min())
                feature_max = float(flat.max())

        yield Segment(
            seg_name=seg_name,
            start_ms=start_ts,
            end_ms=end_ts,
            n_events=count,
            start_idx=start_idx,
            end_idx=end_idx,
            mid_ms=mid_ts,
            duration_ms=duration_ms,
            seg_idx=seg_idx,
            flags=seg_flags,
            feature_mean=feature_mean,
            feature_std=feature_std,
            feature_min=feature_min,
            feature_max=feature_max,
        )

        seg_idx += 1

        if end_idx <= start_idx:
            start_idx += 1
            continue

        stride = int(round(count * (1.0 - overlap)))
        stride = max(1, stride)
        start_idx = start_idx + stride


def _canonical_video_id(video_id: str) -> str:
    token = video_id.strip()
    if token.startswith("video:"):
        token = token[len("video:") :]
    return token.replace(":", "/")


def write_segments(
    split: str,
    video_id: str,
    segments: List[Segment],
    output_root: Path,
) -> None:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    safe_id = (
        video_id.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )
    out_path = split_dir / f"{safe_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for segment in segments:
            f.write(json.dumps(asdict(segment)) + "\n")


def parse_manifest(manifest_path: Path) -> List[str]:
    with manifest_path.open("r", encoding="utf-8") as f:
        video_ids = [line.strip() for line in f if line.strip()]
    return video_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute event segments.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Which dataset split(s) to process.",
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=Path("HDEvent-Net/Data/UCF_Crime"),
        help="Directory containing train/val/test manifest files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory with raw event streams.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("HDEvent-Net/Data/segments"),
        help="Directory to write segment JSONL files.",
    )
    parser.add_argument(
        "--feature-split",
        type=str,
        default=None,
        help="Optional subdirectory under data_root for feature files (e.g., event_thr_10).",
    )
    parser.add_argument("--target-events", type=int, default=10_000)
    parser.add_argument(
        "--dt-min",
        type=int,
        default=1_000,
        help="Minimum segment span in milliseconds (default: 1000).",
    )
    parser.add_argument(
        "--dt-max",
        type=int,
        default=10_000,
        help="Maximum segment span in milliseconds (default: 10000).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Fractional overlap between consecutive windows.",
    )
    parser.add_argument(
        "--video-prefix",
        type=str,
        default="video",
        help="Namespace prefix for video entities.",
    )
    parser.add_argument(
        "--segment-prefix",
        type=str,
        default="seg",
        help="Namespace prefix for segment entities.",
    )
    parser.add_argument(
        "--sample-period-ms",
        type=float,
        default=1.0,
        help="Assumed time delta between successive temporal features.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_logger(args.verbose)

    if args.split == "all":
        splits = ("train", "val", "test")
    else:
        splits = (args.split,)

    logger.info("Processing splits: %s", ", ".join(splits))
    for split in splits:
        manifest_path = args.manifests_root / f"{split}.txt"
        if not manifest_path.exists():
            logger.warning("Manifest not found: %s", manifest_path)
            continue

        video_ids = parse_manifest(manifest_path)
        logger.info("Split %s contains %d videos", split, len(video_ids))

        for video_id in video_ids:
            events = load_event_stream(
                video_id,
                split,
                args.data_root,
                logger,
                feature_split=args.feature_split,
                sample_period_ms=args.sample_period_ms,
            )

            canonical_vid = _canonical_video_id(video_id)
            canonical_video_entity = f"{args.video_prefix}:{canonical_vid}"

            def name_factory(idx: int, start_idx: int, end_idx: int) -> str:
                return (
                    f"{args.segment_prefix}:"
                    f"{split}/{canonical_vid}:{idx:04d}"
                )

            segments = list(
                adaptive_segment(
                    events=events,
                    target_events=args.target_events,
                    dt_min_ms=args.dt_min,
                    dt_max_ms=args.dt_max,
                    overlap=args.overlap,
                    name_factory=name_factory,
                )
            )

            for segment in segments:
                segment.video_id = canonical_video_entity

            write_segments(split, video_id, segments, args.output_root)
            logger.debug(
                "Wrote %d segments for %s/%s", len(segments), split, video_id
            )


if __name__ == "__main__":
    main()
