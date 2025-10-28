#!/usr/bin/env python3
"""
Enumerate UCFCrime event feature files with class and augmentation metadata.

Example:
    python scripts/tools/list_ucfcrime_videos.py --data-root /mnt/Data_1/UCFCrime_dataset/vitb \
        --split event_thr_10 --output manifests/all_videos_event_thr_10.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List UCFCrime videos with augmentation metadata.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing split folders (e.g., /mnt/.../UCFCrime_dataset/vitb).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="event_thr_10",
        help="Split subdirectory to enumerate (default: event_thr_10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file to write the listing; prints to stdout if omitted.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional subset of class names to include.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="\t",
        help="Delimiter for the output file (default: tab).",
    )
    return parser.parse_args()


def iter_videos(
    data_root: Path,
    split: str,
    classes: Optional[List[str]] = None,
) -> Iterable[Tuple[str, str, Optional[int], Path]]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if classes is None:
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    else:
        classes = sorted(classes)

    for class_name in classes:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for npy_path in sorted(class_dir.glob("*.npy")):
            stem = npy_path.stem
            video_id = stem
            aug_idx: Optional[int] = None
            if "__" in stem:
                base, aug_candidate = stem.rsplit("__", 1)
                if aug_candidate.isdigit():
                    video_id = base
                    aug_idx = int(aug_candidate)
            yield class_name, video_id, aug_idx, npy_path


def main() -> None:
    args = parse_args()
    rows = list(iter_videos(args.data_root, args.split, args.classes))
    lines = [
        args.delimiter.join(
            (
                class_name,
                video_id,
                str(aug_idx) if aug_idx is not None else "",
                str(path),
            )
        )
        for class_name, video_id, aug_idx, path in rows
    ]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines))
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
