#!/usr/bin/env bash

# Build per-class train/val/test manifests (default 80/10/10) from the rich event manifest.
# Defaults: all classes, augmentation index 5.
# Example:
#   bash scripts/generate_split_from_manifest.sh \
#       --manifest HDEvent-Net/manifests/event_thr_10_manifest.txt \
#       --output-root HDEvent-Net/Data/UCF_Crime \
#       --variant vitb \
#       --feature-split event_thr_10

set -euo pipefail

usage() {
    sed -n '3,17p' "${BASH_SOURCE[0]}"
}

PYTHON_BIN="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${PROJECT_ROOT}/manifests/event_thr_10.txt"
OUTPUT_ROOT="${PROJECT_ROOT}/Data/UCF_Crime"
VARIANT="vitb"
FEATURE_SPLIT="event_thr_10"
CLASSES="all"
AUGMENTATIONS="5"
DELIM=$'\t'
TRAIN_RATIO="0.8"
VAL_RATIO="0.1"
TEST_RATIO="0.1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --feature-split)
            FEATURE_SPLIT="$2"
            shift 2
            ;;
        --classes)
            CLASSES="$2"
            shift 2
            ;;
        --augmentations)
            AUGMENTATIONS="$2"
            shift 2
            ;;
        --delimiter)
            DELIM="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --val-ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        --test-ratio)
            TEST_RATIO="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${MANIFEST}" || -z "${OUTPUT_ROOT}" ]]; then
    echo "[ERROR] --manifest and --output-root are required."
    usage
    exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
    echo "[ERROR] Manifest not found: ${MANIFEST}"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" - "$MANIFEST" "$OUTPUT_ROOT" "$VARIANT" "$FEATURE_SPLIT" "$DELIM" "$CLASSES" "$AUGMENTATIONS" "$TRAIN_RATIO" "$VAL_RATIO" "$TEST_RATIO" <<'PY'
import sys
from pathlib import Path
from typing import Optional

manifest_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
variant = sys.argv[3]
feature_split = sys.argv[4]
delimiter = sys.argv[5].encode("utf-8").decode("unicode_escape")
classes_arg = sys.argv[6]
augmentations_arg = sys.argv[7]
train_ratio = float(sys.argv[8])
val_ratio = float(sys.argv[9])
test_ratio = float(sys.argv[10])

total = round(train_ratio + val_ratio + test_ratio, 6)
if abs(total - 1.0) > 1e-6:
    raise SystemExit(f"Train/val/test ratios must sum to 1.0 (got {total})")

allowed_classes = None
if classes_arg.lower() != "all":
    allowed_classes = {cls.strip() for cls in classes_arg.split(",") if cls.strip()}
    if not allowed_classes:
        raise SystemExit("No valid classes specified.")

allowed_aug = None
if augmentations_arg.lower() != "all":
    allowed_aug = set()
    for token in augmentations_arg.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "base"}:
            allowed_aug.add("none")
        else:
            allowed_aug.add(token)
    if not allowed_aug:
        raise SystemExit("No valid augmentations specified.")


def build_identifier(class_name: str, video_id: str, aug_idx: Optional[str]) -> str:
    suffix = ""
    if aug_idx:
        suffix = f"__{aug_idx}"
    return f"video:{variant}:{feature_split}:{class_name}/{video_id}{suffix}"


by_class = {}
with manifest_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        parts = line.split(delimiter)
        if len(parts) < 2:
            continue

        class_name = parts[0].strip()
        video_id = parts[1].strip()
        aug_idx = parts[2].strip() if len(parts) >= 3 else ""
        aug_token = aug_idx.lower() if aug_idx else "none"

        if allowed_classes is not None and class_name not in allowed_classes:
            continue
        if allowed_aug is not None and aug_token not in allowed_aug:
            continue

        identifier = build_identifier(class_name, video_id, aug_idx if aug_idx else None)
        by_class.setdefault(class_name, []).append(identifier)

splits = {"train": [], "val": [], "test": []}
for class_name, identifiers in by_class.items():
    identifiers = sorted(dict.fromkeys(identifiers))
    n = len(identifiers)
    train_cut = int(n * train_ratio)
    val_cut = int(n * val_ratio)
    if train_cut + val_cut > n:
        val_cut = max(0, n - train_cut)
    test_cut = n - train_cut - val_cut

    splits["train"].extend(identifiers[:train_cut])
    splits["val"].extend(identifiers[train_cut:train_cut + val_cut])
    splits["test"].extend(identifiers[train_cut + val_cut:])

for split_name, items in splits.items():
    out_path = output_root / f"{split_name}.txt"
    with out_path.open("w", encoding="utf-8") as writer:
        for identifier in sorted(items):
            writer.write(f"{identifier}\n")
    print(f"[INFO] Wrote {len(items)} entries to {out_path}")
PY
