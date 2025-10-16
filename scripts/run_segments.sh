#!/usr/bin/env bash

# Wrapper to run attribute_similarity.py with segment export enabled.
# Adjust defaults as needed for your environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="/mnt/Data_1/UCFCrime_dataset"
VARIANT="vitb"
SPLIT="event_thr_10"
CHECKPOINT="ckpt/vitb.pt"
CLASSES=("Abuse" "Arrest" "Arson")
MAX_SAMPLES=50
AUG_IDX=5
TOP_K=20
SEG_TOP_K=5
SEG_THRESHOLD=5.0
SEG_AGG="max"
SEGMENTS_ROOT="Data/segments"
SEGMENTS_SPLIT="test"
SEGMENT_OUTPUT_DIR="Data/segment_attributes"
SAMPLE_PERIOD_MS=4

python Utils/attribute_similarity.py "${DATA_ROOT}" \
  --variant "${VARIANT}" \
  --split "${SPLIT}" \
  --checkpoint "${CHECKPOINT}" \
  --classes "${CLASSES[@]}" \
  --use-temporal \
  --use-logits \
  --max-samples "${MAX_SAMPLES}" \
  --augmentation-idx "${AUG_IDX}" \
  --top-k "${TOP_K}" \
  --segments-root "${SEGMENTS_ROOT}" \
  --segments-split "${SEGMENTS_SPLIT}" \
  --segment-output-dir "${SEGMENT_OUTPUT_DIR}" \
  --segment-top-k "${SEG_TOP_K}" \
  --segment-threshold "${SEG_THRESHOLD}" \
  --segment-aggregate "${SEG_AGG}" \
  --sample-period-ms "${SAMPLE_PERIOD_MS}" \
  --log-level info
