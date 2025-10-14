#!/usr/bin/env bash

# End-to-end pipeline for segmenting event videos and regenerating KG artefacts.
# Usage:
#   bash scripts/build_event_kg.sh --data-root /path/to/UCFCrime_dataset/vitb \
#       [--feature-split event_thr_10] [--segments-root Data/segments] \
#       [--manifest-root Data/UCF_Crime] [--output-root Data/UCF_Crime] \
#       [--target-events 10000] [--dt-min 10] [--dt-max 60] [--overlap 0.5] \
#       [--sample-period-ms 4] [--python python3]
#
# The script expects to be run from the repository root (HDEvent-Net).

set -euo pipefail

print_usage() {
    sed -n '3,15p' "${BASH_SOURCE[0]}"
}

PYTHON="python3"
FEATURE_SPLIT="event_thr_10"
SEGMENTS_ROOT="Data/segments"
MANIFEST_ROOT="Data/UCF_Crime"
OUTPUT_ROOT="Data/UCF_Crime"
ATTRIBUTES_PATH="Data/attributes.json"
TARGET_EVENTS=10000
DT_MIN=10
DT_MAX=60
OVERLAP=0.5
SAMPLE_PERIOD_MS=4
SPLITS=("train" "val" "test")
VERBOSE=0

DATA_ROOT="/mnt/Data_1/UCFCrime_dataset/vitb"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --feature-split)
            FEATURE_SPLIT="$2"
            shift 2
            ;;
        --segments-root)
            SEGMENTS_ROOT="$2"
            shift 2
            ;;
        --manifest-root)
            MANIFEST_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --attributes)
            ATTRIBUTES_PATH="$2"
            shift 2
            ;;
        --target-events)
            TARGET_EVENTS="$2"
            shift 2
            ;;
        --dt-min)
            DT_MIN="$2"
            shift 2
            ;;
        --dt-max)
            DT_MAX="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP="$2"
            shift 2
            ;;
        --sample-period-ms)
            SAMPLE_PERIOD_MS="$2"
            shift 2
            ;;
        --splits)
            IFS=',' read -r -a SPLITS <<< "$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "${DATA_ROOT}" ]]; then
    echo "[ERROR] --data-root is required."
    print_usage
    exit 1
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PROJECT_ROOT}"

VERBOSE_ARG=()
if [[ ${VERBOSE} -eq 1 ]]; then
    VERBOSE_ARG=(--verbose)
fi

echo "[INFO] Precomputing segments into ${SEGMENTS_ROOT} for splits: ${SPLITS[*]}"
for split in "${SPLITS[@]}"; do
    ${PYTHON} Utils/precompute_segments.py \
        --split "${split}" \
        --manifests-root "${MANIFEST_ROOT}" \
        --data-root "${DATA_ROOT}" \
        --feature-split "${FEATURE_SPLIT}" \
        --output-root "${SEGMENTS_ROOT}" \
        --target-events "${TARGET_EVENTS}" \
        --dt-min "${DT_MIN}" \
        --dt-max "${DT_MAX}" \
        --overlap "${OVERLAP}" \
        --sample-period-ms "${SAMPLE_PERIOD_MS}" \
        "${VERBOSE_ARG[@]}"
done

echo "[INFO] Building structural triples into ${OUTPUT_ROOT}"
${PYTHON} Utils/KG_builder.py build-triples \
    --segments-root "${SEGMENTS_ROOT}" \
    --manifest-root "${MANIFEST_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --splits "${SPLITS[@]}"

echo "[INFO] Regenerating entity and relation ID maps"
${PYTHON} Utils/build_kg_indices.py \
    --attributes "${ATTRIBUTES_PATH}" \
    --segments-root "${SEGMENTS_ROOT}" \
    --manifest-root "${MANIFEST_ROOT}" \
    --splits "${SPLITS[@]}" \
    --entity-output "${OUTPUT_ROOT}/entity2id.txt" \
    --relation-output "${OUTPUT_ROOT}/relation2id.txt" \
    "${VERBOSE_ARG[@]}"

echo "[INFO] Pipeline completed. Generated files are located under ${OUTPUT_ROOT}."
