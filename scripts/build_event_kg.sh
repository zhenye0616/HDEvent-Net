#!/usr/bin/env bash

# End-to-end pipeline for segmenting event videos and regenerating KG artefacts.
# Usage:
#   bash scripts/build_event_kg.sh --data-root /path/to/UCFCrime_dataset/vitb \
#       [--feature-split event_thr_10] [--variant vitb] \
#       [--segments-root Data/segments] [--manifest-root Data/manifests] \
#       [--regenerate-manifests --manifest-source manifests/event_thr_10.txt] \
#       [--output-root Data/UCF_Crime] [--target-events 10000] \
#       [--dt-min 1000] [--dt-max 10000] [--overlap 0.5] \
#       [--sample-period-ms 4] [--python python3]
#
# The script expects to be run from the repository root (HDEvent-Net).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

print_usage() {
    sed -n '3,15p' "${BASH_SOURCE[0]}"
}

PYTHON="python3"
FEATURE_SPLIT="event_thr_10"
VARIANT=""
SEGMENTS_ROOT="Data/segments"
MANIFEST_ROOT="Data/manifests"
MANIFEST_SOURCE=""
OUTPUT_ROOT="Data/UCF_Crime"
SEGMENT_ATTRS_ROOT="Data/segment_attributes"
ATTRIBUTES_PATH="Data/attributes.json"
TARGET_EVENTS=10000
DT_MIN=1000
DT_MAX=10000
OVERLAP=0.5
SAMPLE_PERIOD_MS=4
SPLITS=("train" "val" "test")
VERBOSE=0
REGENERATE_MANIFESTS=0
MANIFEST_CLASSES="all"
MANIFEST_AUGMENTATIONS="5"
MANIFEST_TRAIN_RATIO="0.8"
MANIFEST_VAL_RATIO="0.1"
MANIFEST_TEST_RATIO="0.1"

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
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --manifest-root)
            MANIFEST_ROOT="$2"
            shift 2
            ;;
        --manifest-source)
            MANIFEST_SOURCE="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --segment-attrs-root)
            SEGMENT_ATTRS_ROOT="$2"
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
        --regenerate-manifests)
            REGENERATE_MANIFESTS=1
            shift 1
            ;;
        --skip-manifest-regeneration)
            REGENERATE_MANIFESTS=0
            shift 1
            ;;
        --manifest-classes)
            MANIFEST_CLASSES="$2"
            shift 2
            ;;
        --manifest-augmentations)
            MANIFEST_AUGMENTATIONS="$2"
            shift 2
            ;;
        --train-ratio)
            MANIFEST_TRAIN_RATIO="$2"
            shift 2
            ;;
        --val-ratio)
            MANIFEST_VAL_RATIO="$2"
            shift 2
            ;;
        --test-ratio)
            MANIFEST_TEST_RATIO="$2"
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

if [[ "${DATA_ROOT}" != /* ]]; then
    DATA_ROOT="${PROJECT_ROOT}/${DATA_ROOT}"
fi
DATA_ROOT="${DATA_ROOT%/}"
if [[ -z "${VARIANT}" ]]; then
    VARIANT="$(basename "${DATA_ROOT}")"
fi
if [[ "${SEGMENTS_ROOT}" != /* ]]; then
    SEGMENTS_ROOT="${PROJECT_ROOT}/${SEGMENTS_ROOT}"
fi
if [[ "${MANIFEST_ROOT}" != /* ]]; then
    MANIFEST_ROOT="${PROJECT_ROOT}/${MANIFEST_ROOT}"
fi
if [[ -n "${MANIFEST_SOURCE}" && "${MANIFEST_SOURCE}" != /* ]]; then
    MANIFEST_SOURCE="${PROJECT_ROOT}/${MANIFEST_SOURCE}"
fi
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi
if [[ -n "${SEGMENT_ATTRS_ROOT}" && "${SEGMENT_ATTRS_ROOT}" != /* ]]; then
    SEGMENT_ATTRS_ROOT="${PROJECT_ROOT}/${SEGMENT_ATTRS_ROOT}"
fi
if [[ "${ATTRIBUTES_PATH}" != /* ]]; then
    ATTRIBUTES_PATH="${PROJECT_ROOT}/${ATTRIBUTES_PATH}"
fi

if [[ -z "${DATA_ROOT}" ]]; then
    echo "[ERROR] --data-root is required."
    print_usage
    exit 1
fi

if [[ ${REGENERATE_MANIFESTS} -eq 1 ]]; then
    if [[ -z "${MANIFEST_SOURCE}" ]]; then
        echo "[ERROR] --manifest-source is required when --regenerate-manifests is set."
        exit 1
    fi
    echo "[INFO] Regenerating manifests in ${MANIFEST_ROOT}"
    mkdir -p "${MANIFEST_ROOT}"
    bash "${SCRIPT_DIR}/tools/generate_split.sh" \
        --manifest "${MANIFEST_SOURCE}" \
        --output-root "${MANIFEST_ROOT}" \
        --variant "${VARIANT}" \
        --feature-split "${FEATURE_SPLIT}" \
        --classes "${MANIFEST_CLASSES}" \
        --augmentations "${MANIFEST_AUGMENTATIONS}" \
        --train-ratio "${MANIFEST_TRAIN_RATIO}" \
        --val-ratio "${MANIFEST_VAL_RATIO}" \
        --test-ratio "${MANIFEST_TEST_RATIO}"
fi

for split in "${SPLITS[@]}"; do
    if [[ ! -f "${MANIFEST_ROOT}/${split}.txt" ]]; then
        echo "[ERROR] Manifest missing: ${MANIFEST_ROOT}/${split}.txt"
        exit 1
    fi
done

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
KG_BUILDER_CMD=(
    ${PYTHON} Utils/KG_builder.py build-triples
    --segments-root "${SEGMENTS_ROOT}"
    --manifest-root "${MANIFEST_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --splits "${SPLITS[@]}"
)
if [[ -n "${SEGMENT_ATTRS_ROOT}" ]]; then
    KG_BUILDER_CMD+=(--segment-attrs-root "${SEGMENT_ATTRS_ROOT}")
fi
"${KG_BUILDER_CMD[@]}"

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
