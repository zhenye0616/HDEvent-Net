#!/usr/bin/env bash

# Build KG triples with optional segment attribute integration.
# Usage: bash scripts/build_triples.sh [--segments-root ...] [--manifest-root ...]
#        [--output-root ...] [--segment-attrs-root ...] [--splits train,val,test]
#        [--python python3]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="python3"
SEGMENTS_ROOT="Data/segments"
MANIFEST_ROOT="Data/UCF_Crime"
OUTPUT_ROOT="Data/UCF_Crime"
SEGMENT_ATTRS_ROOT=""
SPLITS=("train" "val" "test")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --segments-root)
            SEGMENTS_ROOT="$2"; shift 2 ;;
        --manifest-root)
            MANIFEST_ROOT="$2"; shift 2 ;;
        --output-root)
            OUTPUT_ROOT="$2"; shift 2 ;;
        --segment-attrs-root)
            SEGMENT_ATTRS_ROOT="$2"; shift 2 ;;
        --splits)
            IFS=',' read -r -a SPLITS <<< "$2"; shift 2 ;;
        --python)
            PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [options]"; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

CMD=(
    "${PYTHON_BIN}" Utils/KG_builder.py build-triples
    --segments-root "${SEGMENTS_ROOT}"
    --manifest-root "${MANIFEST_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --splits "${SPLITS[@]}"
)

if [[ -n "${SEGMENT_ATTRS_ROOT}" ]]; then
    CMD+=(--segment-attrs-root "${SEGMENT_ATTRS_ROOT}")
fi

"${CMD[@]}"
