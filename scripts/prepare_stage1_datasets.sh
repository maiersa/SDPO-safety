#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/datasets}"
PREPARE_GSM8K="${PREPARE_GSM8K:-true}"
PREPARE_MATH="${PREPARE_MATH:-true}"
PREPARE_OPENMATHINSTRUCT="${PREPARE_OPENMATHINSTRUCT:-false}"
FORCE="${FORCE:-false}"

GSM8K_DIR="$DATA_ROOT/gsm8k"
MATH_DIR="$DATA_ROOT/math"
OPENMATHINSTRUCT_DIR="$DATA_ROOT/openmathinstruct"

mkdir -p "$DATA_ROOT"

prepare_dataset() {
    local name="$1"
    local out_dir="$2"
    local script_path="$3"

    if [[ -f "$out_dir/train.parquet" && -f "$out_dir/test.parquet" && "$FORCE" != "true" ]]; then
        echo "[$name] Found existing parquet files in $out_dir; skipping. Set FORCE=true to rebuild."
        return 0
    fi

    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    echo "[$name] Writing parquet files to $out_dir"
    "$PYTHON_BIN" "$script_path" --local_save_dir "$out_dir"

    echo "[$name] Done"
    ls -lh "$out_dir"
}

if [[ "$PREPARE_GSM8K" == "true" ]]; then
    prepare_dataset "GSM8K" "$GSM8K_DIR" "$PROJECT_ROOT/examples/data_preprocess/gsm8k.py"
fi

if [[ "$PREPARE_MATH" == "true" ]]; then
    prepare_dataset "MATH" "$MATH_DIR" "$PROJECT_ROOT/examples/data_preprocess/math_dataset.py"
fi

if [[ "$PREPARE_OPENMATHINSTRUCT" == "true" ]]; then
    prepare_dataset "OpenMathInstruct" "$OPENMATHINSTRUCT_DIR" "$PROJECT_ROOT/examples/data_preprocess/openmathinstruct.py"
fi

echo

echo "Stage 1 datasets ready."
echo "GSM8K: $GSM8K_DIR"
echo "MATH:  $MATH_DIR"
if [[ "$PREPARE_OPENMATHINSTRUCT" == "true" ]]; then
    echo "OPENMATHINSTRUCT: $OPENMATHINSTRUCT_DIR"
fi
