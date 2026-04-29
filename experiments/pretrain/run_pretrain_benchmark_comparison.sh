#!/usr/bin/env bash

set -euo pipefail

export USER=${USER:-$(whoami)}

CONDA_ENV=${CONDA_ENV:-/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo}
REPO_DIR=${REPO_DIR:-/dlabscratch1/${USER}/projects/SDPO-safety}
PYTHON_BIN=${PYTHON_BIN:-${CONDA_ENV}/bin/python}

TASKS=${TASKS:-gsm8k}
PROMPT_STYLE=${PROMPT_STYLE:-rlx}
ANSWER_FORMAT=${ANSWER_FORMAT:-auto}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-}
NUM_SAMPLES=${NUM_SAMPLES:-32}
PASS_AT_K=${PASS_AT_K:-1,8,32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
NUM_FEWSHOT_BASE=${NUM_FEWSHOT_BASE:-8}
NUM_FEWSHOT_TRAINED=${NUM_FEWSHOT_TRAINED:-0}

BACKEND=${BACKEND:-vllm}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
ENFORCE_EAGER=${ENFORCE_EAGER:-false}
SEED=${SEED:-1}
DEVICE_MAP=${DEVICE_MAP:-auto}
TORCH_DTYPE=${TORCH_DTYPE:-float16}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
ADD_DEFAULT_STOPS=${ADD_DEFAULT_STOPS:-true}

GSM8K_TRAIN_PATH=${GSM8K_TRAIN_PATH:-datasets/gsm8k/train.parquet}
GSM8K_EVAL_PATH=${GSM8K_EVAL_PATH:-datasets/gsm8k/test.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/pretrain_benchmarks/rlx_comparison}
COMBINE_SUMMARIES=${COMBINE_SUMMARIES:-true}

STAGE1_ROOT=${STAGE1_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-stage1}
STAGE2_ROOT=${STAGE2_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-stage2}
STAGE3_ROOT=${STAGE3_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-stage3}

BASE_CHECKPOINTS=${BASE_CHECKPOINTS:-}
TRAINED_CHECKPOINTS=${TRAINED_CHECKPOINTS:-}

normalize_list() {
    local raw="$1"
    raw="${raw//,/ }"
    echo "$raw" | xargs
}

sanitize_label() {
    local label="$1"
    label="${label//@/-}"
    label="${label//\//-}"
    label="${label// /-}"
    printf '%s' "$label"
}

resolve_model_path() {
    local ckpt="$1"
    local name path

    if [[ "$ckpt" == *"="* ]]; then
        name="${ckpt%%=*}"
        path="${ckpt#*=}"
    else
        name="$(sanitize_label "$ckpt")"
        path="$ckpt"
    fi

    if [[ -d "$path" ]]; then
        printf '%s=%s' "$name" "$path"
        return 0
    fi

    if [[ "$path" == stage1-* ]]; then
        printf '%s=%s/%s' "$name" "$STAGE1_ROOT" "$path"
        return 0
    fi

    if [[ "$path" == stage2-* ]]; then
        printf '%s=%s/%s' "$name" "$STAGE2_ROOT" "$path"
        return 0
    fi

    if [[ "$path" == stage3-* ]]; then
        printf '%s=%s/%s' "$name" "$STAGE3_ROOT" "$path"
        return 0
    fi

    if [[ "$path" == main && -d "${STAGE3_ROOT}/main" ]]; then
        printf '%s=%s' "$name" "${STAGE3_ROOT}/main"
        return 0
    fi

    printf '%s=%s' "$name" "$path"
}

checkpoint_name() {
    local spec="$1"
    if [[ "$spec" == *"="* ]]; then
        sanitize_label "${spec%%=*}"
    else
        sanitize_label "$spec"
    fi
}

run_group() {
    local group_name="$1"
    local prompt_mode="$2"
    local num_fewshot="$3"
    local raw_checkpoints="$4"
    local ckpt resolved name output_dir

    raw_checkpoints="$(normalize_list "$raw_checkpoints")"
    if [[ -z "$raw_checkpoints" ]]; then
        echo "No ${group_name} checkpoints requested."
        return 0
    fi

    for ckpt in $raw_checkpoints; do
        resolved="$(resolve_model_path "$ckpt")"
        name="$(checkpoint_name "$resolved")"
        output_dir="${OUTPUT_ROOT}/${group_name}/${name}"

        echo "=============================================================="
        echo "Running pretrain benchmark"
        echo "Group: $group_name"
        echo "Prompt mode: $prompt_mode"
        echo "Checkpoint: $resolved"
        echo "Output dir: $output_dir"
        echo "=============================================================="

        CHECKPOINTS="$resolved" \
        PROMPT_MODE="$prompt_mode" \
        PROMPT_STYLE="$PROMPT_STYLE" \
        ANSWER_FORMAT="$ANSWER_FORMAT" \
        NUM_FEWSHOT="$num_fewshot" \
        TASKS="$TASKS" \
        TEMPERATURE="$TEMPERATURE" \
        TOP_P="$TOP_P" \
        TOP_K="$TOP_K" \
        NUM_SAMPLES="$NUM_SAMPLES" \
        PASS_AT_K="$PASS_AT_K" \
        MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        MAX_PROMPT_TOKENS="$MAX_PROMPT_TOKENS" \
        BATCH_SIZE="$BATCH_SIZE" \
        MAX_EXAMPLES="$MAX_EXAMPLES" \
        OUTPUT_DIR="$output_dir" \
        GSM8K_TRAIN_PATH="$GSM8K_TRAIN_PATH" \
        GSM8K_EVAL_PATH="$GSM8K_EVAL_PATH" \
        BACKEND="$BACKEND" \
        TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
        GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        ENFORCE_EAGER="$ENFORCE_EAGER" \
        SEED="$SEED" \
        DEVICE_MAP="$DEVICE_MAP" \
        TORCH_DTYPE="$TORCH_DTYPE" \
        TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
        ADD_DEFAULT_STOPS="$ADD_DEFAULT_STOPS" \
        PYTHON_BIN="$PYTHON_BIN" \
        bash "${REPO_DIR}/experiments/pretrain/run_pretrain_benchmark_eval.sh"
    done
}

combine_summaries() {
    if [[ "$COMBINE_SUMMARIES" != "true" ]]; then
        return 0
    fi

    "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted(root.glob("*/*/*/summary.csv"))
rows = []
fieldnames = None
for path in paths:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row["summary_csv"] = str(path)
            rows.append(row)

if not rows or fieldnames is None:
    print(f"No summary.csv files found under {root}")
    raise SystemExit(0)

fieldnames = fieldnames + ["summary_csv"]
out = root / "combined_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote combined summary: {out}")
PY
}

cd "$REPO_DIR"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

run_group "base" "base" "$NUM_FEWSHOT_BASE" "$BASE_CHECKPOINTS"
run_group "trained" "trained" "$NUM_FEWSHOT_TRAINED" "$TRAINED_CHECKPOINTS"
combine_summaries
