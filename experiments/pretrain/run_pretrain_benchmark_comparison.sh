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
BATCH_SIZE=${BATCH_SIZE:-64}
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
MATH_TRAIN_PATH=${MATH_TRAIN_PATH:-datasets/math/train.parquet}
MATH_EVAL_PATH=${MATH_EVAL_PATH:-datasets/math/test.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/pretrain_benchmarks/rlx_comparison}
COMBINE_SUMMARIES=${COMBINE_SUMMARIES:-true}
SKIP_COMPLETED_BENCHMARKS=${SKIP_COMPLETED_BENCHMARKS:-true}
LOG_TO_FILE=${LOG_TO_FILE:-true}
BENCHMARK_LOG_DIR=${BENCHMARK_LOG_DIR:-${OUTPUT_ROOT}/logs}
AUTO_MERGE_FSDP=${AUTO_MERGE_FSDP:-true}
MERGER_BACKEND=${MERGER_BACKEND:-fsdp}
CHECKPOINT_SELECTION=${CHECKPOINT_SELECTION:-latest}
OPSD_CHECKPOINT_SELECTION=${OPSD_CHECKPOINT_SELECTION:-best}
GRPO_CHECKPOINT_SELECTION=${GRPO_CHECKPOINT_SELECTION:-latest}

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

normalized_tasks() {
    normalize_list "$TASKS"
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

completed_summary_csv() {
    local output_dir="$1"
    local task="$2"
    if [[ ! -d "$output_dir" ]]; then
        return 0
    fi
    "$PYTHON_BIN" - "$output_dir" "$task" "$PROMPT_STYLE" "$NUM_SAMPLES" "$PASS_AT_K" <<'PY'
import csv
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
task = sys.argv[2].lower()
prompt_style = sys.argv[3]
num_samples = sys.argv[4]
pass_at_k = [f"pass@{part.strip()}" for part in sys.argv[5].split(",") if part.strip()]
accepted_prompt_styles = {prompt_style}
if task == "math" and prompt_style == "rlx":
    # MATH is forced to boxed prompting internally even in mixed
    # TASKS=gsm8k,math runs where the global prompt style is rlx.
    accepted_prompt_styles.add("boxed")

for path in sorted(output_dir.glob("*/summary.csv"), reverse=True):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    for row in rows:
        if row.get("task", "").lower() != task:
            continue
        has_prompt_style = row.get("prompt_style") in accepted_prompt_styles
        has_num_samples = row.get("num_samples") == num_samples
        has_metrics = all(metric in row and row.get(metric, "") != "" for metric in pass_at_k)
        if has_prompt_style and has_num_samples and has_metrics:
            print(path)
            raise SystemExit(0)
PY
}

resolve_checkpoint_selection() {
    local spec="$1"
    local name path selection step_file step selected_dir

    name="${spec%%=*}"
    path="${spec#*=}"
    selection="$CHECKPOINT_SELECTION"
    if [[ "$name" == opsd* || "$(basename "$path")" == OPSD-* ]]; then
        selection="$OPSD_CHECKPOINT_SELECTION"
    elif [[ "$name" == grpo* || "$(basename "$path")" == GRPO-* ]]; then
        selection="$GRPO_CHECKPOINT_SELECTION"
    fi

    # If the caller already points at a concrete checkpoint or HF model dir,
    # leave it untouched. Selection only applies to run roots containing
    # best/latest tracker files and global_step_* children.
    if [[ "$(basename "$path")" == global_step_* ]] || [[ -d "${path}/actor" ]] || [[ -f "${path}/config.json" ]]; then
        printf '%s=%s' "$name" "$path"
        return 0
    fi

    if [[ ! -d "$path" ]] || ! compgen -G "${path}/global_step_*" >/dev/null; then
        printf '%s=%s' "$name" "$path"
        return 0
    fi

    case "$selection" in
        best)
            step_file="${path}/best_checkpointed_iteration.txt"
            ;;
        latest|last)
            step_file="${path}/latest_checkpointed_iteration.txt"
            ;;
        none|explicit)
            printf '%s=%s' "$name" "$path"
            return 0
            ;;
        *)
            echo "Unknown checkpoint selection=${selection}; expected best, latest, last, none, or explicit." >&2
            return 1
            ;;
    esac

    if [[ ! -f "$step_file" ]]; then
        echo "Missing checkpoint selection tracker: $step_file" >&2
        return 1
    fi

    step="$(tr -d '[:space:]' < "$step_file")"
    selected_dir="${path}/global_step_${step}"
    if [[ ! -d "$selected_dir" ]]; then
        echo "Selected checkpoint directory does not exist: $selected_dir" >&2
        return 1
    fi

    echo "Selected ${selection} checkpoint for ${name}: global_step_${step}" >&2
    printf '%s=%s' "$name" "$selected_dir"
}

ensure_eval_checkpoint() {
    local spec="$1"
    local name path target_dir

    name="${spec%%=*}"
    path="${spec#*=}"

    if [[ -f "${path}/config.json" ]] && compgen -G "${path}/model*.safetensors" >/dev/null; then
        printf '%s=%s' "$name" "$path"
        return 0
    fi

    if [[ -d "${path}/actor" ]]; then
        path="${path}/actor"
    fi

    if [[ -d "${path}/hf_merged" ]]; then
        printf '%s=%s' "$name" "${path}/hf_merged"
        return 0
    fi

    if [[ -d "$path" ]] && compgen -G "${path}/model_world_size_*_rank_*.pt" >/dev/null; then
        target_dir="${path}/hf_merged"
        if [[ "$AUTO_MERGE_FSDP" != "true" ]]; then
            echo "FSDP actor checkpoint needs merging: $path" >&2
            echo "Set AUTO_MERGE_FSDP=true or merge it manually to ${target_dir}" >&2
            return 1
        fi
        echo "Merging FSDP actor checkpoint for evaluation"
        echo "Local dir: $path"
        echo "Target dir: $target_dir"
        "$PYTHON_BIN" -m verl.model_merger merge \
            --backend "$MERGER_BACKEND" \
            --local_dir "$path" \
            --target_dir "$target_dir"
        printf '%s=%s' "$name" "$target_dir"
        return 0
    fi

    printf '%s=%s' "$name" "$path"
}

run_group() {
    local group_name="$1"
    local prompt_mode="$2"
    local num_fewshot="$3"
    local raw_checkpoints="$4"
    local ckpt resolved name output_dir existing_summary log_path status task

    raw_checkpoints="$(normalize_list "$raw_checkpoints")"
    if [[ -z "$raw_checkpoints" ]]; then
        echo "No ${group_name} checkpoints requested."
        return 0
    fi

    for ckpt in $raw_checkpoints; do
        resolved="$(resolve_model_path "$ckpt")"
        resolved="$(resolve_checkpoint_selection "$resolved")"
        resolved="$(ensure_eval_checkpoint "$resolved")"
        name="$(checkpoint_name "$resolved")"
        output_dir="${OUTPUT_ROOT}/${group_name}/${name}"

        for task in $(normalized_tasks); do
            existing_summary="$(completed_summary_csv "$output_dir" "$task")"
            if [[ "$SKIP_COMPLETED_BENCHMARKS" == "true" && -n "$existing_summary" ]]; then
                echo "=============================================================="
                echo "Skipping completed benchmark"
                echo "Group: $group_name"
                echo "Task: $task"
                echo "Checkpoint: $resolved"
                echo "Existing summary: $existing_summary"
                echo "Set SKIP_COMPLETED_BENCHMARKS=false to rerun."
                echo "=============================================================="
                continue
            fi

            echo "=============================================================="
            echo "Running pretrain benchmark"
            echo "Group: $group_name"
            echo "Task: $task"
            echo "Prompt mode: $prompt_mode"
            echo "Checkpoint: $resolved"
            echo "Output dir: $output_dir"
            echo "Batch size: $BATCH_SIZE"
            echo "=============================================================="

            if [[ "$LOG_TO_FILE" == "true" ]]; then
                mkdir -p "$BENCHMARK_LOG_DIR"
                log_path="${BENCHMARK_LOG_DIR}/${group_name}_${name}_${task}_$(date -u +%Y%m%d_%H%M%S).log"
                echo "Writing benchmark log: $log_path"
                set +e
                CHECKPOINTS="$resolved" \
                PROMPT_MODE="$prompt_mode" \
                PROMPT_STYLE="$PROMPT_STYLE" \
                ANSWER_FORMAT="$ANSWER_FORMAT" \
                NUM_FEWSHOT="$num_fewshot" \
                TASKS="$task" \
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
                MATH_TRAIN_PATH="$MATH_TRAIN_PATH" \
                MATH_EVAL_PATH="$MATH_EVAL_PATH" \
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
                bash "${REPO_DIR}/experiments/pretrain/run_pretrain_benchmark_eval.sh" 2>&1 | tee -a "$log_path"
                status=${PIPESTATUS[0]}
                set -e
                if [[ "$status" -ne 0 ]]; then
                    echo "Benchmark failed with status $status. Log: $log_path" >&2
                    return "$status"
                fi
            else
                CHECKPOINTS="$resolved" \
                PROMPT_MODE="$prompt_mode" \
                PROMPT_STYLE="$PROMPT_STYLE" \
                ANSWER_FORMAT="$ANSWER_FORMAT" \
                NUM_FEWSHOT="$num_fewshot" \
                TASKS="$task" \
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
                MATH_TRAIN_PATH="$MATH_TRAIN_PATH" \
                MATH_EVAL_PATH="$MATH_EVAL_PATH" \
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
            fi
        done
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
