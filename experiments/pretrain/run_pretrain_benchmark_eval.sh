#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Comma-separated task list. First implemented task: gsm8k.
TASKS="${TASKS:-gsm8k}"

# Comma-separated checkpoint specs. Each item may be PATH or NAME=PATH.
# Example:
#   CHECKPOINTS="base=/path/to/base,opsd_step_100=/path/to/actor/hf"
CHECKPOINTS="${CHECKPOINTS:?Set CHECKPOINTS to a comma-separated list of checkpoint paths or NAME=PATH specs.}"

# RL Excursions-style defaults:
#   base checkpoints: PROMPT_MODE=base   -> 8-shot
#   trained ckpts:    PROMPT_MODE=trained -> 0-shot
PROMPT_MODE="${PROMPT_MODE:-trained}"
PROMPT_STYLE="${PROMPT_STYLE:-rlx}"
ANSWER_FORMAT="${ANSWER_FORMAT:-auto}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
PASS_AT_K="${PASS_AT_K:-1,8,32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
NUM_FEWSHOT="${NUM_FEWSHOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pretrain_benchmarks}"
GSM8K_TRAIN_PATH="${GSM8K_TRAIN_PATH:-datasets/gsm8k/train.parquet}"
GSM8K_EVAL_PATH="${GSM8K_EVAL_PATH:-datasets/gsm8k/test.parquet}"
BACKEND="${BACKEND:-hf}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
SEED="${SEED:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"
ADD_DEFAULT_STOPS="${ADD_DEFAULT_STOPS:-true}"

args=(
  "$PROJECT_ROOT/scripts/eval_pretrain_benchmarks.py"
  --tasks "$TASKS"
  --prompt-mode "$PROMPT_MODE"
  --prompt-style "$PROMPT_STYLE"
  --answer-format "$ANSWER_FORMAT"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --num-samples "$NUM_SAMPLES"
  --pass-at-k "$PASS_AT_K"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --batch-size "$BATCH_SIZE"
  --output-dir "$OUTPUT_DIR"
  --gsm8k-train-path "$GSM8K_TRAIN_PATH"
  --gsm8k-eval-path "$GSM8K_EVAL_PATH"
  --backend "$BACKEND"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --seed "$SEED"
  --device-map "$DEVICE_MAP"
  --torch-dtype "$TORCH_DTYPE"
)

IFS=',' read -r -a checkpoint_specs <<< "$CHECKPOINTS"
for spec in "${checkpoint_specs[@]}"; do
  [[ -n "$spec" ]] && args+=(--checkpoint "$spec")
done

if [[ -n "$TOP_K" ]]; then
  args+=(--top-k "$TOP_K")
fi

if [[ -n "$MAX_PROMPT_TOKENS" ]]; then
  args+=(--max-prompt-tokens "$MAX_PROMPT_TOKENS")
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
  args+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ "$ENFORCE_EAGER" == "true" ]]; then
  args+=(--enforce-eager)
fi

if [[ -n "$MAX_EXAMPLES" ]]; then
  args+=(--max-examples "$MAX_EXAMPLES")
fi

if [[ -n "$NUM_FEWSHOT" ]]; then
  args+=(--num-fewshot "$NUM_FEWSHOT")
fi

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  args+=(--trust-remote-code)
fi

if [[ "$ADD_DEFAULT_STOPS" == "false" ]]; then
  args+=(--no-add-default-stops)
fi

# Comma-separated literal stop strings. To pass newlines from bash, use ANSI-C
# quoting, e.g. STOP_SEQUENCES=$'\nQuestion:',$'\n\nQuestion:'.
if [[ -n "${STOP_SEQUENCES:-}" ]]; then
  IFS=',' read -r -a stop_specs <<< "$STOP_SEQUENCES"
  for stop in "${stop_specs[@]}"; do
    args+=(--stop-sequence "$stop")
  done
fi

"$PYTHON_BIN" "${args[@]}"
