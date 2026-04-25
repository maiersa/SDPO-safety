#!/usr/bin/env bash

set -euo pipefail

# This script is intended to run inside the Run:AI container.
# It does not call runai submit.

CONFIG_NAME=${CONFIG_NAME:-"baseline_grpo"}
DATA_PATH=${DATA_PATH:-"datasets/tooluse"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
LR=${LR:-"1e-5"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
TOKENIZER_PATH=${TOKENIZER_PATH:-""}

# Optional smoke-test knobs
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-"-1"}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-"8"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-"1"}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-""}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-"False"}
VAL_ONLY=${VAL_ONLY:-"False"}
TEST_FREQ=${TEST_FREQ:-"50"}
SAVE_FREQ=${SAVE_FREQ:-"50"}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-"0"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-""}
VAL_GENERATION_N=${VAL_GENERATION_N:-"4"}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-"True"}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-"0.6"}
VAL_TOP_P=${VAL_TOP_P:-""}
VAL_TOP_K=${VAL_TOP_K:-""}
VALIDATION_GENERATIONS_ONLY=${VALIDATION_GENERATIONS_ONLY:-"False"}
RESUME_MODE=${RESUME_MODE:-"auto"}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-""}

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-"0.8"}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-"16384"}
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-"2"}
ACTOR_PPO_MICRO_BATCH_SIZE=${ACTOR_PPO_MICRO_BATCH_SIZE:-"2"}
ASYNC_REWARD_FUNCTION=${ASYNC_REWARD_FUNCTION:-"True"}
REWARD_MAX_WORKERS=${REWARD_MAX_WORKERS:-"16"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-""}
DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-""}
DATA_MAX_RESPONSE_LENGTH=${DATA_MAX_RESPONSE_LENGTH:-""}

CONDA_ENV=${CONDA_ENV:-"default"}
REPO_DIR=${REPO_DIR:-"/dlabscratch1/${USER}/projects/SDPO-safety"}
LOG_DIR=${LOG_DIR:-"/dlabscratch1/${USER}/output"}
CKPT_DIR=${CKPT_DIR:-"/dlabscratch1/${USER}/checkpoints"}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-"1"}
PROJECT_NAME=${PROJECT_NAME:-"GRPO-MathTeacher-${USER}"}

SUFFIX=${SUFFIX:-"runai_grpo"}

# Persist startup and failure diagnostics to mounted storage so logs survive pod restarts.
mkdir -p "$LOG_DIR/runai_debug"
RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ID="${HOSTNAME:-runai}-${RUN_TS}"
RUN_LOG="$LOG_DIR/runai_debug/${RUN_ID}.log"

exec > >(tee -a "$RUN_LOG") 2>&1

on_error() {
    local exit_code="$?"
    local line_no="$1"
    echo "ERROR: runai_grpo_worker.sh failed at line ${line_no} with exit code ${exit_code}"
    echo "Log file: $RUN_LOG"
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME=${EXP_NAME:-"RUNAI-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${MODEL_NAME}-${SUFFIX}"}

if [[ -z "$VALIDATION_DATA_DIR" && "$LOG_VAL_GENERATIONS" != "0" ]]; then
    VALIDATION_DATA_DIR="$LOG_DIR/validation_generations/$EXP_NAME"
fi

candidate_mount_alias() {
    local path="$1"
    if [[ "$path" == /mnt/dlabscratch1/* ]]; then
        echo "${path#/mnt}"
    elif [[ "$path" == /dlabscratch1/* ]]; then
        echo "/mnt${path}"
    else
        echo ""
    fi
}

activate_conda_env() {
    local requested="$1"
    local -a candidates=()
    local -A seen=()

    candidates+=("$requested")

    # If only a name is provided, try the persistent repo-local environment too.
    if [[ "$requested" != /* ]]; then
        candidates+=("$REPO_DIR/.conda")
    fi

    # Add mount-alias alternatives between /dlabscratch1 and /mnt/dlabscratch1.
    local alt_requested
    alt_requested="$(candidate_mount_alias "$requested")"
    if [[ -n "$alt_requested" ]]; then
        candidates+=("$alt_requested")
    fi

    local repo_env="$REPO_DIR/.conda"
    local alt_repo_env
    alt_repo_env="$(candidate_mount_alias "$repo_env")"
    if [[ -n "$alt_repo_env" ]]; then
        candidates+=("$alt_repo_env")
    fi

    for env_target in "${candidates[@]}"; do
        if [[ -n "${seen[$env_target]:-}" ]]; then
            continue
        fi
        seen[$env_target]=1

        if [[ "$env_target" == /* && ! -d "$env_target" ]]; then
            continue
        fi

        if conda activate "$env_target"; then
            echo "Activated conda env: $env_target"
            return 0
        fi
    done

    echo "ERROR: Could not activate CONDA_ENV='$requested'"
    echo "Tried candidates: ${candidates[*]}"
    conda info --envs || true
    return 1
}

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    activate_conda_env "$CONDA_ENV"
else
    echo "WARNING: /opt/conda/etc/profile.d/conda.sh not found; skipping conda activation"
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export USER=${USER:-$(whoami)}

echo "Worker host: ${HOSTNAME:-unknown}"
echo "Worker log: $RUN_LOG"
echo "Python executable: $(command -v python || echo 'python not found')"
python --version || true

# Training config expects parquet files. If only JSON exists, generate parquet on the fly.
if [[ ! -f "$DATA_PATH/train.parquet" || ! -f "$DATA_PATH/test.parquet" ]]; then
    if [[ -f "$DATA_PATH/train.json" && -f "$DATA_PATH/test.json" ]]; then
        echo "Parquet dataset not found under $DATA_PATH. Running data/preprocess.py ..."
        python data/preprocess.py --data_source "$DATA_PATH"
    fi
fi

if [[ ! -f "$DATA_PATH/train.parquet" || ! -f "$DATA_PATH/test.parquet" ]]; then
    echo "ERROR: Missing parquet dataset files under $DATA_PATH"
    echo "Expected: $DATA_PATH/train.parquet and $DATA_PATH/test.parquet"
    echo "Either generate them manually with: python data/preprocess.py --data_source $DATA_PATH"
    echo "or provide data path containing parquet files."
    exit 1
fi

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
data.train_max_samples=$TRAIN_MAX_SAMPLES \
data.val_max_samples=$VAL_MAX_SAMPLES \
trainer.group_name=GRPO-runai \
trainer.project_name=$PROJECT_NAME \
trainer.n_gpus_per_node=$TRAINER_GPUS_PER_NODE \
trainer.total_epochs=$TOTAL_EPOCHS \
trainer.val_before_train=$VAL_BEFORE_TRAIN \
trainer.val_only=$VAL_ONLY \
trainer.validation_generations_only=$VALIDATION_GENERATIONS_ONLY \
trainer.test_freq=$TEST_FREQ \
trainer.save_freq=$SAVE_FREQ \
trainer.max_actor_ckpt_to_keep=2 \
trainer.resume_mode=$RESUME_MODE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_PPO_MICRO_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
reward_model.launch_reward_fn_async=$ASYNC_REWARD_FUNCTION \
+reward_model.reward_kwargs.max_workers=$REWARD_MAX_WORKERS \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_GENERATION_N \
actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
vars.dir=$REPO_DIR \
vars.log_dir=$LOG_DIR \
vars.ckpt_dir=$CKPT_DIR"

if [[ -n "$TOKENIZER_PATH" ]]; then
    ARGS="$ARGS actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH critic.model.tokenizer_path=$TOKENIZER_PATH"
fi

if [[ "$LOG_VAL_GENERATIONS" != "0" ]]; then
    ARGS="$ARGS trainer.log_val_generations=$LOG_VAL_GENERATIONS"
fi

if [[ -n "$VAL_TOP_P" ]]; then
    ARGS="$ARGS actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P"
fi

if [[ -n "$VAL_TOP_K" ]]; then
    ARGS="$ARGS actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K"
fi

if [[ -n "$VALIDATION_DATA_DIR" ]]; then
    mkdir -p "$VALIDATION_DATA_DIR"
    ARGS="$ARGS trainer.validation_data_dir=$VALIDATION_DATA_DIR"
fi

if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
    ARGS="$ARGS trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
fi

if [[ -n "$RESUME_FROM_PATH" ]]; then
    ARGS="$ARGS trainer.resume_from_path=$RESUME_FROM_PATH"
fi

if [[ -n "$MAX_MODEL_LEN" ]]; then
    ARGS="$ARGS max_model_len=$MAX_MODEL_LEN actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
fi

if [[ -n "$DATA_MAX_PROMPT_LENGTH" ]]; then
    ARGS="$ARGS data.max_prompt_length=$DATA_MAX_PROMPT_LENGTH"
fi

if [[ -n "$DATA_MAX_RESPONSE_LENGTH" ]]; then
    ARGS="$ARGS data.max_response_length=$DATA_MAX_RESPONSE_LENGTH"
fi

echo "----------------------------------------------------------------"
echo "Starting Run:AI GRPO worker"
echo "Experiment: $EXP_NAME"
echo "Repo: $REPO_DIR"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Tokenizer path: ${TOKENIZER_PATH:-<default>}"
echo "Conda env requested: $CONDA_ENV"
echo "Async reward fn: $ASYNC_REWARD_FUNCTION"
echo "Reward max workers: $REWARD_MAX_WORKERS"
echo "Rollout GPU memory utilization: $ROLLOUT_GPU_MEMORY_UTILIZATION"
echo "Rollout max batched tokens: $ROLLOUT_MAX_NUM_BATCHED_TOKENS"
echo "Rollout/ref logprob micro-batch per GPU: $ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE"
echo "Actor PPO micro-batch per GPU: $ACTOR_PPO_MICRO_BATCH_SIZE"
echo "Validation sampling: do_sample=$VAL_DO_SAMPLE temperature=$VAL_TEMPERATURE top_p=${VAL_TOP_P:-<default>} top_k=${VAL_TOP_K:-<default>} n=$VAL_GENERATION_N"
echo "----------------------------------------------------------------"

bash training/verl_training.sh "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
