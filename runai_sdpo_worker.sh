#!/usr/bin/env bash

set -euo pipefail

# This script is intended to run inside the Run:AI container.
# It does not call runai submit.

CONFIG_NAME=${CONFIG_NAME:-"sdpo"}
DATA_PATH=${DATA_PATH:-"datasets/tooluse"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
LR=${LR:-"1e-5"}
ALPHA=${ALPHA:-"0.5"}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-100}
DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS:-"True"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}

# Optional smoke-test knobs
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-"-1"}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-"-1"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-"1"}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-""}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-"False"}
TEST_FREQ=${TEST_FREQ:-"-1"}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-"0"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-""}
VALIDATION_GENERATIONS_ONLY=${VALIDATION_GENERATIONS_ONLY:-"False"}

CONDA_ENV=${CONDA_ENV:-"default"}
REPO_DIR=${REPO_DIR:-"/dlabscratch1/${USER}/projects/SDPO-safety"}
LOG_DIR=${LOG_DIR:-"/dlabscratch1/${USER}/output"}
CKPT_DIR=${CKPT_DIR:-"/dlabscratch1/${USER}/checkpoints"}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-"1"}

SUFFIX=${SUFFIX:-"runai_sdpo"}

mkdir -p "$LOG_DIR/runai_debug"
RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ID="${HOSTNAME:-runai}-${RUN_TS}"
RUN_LOG="$LOG_DIR/runai_debug/${RUN_ID}.log"

exec > >(tee -a "$RUN_LOG") 2>&1

on_error() {
    local exit_code="$?"
    local line_no="$1"
    echo "ERROR: runai_sdpo_worker.sh failed at line ${line_no} with exit code ${exit_code}"
    echo "Log file: $RUN_LOG"
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME=${EXP_NAME:-"RUNAI-SDPO-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${MODEL_NAME}-${SUFFIX}"}

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

    if [[ "$requested" != /* ]]; then
        candidates+=("$REPO_DIR/.conda")
    fi

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
trainer.group_name=SDPO-runai \
trainer.n_gpus_per_node=$TRAINER_GPUS_PER_NODE \
trainer.total_epochs=$TOTAL_EPOCHS \
trainer.val_before_train=$VAL_BEFORE_TRAIN \
trainer.test_freq=$TEST_FREQ \
trainer.save_freq=50 \
trainer.max_actor_ckpt_to_keep=2 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16 \
vars.dir=$REPO_DIR \
vars.log_dir=$LOG_DIR \
vars.ckpt_dir=$CKPT_DIR"

if [[ "$LOG_VAL_GENERATIONS" != "0" ]]; then
    ARGS="$ARGS trainer.log_val_generations=$LOG_VAL_GENERATIONS"
fi

if [[ -n "$VALIDATION_DATA_DIR" ]]; then
    mkdir -p "$VALIDATION_DATA_DIR"
    ARGS="$ARGS trainer.validation_data_dir=$VALIDATION_DATA_DIR"
fi

if [[ "$VALIDATION_GENERATIONS_ONLY" != "False" ]]; then
    ARGS="$ARGS trainer.validation_generations_only=$VALIDATION_GENERATIONS_ONLY"
fi

if [[ -n "$TOTAL_TRAINING_STEPS" ]]; then
    ARGS="$ARGS trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
fi

echo "----------------------------------------------------------------"
echo "Starting Run:AI SDPO worker"
echo "Experiment: $EXP_NAME"
echo "Repo: $REPO_DIR"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Conda env requested: $CONDA_ENV"
echo "----------------------------------------------------------------"

bash training/verl_training.sh "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
