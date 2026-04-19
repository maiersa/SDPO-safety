#!/bin/bash

# Stage 1 launcher using the same rs-based submission style as the working Baseten commands.
#
# Examples:
#   DRY_RUN=true MODE=base_eval SINGLE_CHECKPOINT=stage1-step1000 SINGLE_DATASET=datasets/gsm8k \
#     MODEL_ROOT=/dlabscratch1/$USER/checkpoints/olmo-32b-stage1 \
#     bash experiments/stage1/run_stage1_threshold.sh
#
#   MODE=sdpo MODEL_ROOT=/dlabscratch1/$USER/checkpoints/olmo-32b-stage1 \
#     bash experiments/stage1/run_stage1_threshold.sh

set -euo pipefail

MODE="${MODE:-all}"
export USER=${USER:-$(whoami)}

MODEL_ROOT="${MODEL_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-stage1}"
GSM8K_DATA="${GSM8K_DATA:-datasets/gsm8k}"
MATH_DATA="${MATH_DATA:-datasets/math}"

CONDA_ENV="${CONDA_ENV:-/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo}"
REPO_DIR="${REPO_DIR:-/dlabscratch1/${USER}/projects/SDPO-safety}"
LOG_DIR="${LOG_DIR:-/dlabscratch1/${USER}/output}"
CKPT_DIR="${CKPT_DIR:-/dlabscratch1/${USER}/checkpoints}"
WANDB_ENTITY="${WANDB_ENTITY:-samaier-epfl}"

RS_BIN="${RS_BIN:-rs}"
RUNAI_GPU="${RUNAI_GPU:-4.0}"
RUNAI_CPU="${RUNAI_CPU:-16}"
RUNAI_MEMORY="${RUNAI_MEMORY:-120G}"
RUNAI_NODE_POOL="${RUNAI_NODE_POOL:-h100}"
RUNAI_LARGE_SHM="${RUNAI_LARGE_SHM:-true}"
TRAINER_GPUS_PER_NODE="${TRAINER_GPUS_PER_NODE:-4}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
SDPO_MINI_BATCH_SIZE="${SDPO_MINI_BATCH_SIZE:-32}"
GRPO_MINI_BATCH_SIZE="${GRPO_MINI_BATCH_SIZE:-8}"
LR="${LR:-1e-5}"
SDPO_ALPHA="${SDPO_ALPHA:-0.5}"
DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
BASE_VAL_MAX_SAMPLES="${BASE_VAL_MAX_SAMPLES:--1}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-8}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-/dlabscratch1/${USER}/output/validation_generations}"
VAL_GENERATION_N="${VAL_GENERATION_N:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-1024}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-2048}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="${SMOKE_TEST:-false}"

CHECKPOINTS=(
    "stage1-step1000"
    "stage1-step4000"
    "stage1-step16000"
    "stage1-step64000"
    "stage1-step128000"
    "stage1-step256000"
    "stage1-step656000"
)

GRPO_CHECKPOINTS=(
    "stage1-step16000"
    "stage1-step64000"
    "stage1-step256000"
)

DATASETS=(
    "$GSM8K_DATA"
    "$MATH_DATA"
)

SEEDS=(0)

if [[ -n "${SINGLE_CHECKPOINT:-}" ]]; then
    CHECKPOINTS=("$SINGLE_CHECKPOINT")
fi

if [[ -n "${GRPO_SINGLE_CHECKPOINT:-}" ]]; then
    GRPO_CHECKPOINTS=("$GRPO_SINGLE_CHECKPOINT")
elif [[ -n "${SINGLE_CHECKPOINT:-}" ]]; then
    GRPO_CHECKPOINTS=("$SINGLE_CHECKPOINT")
fi

if [[ -n "${SINGLE_DATASET:-}" ]]; then
    DATASETS=("$SINGLE_DATASET")
fi

if [[ "$SMOKE_TEST" == "true" ]]; then
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    SDPO_MINI_BATCH_SIZE="${SDPO_MINI_BATCH_SIZE:-8}"
    GRPO_MINI_BATCH_SIZE="${GRPO_MINI_BATCH_SIZE:-8}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
    BASE_VAL_MAX_SAMPLES="${BASE_VAL_MAX_SAMPLES:-8}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-512}"
    DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-1024}"
    ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
fi

print_or_run() {
    local cmd="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        printf '%s\n\n' "$cmd"
    else
        eval "$cmd"
    fi
}

dataset_name() {
    basename "$1"
}

large_shm_flag() {
    if [[ "$RUNAI_LARGE_SHM" == "true" ]]; then
        printf '%s' '--large-shm'
    fi
}

submit_base_eval() {
    local ckpt="$1"
    local data_path="$2"
    local seed="$3"
    local data_name exp_name job_name model_path suffix large_shm

    data_name="$(dataset_name "$data_path")"
    exp_name="STAGE1-BASE-${data_name}-${ckpt}-seed${seed}"
    suffix="stage1_base_${data_name}_${ckpt}_seed${seed}"
    job_name="stage1-base-${USER}-${data_name}-${ckpt}"
    model_path="${MODEL_ROOT}/${ckpt}"
    large_shm="$(large_shm_flag)"

    print_or_run "${RS_BIN} ${job_name} --gpu ${RUNAI_GPU} --cpu ${RUNAI_CPU} --memory ${RUNAI_MEMORY} --node-pools ${RUNAI_NODE_POOL} ${large_shm} -- env \
USER=$USER \
CONDA_ENV=${CONDA_ENV} \
CONFIG_NAME=baseline_grpo \
DATA_PATH=${data_path} \
SUFFIX=${suffix} \
EXP_NAME=${exp_name} \
REPO_DIR=${REPO_DIR} \
LOG_DIR=${LOG_DIR} \
CKPT_DIR=${CKPT_DIR} \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} \
MINI_BATCH_SIZE=${GRPO_MINI_BATCH_SIZE} \
LR=${LR} \
MODEL_PATH=${model_path} \
TRAIN_MAX_SAMPLES=-1 \
VAL_MAX_SAMPLES=${BASE_VAL_MAX_SAMPLES} \
TOTAL_EPOCHS=1 \
VAL_ONLY=True \
VAL_BEFORE_TRAIN=True \
TEST_FREQ=-1 \
SAVE_FREQ=-1 \
VAL_GENERATION_N=${VAL_GENERATION_N} \
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE} \
WANDB_ENTITY=${WANDB_ENTITY} \
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS} \
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR}/${exp_name} \
VALIDATION_GENERATIONS_ONLY=False \
MAX_MODEL_LEN=${MAX_MODEL_LEN} \
DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH} \
DATA_MAX_RESPONSE_LENGTH=${DATA_MAX_RESPONSE_LENGTH} \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
bash ${REPO_DIR}/runai_grpo_worker.sh"
}

submit_sdpo() {
    local ckpt="$1"
    local data_path="$2"
    local seed="$3"
    local data_name exp_name job_name model_path suffix large_shm

    data_name="$(dataset_name "$data_path")"
    exp_name="STAGE1-SDPO-${data_name}-${ckpt}-seed${seed}"
    suffix="stage1_sdpo_${data_name}_${ckpt}_seed${seed}"
    job_name="stage1-sdpo-${USER}-${data_name}-${ckpt}"
    model_path="${MODEL_ROOT}/${ckpt}"
    large_shm="$(large_shm_flag)"

    print_or_run "${RS_BIN} ${job_name} --gpu ${RUNAI_GPU} --cpu ${RUNAI_CPU} --memory ${RUNAI_MEMORY} --node-pools ${RUNAI_NODE_POOL} ${large_shm} -- env \
USER=$USER \
CONDA_ENV=${CONDA_ENV} \
CONFIG_NAME=sdpo \
DATA_PATH=${data_path} \
SUFFIX=${suffix} \
EXP_NAME=${exp_name} \
REPO_DIR=${REPO_DIR} \
LOG_DIR=${LOG_DIR} \
CKPT_DIR=${CKPT_DIR} \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} \
PPO_MINI_BATCH_SIZE=${SDPO_MINI_BATCH_SIZE} \
LR=${LR} \
ALPHA=${SDPO_ALPHA} \
DISTILLATION_TOPK=${DISTILLATION_TOPK} \
DONT_REPROMPT_ON_SELF_SUCCESS=True \
MODEL_PATH=${model_path} \
TRAIN_MAX_SAMPLES=-1 \
VAL_MAX_SAMPLES=-1 \
TOTAL_EPOCHS=${TOTAL_EPOCHS} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} \
VAL_BEFORE_TRAIN=True \
TEST_FREQ=${TOTAL_TRAINING_STEPS} \
SAVE_FREQ=50 \
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE} \
WANDB_ENTITY=${WANDB_ENTITY} \
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS} \
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR}/${exp_name} \
VALIDATION_GENERATIONS_ONLY=False \
MAX_MODEL_LEN=${MAX_MODEL_LEN} \
DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH} \
DATA_MAX_RESPONSE_LENGTH=${DATA_MAX_RESPONSE_LENGTH} \
bash ${REPO_DIR}/runai_sdpo_worker.sh"
}

submit_grpo() {
    local ckpt="$1"
    local data_path="$2"
    local seed="$3"
    local data_name exp_name job_name model_path suffix large_shm

    data_name="$(dataset_name "$data_path")"
    exp_name="STAGE1-GRPO-${data_name}-${ckpt}-seed${seed}"
    suffix="stage1_grpo_${data_name}_${ckpt}_seed${seed}"
    job_name="stage1-grpo-${USER}-${data_name}-${ckpt}"
    model_path="${MODEL_ROOT}/${ckpt}"
    large_shm="$(large_shm_flag)"

    print_or_run "${RS_BIN} ${job_name} --gpu ${RUNAI_GPU} --cpu ${RUNAI_CPU} --memory ${RUNAI_MEMORY} --node-pools ${RUNAI_NODE_POOL} ${large_shm} -- env \
USER=$USER \
CONDA_ENV=${CONDA_ENV} \
CONFIG_NAME=baseline_grpo \
DATA_PATH=${data_path} \
SUFFIX=${suffix} \
EXP_NAME=${exp_name} \
REPO_DIR=${REPO_DIR} \
LOG_DIR=${LOG_DIR} \
CKPT_DIR=${CKPT_DIR} \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} \
MINI_BATCH_SIZE=${GRPO_MINI_BATCH_SIZE} \
LR=${LR} \
MODEL_PATH=${model_path} \
TRAIN_MAX_SAMPLES=-1 \
VAL_MAX_SAMPLES=-1 \
TOTAL_EPOCHS=${TOTAL_EPOCHS} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} \
VAL_BEFORE_TRAIN=True \
TEST_FREQ=${TOTAL_TRAINING_STEPS} \
SAVE_FREQ=50 \
VAL_GENERATION_N=${VAL_GENERATION_N} \
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE} \
WANDB_ENTITY=${WANDB_ENTITY} \
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS} \
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR}/${exp_name} \
VALIDATION_GENERATIONS_ONLY=False \
MAX_MODEL_LEN=${MAX_MODEL_LEN} \
DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH} \
DATA_MAX_RESPONSE_LENGTH=${DATA_MAX_RESPONSE_LENGTH} \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
bash ${REPO_DIR}/runai_grpo_worker.sh"
}

if [[ "$MODE" == "base_eval" || "$MODE" == "all" ]]; then
    for ckpt in "${CHECKPOINTS[@]}"; do
        for data_path in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                submit_base_eval "$ckpt" "$data_path" "$seed"
            done
        done
    done
fi

if [[ "$MODE" == "sdpo" || "$MODE" == "all" ]]; then
    for ckpt in "${CHECKPOINTS[@]}"; do
        for data_path in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                submit_sdpo "$ckpt" "$data_path" "$seed"
            done
        done
    done
fi

if [[ "$MODE" == "grpo_reference" || "$MODE" == "all" ]]; then
    for ckpt in "${GRPO_CHECKPOINTS[@]}"; do
        for data_path in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                submit_grpo "$ckpt" "$data_path" "$seed"
            done
        done
    done
fi
