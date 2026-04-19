#!/usr/bin/env bash

set -euo pipefail

export USER=${USER:-$(whoami)}

CONDA_ENV=${CONDA_ENV:-/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo}
REPO_DIR=${REPO_DIR:-/dlabscratch1/${USER}/projects/SDPO-safety}
LOG_DIR=${LOG_DIR:-/dlabscratch1/${USER}/output}
CKPT_DIR=${CKPT_DIR:-/dlabscratch1/${USER}/checkpoints}
WANDB_ENTITY=${WANDB_ENTITY:-samaier-epfl}
DATA_PATH=${DATA_PATH:-datasets/gsm8k}
TOKENIZER_PATH=${TOKENIZER_PATH:-allenai/Olmo-3-7B-Instruct}

CHECKPOINTS=${CHECKPOINTS:-"stage2-step47684 stage2-step32000 stage2-step16000 stage1-step1413814 stage1-step656000"}
LRS=${LRS:-"1e-5"}
MINI_BATCH_SIZES=${MINI_BATCH_SIZES:-"8"}
SEEDS=${SEEDS:-"0"}

STAGE1_ROOT=${STAGE1_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-stage1}
STAGE2_ROOT=${STAGE2_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-stage2}
THINK_SFT_ROOT=${THINK_SFT_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-think-sft}
INSTRUCT_SFT_ROOT=${INSTRUCT_SFT_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-instruct-sft}
INSTRUCT_DPO_ROOT=${INSTRUCT_DPO_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-instruct-dpo}
INSTRUCT_RL_ROOT=${INSTRUCT_RL_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-instruct}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
VAL_ONLY=${VAL_ONLY:-False}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-4}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
VAL_GENERATION_N=${VAL_GENERATION_N:-8}
VALIDATION_GENERATIONS_ONLY=${VALIDATION_GENERATIONS_ONLY:-False}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-1024}
DATA_MAX_RESPONSE_LENGTH=${DATA_MAX_RESPONSE_LENGTH:-1024}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}
ACTOR_PPO_MICRO_BATCH_SIZE=${ACTOR_PPO_MICRO_BATCH_SIZE:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-false}
SKIP_COMPLETED=${SKIP_COMPLETED:-true}
RESUME_MODE=${RESUME_MODE:-auto}

normalize_list() {
    local raw="$1"
    raw="${raw//,/ }"
    echo "$raw" | xargs
}

CHECKPOINTS="$(normalize_list "$CHECKPOINTS")"
LRS="$(normalize_list "$LRS")"
MINI_BATCH_SIZES="$(normalize_list "$MINI_BATCH_SIZES")"
SEEDS="$(normalize_list "$SEEDS")"

resolve_model_path() {
    local ckpt="$1"
    local variant revision

    if [[ -d "$ckpt" ]]; then
        printf '%s' "$ckpt"
        return 0
    fi

    if [[ "$ckpt" == stage2-* ]]; then
        printf '%s' "${STAGE2_ROOT}/${ckpt}"
        return 0
    fi

    if [[ "$ckpt" == stage1-* ]]; then
        printf '%s' "${STAGE1_ROOT}/${ckpt}"
        return 0
    fi

    if [[ "$ckpt" == *"@"* ]]; then
        variant="${ckpt%%@*}"
        revision="${ckpt#*@}"
        case "$variant" in
            think-sft)
                printf '%s' "${THINK_SFT_ROOT}/${revision}"
                ;;
            instruct-sft)
                printf '%s' "${INSTRUCT_SFT_ROOT}/${revision}"
                ;;
            instruct-dpo)
                printf '%s' "${INSTRUCT_DPO_ROOT}/${revision}"
                ;;
            instruct|instruct-rl)
                printf '%s' "${INSTRUCT_RL_ROOT}/${revision}"
                ;;
            *)
                printf '%s' "$ckpt"
                ;;
        esac
        return 0
    fi

    printf '%s' "$ckpt"
}

resolve_tokenizer_path() {
    local ckpt="$1"
    local model_path="$2"

    case "$ckpt" in
        think-sft@*|instruct-sft@*|instruct-dpo@*|instruct@*|instruct-rl@*)
            printf '%s' "$model_path"
            ;;
        *)
            printf '%s' "$TOKENIZER_PATH"
            ;;
    esac
}

sanitize_ckpt_label() {
    local ckpt="$1"
    ckpt="${ckpt//@/-}"
    ckpt="${ckpt//\//-}"
    printf '%s' "$ckpt"
}

checkpoint_dir_for_exp() {
    local exp_name="$1"
    printf '%s' "${CKPT_DIR}/${exp_name}"
}

latest_saved_step() {
    local exp_dir="$1"
    local tracker_file="$exp_dir/latest_checkpointed_iteration.txt"
    if [[ -f "$tracker_file" ]]; then
        tr -d '[:space:]' < "$tracker_file"
        return 0
    fi

    local latest_dir
    latest_dir="$(find "$exp_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    if [[ -n "$latest_dir" ]]; then
        basename "$latest_dir" | sed 's/^global_step_//'
        return 0
    fi

    return 1
}

is_completed_run() {
    local exp_dir="$1"
    local latest_step
    if ! latest_step="$(latest_saved_step "$exp_dir")"; then
        return 1
    fi

    [[ -n "${TOTAL_TRAINING_STEPS}" && "$latest_step" -ge "$TOTAL_TRAINING_STEPS" ]]
}

run_one() {
    local ckpt="$1"
    local lr="$2"
    local mini_batch="$3"
    local seed="$4"
    local model_path tokenizer_path ckpt_label exp_name suffix val_dir exp_ckpt_dir latest_step

    model_path="$(resolve_model_path "$ckpt")"
    if [[ ! -d "$model_path" ]]; then
        echo "Missing checkpoint directory: $model_path" >&2
        return 1
    fi
    tokenizer_path="$(resolve_tokenizer_path "$ckpt" "$model_path")"
    ckpt_label="$(sanitize_ckpt_label "$ckpt")"

    exp_name="GRPO-REVERSE-gsm8k-${ckpt_label}-lr${lr}-mb${mini_batch}-seed${seed}"
    suffix="grpo_reverse_gsm8k_${ckpt_label}_lr${lr}_mb${mini_batch}_seed${seed}"
    val_dir="${LOG_DIR}/validation_generations/${exp_name}"
    exp_ckpt_dir="$(checkpoint_dir_for_exp "$exp_name")"

    if [[ "$SKIP_COMPLETED" == "true" ]] && [[ -d "$exp_ckpt_dir" ]] && is_completed_run "$exp_ckpt_dir"; then
        latest_step="$(latest_saved_step "$exp_ckpt_dir")"
        echo "=============================================================="
        echo "Skipping completed GRPO sweep item"
        echo "Experiment: $exp_name"
        echo "Checkpoint dir: $exp_ckpt_dir"
        echo "Latest saved step: $latest_step"
        echo "=============================================================="
        return 0
    fi

    if [[ -d "$exp_ckpt_dir" ]] && latest_step="$(latest_saved_step "$exp_ckpt_dir")"; then
        echo "Found existing checkpoint state for $exp_name at step $latest_step"
        echo "Resume mode: $RESUME_MODE"
    fi

    echo "=============================================================="
    echo "Running GRPO sweep item"
    echo "Checkpoint: $ckpt"
    echo "Model path: $model_path"
    echo "Tokenizer path: $tokenizer_path"
    echo "LR: $lr | Mini-batch: $mini_batch | Seed: $seed"
    echo "Experiment: $exp_name"
    echo "=============================================================="

    USER=$USER \
    CONDA_ENV=$CONDA_ENV \
    CONFIG_NAME=baseline_grpo \
    DATA_PATH=$DATA_PATH \
    SUFFIX=$suffix \
    EXP_NAME=$exp_name \
    REPO_DIR=$REPO_DIR \
    LOG_DIR=$LOG_DIR \
    CKPT_DIR=$CKPT_DIR \
    TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE \
    ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE \
    MINI_BATCH_SIZE=$mini_batch \
    LR=$lr \
    MODEL_PATH=$model_path \
    TOKENIZER_PATH=$tokenizer_path \
    TRAIN_MAX_SAMPLES=-1 \
    VAL_MAX_SAMPLES=-1 \
    TOTAL_EPOCHS=$TOTAL_EPOCHS \
    TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS \
    VAL_ONLY=$VAL_ONLY \
    VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN \
    TEST_FREQ=$TEST_FREQ \
    SAVE_FREQ=$SAVE_FREQ \
    VAL_GENERATION_N=$VAL_GENERATION_N \
    TRAINER_GPUS_PER_NODE=$TRAINER_GPUS_PER_NODE \
    WANDB_ENTITY=$WANDB_ENTITY \
    LOG_VAL_GENERATIONS=$LOG_VAL_GENERATIONS \
    VALIDATION_DATA_DIR=$val_dir \
    VALIDATION_GENERATIONS_ONLY=$VALIDATION_GENERATIONS_ONLY \
    MAX_MODEL_LEN=$MAX_MODEL_LEN \
    DATA_MAX_PROMPT_LENGTH=$DATA_MAX_PROMPT_LENGTH \
    DATA_MAX_RESPONSE_LENGTH=$DATA_MAX_RESPONSE_LENGTH \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE \
    ACTOR_PPO_MICRO_BATCH_SIZE=$ACTOR_PPO_MICRO_BATCH_SIZE \
    SKIP_COMPLETED=$SKIP_COMPLETED \
    RESUME_MODE=$RESUME_MODE \
    bash "$REPO_DIR/runai_grpo_worker.sh"
}

for ckpt in ${CHECKPOINTS}; do
    for lr in ${LRS}; do
        for mini_batch in ${MINI_BATCH_SIZES}; do
            for seed in ${SEEDS}; do
                if ! run_one "$ckpt" "$lr" "$mini_batch" "$seed"; then
                    if [[ "$CONTINUE_ON_ERROR" == "true" ]]; then
                        echo "Continuing after failure for ${ckpt}" >&2
                    else
                        exit 1
                    fi
                fi
            done
        done
    done
done
