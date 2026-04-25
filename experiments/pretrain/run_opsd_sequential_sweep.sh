#!/usr/bin/env bash

set -euo pipefail

export USER=${USER:-$(whoami)}

CONDA_ENV=${CONDA_ENV:-/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo}
REPO_DIR=${REPO_DIR:-/dlabscratch1/${USER}/projects/SDPO-safety}
LOG_DIR=${LOG_DIR:-/dlabscratch1/${USER}/output}
CKPT_DIR=${CKPT_DIR:-/dlabscratch1/${USER}/checkpoints}
WANDB_ENTITY=${WANDB_ENTITY:-samaier-epfl}
DATA_PATH=${DATA_PATH:-datasets/gsm8k}
TOKENIZER_PATH=${TOKENIZER_PATH:-}
CONFIG_NAME=${CONFIG_NAME:-sdpo_math_teacher}
EXP_PREFIX=${EXP_PREFIX:-OPSD-REVERSE}
SUFFIX_PREFIX=${SUFFIX_PREFIX:-opsd_reverse}

CHECKPOINTS=${CHECKPOINTS:-"stage2-step47684 stage2-step32000 stage2-step16000 stage1-step1413814 stage1-step656000"}
LRS=${LRS:-"1e-5"}
ALPHAS=${ALPHAS:-"0.5"}
DISTILLATION_TOPKS=${DISTILLATION_TOPKS:-"100"}
SEEDS=${SEEDS:-"0"}

STAGE1_ROOT=${STAGE1_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-stage1}
STAGE2_ROOT=${STAGE2_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-stage2}
STAGE3_ROOT=${STAGE3_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-stage3}
THINK_SFT_ROOT=${THINK_SFT_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-think-sft}
INSTRUCT_SFT_ROOT=${INSTRUCT_SFT_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-instruct-sft}
INSTRUCT_DPO_ROOT=${INSTRUCT_DPO_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-instruct-dpo}
INSTRUCT_RL_ROOT=${INSTRUCT_RL_ROOT:-/dlabscratch1/${USER}/checkpoints/olmo-7b-instruct}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-50}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-4}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
VAL_GENERATION_N=${VAL_GENERATION_N:-8}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-}
VAL_TOP_K=${VAL_TOP_K:-}
VALIDATION_GENERATIONS_ONLY=${VALIDATION_GENERATIONS_ONLY:-False}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-1024}
DATA_MAX_RESPONSE_LENGTH=${DATA_MAX_RESPONSE_LENGTH:-768}
DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS:-True}
ROLLOUT_SOURCE=${ROLLOUT_SOURCE:-student}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-false}
SKIP_COMPLETED=${SKIP_COMPLETED:-true}
RESUME_MODE=${RESUME_MODE:-auto}

# Leave these empty by default for math-teacher OPSD runs.
# The privileged teacher prompt comes from teacher_prompt_type=math_reference in the config,
# so we do not need Hydra CLI overrides containing literal braces.
SELF_DISTILL_REPROMPT_TEMPLATE=${SELF_DISTILL_REPROMPT_TEMPLATE:-}
SELF_DISTILL_SOLUTION_TEMPLATE=${SELF_DISTILL_SOLUTION_TEMPLATE:-}
SELF_DISTILL_FEEDBACK_TEMPLATE=${SELF_DISTILL_FEEDBACK_TEMPLATE:-}

normalize_list() {
    local raw="$1"
    raw="${raw//,/ }"
    # Collapse repeated whitespace so bash word-splitting behaves predictably.
    echo "$raw" | xargs
}

dataset_label() {
    local raw="$1"
    local label

    label="$(basename "$raw")"
    label="${label%/}"
    label="${label// /-}"
    label="${label//\//-}"
    printf '%s' "$label"
}

CHECKPOINTS="$(normalize_list "$CHECKPOINTS")"
LRS="$(normalize_list "$LRS")"
ALPHAS="$(normalize_list "$ALPHAS")"
DISTILLATION_TOPKS="$(normalize_list "$DISTILLATION_TOPKS")"
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

    if [[ "$ckpt" == stage3-* ]]; then
        printf '%s' "${STAGE3_ROOT}/${ckpt}"
        return 0
    fi

    if [[ "$ckpt" == stage1-* ]]; then
        printf '%s' "${STAGE1_ROOT}/${ckpt}"
        return 0
    fi

    # Allow a Stage 3 sweep to reference the final checkpoint as "main"
    # while keeping the rest of the list in the stage3-step* naming scheme.
    if [[ "$ckpt" == main && -d "${STAGE3_ROOT}/main" ]]; then
        printf '%s' "${STAGE3_ROOT}/main"
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
    local alpha="$3"
    local topk="$4"
    local seed="$5"
    local model_path tokenizer_path ckpt_label data_label exp_name suffix val_dir exp_ckpt_dir latest_step

    model_path="$(resolve_model_path "$ckpt")"
    if [[ ! -d "$model_path" ]]; then
        echo "Missing checkpoint directory: $model_path" >&2
        return 1
    fi
    tokenizer_path="$TOKENIZER_PATH"
    ckpt_label="$(sanitize_ckpt_label "$ckpt")"
    data_label="$(dataset_label "$DATA_PATH")"

    exp_name="${EXP_PREFIX}-${data_label}-${ckpt_label}-lr${lr}-a${alpha}-k${topk}-seed${seed}"
    suffix="${SUFFIX_PREFIX}_${data_label}_${ckpt_label}_lr${lr}_a${alpha}_k${topk}_seed${seed}"
    val_dir="${LOG_DIR}/validation_generations/${exp_name}"
    exp_ckpt_dir="$(checkpoint_dir_for_exp "$exp_name")"

    if [[ "$SKIP_COMPLETED" == "true" ]] && [[ -d "$exp_ckpt_dir" ]] && is_completed_run "$exp_ckpt_dir"; then
        latest_step="$(latest_saved_step "$exp_ckpt_dir")"
        echo "=============================================================="
        echo "Skipping completed OPSD sweep item"
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
    echo "Running OPSD sweep item"
    echo "Checkpoint: $ckpt"
    echo "Model path: $model_path"
    echo "Tokenizer path: ${tokenizer_path:-<checkpoint default>}"
    echo "LR: $lr | Alpha: $alpha | Top-k: $topk | Seed: $seed"
    echo "Experiment: $exp_name"
    echo "=============================================================="

    USER=$USER \
    CONDA_ENV=$CONDA_ENV \
    CONFIG_NAME=$CONFIG_NAME \
    DATA_PATH=$DATA_PATH \
    SUFFIX=$suffix \
    EXP_NAME=$exp_name \
    REPO_DIR=$REPO_DIR \
    LOG_DIR=$LOG_DIR \
    CKPT_DIR=$CKPT_DIR \
    TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE \
    ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE \
    PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE \
    LR=$lr \
    ALPHA=$alpha \
    DISTILLATION_TOPK=$topk \
    DONT_REPROMPT_ON_SELF_SUCCESS=$DONT_REPROMPT_ON_SELF_SUCCESS \
    ROLLOUT_SOURCE=$ROLLOUT_SOURCE \
    MODEL_PATH=$model_path \
    TOKENIZER_PATH=$tokenizer_path \
    TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES \
    VAL_MAX_SAMPLES=$VAL_MAX_SAMPLES \
    TOTAL_EPOCHS=$TOTAL_EPOCHS \
    TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS \
    VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN \
    TEST_FREQ=$TEST_FREQ \
    SAVE_FREQ=$SAVE_FREQ \
    TRAINER_GPUS_PER_NODE=$TRAINER_GPUS_PER_NODE \
    WANDB_ENTITY=$WANDB_ENTITY \
    LOG_VAL_GENERATIONS=$LOG_VAL_GENERATIONS \
    VAL_GENERATION_N=$VAL_GENERATION_N \
    VAL_DO_SAMPLE=$VAL_DO_SAMPLE \
    VAL_TEMPERATURE=$VAL_TEMPERATURE \
    VAL_TOP_P=$VAL_TOP_P \
    VAL_TOP_K=$VAL_TOP_K \
    VALIDATION_DATA_DIR=$val_dir \
    VALIDATION_GENERATIONS_ONLY=$VALIDATION_GENERATIONS_ONLY \
    MAX_MODEL_LEN=$MAX_MODEL_LEN \
    DATA_MAX_PROMPT_LENGTH=$DATA_MAX_PROMPT_LENGTH \
    DATA_MAX_RESPONSE_LENGTH=$DATA_MAX_RESPONSE_LENGTH \
    RESUME_MODE=$RESUME_MODE \
    SELF_DISTILL_REPROMPT_TEMPLATE=$SELF_DISTILL_REPROMPT_TEMPLATE \
    SELF_DISTILL_SOLUTION_TEMPLATE=$SELF_DISTILL_SOLUTION_TEMPLATE \
    SELF_DISTILL_FEEDBACK_TEMPLATE=$SELF_DISTILL_FEEDBACK_TEMPLATE \
    bash "$REPO_DIR/runai_sdpo_worker.sh"
}

for ckpt in ${CHECKPOINTS}; do
    for lr in ${LRS}; do
        for alpha in ${ALPHAS}; do
            for topk in ${DISTILLATION_TOPKS}; do
                for seed in ${SEEDS}; do
                    if ! run_one "$ckpt" "$lr" "$alpha" "$topk" "$seed"; then
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
done
