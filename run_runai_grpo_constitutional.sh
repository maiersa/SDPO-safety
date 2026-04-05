#!/bin/bash

# RunAI GRPO Launcher - Constitutional (Baseten-style)
# Submits GRPO training with constitutional judge to the RunAI cluster
#
# Usage:
#   OPENAI_API_KEY=sk-... bash run_runai_grpo_constitutional.sh

set -euo pipefail

CONFIG_NAME="baseline_grpo_constitutional"
DATA_PATH="${DATA_PATH:-datasets/beavertails_safety}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
LR=${LR:-1e-5}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B-Instruct-2507"}

# Optional smoke-test knobs
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-"-1"}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-"8"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-"1"}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-""}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-"False"}
TEST_FREQ=${TEST_FREQ:-"50"}
VAL_GENERATION_N=${VAL_GENERATION_N:-"4"}

SUFFIX=${1:-"runai_grpo_constitutional"}

export USER=${USER:-$(whoami)}
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

RUNAI_BIN=${RUNAI_BIN:-""}
RUNAI_IMAGE=${RUNAI_IMAGE:-"ghcr.io/jkminder/dlab-runai-images/pytorch:master"}
RUNAI_PVC=${RUNAI_PVC:-"runai-dlab-${USER}-scratch:/mnt"}
RUNAI_GPU=${RUNAI_GPU:-"1.0"}
RUNAI_CPU=${RUNAI_CPU:-"16"}
RUNAI_MEMORY=${RUNAI_MEMORY:-"120G"}
RUNAI_NODE_POOL=${RUNAI_NODE_POOL:-"default"}
RUNAI_LARGE_SHM=${RUNAI_LARGE_SHM:-"true"}

REPO_DIR=${REPO_DIR:-"/dlabscratch1/${USER}/projects/SDPO-safety"}
if [[ -z "${CONDA_ENV:-}" ]]; then
    if [[ -d "$REPO_DIR/.conda" ]]; then
        CONDA_ENV="$REPO_DIR/.conda"
    else
        CONDA_ENV="default"
    fi
fi
LOG_DIR=${LOG_DIR:-"/dlabscratch1/${USER}/output"}
CKPT_DIR=${CKPT_DIR:-"/dlabscratch1/${USER}/checkpoints"}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-"1"}
DRY_RUN=${DRY_RUN:-"false"}

# Judge configuration (OpenAI gpt-5.4-mini default)
export JUDGE_PROVIDER=${JUDGE_PROVIDER:-openai}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.4-mini}

if [[ -z "$RUNAI_BIN" ]]; then
    if command -v runai >/dev/null 2>&1; then
        RUNAI_BIN="runai"
    elif command -v runai-rcp-prod >/dev/null 2>&1; then
        RUNAI_BIN="runai-rcp-prod"
    elif [[ "$DRY_RUN" == "true" ]]; then
        RUNAI_BIN="runai"
    else
        echo "ERROR: Could not find runai binary. Install Run:AI aliases or set RUNAI_BIN."
        exit 1
    fi
fi

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="RUNAI-GRPO-Constitutional-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${MODEL_NAME}-${SUFFIX}"
JOB_NAME="grpo-const-${USER}-${SUFFIX}"
WORKER_SCRIPT=${WORKER_SCRIPT:-"$REPO_DIR/runai_grpo_worker.sh"}

RUN_CMD="env \
USER=$USER \
SUFFIX=$SUFFIX \
EXP_NAME=$EXP_NAME \
CONFIG_NAME=$CONFIG_NAME \
DATA_PATH=$DATA_PATH \
TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE \
ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE \
MINI_BATCH_SIZE=$MINI_BATCH_SIZE \
LR=$LR \
MODEL_PATH=$MODEL_PATH \
TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES \
VAL_MAX_SAMPLES=$VAL_MAX_SAMPLES \
TOTAL_EPOCHS=$TOTAL_EPOCHS \
TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS \
VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN \
TEST_FREQ=$TEST_FREQ \
VAL_GENERATION_N=$VAL_GENERATION_N \
JUDGE_PROVIDER=$JUDGE_PROVIDER \
JUDGE_MODEL=$JUDGE_MODEL \
CONDA_ENV=$CONDA_ENV \
REPO_DIR=$REPO_DIR \
LOG_DIR=$LOG_DIR \
CKPT_DIR=$CKPT_DIR \
TRAINER_GPUS_PER_NODE=$TRAINER_GPUS_PER_NODE \
bash $WORKER_SCRIPT"

RUNAI_CMD=(
    "$RUNAI_BIN" submit "$JOB_NAME"
    -i "$RUNAI_IMAGE"
    --pvc "$RUNAI_PVC"
    -g "$RUNAI_GPU"
    --cpu "$RUNAI_CPU"
    --memory "$RUNAI_MEMORY"
    --node-pools "$RUNAI_NODE_POOL"
)

if [[ "$RUNAI_LARGE_SHM" == "true" ]]; then
    RUNAI_CMD+=(--large-shm)
fi

# Forward API keys into the job
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    RUN_CMD="export OPENAI_API_KEY=$OPENAI_API_KEY && $RUN_CMD"
fi
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    RUN_CMD="export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY && $RUN_CMD"
fi

RUNAI_CMD+=(-- bash -lc "$RUN_CMD")

echo "================================================================"
echo "Submitting Run:AI GRPO Consti tutional Job"
echo "RunAI binary: $RUNAI_BIN"
echo "Job name: $JOB_NAME"
echo "Experiment: $EXP_NAME"
echo "Image: $RUNAI_IMAGE"
echo "Node pool: $RUNAI_NODE_POOL"
echo "GPUs: $RUNAI_GPU (trainer.n_gpus_per_node=$TRAINER_GPUS_PER_NODE)"
echo "Conda env: $CONDA_ENV"
echo "Repo dir: $REPO_DIR"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Judge: $JUDGE_PROVIDER/$JUDGE_MODEL"
echo "Smoke knobs: train_max_samples=$TRAIN_MAX_SAMPLES val_max_samples=$VAL_MAX_SAMPLES total_steps=${TOTAL_TRAINING_STEPS:-unset}"
echo "Worker script: $WORKER_SCRIPT"
echo "================================================================"

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'DRY_RUN command: '
    printf '%q ' "${RUNAI_CMD[@]}"
    echo
else
    "${RUNAI_CMD[@]}"
fi
