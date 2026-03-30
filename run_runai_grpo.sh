#!/bin/bash

set -euo pipefail

# Usage:
#   ./run_runai_grpo.sh [experiment_name_suffix]
#
# Optional environment overrides:
#   RUNAI_BIN                (default: autodetect runai, then runai-rcp-prod)
#   RUNAI_IMAGE              (default: ghcr.io/jkminder/dlab-runai-images/pytorch:master)
#   RUNAI_PVC                (default: runai-dlab-${USER}-scratch:/mnt)
#   RUNAI_GPU                (default: 1.0)
#   RUNAI_CPU                (default: 16)
#   RUNAI_MEMORY             (default: 120G)
#   RUNAI_NODE_POOL          (default: default; use h100 or v100 when needed)
#   RUNAI_LARGE_SHM          (default: true)
#   REPO_DIR                 (default: /dlabscratch1/${USER}/projects/SDPO-safety)
#   CONDA_ENV                (default: $REPO_DIR/.conda if present, otherwise default)
#   LOG_DIR                  (default: /dlabscratch1/${USER}/output)
#   CKPT_DIR                 (default: /dlabscratch1/${USER}/checkpoints)
#   TRAINER_GPUS_PER_NODE    (default: 1)
#   DRY_RUN                  (default: false)
#   WORKER_SCRIPT            (default: /dlabscratch1/${USER}/projects/SDPO-safety/runai_grpo_worker.sh)

CONFIG_NAME="baseline_grpo"
DATA_PATH="datasets/tooluse"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=8
LR=1e-5
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

SUFFIX=${1:-"runai_grpo"}

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
EXP_NAME="RUNAI-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${MODEL_NAME}-${SUFFIX}"
JOB_NAME="grpo-${USER}-${SUFFIX}"

HYDRA_ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=GRPO-runai \
trainer.n_gpus_per_node=$TRAINER_GPUS_PER_NODE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16 \
vars.dir=$REPO_DIR \
vars.log_dir=$LOG_DIR \
vars.ckpt_dir=$CKPT_DIR"
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

RUNAI_CMD+=(-- bash -lc "$RUN_CMD")

echo "----------------------------------------------------------------"
echo "Submitting Run:AI GRPO Job"
echo "RunAI binary: $RUNAI_BIN"
echo "Job name: $JOB_NAME"
echo "Experiment: $EXP_NAME"
echo "Image: $RUNAI_IMAGE"
echo "Node pool: $RUNAI_NODE_POOL"
echo "GPUs: $RUNAI_GPU (trainer.n_gpus_per_node=$TRAINER_GPUS_PER_NODE)"
echo "Conda env: $CONDA_ENV"
echo "Repo dir: $REPO_DIR"
echo "Worker script: $WORKER_SCRIPT"
echo "----------------------------------------------------------------"

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'DRY_RUN command: '
    printf '%q ' "${RUNAI_CMD[@]}"
    echo
else
    "${RUNAI_CMD[@]}"
fi
