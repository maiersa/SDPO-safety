#!/usr/bin/env bash

set -euo pipefail

# Minimal rs-based submitter.
# Usage:
#   ./run_rs_grpo_submit.sh [job_name]
#
# Optional env vars:
#   GPU=4.0
#   CPU=32
#   MEMORY=240G
#   NODE_POOL=h100
#   LARGE_SHM=true
#   WORKER_SCRIPT=/dlabscratch1/${USER}/projects/SDPO-safety/runai_grpo_worker.sh
#   DRY_RUN=false

JOB_NAME=${1:-"sdpo-grpo-${USER}"}

GPU=${GPU:-"4.0"}
CPU=${CPU:-"32"}
MEMORY=${MEMORY:-"240G"}
NODE_POOL=${NODE_POOL:-"h100"}
LARGE_SHM=${LARGE_SHM:-"true"}
WORKER_SCRIPT=${WORKER_SCRIPT:-"/dlabscratch1/${USER}/projects/SDPO-safety/runai_grpo_worker.sh"}
DRY_RUN=${DRY_RUN:-"false"}

if ! command -v rs >/dev/null 2>&1; then
    echo "ERROR: rs alias was not found. Source your .runai_aliases first."
    exit 1
fi

CMD=(
    rs "$JOB_NAME"
    --gpu "$GPU"
    --cpu "$CPU"
    --memory "$MEMORY"
    --node-pools "$NODE_POOL"
)

if [[ "$LARGE_SHM" == "true" ]]; then
    CMD+=(--large-shm)
fi

CMD+=(-- bash "$WORKER_SCRIPT")

echo "Submitting with rs"
echo "job=$JOB_NAME gpu=$GPU cpu=$CPU memory=$MEMORY node_pool=$NODE_POOL worker=$WORKER_SCRIPT"

if [[ "$DRY_RUN" == "true" ]]; then
    printf 'DRY_RUN command: '
    printf '%q ' "${CMD[@]}"
    echo
else
    "${CMD[@]}"
fi
