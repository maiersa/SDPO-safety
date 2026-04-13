#!/usr/bin/env bash

set -euo pipefail

MODELS_JSON=${MODELS_JSON:-"scripts/bullshitbench_models.json"}
OUT_DIR=${OUT_DIR:-"/dlabscratch1/${USER}/output/bullshitbench_$(date +%Y%m%d_%H%M%S)"}
REJECTION_JUDGE_MODEL=${REJECTION_JUDGE_MODEL:-"gpt-5.4-mini"}
REJECTION_JUDGE_API_KEY=${REJECTION_JUDGE_API_KEY:-"${OPENAI_API_KEY:-}"}
SKIP_JUDGE=${SKIP_JUDGE:-"True"}
DISABLE_REJECTION_JUDGE=${DISABLE_REJECTION_JUDGE:-"False"}
EXTRA_EVAL_ARGS=${EXTRA_EVAL_ARGS:-""}

CONDA_ENV=${CONDA_ENV:-"default"}
REPO_DIR=${REPO_DIR:-"/dlabscratch1/${USER}/projects/SDPO-safety"}
LOG_DIR=${LOG_DIR:-"/dlabscratch1/${USER}/output"}

mkdir -p "$LOG_DIR/runai_debug"
RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ID="${HOSTNAME:-runai}-${RUN_TS}"
RUN_LOG="$LOG_DIR/runai_debug/${RUN_ID}.log"

exec > >(tee -a "$RUN_LOG") 2>&1

on_error() {
    local exit_code="$?"
    local line_no="$1"
    echo "ERROR: runai_bullshitbench_worker.sh failed at line ${line_no} with exit code ${exit_code}"
    echo "Log file: $RUN_LOG"
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

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
    # shellcheck disable=SC1091
    source /opt/conda/etc/profile.d/conda.sh
    activate_conda_env "$CONDA_ENV"
else
    echo "WARNING: /opt/conda/etc/profile.d/conda.sh not found; skipping conda activation"
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export USER=${USER:-$(whoami)}

mkdir -p "$OUT_DIR"

EVAL_ARGS=(
    --models-json "$MODELS_JSON"
    --output-dir "$OUT_DIR"
    --rejection-judge-model "$REJECTION_JUDGE_MODEL"
)

if [[ -n "$REJECTION_JUDGE_API_KEY" ]]; then
    EVAL_ARGS+=(--rejection-judge-api-key "$REJECTION_JUDGE_API_KEY")
fi

if [[ "$SKIP_JUDGE" != "False" ]]; then
    EVAL_ARGS+=(--skip-judge)
fi

if [[ "$DISABLE_REJECTION_JUDGE" != "False" ]]; then
    EVAL_ARGS+=(--disable-rejection-judge)
fi

if [[ -n "$EXTRA_EVAL_ARGS" ]]; then
    # Intentionally allow word splitting so users can pass normal CLI fragments.
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_EVAL_ARGS)
    EVAL_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "----------------------------------------------------------------"
echo "Starting Run:AI BullshitBench worker"
echo "Repo: $REPO_DIR"
echo "Models JSON: $MODELS_JSON"
echo "Output dir: $OUT_DIR"
echo "Conda env requested: $CONDA_ENV"
echo "Rejection judge model: $REJECTION_JUDGE_MODEL"
echo "Skip constitutional judge: $SKIP_JUDGE"
echo "Disable rejection judge: $DISABLE_REJECTION_JUDGE"
echo "Worker host: ${HOSTNAME:-unknown}"
echo "Worker log: $RUN_LOG"
echo "Python executable: $(command -v python || echo 'python not found')"
python --version || true
echo "----------------------------------------------------------------"

python scripts/eval_bullshitbench_models.py "${EVAL_ARGS[@]}"

python scripts/plot_bullshitbench_results.py \
    --summary-csv "$OUT_DIR/comparison_summary.csv"

echo
echo "BullshitBench results written to: $OUT_DIR"
