#!/usr/bin/env bash

set -euo pipefail

export USER=${USER:-$(whoami)}

CONDA_ENV=${CONDA_ENV:-/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo}
REPO_DIR=${REPO_DIR:-/dlabscratch1/${USER}/projects/SDPO-safety}
LOG_DIR=${LOG_DIR:-/dlabscratch1/${USER}/output}
CKPT_DIR=${CKPT_DIR:-/dlabscratch1/${USER}/checkpoints}

STAGE1_ROOT=${STAGE1_ROOT:-${CKPT_DIR}/olmo-7b-stage1}
STAGE2_ROOT=${STAGE2_ROOT:-${CKPT_DIR}/olmo-7b-stage2}
STAGE3_ROOT=${STAGE3_ROOT:-${CKPT_DIR}/olmo-7b-stage3}
THINK_ROOT=${THINK_ROOT:-${CKPT_DIR}/olmo-7b-think}
THINK_DPO_ROOT=${THINK_DPO_ROOT:-${CKPT_DIR}/olmo-7b-think-dpo}
THINK_SFT_ROOT=${THINK_SFT_ROOT:-${CKPT_DIR}/olmo-7b-think-sft}
INSTRUCT_SFT_ROOT=${INSTRUCT_SFT_ROOT:-${CKPT_DIR}/olmo-7b-instruct-sft}
INSTRUCT_DPO_ROOT=${INSTRUCT_DPO_ROOT:-${CKPT_DIR}/olmo-7b-instruct-dpo}
INSTRUCT_RL_ROOT=${INSTRUCT_RL_ROOT:-${CKPT_DIR}/olmo-7b-instruct}
RL_ZERO_MATH_ROOT=${RL_ZERO_MATH_ROOT:-${CKPT_DIR}/olmo-7b-rl-zero-math}

CHECKPOINTS=${CHECKPOINTS:-"stage1-step500000 stage1-step656000"}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-${ALIGNMENT_GPUS:-1}}
ALIGNMENT_GPUS=${ALIGNMENT_GPUS:-$TRAINER_GPUS_PER_NODE}
ALIGNMENT_GPUS=${ALIGNMENT_GPUS:-1}

ALIGNMENT_EXP_NAME=${ALIGNMENT_EXP_NAME:-opsd_alignment_smoke}
ALIGNMENT_OUTPUT_DIR=${ALIGNMENT_OUTPUT_DIR:-${LOG_DIR}/opsd_alignment/${ALIGNMENT_EXP_NAME}}
ALIGNMENT_CONFIG=${ALIGNMENT_CONFIG:-${ALIGNMENT_OUTPUT_DIR}/config.yaml}
QUESTIONS_PATH=${QUESTIONS_PATH:-${REPO_DIR}/opsd_alignment/data/questions.jsonl}

NUM_SYNTHETIC=${NUM_SYNTHETIC:-20}
QUESTION_LIMIT=${QUESTION_LIMIT:-1}
STUDENT_ROLLOUTS_PER_QUESTION=${STUDENT_ROLLOUTS_PER_QUESTION:-1}
NODES_PER_ROLLOUT=${NODES_PER_ROLLOUT:-1}
TOP_K_STUDENT=${TOP_K_STUDENT:-2}
TOP_K_TEACHER=${TOP_K_TEACHER:-2}
FORCED_ROLLOUTS_PER_CANDIDATE=${FORCED_ROLLOUTS_PER_CANDIDATE:-2}
MAX_POSITIONS_PER_ROLLOUT=${MAX_POSITIONS_PER_ROLLOUT:-32}
DISTILLATION_OBJECTIVE=${DISTILLATION_OBJECTIVE:-forward_kl}
JSD_ALPHA=${JSD_ALPHA:-0.5}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
TORCH_DTYPE=${TORCH_DTYPE:-bf16}
GENERATION_DEVICE=${GENERATION_DEVICE:-auto}
SHARD_DEVICE=${SHARD_DEVICE:-cuda:0}
TEACHER_CONTEXT_FOR_SELECTION=${TEACHER_CONTEXT_FOR_SELECTION:-full_solution}
SKIP_COMPLETED=${SKIP_COMPLETED:-true}
OVERWRITE=${OVERWRITE:-false}

normalize_list() {
    local raw="$1"
    raw="${raw//,/ }"
    echo "$raw" | xargs
}

activate_conda_env() {
    local requested="$1"
    if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
        source /opt/conda/etc/profile.d/conda.sh
        conda activate "$requested"
        echo "Activated conda env: $requested"
    else
        echo "WARNING: /opt/conda/etc/profile.d/conda.sh not found; skipping conda activation"
    fi
}

run_sharded_stage() {
    local stage_name="$1"
    shift
    local num_shards="$ALIGNMENT_GPUS"
    local -a pids=()

    if [[ "$num_shards" -le 1 ]]; then
        "$@" --num-shards 1 --shard-index 0
        return 0
    fi

    echo "Running ${stage_name} across ${num_shards} GPU shards"
    for shard_idx in $(seq 0 $((num_shards - 1))); do
        (
            export CUDA_VISIBLE_DEVICES="$shard_idx"
            "$@" --num-shards "$num_shards" --shard-index "$shard_idx"
        ) &
        pids+=("$!")
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "ERROR: ${stage_name} failed in at least one shard" >&2
        return 1
    fi
}

overwrite_args=()
if [[ "$OVERWRITE" == "true" ]]; then
    overwrite_args=(--overwrite)
fi

mkdir -p "$ALIGNMENT_OUTPUT_DIR" "$LOG_DIR/opsd_alignment"
RUN_LOG="${ALIGNMENT_OUTPUT_DIR}/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

trap 'echo "ERROR: run_opsd_alignment_smoke.sh failed at line $LINENO" >&2' ERR

activate_conda_env "$CONDA_ENV"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

CHECKPOINTS="$(normalize_list "$CHECKPOINTS")"

if [[ ! -f "$QUESTIONS_PATH" ]]; then
    echo "Question file missing, building synthetic questions: $QUESTIONS_PATH"
    python -m opsd_alignment.scripts.build_questions \
        --output "$QUESTIONS_PATH" \
        --num-synthetic "$NUM_SYNTHETIC" \
        --overwrite
fi

if [[ "$SKIP_COMPLETED" == "true" && -f "${ALIGNMENT_OUTPUT_DIR}/plots/mean_alignment_by_checkpoint_context.png" ]]; then
    echo "Skipping completed OPSD alignment smoke run: $ALIGNMENT_OUTPUT_DIR"
    exit 0
fi

export STAGE1_ROOT STAGE2_ROOT STAGE3_ROOT THINK_ROOT THINK_DPO_ROOT THINK_SFT_ROOT
export INSTRUCT_SFT_ROOT INSTRUCT_DPO_ROOT INSTRUCT_RL_ROOT RL_ZERO_MATH_ROOT
export ALIGNMENT_EXP_NAME TEMPERATURE TOP_P STUDENT_ROLLOUTS_PER_QUESTION QUESTION_LIMIT
export NODES_PER_ROLLOUT TOP_K_STUDENT TOP_K_TEACHER FORCED_ROLLOUTS_PER_CANDIDATE
export DISTILLATION_OBJECTIVE JSD_ALPHA

python - "$ALIGNMENT_CONFIG" "$ALIGNMENT_OUTPUT_DIR" "$QUESTIONS_PATH" "$CHECKPOINTS" <<'PYCONFIG'
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
output_dir = sys.argv[2]
questions_path = sys.argv[3]
checkpoints = sys.argv[4].split()

def resolve_model_path(ckpt: str) -> str:
    roots = {
        "stage1-": os.environ["STAGE1_ROOT"],
        "stage2-": os.environ["STAGE2_ROOT"],
        "stage3-": os.environ["STAGE3_ROOT"],
    }
    if Path(ckpt).is_dir():
        return ckpt
    for prefix, root in roots.items():
        if ckpt.startswith(prefix):
            return str(Path(root) / ckpt)
    if ckpt == "main" and (Path(os.environ["STAGE3_ROOT"]) / "main").exists():
        return str(Path(os.environ["STAGE3_ROOT"]) / "main")
    if "@" in ckpt:
        variant, revision = ckpt.split("@", 1)
        env_name = {
            "think": "THINK_ROOT",
            "think-dpo": "THINK_DPO_ROOT",
            "think-sft": "THINK_SFT_ROOT",
            "instruct-sft": "INSTRUCT_SFT_ROOT",
            "instruct-dpo": "INSTRUCT_DPO_ROOT",
            "instruct": "INSTRUCT_RL_ROOT",
            "instruct-rl": "INSTRUCT_RL_ROOT",
            "rl-zero-math": "RL_ZERO_MATH_ROOT",
        }.get(variant)
        if env_name:
            return str(Path(os.environ[env_name]) / revision)
    return ckpt

def sanitize(value: str) -> str:
    return value.replace("@", "-").replace("/", "-").replace(" ", "-")

models = []
for ckpt in checkpoints:
    model_path = resolve_model_path(ckpt)
    if not Path(model_path).exists() and model_path.startswith("/"):
        raise SystemExit(f"Missing checkpoint directory: {model_path}")
    models.append((sanitize(ckpt), model_path, 1024 if "think" in ckpt else 512))

lines = [
    f"experiment_name: {os.environ.get('ALIGNMENT_EXP_NAME', 'opsd_alignment_smoke')}",
    "seed: 17",
    "",
    "models:",
]
for name, model_path, max_new_tokens in models:
    lines.extend([
        f"  - name: {name}",
        "    backend: hf",
        f"    path: {model_path}",
        "    trust_remote_code: true",
        f"    max_new_tokens: {max_new_tokens}",
    ])
lines.extend([
    "",
    "teacher_contexts:",
    "  - answer_only",
    "  - full_solution",
    "",
    "generation:",
    f"  temperature: {os.environ['TEMPERATURE']}",
    f"  top_p: {os.environ['TOP_P']}",
    f"  student_rollouts_per_question: {os.environ['STUDENT_ROLLOUTS_PER_QUESTION']}",
    "",
    "diagnostic:",
    f"  questions_limit: {os.environ['QUESTION_LIMIT']}",
    f"  nodes_per_rollout: {os.environ['NODES_PER_ROLLOUT']}",
    f"  top_k_student: {os.environ['TOP_K_STUDENT']}",
    f"  top_k_teacher: {os.environ['TOP_K_TEACHER']}",
    f"  forced_rollouts_per_candidate: {os.environ['FORCED_ROLLOUTS_PER_CANDIDATE']}",
    f"  distillation_objective: {os.environ['DISTILLATION_OBJECTIVE']}",
    f"  jsd_alpha: {os.environ['JSD_ALPHA']}",
    "  min_gradient_norm: 1.0e-8",
    "",
    "paths:",
    f"  questions: {questions_path}",
    f"  output_dir: {output_dir}",
])
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote alignment smoke config: {config_path}")
PYCONFIG

python -m opsd_alignment.scripts.validate_config --config "$ALIGNMENT_CONFIG"

COMMON_GPU_ARGS=(--config "$ALIGNMENT_CONFIG" --torch-dtype "$TORCH_DTYPE")

python -m opsd_alignment.scripts.generate_student_rollouts \
    "${COMMON_GPU_ARGS[@]}" \
    --question-limit "$QUESTION_LIMIT" \
    --device "$GENERATION_DEVICE" \
    "${overwrite_args[@]}"

python -m opsd_alignment.scripts.select_nodes \
    "${COMMON_GPU_ARGS[@]}" \
    --teacher-context "$TEACHER_CONTEXT_FOR_SELECTION" \
    --max-positions-per-rollout "$MAX_POSITIONS_PER_ROLLOUT" \
    --device "$GENERATION_DEVICE" \
    "${overwrite_args[@]}"

run_sharded_stage "teacher/student distributions" \
    python -m opsd_alignment.scripts.compute_teacher_student_distributions \
    "${COMMON_GPU_ARGS[@]}" \
    --device "$SHARD_DEVICE" \
    "${overwrite_args[@]}"

if [[ "$ALIGNMENT_GPUS" -gt 1 ]]; then
    python -m opsd_alignment.scripts.merge_jsonl \
        --config "$ALIGNMENT_CONFIG" \
        --artifact distributions \
        "${overwrite_args[@]}"
    distribution_input_args=(--distribution-glob "${ALIGNMENT_OUTPUT_DIR}/distributions/teacher_student_distributions.shard*-of-*.jsonl")
else
    distribution_input_args=()
fi

run_sharded_stage "branch success" \
    python -m opsd_alignment.scripts.estimate_success_branches \
    "${COMMON_GPU_ARGS[@]}" \
    "${distribution_input_args[@]}" \
    --device "$SHARD_DEVICE" \
    "${overwrite_args[@]}"

if [[ "$ALIGNMENT_GPUS" -gt 1 ]]; then
    python -m opsd_alignment.scripts.merge_jsonl \
        --config "$ALIGNMENT_CONFIG" \
        --artifact branches \
        "${overwrite_args[@]}"
fi

python -m opsd_alignment.scripts.compute_gradients_and_alignment \
    --config "$ALIGNMENT_CONFIG" \
    --objective "$DISTILLATION_OBJECTIVE" \
    --jsd-alpha "$JSD_ALPHA" \
    "${overwrite_args[@]}"

python -m opsd_alignment.scripts.aggregate_results \
    --config "$ALIGNMENT_CONFIG" \
    --group-by checkpoint_teacher \
    "${overwrite_args[@]}"

python -m opsd_alignment.scripts.plot_results \
    --config "$ALIGNMENT_CONFIG" \
    "${overwrite_args[@]}"

echo "OPSD alignment smoke run complete: $ALIGNMENT_OUTPUT_DIR"
