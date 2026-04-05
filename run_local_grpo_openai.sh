#!/bin/bash

# Usage: OPENAI_API_KEY=... ./run_local_grpo_openai.sh [experiment_name_suffix]

CONFIG_NAME="baseline_grpo_api"
DATA_PATH="datasets/tooluse"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=4
MINI_BATCH_SIZE=8
LR=1e-5
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export N_GPUS_PER_NODE=1

# Judge config
export JUDGE_PROVIDER=${JUDGE_PROVIDER:-openai}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-4.1-mini}

SUFFIX=${1:-"local_grpo_openai"}

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  echo "Run like: OPENAI_API_KEY=sk-... bash run_local_grpo_openai.sh"
  exit 1
fi

EXP_NAME="LOCAL-GRPO-OPENAI-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=GRPO-openai-local \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16"

echo "----------------------------------------------------------------"
echo "Starting Local GRPO Training (OpenAI Judge)"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Judge Provider: $JUDGE_PROVIDER"
echo "Judge Model: $JUDGE_MODEL"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
