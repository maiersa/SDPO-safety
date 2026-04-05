#!/bin/bash

# GRPO Training with Constitutional Judge (Baseten-style setup)
# Uses: Qwen3-4B-Instruct, BeaverTails safety subset, external judge (OpenAI or Anthropic)
# 
# Usage: OPENAI_API_KEY=sk-... bash run_local_grpo_constitutional.sh

CONFIG_NAME="baseline_grpo_constitutional"

# Baseten setup: BeaverTails safety dataset
DATA_PATH="datasets/beavertails_safety"

# Baseten model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

# GRPO hyperparameters (Baseten: 4 rollouts, 300 steps)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=4
MINI_BATCH_SIZE=8
LR=1e-5
export N_GPUS_PER_NODE=1

# Judge configuration (defaults to OpenAI gpt-5.4-mini, can override)
export JUDGE_PROVIDER=${JUDGE_PROVIDER:-openai}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.4-mini}

SUFFIX=${1:-"constitutional_grpo"}

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

# Validate API key for judge
if [[ "$JUDGE_PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Run like:"
  echo "  OPENAI_API_KEY=sk-... bash run_local_grpo_constitutional.sh"
  exit 1
fi

if [[ "$JUDGE_PROVIDER" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set. Run like:"
  echo "  ANTHROPIC_API_KEY=sk-ant-... bash run_local_grpo_constitutional.sh"
  exit 1
fi

# Check if BeaverTails data exists, warn if not
if [[ ! -f "$DATA_PATH/train.parquet" ]]; then
  echo ""
  echo "⚠️  WARNING: BeaverTails data not found at $DATA_PATH"
  echo "   To prepare BeaverTails, run:"
  echo "     bash scripts/prepare_beavertails.sh"
  echo ""
  echo "   Falling back to datasets/tooluse for now..."
  DATA_PATH="datasets/tooluse"
fi

EXP_NAME="GRPO-Constitutional-$(basename "$MODEL_PATH")-steps300-rollout${ROLLOUT_BATCH_SIZE}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=GRPO-Constitutional \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16"

echo "================================================================"
echo "GRPO Training with Constitutional Judge"
echo "================================================================"
echo "Experiment: $EXP_NAME"
echo "Config: $CONFIG_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Judge Provider: $JUDGE_PROVIDER"
echo "Judge Model: $JUDGE_MODEL"
echo "Rollout/Prompt: $ROLLOUT_BATCH_SIZE"
echo "================================================================"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
