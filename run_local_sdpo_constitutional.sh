#!/bin/bash

# SDPO Training with Constitutional Teacher (Baseten-style setup)
# On-policy self-distillation with privileged teacher context (constitution)
#
# Uses: Qwen3-4B-Instruct, BeaverTails safety subset  
# 150 training steps, 4 rollouts per prompt, dense token-level supervision
#
# Usage: bash run_local_sdpo_constitutional.sh

CONFIG_NAME="sdpo_constitutional"

# Baseten setup: BeaverTails safety dataset
DATA_PATH="datasets/beavertails_safety"

# Baseten model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"

# SDPO hyperparameters (Baseten: 4 rollouts, 150 steps)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=32
LR=1e-5
ALPHA=0.5  # KL divergence weight in distillation
DISTILLATION_TOPK=100
DONT_REPROMPT_ON_SELF_SUCCESS=True
export N_GPUS_PER_NODE=1

SUFFIX=${1:-"constitutional_sdpo"}

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

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

EXP_NAME="SDPO-Constitutional-$(basename "$MODEL_PATH")-steps150-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-Constitutional \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16"

echo "================================================================"
echo "SDPO Training with Constitutional Teacher"
echo "================================================================"
echo "Experiment: $EXP_NAME"
echo "Config: $CONFIG_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "Teacher Context: Constitution from data/constitution.txt"
echo "Rollout/Prompt: $ROLLOUT_BATCH_SIZE"
echo "Alpha (KL weight): $ALPHA"
echo "================================================================"
echo ""
echo "Teacher receives privileged constitutional context for dense supervision."
echo "Student learns token-level targets from teacher through KL divergence."
echo ""

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
