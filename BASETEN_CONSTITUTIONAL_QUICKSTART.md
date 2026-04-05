# Baseten Constitutional Alignment Replication: Quick Start

This guide helps you run the Baseten "Dense, on-policy, or both?" experiments with constitutional alignment using GRPO and SDPO.

## Architecture Overview

```
Two pathways for constitutional alignment:

┌─ GRPO (sparse, on-policy RL) ─────────────────────┐
│                                                     │
│  Student generates 4 responses on-policy           │
│  ↓                                                  │
│  External judge (OpenAI gpt-5.4-mini) scores each  │
│  ↓                                                  │
│  Scores: 0.7 × correctness + 0.3 × tone            │
│  ↓                                                  │
│  GRPO computes advantages relative to group mean   │
│  ↓                                                  │
│  300 training steps, LoRA adapter                  │
└─────────────────────────────────────────────────────┘

┌─ SDPO (dense, on-policy distillation) ──────────────┐
│                                                      │
│  Student generates 4 responses on-policy            │
│  ↓                                                   │
│  Teacher (frozen copy) sees constitution context    │
│  ↓                                                   │
│  Teacher generates aligned feedback/next-tokens     │
│  ↓                                                   │
│  Student learns token-level KL divergence from      │
│  teacher's feedback, weighted by success            │
│  ↓                                                   │
│  150 training steps, on-policy self-distillation    │
└────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. API Keys

**For GRPO (external judge):**
```bash
export OPENAI_API_KEY="sk-..."  # OpenAI gpt-5.4-mini (default)
# OR
export ANTHROPIC_API_KEY="sk-ant-..."  # Claude if using anthropic provider
```

**For SDPO:**
- No external API needed; uses frozen copy of base model as teacher

### 2. Prepare Training Data

Option A: **Use BeaverTails safety dataset (recommended for replication)**
```bash
bash scripts/prepare_beavertails.sh
# Downloads PKU-Alignment/BeaverTails, filters for safety categories
# Creates: datasets/beavertails_safety/{train,test}.parquet
```

Option B: **Use existing ToolUse dataset**
```bash
# Already available in datasets/tooluse/
# Note: original Baseten used BeaverTails, not ToolUse
```

## Running Experiments

### Local Testing (single GPU)

**GRPO with Constitutional Judge:**
```bash
OPENAI_API_KEY="sk-..." bash run_local_grpo_constitutional.sh
```

Expected output:
- Experiment: `GRPO-Constitutional-Qwen3-4B-Instruct-2507-steps300-rollout4-*`
- Checkpoints saved to: `outputs/`

**SDPO with Constitutional Teacher:**
```bash
bash run_local_sdpo_constitutional.sh
```

Expected output:
- Experiment: `SDPO-Constitutional-Qwen3-4B-Instruct-2507-steps150-alpha0.5-rollout4-*`
- Teacher receives constitution in prompt automatically

### RunAI Cluster Submission

**GRPO Constitutional (multi-GPU):**
```bash
OPENAI_API_KEY="sk-..." bash run_runai_grpo_constitutional.sh
```

**SDPO Constitutional (multi-GPU):**
```bash
bash run_runai_sdpo_constitutional.sh
```

### Smoke Test (tiny end-to-end run)

Before a full run, execute a tiny pipeline to verify data loading, rollout, judge calls, optimization, checkpointing, and logging.

Recommended smoke settings:
- `TRAIN_MAX_SAMPLES=32`
- `VAL_MAX_SAMPLES=8`
- `TOTAL_TRAINING_STEPS=10`
- `ROLLOUT_BATCH_SIZE=2` (to reduce judge/API load)
- `TRAIN_BATCH_SIZE=8`

**GRPO smoke test on RunAI:**
```bash
OPENAI_API_KEY="sk-..." \
TRAIN_MAX_SAMPLES=32 \
VAL_MAX_SAMPLES=8 \
TOTAL_TRAINING_STEPS=10 \
ROLLOUT_BATCH_SIZE=2 \
TRAIN_BATCH_SIZE=8 \
MINI_BATCH_SIZE=4 \
bash run_runai_grpo_constitutional.sh smoke
```

**SDPO smoke test on RunAI:**
```bash
TRAIN_MAX_SAMPLES=32 \
VAL_MAX_SAMPLES=8 \
TOTAL_TRAINING_STEPS=10 \
ROLLOUT_BATCH_SIZE=2 \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
bash run_runai_sdpo_constitutional.sh smoke
```

These knobs are now passed through the RunAI wrappers into worker hydra overrides.

### Customize Hyperparameters

Environment variables override defaults:

```bash
# Change data or model
export DATA_PATH="datasets/tooluse"
export MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

# GRPO-specific
export ROLLOUT_BATCH_SIZE=4      # samples per prompt
export MINI_BATCH_SIZE=8
export TRAIN_BATCH_SIZE=32

# SDPO-specific
export ALPHA=0.5                 # KL weight (0=reverse KL, 1=forward KL)
export DISTILLATION_TOPK=100     # top-k vocabulary for teacher
export DONT_REPROMPT_ON_SELF_SUCCESS=True

# Judge (GRPO only)
export JUDGE_PROVIDER="openai"   # or "anthropic"
export JUDGE_MODEL="gpt-5.4-mini"
```

Then run the script:
```bash
OPENAI_API_KEY="sk-..." bash run_local_grpo_constitutional.sh
```

## What Changed

### Judge Changes
- **Default model**: changed from `gpt-4.1-mini` to `gpt-5.4-mini`
- **Error handling**: now logs failures with context instead of silently returning 0.5
- **Failure tracking**: module-level counter tracks judge API failure rate

### Trainer Integration
- **New configs**:
  - `verl/trainer/config/baseline_grpo_constitutional.yaml` — GRPO with judge
  - `verl/trainer/config/sdpo_constitutional.yaml` — SDPO with teacher
  
- **New launchers**:
  - `run_local_grpo_constitutional.sh` — local GRPO
  - `run_local_sdpo_constitutional.sh` — local SDPO
  - `run_runai_grpo_constitutional.sh` — RunAI GRPO
  - `run_runai_sdpo_constitutional.sh` — RunAI SDPO

### Data Preparation
- **New script**: `scripts/prepare_beavertails.sh` downloads and filters BeaverTails safety

### Defaults (Baseten-aligned)
- **Model**: Qwen3-4B-Instruct-2507
- **Data**: BeaverTails safety subset (violence, incitement, aiding_abetting, privacy_violation)
- **GRPO steps**: 300 (changed from 150)
- **SDPO steps**: 150 (default)
- **Rollouts/prompt**: 4 (changed from 8)

## Monitoring Training

### Local runs:
```bash
tail -f outputs/*/train.log
```

### RunAI cluster:
```bash
runai list  # see all jobs
runai logs <job-name> -p dlab-samaier  # watch live logs
runai describe job <job-name> -p dlab-samaier  # status & details
```

## Expected Behavior

### GRPO logs:
```
...
Judge API call (provider=openai, model=gpt-5.4-mini)
  Question length: 124
  Response length: 256
  Score: {"correctness": 8, "tone": 9}
  Combined (0.7×0.8 + 0.3×0.9): 0.83
...
rollout_corr/loss: 0.234
actor/policy_loss: 0.187
...
```

### SDPO logs:
```
...
self_distillation/success_sample_fraction: 0.45
self_distillation/reprompt_sample_fraction: 0.50
actor/policy_loss: 0.156  # KL divergence from teacher
...
```

### Judge failure warnings:
```
WARNING: Judge API call failed (openai/gpt-5.4-mini): ...
         Question length: 512, Response length: 2048
         Failures: 2/150 (1.3%)
```

If failure rate climbs above ~5%, investigate:
- API rate limits (add exponential backoff)
- Long responses causing timeouts (truncate before judge)
- Model overload (reduce ROLLOUT_BATCH_SIZE)

## Files Created/Modified

| File | Purpose | Status |
|------|---------|--------|
| `constitution_judge.py` | Judge with logging + gpt-5.4-mini | ✓ Updated |
| `constitution_teacher.py` | Teacher prompt builder | ✓ Already existed |
| `baseline_grpo_constitutional.yaml` | GRPO config | ✓ Created |
| `sdpo_constitutional.yaml` | SDPO config | ✓ Created |
| `run_local_grpo_constitutional.sh` | Local GRPO launcher | ✓ Created |
| `run_local_sdpo_constitutional.sh` | Local SDPO launcher | ✓ Created |
| `run_runai_grpo_constitutional.sh` | RunAI GRPO launcher | ✓ Created |
| `run_runai_sdpo_constitutional.sh` | RunAI SDPO launcher | ✓ Created |
| `scripts/prepare_beavertails.sh` | BeaverTails downloader | ✓ Created |

## Next Steps

1. ✅ Set up API keys:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. 📥 Prepare training data:
   ```bash
   bash scripts/prepare_beavertails.sh
   ```

3. 🚀 Run GRPO on RunAI:
   ```bash
   OPENAI_API_KEY="sk-..." bash run_runai_grpo_constitutional.sh
   ```

4. 🚀 Run SDPO on RunAI:
   ```bash
   bash run_runai_sdpo_constitutional.sh
   ```

5. 📊 Monitor progress:
   ```bash
   runai list
   runai logs <job-name> -p dlab-samaier | tail -n 20
   ```

6. 📈 Evaluate on BullshitBench after training completes

---

**Questions?** See `IMPLEMENTATION_GUIDE_CONSTITUTIONAL_ALIGNMENT.md` for detailed setup instructions.
