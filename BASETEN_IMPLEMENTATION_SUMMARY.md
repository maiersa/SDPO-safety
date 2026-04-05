# Implementation Summary: Baseten Constitutional Alignment Replication

## Changes Made

### 1. Judge Improvements (constitution_judge.py)

**Error Handling → Failure Logging**
- **Before**: Silently returned neutral score (0.5) on API failures
- **After**: Logs failure with context and re-raises exception
- **Benefit**: Catches judge API problems instead of masking them

**Changes**:
```python
# Added logging infrastructure
import logging
logger = logging.getLogger(__name__)
_judge_failure_count = 0
_judge_call_count = 0

# Changed default model
# Before: "gpt-4.1-mini"
# After: "gpt-5.4-mini"

# Changed error handling
# Before: return {"score": 0.5, ...}  # silent failure
# After: raise  # let trainer see the error, log with context
```

**Logging Output Example**:
```
WARNING: Judge API call failed (openai/gpt-5.4-mini): APIConnectionError: ...
         Question length: 124, Response length: 256
         Failures: 2/150 (1.3%)
```

---

### 2. Constitutional GRPO (Baseline Model for GRPO)

**New Config**: `verl/trainer/config/baseline_grpo_constitutional.yaml`
- Sets rollout.n = 4 (Baseten style, was 8)
- Points to constitution judge in custom_reward_function
- Group name: grpo_constitutional

**New Launcher**: `run_local_grpo_constitutional.sh`
- Defaults: BeaverTails data, gpt-5.4-mini judge, OPENAI_API_KEY required
- Validates API key before launching
- Warns if BeaverTails not found, falls back to tooluse

**New RunAI Launcher**: `run_runai_grpo_constitutional.sh`
- Submits to RunAI cluster with proper env var forwarding
- Forwards OPENAI_API_KEY + JUDGE_PROVIDER/JUDGE_MODEL into container
- Multi-GPU ready with configurable TRAINER_GPUS_PER_NODE

---

### 3. Constitutional SDPO (Dense On-Policy Distillation)

**New Config**: `verl/trainer/config/sdpo_constitutional.yaml`
- Sets rollout.n = 4 (Baseten style, was 8)
- Includes placeholder for constitutional teacher integration
- Group name: sdpo_constitutional

**New Launcher**: `run_local_sdpo_constitutional.sh`
- Defaults: BeaverTails data, 150 training steps, alpha=0.5
- No external API needed (uses frozen copy of model as teacher)
- Warns if BeaverTails not found, falls back to tooluse

**New RunAI Launcher**: `run_runai_sdpo_constitutional.sh`
- Submits to RunAI cluster
- Multi-GPU ready
- Default: 1 GPU, configurable via TRAINER_GPUS_PER_NODE

---

### 4. Training Data Preparation

**New Script**: `scripts/prepare_beavertails.sh`
- Downloads PKU-Alignment/BeaverTails from HuggingFace
- Filters for safety harm categories:
  - violence
  - incitement
  - aiding_abetting
  - privacy_violation
- Creates `datasets/beavertails_safety/{train,test}.parquet`
- Auto-creates placeholder if download fails

---

### 5. Documentation

**New Quick Start**: `BASETEN_CONSTITUTIONAL_QUICKSTART.md`
- Architecture overview (GRPO vs SDPO)
- Step-by-step setup instructions
- Parameter customization examples
- Local vs RunAI submission
- Expected logs and monitoring
- Troubleshooting guide

---

## How to Use

### Setup (One-time)

```bash
# 1. Set OpenAI API key for GRPO judge
export OPENAI_API_KEY="sk-..."

# 2. Download and prepare BeaverTails (optional, falls back to tooluse)
bash scripts/prepare_beavertails.sh
```

### Local Testing

```bash
# Test GRPO with constitutional judge
OPENAI_API_KEY="sk-..." bash run_local_grpo_constitutional.sh

# Test SDPO with constitutional teacher
bash run_local_sdpo_constitutional.sh
```

### RunAI Cluster (Production)

```bash
# Submit GRPO job
OPENAI_API_KEY="sk-..." bash run_runai_grpo_constitutional.sh

# Submit SDPO job
bash run_runai_sdpo_constitutional.sh

# Monitor
runai logs grpo-const-${USER}-* -p dlab-samaier | tail -20
```

### Customize Hyperparameters

```bash
# Change model or data
export MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
export DATA_PATH="datasets/tooluse"

# GRPO-specific
export ROLLOUT_BATCH_SIZE=8
export MINI_BATCH_SIZE=16

# SDPO-specific
export ALPHA=0.5
export DISTILLATION_TOPK=100

# Judge-specific (GRPO)
export JUDGE_MODEL="claude-3-5-sonnet-20241022"
export JUDGE_PROVIDER="anthropic"

# Run
OPENAI_API_KEY="sk-..." bash run_local_grpo_constitutional.sh
```

---

## Architecture Comparison

### GRPO (Sparse, On-Policy RL)
```
Student → Generate 4 responses
         → Judge API scores each
         → GRPO advantage computation
         → Optimize policy via PG
         → 300 steps, LoRA adapter
```

Score combination: `0.7 × correctness + 0.3 × tone normalized to [0,1]`

Judge: External frontier model (OpenAI gpt-5.4-mini by default)

### SDPO (Dense, On-Policy Distillation)
```
Student → Generate 4 responses (on-policy)
        ↓
Teacher (frozen copy + constitution context)
        → Generate feedback/next-token distribution
        ↓
KL divergence loss, weighted by student success
        → Optimize policy via supervised learning
        → 150 steps, on-policy self-distillation
```

Teacher context: Automatically receives constitution via data loading

---

## Files Modified/Created

| File | Type | Status |
|------|------|--------|
| `verl/utils/reward_score/feedback/constitution_judge.py` | Modified | Logging + gpt-5.4-mini + exception re-raise |
| `verl/trainer/config/baseline_grpo_constitutional.yaml` | Created | GRPO config, 4 rollouts |
| `verl/trainer/config/sdpo_constitutional.yaml` | Created | SDPO config, 4 rollouts |
| `run_local_grpo_constitutional.sh` | Created | Local GRPO launcher |
| `run_local_sdpo_constitutional.sh` | Created | Local SDPO launcher |
| `run_runai_grpo_constitutional.sh` | Created | RunAI GRPO launcher |
| `run_runai_sdpo_constitutional.sh` | Created | RunAI SDPO launcher |
| `scripts/prepare_beavertails.sh` | Created | BeaverTails downloader |
| `BASETEN_CONSTITUTIONAL_QUICKSTART.md` | Created | Quick start guide |

---

## Key Decision Points

### Error Handling
**Decision**: Fail explicitly instead of returning neutral score
**Rationale**: Judge failures indicate real problems (rate limits, model issues, timeouts). Masking them silently would make them invisible in logs. Better to fail loudly and let the trainer/user investigate.

### Model Upgrade
**Decision**: Default to gpt-5.4-mini (was gpt-4.1-mini)
**Rationale**: Matches user's stated availability of gpt-5.4-mini; higher quality judgments for constitutional alignment.

### Defaults
**Decision**: BeaverTails as default data path, 4 rollouts per prompt
**Rationale**: Matches Baseten setup exactly for faithful replication. Graceful fallback to tooluse if BeaverTails unavailable.

### Teacher Integration
**Decision**: Use existing frozen model + constitution context pattern
**Rationale**: Avoids modifying core trainer logic. Constitution is passed via data loading. Teacher prompt builder (`constitution_teacher.py`) is available for future hook-ups.

---

## Next Steps for User

1. ✅ **Export API key**:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. 📥 **Prepare data** (optional):
   ```bash
   bash scripts/prepare_beavertails.sh
   ```

3. 🚀 **Run GRPO on RunAI**:
   ```bash
   OPENAI_API_KEY="sk-..." bash run_runai_grpo_constitutional.sh
   ```

4. 🚀 **Run SDPO on RunAI**:
   ```bash
   bash run_runai_sdpo_constitutional.sh
   ```

5. 📊 **Monitor & Evaluate**:
   ```bash
   runai logs <job-name> -p dlab-samaier
   # After training: evaluate on BullshitBench
   python scripts/eval_bullshitbench.py outputs/GRPO-Constitutional-*.pt
   python scripts/eval_bullshitbench.py outputs/SDPO-Constitutional-*.pt
   ```

---

## Troubleshooting

### "OPENAI_API_KEY is not set" (GRPO)
```bash
export OPENAI_API_KEY="sk-..."
bash run_local_grpo_constitutional.sh
```

### "BeaverTails data not found"
```bash
bash scripts/prepare_beavertails.sh  # Download & prep
# Or just use fallback:
bash run_local_grpo_constitutional.sh  # Falls back to tooluse
```

### Judge API failing frequently (failure rate > 5%)
- Check model availability: `JUDGE_MODEL="gpt-5.4-mini"`
- Try Anthropic: `JUDGE_PROVIDER="anthropic" ANTHROPIC_API_KEY="sk-ant-..." bash run_local_grpo_constitutional.sh`
- Reduce batch size: `ROLLOUT_BATCH_SIZE=2` (smaller prompts/responses)

### RunAI job stuck waiting
```bash
runai list                                    # find job name
runai describe job grpo-const-${USER}-* -p dlab-samaier
runai delete job grpo-const-${USER}-* -p dlab-samaier  # cancel if needed
```

---

For detailed setup instructions, see: `BASETEN_CONSTITUTIONAL_QUICKSTART.md`
