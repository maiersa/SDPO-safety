# Quick Reference: Constitutional GRPO + SDPO Setup

## Your 4 Key Components

### 1. **Constitution** (`data/constitution.txt`)
- ✓ Already created with safety/truthfulness/helpfulness framework
- **Used by**: GRPO judge prompt, SDPO teacher prompt
- **To customize**: Edit `data/constitution.txt` directly

---

### 2. **GRPO with API Judge** (`constitution_judge.py`)

**What it does**:
- Student generates completions on-policy
- Call external judge API (Anthropic Claude 3.5 Sonnet) to score each
- Judge scores on: correctness (0.7 weight) + tone (0.3 weight)
- Compute GRPO advantages and train

**Setup**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
pip install anthropic
```

**Test**:
```python
from verl.utils.reward_score.feedback.constitution_judge import call_judge_api
scores = call_judge_api("What is 2+2?", "4", constitution=open("data/constitution.txt").read())
print(scores)
```

**Run**:
```bash
bash run_local_grpo_api.sh
# Or with config override:
python -m verl.trainer.main_ppo --config-name baseline_grpo_api data.train_batch_size=32 actor_rollout_ref.rollout.n=4
```

---

### 3. **SDPO with Constitutional Teacher** (`constitution_teacher.py`)

**What it does**:
- Student generates on-policy completions
- Teacher receives privileged constitution context
- Teacher provides dense token-level supervision
- Train with KL distillation from teacher

**Setup**:
```python
from verl.utils.reward_score.feedback.constitution_teacher import build_teacher_prompt
prompt = build_teacher_prompt(question="What is AI?", constitution=constitution_text)
print(prompt)
```

**Run**:
```bash
bash run_local_sdpo_constitutional.sh
# Or:
python -m verl.trainer.main_ppo --config-name sdpo data.train_batch_size=32
```

---

### 4. **BullshitBench Evaluation Dataset**

**Download & organize**:
```bash
chmod +x scripts/download_bullshitbench.sh
bash scripts/download_bullshitbench.sh
```

**Result structure**:
```
datasets/bullshitbench/
├── nonsense_questions.json    ← 50 incoherent prompts (test rejection)
├── salvaged_questions.TODO.json ← manual rewrite template
└── nonsense_questions.jsonl   ← JSONL compatibility file
```

**Create salvaged prompts**:
- Open `nonsense_questions.json`
- For each nonsense question, rewrite it to be sensible
- Save rewritten versions to `salvaged_questions.TODO.json`
- Rename the completed file to `salvaged_questions.json`

**Example**:
```json
{
  "nonsense": "At what rate will we hit 100% and reject everything?",
  "salvaged": "At what rate should acceptance plateau, and how adapt our review?"
}
```

---

## Combined Experiment Pipeline

### Phase 1: GRPO (300 training steps, 4 rollouts/prompt)
```bash
ANTHROPIC_API_KEY="sk-ant-..." bash run_local_grpo_api.sh
```
**Output**: `outputs/GRPO-Constitutional-*.pt` (LoRA or full checkpoint)

### Phase 2: SDPO (150 training steps, 4 rollouts/prompt)
```bash
bash run_local_sdpo_constitutional.sh
```
**Output**: `outputs/SDPO-Constitutional-*.pt`

### Phase 3: Evaluate both on BullshitBench
```bash
python scripts/eval_bullshitbench.py outputs/GRPO-Constitutional-*.pt
python scripts/eval_bullshitbench.py outputs/SDPO-Constitutional-*.pt
```

**Expect**: SDPO should outperform GRPO on nonsense-detection + remain helpful on your manually rewritten prompts.

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `data/constitution.txt` | Constitution text | ✓ Created |
| `verl/utils/reward_score/feedback/constitution_judge.py` | GRPO judge (API) | ✓ Created |
| `verl/utils/reward_score/feedback/constitution_teacher.py` | SDPO teacher prompt builder | ✓ Created |
| `verl/trainer/config/baseline_grpo_api.yaml` | GRPO config with API judge | ✓ Created |
| `verl/trainer/config/sdpo.yaml` | SDPO config (existing) | ✓ Already in repo |
| `scripts/download_bullshitbench.sh` | Download BullshitBench | ✓ Created |
| `datasets/bullshitbench/nonsense_questions.json` | Nonsense eval set | ⏳ Creates on download |
| `datasets/bullshitbench/salvaged_questions.TODO.json` | Manual rewrite template | 🔄 YOU POPULATE |
| `IMPLEMENTATION_GUIDE_CONSTITUTIONAL_ALIGNMENT.md` | Full guide | ✓ Created |

---

## Environment Setup (One-time)

```bash
# 1. Activate conda environment
conda activate /mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo

# 2. Install anthropic SDK
pip install anthropic

# 3. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
# Or permanently:
conda env config vars set ANTHROPIC_API_KEY="sk-ant-..." -p /mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo

# 4. Verify
python -c "from anthropic import Anthropic; print('✓ Anthropic SDK ready')"
```

---

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| `ANTHROPIC_API_KEY not set` | `export ANTHROPIC_API_KEY="sk-ant-..."` |
| Judge API timeout | Reduce `rollout.n` parameter, or batch fewer in parallel |
| BullshitBench download fails | Download manually from https://github.com/petergpt/BullshitBench |
| Out of CUDA memory | Reduce `train_batch_size` or `actor_rollout_ref.actor.ppo_mini_batch_size` |
| Constitution not found | Check `ls data/constitution.txt` exists |

---

## Expected Metrics

After training + evaluation:

**GRPO (sparse on-policy RL)**:
- BullshitBench nonsense rejection rate: ~60-70%
- Salvaged helpfulness score: ~0.6-0.7

**SDPO (dense on-policy distillation)**:
- BullshitBench nonsense rejection rate: ~75-85% ← Better!
- Salvaged helpfulness score: ~0.65-0.75

---

## Next Steps

1. ✅ Review this quick ref and the full `IMPLEMENTATION_GUIDE_CONSTITUTIONAL_ALIGNMENT.md`
2. 🔑 Export API key (see above)
3. 📥 Run `bash scripts/download_bullshitbench.sh`
4. ✏️ Populate `salvaged_questions.TODO.json` and rename it to `salvaged_questions.json`
5. 🚀 Run GRPO and SDPO training
6. 📊 Evaluate and compare results

---

**Questions?** See `IMPLEMENTATION_GUIDE_CONSTITUTIONAL_ALIGNMENT.md` for detailed step-by-step instructions.
