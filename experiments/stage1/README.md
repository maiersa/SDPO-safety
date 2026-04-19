# Stage 1: SDPO Feasibility Threshold on Intermediate Pretraining Checkpoints

## Goal

Stage 1 asks a narrow question:

When does SDPO first become net-positive over the base model during pretraining, and how does that threshold differ between `GSM8K` and `MATH`?

This stage focuses only on capability. Safety alignment is intentionally deferred to Stage 2.

## Main Hypotheses

1. `SDPO` becomes effective only after the base policy reaches a minimum capability threshold.
2. That threshold appears earlier on `GSM8K` than on `MATH`.
3. The first successful SDPO checkpoint is likely earlier than we would expect from final-model intuition.
4. Near the threshold, results will be more seed-sensitive than for late checkpoints.

## Primary Readout

The main quantity we care about is:

`delta = metric(post-train) - metric(base checkpoint)`

We should report this separately for:

- `pass@1`
- `pass@8`
- `pass@16`

The first positive and repeatable `delta` is our operational SDPO feasibility threshold.

## Stage 1 Experiment Table

| Wave | Purpose | Checkpoints | Algorithm | Datasets | Seeds | Budget | Decision Rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Base capability map | `stage1-step1000`, `4000`, `16000`, `64000`, `128000`, `256000`, `656000` | Base eval only | `GSM8K`, `MATH` | 1 | No training; validation-only | Find where the raw policy becomes non-trivial and where prompt/format failures dominate |
| B | Coarse SDPO sweep | Same 7 checkpoints | `SDPO` | `GSM8K`, `MATH` | 1 | `150` training steps, fixed hyperparameters | Identify the first checkpoint with positive delta over base |
| C | Threshold zoom-in | 2 checkpoints below and 2 above the first positive checkpoint from Wave B | `SDPO` | `GSM8K`, `MATH` | 3 | Same budget as Wave B | Confirm whether the threshold is real or just noise |
| D | GRPO reference subset | Earliest promising checkpoint, threshold checkpoint, and one late checkpoint | `GRPO` | `GSM8K`, `MATH` | 1 to 3 | Same rollout budget as SDPO | Compare whether SDPO turns on earlier, later, or at the same point |

## Recommended Initial Checkpoint Grid

These are chosen to be coarse and approximately log-spaced across the `stage1` trajectory while keeping the first pass affordable:

| Label | Branch Name | Why It Is Included |
| --- | --- | --- |
| C0 | `stage1-step1000` | Earliest checkpoint worth testing after excluding the random initialization |
| C1 | `stage1-step4000` | Very early checkpoint likely still near or below threshold |
| C2 | `stage1-step16000` | Early-mid candidate |
| C3 | `stage1-step64000` | Mid-stage candidate |
| C4 | `stage1-step128000` | Mid-late candidate |
| C5 | `stage1-step256000` | Strong checkpoint likely above threshold |
| C6 | `stage1-step656000` | Late Stage 1 anchor |

If Wave B shows the transition between two checkpoints, we densify only that interval.

## What Stays Fixed in Stage 1

To make the threshold interpretable, we should hold the post-training recipe constant across checkpoints.

### SDPO defaults

- Config: `sdpo`
- Train batch size: `32`
- Rollout samples per prompt: `8`
- PPO mini-batch size: `32`
- Learning rate: `1e-5`
- Alpha: `0.5`
- Distillation top-k: `100`
- Total training steps: `150`

### GRPO reference defaults

- Config: `baseline_grpo`
- Train batch size: `32`
- Rollout samples per prompt: `8`
- PPO mini-batch size: `8`
- Learning rate: `1e-5`
- Total training steps: `150`

### Evaluation defaults

- Evaluate both the raw checkpoint and the post-trained model
- Use identical answer extraction and reward logic across all checkpoints
- Keep prompting as standardized as possible across checkpoints
- Report `pass@1`, `pass@8`, and `pass@16`

## Datasets

The repo already has compatible preprocessing utilities:

- `GSM8K`: [examples/data_preprocess/gsm8k.py](/dlabscratch1/samaier/projects/SDPO-safety/examples/data_preprocess/gsm8k.py)
- `MATH`: [examples/data_preprocess/math_dataset.py](/dlabscratch1/samaier/projects/SDPO-safety/examples/data_preprocess/math_dataset.py)

Recommended dataset directories for Stage 1:

- `datasets/gsm8k`
- `datasets/math`

Each should contain:

- `train.parquet`
- `test.parquet`

## Success Criteria

We should consider Stage 1 successful if it answers all three of these:

1. What is the earliest checkpoint where SDPO improves `GSM8K` over base?
2. What is the earliest checkpoint where SDPO improves `MATH` over base?
3. Are those thresholds clearly different, or roughly aligned?

## Logging and Analysis

For every run, save:

- checkpoint label
- dataset
- algorithm
- seed
- base score
- post-train score
- delta over base

The Stage 1 figure should be:

`x-axis = checkpoint progress`

`y-axis = delta over base`

with separate lines for:

- `SDPO on GSM8K`
- `SDPO on MATH`
- optional `GRPO on GSM8K`
- optional `GRPO on MATH`

## Practical Note on Model Revisions

The current repo config exposes `actor_rollout_ref.model.path`, but does not appear to expose a first-class Hugging Face `revision` field in the trainer config.

For that reason, the cleanest execution path is:

1. Materialize each target checkpoint as its own local model directory.
2. Point `actor_rollout_ref.model.path` to that local directory.

The helper script in this folder assumes that layout.
