# OPSD Gradient-Alignment Diagnostic

This package implements an offline diagnostic for asking whether a privileged OPSD teacher pushes a student in reward-improving token directions.

## Pipeline

1. Generate and grade student rollouts.
2. Select a small number of diagnostic token positions.
3. Compute student/teacher candidate-set distributions at each selected node.
4. Force each candidate token and estimate downstream student success.
5. Compute ideal gradients, OPSD gradients, and cosine alignment.
6. Aggregate and plot results.

## Model Path Configuration

The committed smoke and main configs use environment variables for checkpoint paths:

```bash
export OLMO3_7B_STAGE1_FINAL=/absolute/path/to/olmo3_7b_stage1_final
export OLMO3_7B_STAGE2_FINAL=/absolute/path/to/olmo3_7b_stage2_final
export OLMO3_7B_THINK_FINAL=/absolute/path/to/olmo3_7b_think_final
```

Before launching GPU jobs, validate the config:

```bash
python -m opsd_alignment.scripts.validate_config \
  --config opsd_alignment/configs/smoke_test.yaml
```

If you only want to check schema/settings before the paths exist:

```bash
python -m opsd_alignment.scripts.validate_config \
  --config opsd_alignment/configs/smoke_test.yaml \
  --skip-model-paths
```

You can also copy `opsd_alignment/configs/local_paths.template.yaml` to a private config and fill in concrete paths.

## Building The Question File

The default question file is:

```text
opsd_alignment/data/questions.jsonl
```

To rebuild the self-contained synthetic smoke set:

```bash
python -m opsd_alignment.scripts.build_questions \
  --output opsd_alignment/data/questions.jsonl \
  --num-synthetic 20 \
  --overwrite
```

To mix in local GSM8K-style records from JSON/JSONL/parquet:

```bash
python -m opsd_alignment.scripts.build_questions \
  --output opsd_alignment/data/questions.jsonl \
  --num-synthetic 20 \
  --gsm8k-source /path/to/gsm8k.jsonl \
  --num-gsm8k 30 \
  --overwrite
```

To mix in local MATH-style records:

```bash
python -m opsd_alignment.scripts.build_questions \
  --output opsd_alignment/data/questions.jsonl \
  --num-synthetic 5 \
  --gsm8k-source /path/to/gsm8k.jsonl \
  --num-gsm8k 30 \
  --math-source /path/to/math.jsonl \
  --num-math 15 \
  --overwrite
```

If the `datasets` package and dataset cache/network are available, `--gsm8k-source openai/gsm8k` also works.

## Single-GPU Smoke Run

Edit `opsd_alignment/configs/smoke_test.yaml` so the model paths point at real checkpoints. Then run a tiny pass first:

```bash
python -m opsd_alignment.scripts.generate_student_rollouts   --config opsd_alignment/configs/smoke_test.yaml   --model-name olmo3_7b_stage1_final   --question-limit 1   --device cuda:0   --torch-dtype bf16   --overwrite
```

```bash
python -m opsd_alignment.scripts.select_nodes   --config opsd_alignment/configs/smoke_test.yaml   --model-name olmo3_7b_stage1_final   --teacher-context full_solution   --max-positions-per-rollout 32   --device cuda:0   --torch-dtype bf16   --overwrite
```

```bash
python -m opsd_alignment.scripts.compute_teacher_student_distributions   --config opsd_alignment/configs/smoke_test.yaml   --model-name olmo3_7b_stage1_final   --device cuda:0   --torch-dtype bf16   --overwrite
```

```bash
python -m opsd_alignment.scripts.estimate_success_branches   --config opsd_alignment/configs/smoke_test.yaml   --model-name olmo3_7b_stage1_final   --device cuda:0   --torch-dtype bf16   --overwrite
```

```bash
python -m opsd_alignment.scripts.compute_gradients_and_alignment   --config opsd_alignment/configs/smoke_test.yaml   --objective forward_kl   --overwrite
```

```bash
python -m opsd_alignment.scripts.aggregate_results   --config opsd_alignment/configs/smoke_test.yaml   --group-by checkpoint_teacher   --overwrite
```

```bash
python -m opsd_alignment.scripts.plot_results   --config opsd_alignment/configs/smoke_test.yaml   --overwrite
```

## Trying JSD or Reverse KL

The config default is forward KL. For a one-off override at stage 5:

```bash
python -m opsd_alignment.scripts.compute_gradients_and_alignment   --config opsd_alignment/configs/smoke_test.yaml   --objective jsd   --jsd-alpha 0.5   --output-file opsd_alignment/outputs/smoke_test/alignments/gradient_alignments_jsd.jsonl   --overwrite
```

Reverse KL:

```bash
python -m opsd_alignment.scripts.compute_gradients_and_alignment   --config opsd_alignment/configs/smoke_test.yaml   --objective reverse_kl   --output-file opsd_alignment/outputs/smoke_test/alignments/gradient_alignments_reverse_kl.jsonl   --overwrite
```

## Run:AI Cluster Smoke Launcher

A Run:AI-style launcher lives at:

```text
experiments/pretrain/run_opsd_alignment_smoke.sh
```

It mirrors the OPSD training sweep conventions for `CHECKPOINTS`, `STAGE*_ROOT`, `CONDA_ENV`, `REPO_DIR`, `LOG_DIR`, and `CKPT_DIR`. The cheap stages run once, and the expensive distribution/branch stages shard across `ALIGNMENT_GPUS` workers inside the job.

Example:

```bash
rs opsd-align-smoke-${USER}-stage1 --gpu 8.0 --cpu 32 --memory 240G --node-pools h100 --large-shm -- env \
USER=$USER \
CONDA_ENV=/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo \
REPO_DIR=/dlabscratch1/${USER}/projects/SDPO-safety \
LOG_DIR=/dlabscratch1/${USER}/output \
CKPT_DIR=/dlabscratch1/${USER}/checkpoints \
STAGE1_ROOT=/dlabscratch1/${USER}/checkpoints/olmo-7b-stage1 \
STAGE2_ROOT=/dlabscratch1/${USER}/checkpoints/olmo-7b-stage2 \
CHECKPOINTS=stage1-step500000,stage1-step656000 \
ALIGNMENT_EXP_NAME=opsd_alignment_smoke_stage1 \
ALIGNMENT_GPUS=8 \
QUESTION_LIMIT=1 \
STUDENT_ROLLOUTS_PER_QUESTION=1 \
NODES_PER_ROLLOUT=1 \
TOP_K_STUDENT=2 \
TOP_K_TEACHER=2 \
FORCED_ROLLOUTS_PER_CANDIDATE=2 \
MAX_POSITIONS_PER_ROLLOUT=32 \
DISTILLATION_OBJECTIVE=forward_kl \
TORCH_DTYPE=bf16 \
GENERATION_DEVICE=auto \
SHARD_DEVICE=cuda:0 \
TRAINER_GPUS_PER_NODE=8 \
SKIP_COMPLETED=true \
OVERWRITE=false \
bash /dlabscratch1/${USER}/projects/SDPO-safety/experiments/pretrain/run_opsd_alignment_smoke.sh
```

Outputs go to:

```text
/dlabscratch1/${USER}/output/opsd_alignment/<ALIGNMENT_EXP_NAME>/
```

## Cluster Sharding Pattern

The expensive stages are distribution computation and branch success estimation. Run many independent workers with deterministic shards:

```bash
CUDA_VISIBLE_DEVICES=0 python -m opsd_alignment.scripts.compute_teacher_student_distributions   --config opsd_alignment/configs/smoke_test.yaml   --model-name olmo3_7b_stage1_final   --device cuda:0   --torch-dtype bf16   --num-shards 8   --shard-index 0   --overwrite
```

Repeat with `CUDA_VISIBLE_DEVICES=1 --shard-index 1`, and so on.

For stage 4, read all distribution shards and shard candidate branches:

```bash
CUDA_VISIBLE_DEVICES=0 python -m opsd_alignment.scripts.estimate_success_branches   --config opsd_alignment/configs/smoke_test.yaml   --distribution-glob 'opsd_alignment/outputs/smoke_test/distributions/teacher_student_distributions.shard*-of-*.jsonl'   --model-name olmo3_7b_stage1_final   --device cuda:0   --torch-dtype bf16   --num-shards 8   --shard-index 0   --overwrite
```

## Merging Shards

Later stages can read shard globs directly, but merging is useful for inspection:

```bash
python -m opsd_alignment.scripts.merge_jsonl   --config opsd_alignment/configs/smoke_test.yaml   --artifact distributions   --overwrite
```

```bash
python -m opsd_alignment.scripts.merge_jsonl   --config opsd_alignment/configs/smoke_test.yaml   --artifact branches   --overwrite
```

Supported artifacts:

```text
distributions
branches
alignments
summaries
```

## Main Outputs

```text
outputs/<experiment>/rollouts/student_rollouts.jsonl
outputs/<experiment>/nodes/selected_nodes.jsonl
outputs/<experiment>/distributions/teacher_student_distributions*.jsonl
outputs/<experiment>/branches/branch_success*.jsonl
outputs/<experiment>/alignments/gradient_alignments.jsonl
outputs/<experiment>/summaries/alignment_summary.{jsonl,csv}
outputs/<experiment>/plots/*.png
```

## Notes

- Branch success is student-only and is deduplicated across teacher contexts.
- `control` teacher context uses the exact same tokenized prefix as the student.
- The current backend is Hugging Face. The runner interface is intentionally narrow so vLLM or SGLang can be added later for larger forced-rollout jobs.
