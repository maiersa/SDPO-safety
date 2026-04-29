# Pretraining Benchmark Evaluation

This benchmark path is separate from OPSD/GRPO training validation. It loads
prepared benchmark parquet files, uses plain-text prompts, samples once per
problem, and computes pass@k from the same sample pool.

## Backends

The runner supports two generation backends:

- `hf`: Hugging Face `transformers`. This is simple and works on one GPU, or
  with `device_map=auto` sharding, but it is slow for pass@32.
- `vllm`: vLLM. This is the preferred backend for full benchmark sweeps and can
  use multiple GPUs through tensor parallelism.

For multi-GPU vLLM, run one process with all target GPUs visible and set:

```bash
BACKEND=vllm
TENSOR_PARALLEL_SIZE=8
GPU_MEMORY_UTILIZATION=0.9
```

The vLLM backend still uses the checkpoint's own tokenizer and the same
plain-text prompt templates as the HF backend.

On the local V100 smoke-test node, vLLM 0.11.0 can import and load OLMo3 after
prepending the conda C++ runtime, but generation currently fails in vLLM's
V1/FlexAttention path for this model. Use the HF backend on V100s, or run vLLM
on H100/A100-class nodes where FlashAttention2 is supported. If vLLM import
fails with a `CXXABI_1.3.15` error, launch with:

```bash
export LD_LIBRARY_PATH=/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo/lib:${LD_LIBRARY_PATH:-}
```

If vLLM hits CUDA graph issues, try:

```bash
ENFORCE_EAGER=true
```

## Checkpoints

Base HF checkpoints can be evaluated directly. Example:

```bash
/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo/bin/python \
  scripts/eval_pretrain_benchmarks.py \
  --backend vllm \
  --tensor-parallel-size 8 \
  --checkpoint stage3_main=/dlabscratch1/samaier/checkpoints/olmo-7b-stage3/main \
  --prompt-mode base \
  --tasks gsm8k \
  --temperature 0.6 \
  --top-p 1.0 \
  --num-samples 32 \
  --pass-at-k 1,8,32 \
  --torch-dtype float16 \
  --output-dir outputs/pretrain_benchmarks/stage3_main
```

FSDP actor checkpoints must be merged to Hugging Face format first:

```bash
/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo/bin/python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /path/to/run/global_step_100/actor \
  --target_dir /path/to/run/global_step_100/actor/hf_merged
```

Then evaluate the merged checkpoint as a trained model:

```bash
/mnt/dlabscratch1/samaier/conda-envs/sdpo-grpo/bin/python \
  scripts/eval_pretrain_benchmarks.py \
  --backend vllm \
  --tensor-parallel-size 8 \
  --checkpoint grpo_step100=/path/to/run/global_step_100/actor/hf_merged \
  --prompt-mode trained \
  --tasks gsm8k \
  --temperature 0.6 \
  --top-p 1.0 \
  --num-samples 32 \
  --pass-at-k 1,8,32 \
  --torch-dtype float16 \
  --output-dir outputs/pretrain_benchmarks/grpo_step100
```

For RL-Excursions-style comparison, use:

- base checkpoint: `--prompt-mode base`, default `8-shot`
- trained checkpoint: `--prompt-mode trained`, default `0-shot`
- prompt style: `--prompt-style rlx`
- decoding: `--temperature 0.6`
- sample count: `--num-samples 32`
- metrics: `--pass-at-k 1,8,32`

The script's `--prompt-mode` is global, so base and trained checkpoints should
be run as separate invocations unless they use the same prompt mode.

## Prompt And Scoring Modes

The default benchmark mode is intended for the RL Excursions comparison:

```bash
PROMPT_STYLE=rlx
ANSWER_FORMAT=auto
```

For GSM8K this builds plain `Question:` / `Answer:` prompts. Base checkpoints
get 8 fixed exemplars from the train split, trained checkpoints default to
0-shot, and `ANSWER_FORMAT=auto` uses flexible numeric extraction from the
completion. This keeps the benchmark independent of verl training validation.

There are also two diagnostic prompt styles for investigating gaps between
benchmark results and training validation:

- `PROMPT_STYLE=boxed`: plain text `Problem:` / `Solution:` prompt asking for
  `\boxed{}`.
- `PROMPT_STYLE=validation_chat`: decoded verl-style text,
  `User: ...\n\nAssistant:`, also asking for `\boxed{}`.

Those diagnostic styles should not replace the RL-style comparison table. They
are useful for checking whether a trained checkpoint learned the OpenMath boxed
answer interface but underperforms when asked for GSM8K `####`-style answers.

## Interpreting Training Validation Differences

The OPSD/GRPO validation metric is not the same measurement as this benchmark:

- Training used `datasets/openmathinstruct`, while the benchmark uses held-out
  `datasets/gsm8k` by default.
- The OpenMathInstruct preprocessing stores every row as `data_source="math"`,
  including rows whose `extra_info.source_dataset` is `gsm8k`.
- That means verl validation routes those rows through the math reward function,
  which expects a final `\boxed{...}` answer.
- The decoded validation prompts are chat-style text like
  `User: <problem> Let's think step by step ... \boxed{}.\n\nAssistant:`.
- The benchmark's RL-style GSM8K mode uses plain `Question:` / `Answer:` prompts
  and evaluates numeric answers from a 32-sample pool with pass@k.
- verl validation aggregates rollout samples as `mean@N`, `best@N`, and
  majority metrics; pass@k is an unbiased estimator from the same generated
  sample pool and is a different statistic.

For the cleanest RL Excursions comparison, keep the benchmark in `rlx` mode and
report base 8-shot versus trained 0-shot. For diagnosing why training validation
looks much higher, run the same checkpoints again with `PROMPT_STYLE=boxed` or
`PROMPT_STYLE=validation_chat` and compare the deltas.

## Launcher

The shell launcher exposes the same knobs through environment variables:

```bash
CHECKPOINTS="stage3_main=/dlabscratch1/samaier/checkpoints/olmo-7b-stage3/main" \
PROMPT_MODE=base \
PROMPT_STYLE=rlx \
BACKEND=vllm \
TENSOR_PARALLEL_SIZE=8 \
NUM_SAMPLES=32 \
PASS_AT_K=1,8,32 \
TEMPERATURE=0.6 \
TOP_P=1.0 \
TORCH_DTYPE=float16 \
OUTPUT_DIR=outputs/pretrain_benchmarks/stage3_main \
experiments/pretrain/run_pretrain_benchmark_eval.sh
```

Useful smoke-test knobs:

```bash
MAX_EXAMPLES=32
NUM_SAMPLES=8
PASS_AT_K=1,8
MAX_NEW_TOKENS=256
```

Validation-style diagnostic run:

```bash
CHECKPOINTS="grpo_step100=/path/to/run/global_step_100/actor/hf_merged" \
PROMPT_MODE=trained \
PROMPT_STYLE=validation_chat \
ANSWER_FORMAT=boxed \
MAX_EXAMPLES=32 \
NUM_SAMPLES=8 \
PASS_AT_K=1,8 \
experiments/pretrain/run_pretrain_benchmark_eval.sh
```

## Outputs

Each run creates a timestamped directory containing:

- `run_config.json`: exact CLI config
- `summary.csv`: compact metrics table
- `summary.json`: same metrics as JSON
- `{checkpoint}__{task}.jsonl`: prompts, completions, extracted answers, and per-sample correctness

To combine a few `summary.csv` files:

```bash
python - <<'PY'
import csv
from pathlib import Path

inputs = [
    Path("outputs/pretrain_benchmarks/base_run/summary.csv"),
    Path("outputs/pretrain_benchmarks/grpo_run/summary.csv"),
]
out = Path("outputs/pretrain_benchmarks/comparison_summary.csv")
rows = []
for path in inputs:
    with path.open(newline="") as f:
        rows.extend(csv.DictReader(f))
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

## Plotting

Use the plotting helper on one or more summary CSVs:

```bash
python scripts/plot_pretrain_benchmarks.py \
  --summary-csv outputs/pretrain_benchmarks/comparison_summary.csv \
  --metrics pass@1,pass@8,pass@32 \
  --title "GSM8K: Base vs GRPO"
```

This writes:

- `pretrain_benchmark_comparison.png`
- `pretrain_benchmark_comparison.pdf`

For quick slice plots where only pass@1/pass@8 are available:

```bash
python scripts/plot_pretrain_benchmarks.py \
  --summary-csv outputs/pretrain_benchmarks/quick32_grpo_main_vs_base/combined_summary.csv \
  --metrics pass@1,pass@8 \
  --title "GSM8K Quick Slice: Stage3 Main vs GRPO Step 100"
```
