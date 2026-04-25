#!/usr/bin/env python3
"""Plain-text benchmark evaluation for pretraining checkpoints.

This runner is intentionally separate from training-time validation. It loads
prepared benchmark parquet files, builds plain-text prompts, samples K
responses once, and reports pass@k from that fixed sample pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_GSM8K_STOPS = ["\n\nQuestion:", "\nQuestion:", "\n\nProblem:", "\nProblem:"]
SOLUTION_CLIP_CHARS = 300


@dataclass(frozen=True)
class TaskSpec:
    name: str
    train_path: Path
    eval_path: Path
    prompt_builder: Callable[[str, list[dict[str, str]], str], str]
    scorer: Callable[[str, str], float]


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        "--tasks",
        dest="tasks",
        action="append",
        default=[],
        help="Benchmark task name. Can be repeated or comma-separated. Currently: gsm8k.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="Checkpoint path or NAME=PATH. Can be passed multiple times.",
    )
    parser.add_argument("--prompt-mode", choices=["base", "trained"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrain_benchmarks"))
    parser.add_argument("--gsm8k-train-path", type=Path, default=Path("datasets/gsm8k/train.parquet"))
    parser.add_argument("--gsm8k-eval-path", type=Path, default=Path("datasets/gsm8k/test.parquet"))
    parser.add_argument("--num-fewshot", type=int, default=None, help="Defaults to 8 for base mode and 0 for trained.")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--pass-at-k", default="1,8,32", help="Comma-separated k values.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Prompt batch size. Each prompt still returns num-samples.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional held-out example cap for smoke tests.")
    parser.add_argument("--stop-sequence", action="append", default=[], help="Stop sequence. Can be repeated.")
    parser.add_argument("--add-default-stops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-kwargs-json", default=None, help="Optional JSON dict passed to from_pretrained.")
    return parser.parse_args()


def normalize_tasks(raw_tasks: list[str]) -> list[str]:
    if not raw_tasks:
        return ["gsm8k"]
    tasks: list[str] = []
    for value in raw_tasks:
        tasks.extend(part.strip().lower() for part in value.split(",") if part.strip())
    return tasks


def parse_checkpoints(values: list[str]) -> list[CheckpointSpec]:
    specs = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            specs.append(CheckpointSpec(sanitize_name(name), Path(path).expanduser()))
        else:
            path = Path(value).expanduser()
            specs.append(CheckpointSpec(sanitize_name(path.name), path))
    return specs


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "checkpoint"


def parse_pass_at_k(value: str, num_samples: int) -> list[int]:
    ks = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not ks:
        raise ValueError("--pass-at-k must contain at least one k.")
    too_large = [k for k in ks if k > num_samples]
    if too_large:
        raise ValueError(f"pass@k values {too_large} exceed --num-samples={num_samples}.")
    return ks


def torch_dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def extract_map(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
    return dict(value)


def load_gsm8k_exemplars(train_path: Path, num_fewshot: int) -> list[dict[str, str]]:
    if num_fewshot <= 0:
        return []
    train_df = pd.read_parquet(train_path)
    if len(train_df) < num_fewshot:
        raise ValueError(f"Requested {num_fewshot} exemplars but {train_path} has {len(train_df)} rows.")
    exemplars = []
    for row in train_df.head(num_fewshot).to_dict("records"):
        extra = extract_map(row["extra_info"])
        answer = str(extra.get("answer", "")).strip()
        if "####" not in answer:
            answer = f"{answer}\n#### {extract_map(row['reward_model'])['ground_truth']}"
        exemplars.append({"question": str(extra["question"]).strip(), "answer": answer})
    return exemplars


def gsm8k_prompt(question: str, exemplars: list[dict[str, str]], prompt_mode: str) -> str:
    suffix = 'Let\'s think step by step and output the final answer after "####".'
    if prompt_mode == "base":
        blocks = []
        for example in exemplars:
            blocks.append(f"Question: {example['question']}\nAnswer: {example['answer']}")
        blocks.append(f"Question: {question}\nAnswer: {suffix}")
        return "\n\n".join(blocks)
    return f"Question: {question}\nAnswer: {suffix}"


def gsm8k_score(completion: str, ground_truth: str) -> float:
    answer = extract_gsm8k_solution(completion, method="flexible")
    return float(answer == ground_truth) if answer is not None else 0.0


def extract_gsm8k_solution(solution_str: str, method: str = "flexible") -> str | None:
    if len(solution_str) > SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-SOLUTION_CLIP_CHARS:]
    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if not solutions:
            return None
        return solutions[-1].replace(",", "").replace("$", "")
    if method != "flexible":
        raise ValueError(f"Unknown GSM8K extraction method: {method}")
    answers = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
    for answer in reversed(answers):
        if answer not in ["", "."]:
            return answer.replace(",", "")
    return None


def task_registry(args: argparse.Namespace) -> dict[str, TaskSpec]:
    return {
        "gsm8k": TaskSpec(
            name="gsm8k",
            train_path=args.gsm8k_train_path,
            eval_path=args.gsm8k_eval_path,
            prompt_builder=gsm8k_prompt,
            scorer=gsm8k_score,
        )
    }


def load_eval_rows(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if max_examples is not None:
        df = df.head(max_examples)
    rows = []
    for row in df.to_dict("records"):
        extra = extract_map(row["extra_info"])
        reward = extract_map(row["reward_model"])
        rows.append(
            {
                "index": extra.get("index", len(rows)),
                "question": str(extra["question"]).strip(),
                "ground_truth": str(reward["ground_truth"]),
                "source_row": row,
            }
        )
    return rows


def apply_stop_sequences(text: str, stop_sequences: list[str]) -> tuple[str, str | None]:
    first_pos: int | None = None
    matched: str | None = None
    for stop in stop_sequences:
        if not stop:
            continue
        pos = text.find(stop)
        if pos >= 0 and (first_pos is None or pos < first_pos):
            first_pos = pos
            matched = stop
    if first_pos is None:
        return text, None
    return text[:first_pos], matched


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    # Stable equivalent of 1 - comb(n - c, k) / comb(n, k).
    product = 1.0
    for i in range(k):
        product *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - product


def load_model_and_tokenizer(args: argparse.Namespace, checkpoint_path: Path):
    model_kwargs = json.loads(args.model_kwargs_json) if args.model_kwargs_json else {}
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map=args.device_map,
        torch_dtype=torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        **model_kwargs,
    )
    model.eval()
    return model, tokenizer


def generate_for_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    args: argparse.Namespace,
) -> list[list[str]]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=args.max_prompt_tokens is not None,
        max_length=args.max_prompt_tokens,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "num_return_sequences": args.num_samples,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.top_k is not None:
        generation_kwargs["top_k"] = args.top_k
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    prompt_width = inputs["input_ids"].shape[1]
    grouped: list[list[str]] = [[] for _ in prompts]
    for output_idx, ids in enumerate(output_ids):
        prompt_idx = output_idx // args.num_samples
        completion_ids = ids[prompt_width:]
        grouped[prompt_idx].append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return grouped


def evaluate_checkpoint_task(
    args: argparse.Namespace,
    checkpoint: CheckpointSpec,
    task: TaskSpec,
    pass_ks: list[int],
    stop_sequences: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    fewshot = args.num_fewshot
    if fewshot is None:
        fewshot = 8 if args.prompt_mode == "base" else 0
    exemplars = load_gsm8k_exemplars(task.train_path, fewshot)
    rows = load_eval_rows(task.eval_path, args.max_examples)

    model, tokenizer = load_model_and_tokenizer(args, checkpoint.path)
    predictions_path = run_dir / f"{checkpoint.name}__{task.name}.jsonl"
    totals = {k: 0.0 for k in pass_ks}

    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [task.prompt_builder(row["question"], exemplars, args.prompt_mode) for row in batch]
            completions_by_prompt = generate_for_prompts(model, tokenizer, prompts, args)
            for row, prompt, completions in zip(batch, prompts, completions_by_prompt, strict=True):
                sample_records = []
                correct_count = 0
                for sample_idx, raw_completion in enumerate(completions):
                    stopped_completion, stop_reason = apply_stop_sequences(raw_completion, stop_sequences)
                    correct = bool(task.scorer(stopped_completion, row["ground_truth"]))
                    correct_count += int(correct)
                    sample_records.append(
                        {
                            "sample_index": sample_idx,
                            "raw_completion": raw_completion,
                            "completion": stopped_completion,
                            "stop_reason": stop_reason,
                            "extracted_answer": extract_gsm8k_solution(stopped_completion, method="flexible"),
                            "correct": correct,
                        }
                    )
                pass_at = {
                    f"pass@{k}": estimate_pass_at_k(args.num_samples, correct_count, k)
                    for k in pass_ks
                }
                for k in pass_ks:
                    totals[k] += pass_at[f"pass@{k}"]
                pred_f.write(
                    json.dumps(
                        {
                            "task": task.name,
                            "checkpoint": checkpoint.name,
                            "checkpoint_path": str(checkpoint.path),
                            "prompt_mode": args.prompt_mode,
                            "index": row["index"],
                            "question": row["question"],
                            "ground_truth": row["ground_truth"],
                            "prompt": prompt,
                            "num_correct": correct_count,
                            "pass_at": pass_at,
                            "samples": sample_records,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            done = min(start + args.batch_size, len(rows))
            print(f"[{checkpoint.name}/{task.name}] evaluated {done}/{len(rows)}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = {f"pass@{k}": totals[k] / len(rows) if rows else math.nan for k in pass_ks}
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint.path),
        "task": task.name,
        "prompt_mode": args.prompt_mode,
        "num_examples": len(rows),
        "num_samples": args.num_samples,
        "num_fewshot": fewshot,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "metrics": metrics,
        "predictions_path": str(predictions_path),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tasks = normalize_tasks(args.tasks)
    registry = task_registry(args)
    unknown = [task for task in tasks if task not in registry]
    if unknown:
        raise ValueError(f"Unsupported task(s): {unknown}. Implemented tasks: {sorted(registry)}")

    pass_ks = parse_pass_at_k(args.pass_at_k, args.num_samples)
    stop_sequences = []
    if args.add_default_stops:
        stop_sequences.extend(DEFAULT_GSM8K_STOPS)
    stop_sequences.extend(args.stop_sequence)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.expanduser() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["stop_sequences"] = stop_sequences
    config["tasks"] = tasks
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    results = []
    for checkpoint in parse_checkpoints(args.checkpoints):
        for task_name in tasks:
            results.append(
                evaluate_checkpoint_task(
                    args=args,
                    checkpoint=checkpoint,
                    task=registry[task_name],
                    pass_ks=pass_ks,
                    stop_sequences=stop_sequences,
                    run_dir=run_dir,
                )
            )

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "checkpoint",
            "checkpoint_path",
            "task",
            "prompt_mode",
            "num_examples",
            "num_samples",
            "num_fewshot",
            "temperature",
            "top_p",
            "max_new_tokens",
            *[f"pass@{k}" for k in pass_ks],
            "predictions_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fieldnames}
            row.update(result["metrics"])
            writer.writerow(row)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
