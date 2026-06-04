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
import os
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


DEFAULT_GSM8K_STOPS = ["\n\nQuestion:", "\nQuestion:", "\n\nProblem:", "\nProblem:", "\n\nUser:", "\nUser:"]
SOLUTION_CLIP_CHARS = 300


@dataclass(frozen=True)
class TaskSpec:
    name: str
    train_path: Path
    eval_path: Path
    prompt_builder: Callable[[str, list[dict[str, str]], str, str], str]
    exemplar_loader: Callable[[Path, int], list[dict[str, str]]]
    row_loader: Callable[[Path, int | None], list[dict[str, Any]]]
    scorer: Callable[[str, str, str], tuple[bool, str | None]]


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class CompletionResult:
    text: str
    token_count: int
    finish_reason: str | None = None
    stop_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        "--tasks",
        dest="tasks",
        action="append",
        default=[],
        help="Benchmark task name. Can be repeated or comma-separated. Currently: gsm8k, math, math500.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        required=True,
        help="Checkpoint path or NAME=PATH. Can be passed multiple times.",
    )
    parser.add_argument("--prompt-mode", choices=["base", "trained"], required=True)
    parser.add_argument(
        "--prompt-style",
        choices=["rlx", "boxed", "validation_chat"],
        default="rlx",
        help=(
            "Prompt family. rlx is the GSM8K Question/Answer prompt with #### answers; "
            "boxed asks for \\boxed{} in plain text; validation_chat mimics the decoded "
            "verl validation prompt without relying on a tokenizer chat template."
        ),
    )
    parser.add_argument(
        "--answer-format",
        choices=["auto", "gsm8k_hash", "boxed", "flexible_numeric"],
        default="auto",
        help="Scoring/extraction format. auto uses flexible_numeric for rlx and boxed for boxed/validation_chat.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pretrain_benchmarks"))
    parser.add_argument("--gsm8k-train-path", type=Path, default=Path("datasets/gsm8k/train.parquet"))
    parser.add_argument("--gsm8k-eval-path", type=Path, default=Path("datasets/gsm8k/test.parquet"))
    parser.add_argument("--math-train-path", type=Path, default=Path("datasets/math/train.parquet"))
    parser.add_argument("--math-eval-path", type=Path, default=Path("datasets/math/test.parquet"))
    parser.add_argument("--math500-eval-path", type=Path, default=Path("datasets/math500/test.parquet"))
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
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="Generation backend.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Optional vLLM max model length.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph capture in vLLM.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-kwargs-json", default=None, help="Optional JSON dict passed to from_pretrained.")
    parser.add_argument(
        "--resume-incomplete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume the latest compatible incomplete run under --output-dir and reuse saved JSONL rows.",
    )
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


def run_config(args: argparse.Namespace, tasks: list[str], stop_sequences: list[str]) -> dict[str, Any]:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["output_dir"] = str(args.output_dir)
    config["stop_sequences"] = stop_sequences
    config["resolved_answer_format"] = {
        task_name: resolve_answer_format(args.prompt_style, args.answer_format, task_name)
        for task_name in tasks
    }
    config["tasks"] = tasks
    return config


RESUME_CONFIG_KEYS = (
    "tasks",
    "checkpoints",
    "prompt_mode",
    "prompt_style",
    "answer_format",
    "num_fewshot",
    "num_samples",
    "pass_at_k",
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "max_prompt_tokens",
    "max_examples",
    "stop_sequences",
    "seed",
    "backend",
    "gsm8k_train_path",
    "gsm8k_eval_path",
    "math_train_path",
    "math_eval_path",
    "math500_eval_path",
)


def compatible_run_config(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    return all(existing.get(key) == current.get(key) for key in RESUME_CONFIG_KEYS)


def find_resume_run_dir(output_dir: Path, config: dict[str, Any]) -> Path | None:
    if not output_dir.exists():
        return None
    for config_path in sorted(output_dir.glob("*/run_config.json"), reverse=True):
        run_dir = config_path.parent
        if (run_dir / "summary.csv").exists() or (run_dir / "summary.json").exists():
            continue
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if compatible_run_config(existing, config):
            return run_dir
    return None


def prediction_key(index: Any) -> str:
    return str(index)


def load_existing_predictions(
    predictions_path: Path,
    checkpoint: CheckpointSpec,
    task: TaskSpec,
    prompt_mode: str,
    prompt_style: str,
    answer_format: str,
    num_samples: int,
) -> dict[str, dict[str, Any]]:
    if not predictions_path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with predictions_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {predictions_path}:{line_number}: {exc}") from exc
            samples = record.get("samples")
            if not isinstance(samples, list) or len(samples) != num_samples:
                continue
            expected = {
                "task": task.name,
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint.path),
                "prompt_mode": prompt_mode,
                "prompt_style": prompt_style,
                "answer_format": answer_format,
            }
            mismatched = [key for key, value in expected.items() if record.get(key) != value]
            if mismatched:
                raise ValueError(
                    f"Existing prediction row in {predictions_path}:{line_number} does not match "
                    f"the requested run fields: {', '.join(mismatched)}"
                )
            records[prediction_key(record.get("index"))] = record
    return records


def ensure_jsonl_append_boundary(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")


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
        ground_truth = str(extract_map(row["reward_model"])["ground_truth"])
        answer = str(extra.get("answer", "")).strip()
        if "####" not in answer:
            answer = f"{answer}\n#### {ground_truth}"
        boxed_answer = answer
        if r"\boxed{" not in boxed_answer:
            boxed_answer = f"{answer}\nTherefore, the final answer is \\boxed{{{ground_truth}}}."
        exemplars.append(
            {
                "question": str(extra["question"]).strip(),
                "answer": answer,
                "boxed_answer": boxed_answer,
                "ground_truth": ground_truth,
            }
        )
    return exemplars


MATH_INSTRUCTION = r"Let's think step by step and output the final answer within \boxed{}."


def prompt_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_py"):
        return value.as_py()
    return list(value)


def strip_math_instruction(text: str) -> str:
    return text.replace(" " + MATH_INSTRUCTION, "").replace(MATH_INSTRUCTION, "").strip()


def load_math_exemplars(train_path: Path, num_fewshot: int) -> list[dict[str, str]]:
    if num_fewshot <= 0:
        return []
    train_df = pd.read_parquet(train_path)
    if len(train_df) < num_fewshot:
        raise ValueError(f"Requested {num_fewshot} exemplars but {train_path} has {len(train_df)} rows.")
    exemplars = []
    for row in train_df.head(num_fewshot).to_dict("records"):
        messages = prompt_messages(row["prompt"])
        question = strip_math_instruction(str(messages[-1]["content"]))
        ground_truth = str(extract_map(row["reward_model"])["ground_truth"])
        exemplars.append(
            {
                "question": question,
                "answer": f"The final answer is \\boxed{{{ground_truth}}}.",
                "boxed_answer": f"The final answer is \\boxed{{{ground_truth}}}.",
                "ground_truth": ground_truth,
            }
        )
    return exemplars


def gsm8k_prompt(question: str, exemplars: list[dict[str, str]], prompt_mode: str, prompt_style: str) -> str:
    if prompt_style == "rlx":
        suffix = 'Let\'s think step by step and output the final answer after "####".'
        if prompt_mode == "base":
            blocks = []
            for example in exemplars:
                blocks.append(f"Question: {example['question']}\nAnswer: {example['answer']}")
            blocks.append(f"Question: {question}\nAnswer: {suffix}")
            return "\n\n".join(blocks)
        return f"Question: {question}\nAnswer: {suffix}"

    boxed_suffix = r"Let's think step by step and output the final answer within \boxed{}."
    if prompt_style == "boxed":
        if prompt_mode == "base":
            blocks = []
            for example in exemplars:
                blocks.append(f"Problem: {example['question']}\nSolution: {example['boxed_answer']}")
            blocks.append(f"Problem: {question}\nSolution: {boxed_suffix}")
            return "\n\n".join(blocks)
        return f"Problem: {question}\nSolution: {boxed_suffix}"

    if prompt_style == "validation_chat":
        # Decoded verl validation prompts for OLMo-style chat tokenizers look like
        # "User: ...\n\nAssistant:" after special tokens are skipped.
        if prompt_mode == "base":
            blocks = []
            for example in exemplars:
                blocks.append(f"User: {example['question']} {boxed_suffix}\n\nAssistant: {example['boxed_answer']}")
            blocks.append(f"User: {question} {boxed_suffix}\n\nAssistant:")
            return "\n\n".join(blocks)
        return f"User: {question} {boxed_suffix}\n\nAssistant:"

    raise ValueError(f"Unknown prompt style: {prompt_style}")


def math_prompt(question: str, exemplars: list[dict[str, str]], prompt_mode: str, prompt_style: str) -> str:
    if prompt_style == "validation_chat":
        if prompt_mode == "base":
            blocks = []
            for example in exemplars:
                blocks.append(f"User: {example['question']} {MATH_INSTRUCTION}\n\nAssistant: {example['boxed_answer']}")
            blocks.append(f"User: {question} {MATH_INSTRUCTION}\n\nAssistant:")
            return "\n\n".join(blocks)
        return f"User: {question} {MATH_INSTRUCTION}\n\nAssistant:"

    # MATH always uses boxed final answers. Treat rlx as boxed here so a mixed
    # gsm8k,math run cannot accidentally ask MATH for GSM8K #### answers.
    if prompt_mode == "base":
        blocks = []
        for example in exemplars:
            blocks.append(f"Problem: {example['question']}\nSolution: {example['boxed_answer']}")
        blocks.append(f"Problem: {question}\nSolution: {MATH_INSTRUCTION}")
        return "\n\n".join(blocks)
    return f"Problem: {question}\nSolution: {MATH_INSTRUCTION}"


def resolve_answer_format(prompt_style: str, answer_format: str, task_name: str = "gsm8k") -> str:
    if answer_format != "auto":
        return answer_format
    if task_name == "math":
        return "boxed"
    if prompt_style == "rlx":
        return "flexible_numeric"
    return "boxed"


def score_completion(completion: str, ground_truth: str, answer_format: str) -> tuple[bool, str | None]:
    if answer_format == "gsm8k_hash":
        answer = extract_gsm8k_solution(completion, method="strict")
    elif answer_format == "flexible_numeric":
        answer = extract_gsm8k_solution(completion, method="flexible")
    elif answer_format == "boxed":
        answer = extract_boxed_solution(completion)
    else:
        raise ValueError(f"Unknown answer format: {answer_format}")
    return normalize_answer(answer) == normalize_answer(ground_truth) if answer is not None else False, answer


def score_math_completion(completion: str, ground_truth: str, answer_format: str) -> tuple[bool, str | None]:
    del answer_format
    answer = extract_boxed_solution(completion)
    if answer is None:
        return False, None
    if normalize_answer(answer) == normalize_answer(ground_truth):
        return True, answer
    try:
        from verl.utils.reward_score.feedback.math import verify as verify_math

        correct, verified_answer = verify_math(completion, ground_truth)
        return bool(correct), verified_answer or answer
    except Exception:
        return False, answer


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    value = value.replace(",", "").replace("$", "")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    if re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def extract_boxed_solution(solution_str: str) -> str | None:
    if len(solution_str) > SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-SOLUTION_CLIP_CHARS:]
    idx = solution_str.rfind(r"\boxed{")
    if idx < 0:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    chars = []
    while i < len(solution_str):
        char = solution_str[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        i += 1
    return None


def gsm8k_score(completion: str, ground_truth: str) -> float:
    correct, _ = score_completion(completion, ground_truth, "flexible_numeric")
    return float(correct)


def extract_gsm8k_solution(solution_str: str, method: str = "flexible") -> str | None:
    if len(solution_str) > SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-SOLUTION_CLIP_CHARS:]
    if method == "strict":
        solutions = re.findall(r"####\s*(\-?[$0-9\.\,]+)", solution_str)
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
            exemplar_loader=load_gsm8k_exemplars,
            row_loader=load_gsm8k_eval_rows,
            scorer=score_completion,
        ),
        "math": TaskSpec(
            name="math",
            train_path=args.math_train_path,
            eval_path=args.math_eval_path,
            prompt_builder=math_prompt,
            exemplar_loader=load_math_exemplars,
            row_loader=load_math_eval_rows,
            scorer=score_math_completion,
        ),
        "math500": TaskSpec(
            name="math500",
            train_path=args.math_train_path,
            eval_path=args.math500_eval_path,
            prompt_builder=math_prompt,
            exemplar_loader=load_math_exemplars,
            row_loader=load_math_eval_rows,
            scorer=score_math_completion,
        ),
    }


def load_gsm8k_eval_rows(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
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


def load_math_eval_rows(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if max_examples is not None:
        df = df.head(max_examples)
    rows = []
    for row in df.to_dict("records"):
        extra = extract_map(row["extra_info"])
        reward = extract_map(row["reward_model"])
        messages = prompt_messages(row["prompt"])
        rows.append(
            {
                "index": extra.get("index", len(rows)),
                "question": strip_math_instruction(str(messages[-1]["content"])),
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


def count_text_tokens(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return None
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "completion_tokens_min": None,
            "completion_tokens_mean": None,
            "completion_tokens_median": None,
            "completion_tokens_p90": None,
            "completion_tokens_p95": None,
            "completion_tokens_p99": None,
            "completion_tokens_max": None,
        }
    ordered = sorted(values)

    def percentile(percent: float) -> int:
        idx = math.ceil((percent / 100.0) * len(ordered)) - 1
        idx = max(0, min(idx, len(ordered) - 1))
        return ordered[idx]

    return {
        "completion_tokens_min": ordered[0],
        "completion_tokens_mean": sum(ordered) / len(ordered),
        "completion_tokens_median": percentile(50),
        "completion_tokens_p90": percentile(90),
        "completion_tokens_p95": percentile(95),
        "completion_tokens_p99": percentile(99),
        "completion_tokens_max": ordered[-1],
    }


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
) -> list[list[CompletionResult]]:
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
    grouped: list[list[CompletionResult]] = [[] for _ in prompts]
    for output_idx, ids in enumerate(output_ids):
        prompt_idx = output_idx // args.num_samples
        completion_ids = ids[prompt_width:]
        grouped[prompt_idx].append(
            CompletionResult(
                text=tokenizer.decode(completion_ids, skip_special_tokens=True),
                token_count=len(completion_ids),
            )
        )
    return grouped


def load_vllm(args: argparse.Namespace, checkpoint_path: Path):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit(f"vLLM backend requested, but importing vllm.LLM failed: {exc}") from exc

    kwargs = {
        "model": str(checkpoint_path),
        "tokenizer": str(checkpoint_path),
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.torch_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.revision is not None:
        kwargs["revision"] = args.revision
    return LLM(**kwargs)


def generate_for_prompts_vllm(
    llm: Any,
    prompts: list[str],
    args: argparse.Namespace,
    stop_sequences: list[str],
) -> list[list[CompletionResult]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k is not None else 0,
        max_tokens=args.max_new_tokens,
        stop=stop_sequences or None,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)
    grouped = []
    for request_output in outputs:
        grouped.append(
            [
                CompletionResult(
                    text=completion.text,
                    token_count=len(completion.token_ids or []),
                    finish_reason=getattr(completion, "finish_reason", None),
                    stop_reason=getattr(completion, "stop_reason", None),
                )
                for completion in request_output.outputs
            ]
        )
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
    answer_format = resolve_answer_format(args.prompt_style, args.answer_format, task.name)
    exemplars = task.exemplar_loader(task.train_path, fewshot)
    rows = task.row_loader(task.eval_path, args.max_examples)
    predictions_path = run_dir / f"{checkpoint.name}__{task.name}.jsonl"
    existing_records = load_existing_predictions(
        predictions_path=predictions_path,
        checkpoint=checkpoint,
        task=task,
        prompt_mode=args.prompt_mode,
        prompt_style=args.prompt_style,
        answer_format=answer_format,
        num_samples=args.num_samples,
    )

    totals = {k: 0.0 for k in pass_ks}
    completion_token_counts: list[int] = []
    raw_completion_token_counts: list[int] = []
    correct_completion_token_counts: list[int] = []
    length_capped = 0

    def add_record_metrics(record: dict[str, Any]) -> None:
        nonlocal length_capped
        samples = record["samples"]
        correct_count = sum(int(bool(sample.get("correct"))) for sample in samples)
        for k in pass_ks:
            totals[k] += estimate_pass_at_k(args.num_samples, correct_count, k)
        for sample in samples:
            raw_tokens = sample.get("raw_completion_tokens")
            if raw_tokens is not None:
                raw_completion_token_counts.append(int(raw_tokens))
            completion_tokens = sample.get("completion_tokens")
            if completion_tokens is not None:
                completion_token_counts.append(int(completion_tokens))
                if sample.get("correct"):
                    correct_completion_token_counts.append(int(completion_tokens))
            length_capped += int(bool(sample.get("hit_max_new_tokens")))

    missing_rows = []
    for row in rows:
        key = prediction_key(row["index"])
        prompt = task.prompt_builder(row["question"], exemplars, args.prompt_mode, args.prompt_style)
        record = existing_records.get(key)
        if record is None:
            missing_rows.append(row)
            continue
        if record.get("question") != row["question"] or record.get("ground_truth") != row["ground_truth"]:
            raise ValueError(f"Existing prediction for {checkpoint.name}/{task.name} index={row['index']} has stale row content.")
        if record.get("prompt") != prompt:
            raise ValueError(f"Existing prediction for {checkpoint.name}/{task.name} index={row['index']} has a stale prompt.")
        add_record_metrics(record)

    if existing_records:
        print(
            f"[{checkpoint.name}/{task.name}] reusing {len(rows) - len(missing_rows)}/{len(rows)} saved examples from {predictions_path}",
            flush=True,
        )

    model = None
    llm = None
    tokenizer = None
    if missing_rows:
        if args.backend == "hf":
            model, tokenizer = load_model_and_tokenizer(args, checkpoint.path)
        else:
            llm = load_vllm(args, checkpoint.path)
            tokenizer = llm.get_tokenizer() if hasattr(llm, "get_tokenizer") else None

    if missing_rows:
        ensure_jsonl_append_boundary(predictions_path)
        with predictions_path.open("a", encoding="utf-8") as pred_f:
            for start in range(0, len(missing_rows), args.batch_size):
                batch = missing_rows[start : start + args.batch_size]
                prompts = [task.prompt_builder(row["question"], exemplars, args.prompt_mode, args.prompt_style) for row in batch]
                if args.backend == "hf":
                    completions_by_prompt = generate_for_prompts(model, tokenizer, prompts, args)
                else:
                    completions_by_prompt = generate_for_prompts_vllm(llm, prompts, args, stop_sequences)
                for row, prompt, completions in zip(batch, prompts, completions_by_prompt, strict=True):
                    sample_records = []
                    correct_count = 0
                    for sample_idx, completion_result in enumerate(completions):
                        raw_completion = completion_result.text
                        stopped_completion, stop_reason = apply_stop_sequences(raw_completion, stop_sequences)
                        correct, extracted_answer = task.scorer(stopped_completion, row["ground_truth"], answer_format)
                        correct_count += int(correct)
                        completion_tokens = count_text_tokens(tokenizer, stopped_completion)
                        raw_completion_tokens = completion_result.token_count
                        hit_max_new_tokens = (
                            completion_result.finish_reason == "length"
                            or raw_completion_tokens >= args.max_new_tokens
                        )
                        sample_records.append(
                            {
                                "sample_index": sample_idx,
                                "raw_completion": raw_completion,
                                "completion": stopped_completion,
                                "stop_reason": stop_reason,
                                "generation_finish_reason": completion_result.finish_reason,
                                "generation_stop_reason": completion_result.stop_reason,
                                "raw_completion_tokens": raw_completion_tokens,
                                "completion_tokens": completion_tokens,
                                "hit_max_new_tokens": hit_max_new_tokens,
                                "extracted_answer": extracted_answer,
                                "correct": correct,
                            }
                        )
                    pass_at = {
                        f"pass@{k}": estimate_pass_at_k(args.num_samples, correct_count, k)
                        for k in pass_ks
                    }
                    record = {
                        "task": task.name,
                        "checkpoint": checkpoint.name,
                        "checkpoint_path": str(checkpoint.path),
                        "prompt_mode": args.prompt_mode,
                        "prompt_style": args.prompt_style,
                        "answer_format": answer_format,
                        "index": row["index"],
                        "question": row["question"],
                        "ground_truth": row["ground_truth"],
                        "prompt": prompt,
                        "num_correct": correct_count,
                        "pass_at": pass_at,
                        "samples": sample_records,
                    }
                    add_record_metrics(record)
                    pred_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done = len(rows) - len(missing_rows) + min(start + args.batch_size, len(missing_rows))
                print(f"[{checkpoint.name}/{task.name}] evaluated {done}/{len(rows)}", flush=True)

    del model
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = {f"pass@{k}": totals[k] / len(rows) if rows else math.nan for k in pass_ks}
    token_metrics = length_stats(completion_token_counts)
    token_metrics.update(
        {f"raw_{key}": value for key, value in length_stats(raw_completion_token_counts).items()}
    )
    token_metrics.update(
        {f"correct_{key}": value for key, value in length_stats(correct_completion_token_counts).items()}
    )
    total_samples = len(rows) * args.num_samples
    token_metrics["num_completion_token_samples"] = len(completion_token_counts)
    token_metrics["num_correct_completion_token_samples"] = len(correct_completion_token_counts)
    token_metrics["hit_max_new_tokens_count"] = length_capped
    token_metrics["hit_max_new_tokens_rate"] = length_capped / total_samples if total_samples else math.nan
    return {
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint.path),
        "task": task.name,
        "prompt_mode": args.prompt_mode,
        "prompt_style": args.prompt_style,
        "answer_format": answer_format,
        "num_examples": len(rows),
        "num_samples": args.num_samples,
        "num_fewshot": fewshot,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "backend": args.backend,
        "tensor_parallel_size": args.tensor_parallel_size if args.backend == "vllm" else None,
        "metrics": metrics,
        "token_metrics": token_metrics,
        "predictions_path": str(predictions_path),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.backend == "hf" and torch.cuda.is_available():
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

    config = run_config(args, tasks, stop_sequences)
    output_dir = args.output_dir.expanduser()
    run_dir = find_resume_run_dir(output_dir, config) if args.resume_incomplete else None
    if run_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Resuming incomplete benchmark run: {run_dir}", flush=True)

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
        token_metric_fields = [
            "completion_tokens_min",
            "completion_tokens_mean",
            "completion_tokens_median",
            "completion_tokens_p90",
            "completion_tokens_p95",
            "completion_tokens_p99",
            "completion_tokens_max",
            "raw_completion_tokens_min",
            "raw_completion_tokens_mean",
            "raw_completion_tokens_median",
            "raw_completion_tokens_p90",
            "raw_completion_tokens_p95",
            "raw_completion_tokens_p99",
            "raw_completion_tokens_max",
            "correct_completion_tokens_min",
            "correct_completion_tokens_mean",
            "correct_completion_tokens_median",
            "correct_completion_tokens_p90",
            "correct_completion_tokens_p95",
            "correct_completion_tokens_p99",
            "correct_completion_tokens_max",
            "num_completion_token_samples",
            "num_correct_completion_token_samples",
            "hit_max_new_tokens_count",
            "hit_max_new_tokens_rate",
        ]
        fieldnames = [
            "checkpoint",
            "checkpoint_path",
            "task",
            "prompt_mode",
            "prompt_style",
            "answer_format",
            "num_examples",
            "num_samples",
            "num_fewshot",
            "temperature",
            "top_p",
            "max_new_tokens",
            "backend",
            "tensor_parallel_size",
            *[f"pass@{k}" for k in pass_ks],
            *token_metric_fields,
            "predictions_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fieldnames}
            row.update(result["metrics"])
            row.update(result["token_metrics"])
            writer.writerow(row)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
