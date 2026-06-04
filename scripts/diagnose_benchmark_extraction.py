#!/usr/bin/env python3
"""Diagnose answer extraction sensitivity in saved benchmark generations.

This is a post-processing script for existing benchmark JSONL files. It does
not change official summaries. The goal is to quantify how much score is lost
to answer-format details such as a model continuing after ``####`` or emitting
multiple ``\\boxed{...}`` answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Callable


SOLUTION_CLIP_CHARS = 300
DEFAULT_PASS_KS = [1, 8, 32]
USE_MATH_VERIFY = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=[],
        help="Benchmark root to scan for prediction JSONLs. Can be repeated.",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        action="append",
        default=[],
        help="Specific prediction JSONL to diagnose. Can be repeated.",
    )
    parser.add_argument(
        "--task",
        choices=["auto", "gsm8k", "math"],
        default="auto",
        help="Task override. Auto infers from the prediction filename or rows.",
    )
    parser.add_argument(
        "--pass-at-k",
        default="1,8,32",
        help="Comma-separated pass@k values.",
    )
    parser.add_argument(
        "--comparison-name",
        default="extraction_diagnostics.csv",
        help="Aggregate CSV name written in each --root.",
    )
    parser.add_argument(
        "--examples-name",
        default="extraction_diagnostic_examples.jsonl",
        help="Recovered-example JSONL name written in each --root.",
    )
    parser.add_argument(
        "--max-examples-per-file",
        type=int,
        default=20,
        help="Maximum diagnostic examples to write per prediction file.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on examples read from each prediction file for fast diagnostics.",
    )
    parser.add_argument(
        "--write-per-run",
        action="store_true",
        help="Also write extraction_diagnostics.csv next to each prediction file.",
    )
    parser.add_argument(
        "--use-math-verify",
        action="store_true",
        help=(
            "Use the repository MATH verifier for candidate equivalence. This is "
            "more faithful but much slower on full 32-sample MATH runs."
        ),
    )
    return parser.parse_args()


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    value = value.replace(",", "").replace("$", "")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(?<=\d)[.;:]+$", "", value)
    if re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def answers_equal(answer: str | None, ground_truth: str | None) -> bool:
    if answer is None:
        return False
    lhs = normalize_answer(answer)
    rhs = normalize_answer(ground_truth)
    if lhs == rhs:
        return True
    return verify_math_answer(answer, ground_truth)


def verify_math_answer(answer: str | None, ground_truth: str | None) -> bool:
    if not USE_MATH_VERIFY:
        return False
    if answer is None or ground_truth is None:
        return False
    try:
        from verl.utils.reward_score.feedback.math import verify as verify_math

        completion = f"\\boxed{{{answer}}}"
        correct, _ = verify_math(completion, str(ground_truth))
        return bool(correct)
    except Exception:
        return False


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - product


def infer_task(path: Path, row: dict[str, Any] | None, override: str) -> str:
    if override != "auto":
        return override
    if row and row.get("task") in {"gsm8k", "math"}:
        return str(row["task"])
    if "__math" in path.name:
        return "math"
    if "__gsm8k" in path.name:
        return "gsm8k"
    raise ValueError(f"Could not infer task for {path}")


def infer_group(root: Path | None, path: Path) -> str:
    if root is None:
        return ""
    try:
        return path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return ""


def parse_pass_ks(value: str) -> list[int]:
    ks = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    return ks or DEFAULT_PASS_KS


def clip_tail(text: str) -> str:
    return text[-SOLUTION_CLIP_CHARS:] if len(text) > SOLUTION_CLIP_CHARS else text


def extract_numeric_last(text: str, *, clipped: bool) -> str | None:
    if clipped:
        text = clip_tail(text)
    answers = re.findall(r"(-?[0-9][0-9.,]*)", text)
    for answer in reversed(answers):
        if answer not in ["", "."]:
            return answer.replace(",", "")
    return None


def extract_hashes(text: str, *, clipped: bool) -> list[str]:
    if clipped:
        text = clip_tail(text)
    return [
        answer.replace(",", "").replace("$", "")
        for answer in re.findall(r"####\s*(-?[$0-9.,]+)", text)
    ]


def extract_hash_first_full(text: str) -> str | None:
    hashes = extract_hashes(text, clipped=False)
    return hashes[0] if hashes else None


def extract_hash_last_full(text: str) -> str | None:
    hashes = extract_hashes(text, clipped=False)
    return hashes[-1] if hashes else None


def extract_hash_last_clipped(text: str) -> str | None:
    hashes = extract_hashes(text, clipped=True)
    return hashes[-1] if hashes else None


def extract_numeric_before_second_hash(text: str) -> str | None:
    parts = text.split("####")
    if len(parts) >= 3:
        text = "####".join(parts[:2])
    return extract_numeric_last(text, clipped=False)


def extract_boxed_values(text: str, *, clipped: bool) -> list[str]:
    if clipped:
        text = clip_tail(text)
    values: list[str] = []
    start = 0
    marker = r"\boxed{"
    while True:
        idx = text.find(marker, start)
        if idx < 0:
            break
        i = idx + len(marker)
        depth = 1
        chars: list[str] = []
        while i < len(text):
            char = text[i]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    values.append("".join(chars).strip())
                    break
            chars.append(char)
            i += 1
        start = max(i + 1, idx + len(marker))
    return values


def extract_boxed_first_full(text: str) -> str | None:
    boxes = extract_boxed_values(text, clipped=False)
    return boxes[0] if boxes else None


def extract_boxed_last_full(text: str) -> str | None:
    boxes = extract_boxed_values(text, clipped=False)
    return boxes[-1] if boxes else None


def extract_boxed_last_clipped(text: str) -> str | None:
    boxes = extract_boxed_values(text, clipped=True)
    return boxes[-1] if boxes else None


def extract_before_think_end(extractor: Callable[[str], str | None], text: str) -> str | None:
    for marker in ["</think>", "<|endoftext|>"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return extractor(text)


def policy_extractors(task: str) -> dict[str, Callable[[str], str | None]]:
    shared = {
        "numeric_last_full": lambda text: extract_numeric_last(text, clipped=False),
        "numeric_last_clipped": lambda text: extract_numeric_last(text, clipped=True),
    }
    if task == "gsm8k":
        return {
            **shared,
            "hash_first_full": extract_hash_first_full,
            "hash_last_full": extract_hash_last_full,
            "hash_last_clipped": extract_hash_last_clipped,
            "numeric_before_second_hash": extract_numeric_before_second_hash,
        }
    return {
        **shared,
        "boxed_first_full": extract_boxed_first_full,
        "boxed_last_full": extract_boxed_last_full,
        "boxed_last_clipped": extract_boxed_last_clipped,
        "boxed_first_before_think_end": lambda text: extract_before_think_end(
            extract_boxed_first_full, text
        ),
        "boxed_last_before_think_end": lambda text: extract_before_think_end(
            extract_boxed_last_full, text
        ),
    }


def oracle_correct(task: str, completion: str, ground_truth: str) -> bool:
    candidates: list[str] = []
    candidates.extend(extract_hashes(completion, clipped=False))
    candidates.extend(extract_boxed_values(completion, clipped=False))
    if task == "gsm8k":
        candidates.append(extract_numeric_before_second_hash(completion))
    return any(answers_equal(candidate, ground_truth) for candidate in candidates)


def load_rows(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_examples is not None and len(rows) >= max_examples:
                    break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def diagnose_file(
    path: Path,
    root: Path | None,
    task_override: str,
    pass_ks: list[int],
    example_f,
    max_examples_per_file: int,
    max_examples: int | None,
) -> dict[str, Any]:
    rows = load_rows(path, max_examples=max_examples)
    task = infer_task(path, rows[0], task_override)
    extractors = policy_extractors(task)
    totals = {name: {k: 0.0 for k in pass_ks} for name in extractors}
    totals["stored_correct"] = {k: 0.0 for k in pass_ks}
    totals["oracle_any_marker"] = {k: 0.0 for k in pass_ks}

    sample_counts = {name: 0 for name in totals}
    marker_counts = {
        "hash_present": 0,
        "boxed_present": 0,
        "second_hash_present": 0,
        "multiple_boxed_present": 0,
        "think_end_present": 0,
        "new_question_present": 0,
    }
    total_samples = 0
    total_chars = 0
    maxed_like_samples = 0
    examples_written = 0

    last_row: dict[str, Any] = rows[-1]
    for row in rows:
        ground_truth = str(row.get("ground_truth", ""))
        samples = row.get("samples", [])
        per_policy_correct = {name: 0 for name in totals}
        for sample in samples:
            completion = str(sample.get("completion") or sample.get("raw_completion") or "")
            total_samples += 1
            total_chars += len(completion)
            maxed_like_samples += int(len(completion) >= 0.95 * (1024 * 4))

            hashes = extract_hashes(completion, clipped=False)
            boxes = extract_boxed_values(completion, clipped=False)
            marker_counts["hash_present"] += int(bool(hashes))
            marker_counts["boxed_present"] += int(bool(boxes))
            marker_counts["second_hash_present"] += int(len(hashes) >= 2)
            marker_counts["multiple_boxed_present"] += int(len(boxes) >= 2)
            marker_counts["think_end_present"] += int("</think>" in completion)
            marker_counts["new_question_present"] += int(
                bool(re.search(r"\n\s*(Question|Problem|User)\s*:", completion))
            )

            stored_correct = bool(sample.get("correct"))
            per_policy_correct["stored_correct"] += int(stored_correct)
            for name, extractor in extractors.items():
                per_policy_correct[name] += int(answers_equal(extractor(completion), ground_truth))
            per_policy_correct["oracle_any_marker"] += int(
                oracle_correct(task, completion, ground_truth)
            )

            if (
                not stored_correct
                and per_policy_correct["oracle_any_marker"] > 0
                and examples_written < max_examples_per_file
            ):
                example_f.write(
                    json.dumps(
                        {
                            "task": task,
                            "checkpoint": row.get("checkpoint"),
                            "group": infer_group(root, path),
                            "index": row.get("index"),
                            "ground_truth": ground_truth,
                            "stored_extracted_answer": sample.get("extracted_answer"),
                            "hashes": hashes,
                            "boxes": boxes,
                            "numeric_before_second_hash": extract_numeric_before_second_hash(
                                completion
                            ),
                            "completion": completion,
                            "predictions_path": str(path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                examples_written += 1

        num_samples = len(samples)
        for name, correct_count in per_policy_correct.items():
            sample_counts[name] += correct_count
            for k in pass_ks:
                totals[name][k] += estimate_pass_at_k(num_samples, correct_count, k)

    num_examples = len(rows)
    result: dict[str, Any] = {
        "task": task,
        "group": infer_group(root, path),
        "checkpoint": last_row.get("checkpoint") or path.parent.parent.name,
        "checkpoint_path": last_row.get("checkpoint_path", ""),
        "prompt_mode": last_row.get("prompt_mode", ""),
        "prompt_style": last_row.get("prompt_style", ""),
        "answer_format": last_row.get("answer_format", ""),
        "num_examples": num_examples,
        "num_samples": int(total_samples / num_examples) if num_examples else 0,
        "total_samples": total_samples,
        "avg_completion_chars": total_chars / total_samples if total_samples else 0.0,
        "maxed_like_rate": maxed_like_samples / total_samples if total_samples else 0.0,
        "predictions_path": str(path),
    }
    for key, value in marker_counts.items():
        result[f"{key}_rate"] = value / total_samples if total_samples else 0.0
    for name in totals:
        result[f"{name}_sample_accuracy"] = (
            sample_counts[name] / total_samples if total_samples else 0.0
        )
        for k in pass_ks:
            result[f"{name}_pass@{k}"] = totals[name][k] / num_examples
    for k in pass_ks:
        result[f"oracle_gain_over_stored_pass@{k}"] = (
            result[f"oracle_any_marker_pass@{k}"] - result[f"stored_correct_pass@{k}"]
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_paths_for_root(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.jsonl")
        if re.search(r"__(gsm8k|math)\.jsonl$", path.name)
        and not path.name.endswith(".rescored.jsonl")
    )


def main() -> None:
    global USE_MATH_VERIFY
    args = parse_args()
    USE_MATH_VERIFY = bool(args.use_math_verify)
    pass_ks = parse_pass_ks(args.pass_at_k)
    if not args.root and not args.prediction:
        raise SystemExit("Pass at least one --root or --prediction.")

    for root in args.root:
        root = root.resolve()
        paths = prediction_paths_for_root(root)
        if not paths:
            print(f"No prediction JSONLs found under {root}")
            continue
        examples_path = root / args.examples_name
        rows = []
        with examples_path.open("w", encoding="utf-8") as example_f:
            for path in paths:
                row = diagnose_file(
                    path=path,
                    root=root,
                    task_override=args.task,
                    pass_ks=pass_ks,
                    example_f=example_f,
                    max_examples_per_file=args.max_examples_per_file,
                    max_examples=args.max_examples,
                )
                rows.append(row)
                if args.write_per_run:
                    write_csv(path.parent / "extraction_diagnostics.csv", [row])
        comparison_path = root / args.comparison_name
        write_csv(comparison_path, rows)
        print(f"diagnosed {len(rows)} prediction files under {root}")
        print(f"wrote {comparison_path}")
        print(f"wrote {examples_path}")

    if args.prediction:
        rows = []
        examples_path = Path(args.examples_name).resolve()
        with examples_path.open("w", encoding="utf-8") as example_f:
            for path in args.prediction:
                row = diagnose_file(
                    path=path.resolve(),
                    root=None,
                    task_override=args.task,
                    pass_ks=pass_ks,
                    example_f=example_f,
                    max_examples_per_file=args.max_examples_per_file,
                    max_examples=args.max_examples,
                )
                rows.append(row)
                if args.write_per_run:
                    write_csv(path.parent / "extraction_diagnostics.csv", [row])
        comparison_path = Path(args.comparison_name).resolve()
        write_csv(comparison_path, rows)
        print(f"diagnosed {len(rows)} explicit prediction files")
        print(f"wrote {comparison_path}")
        print(f"wrote {examples_path}")


if __name__ == "__main__":
    main()
