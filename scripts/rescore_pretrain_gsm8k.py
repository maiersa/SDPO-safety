#!/usr/bin/env python3
"""Rescore saved GSM8K benchmark generations.

This script is a post-processing pass over existing ``*__gsm8k.jsonl`` files.
It does not rerun generation and does not overwrite raw predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PASS_AT_PATTERN = re.compile(r"^pass@(\d+)$")
SOLUTION_CLIP_CHARS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Benchmark root containing saved *__gsm8k.jsonl files.",
    )
    parser.add_argument(
        "--pass-at-k",
        default=None,
        help="Comma-separated pass@k values. Defaults to the original summary/config values, or 1,8,32.",
    )
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help="Write sibling *__.rescored.jsonl files with corrected sample fields.",
    )
    parser.add_argument(
        "--write-summaries",
        action="store_true",
        help="Write summary.rescored.json/csv files in each run directory.",
    )
    parser.add_argument(
        "--comparison-name",
        default="rescored_comparison.csv",
        help="Name of the root-level aggregate CSV.",
    )
    parser.add_argument(
        "--recovered-name",
        default="rescored_recovered_examples.jsonl",
        help="Name of the root-level recovered-example audit JSONL.",
    )
    parser.add_argument(
        "--max-recovered-per-run",
        type=int,
        default=20,
        help="Maximum recovered samples to write per prediction file.",
    )
    parser.add_argument(
        "--require-summary",
        action="store_true",
        help="Only rescore prediction files in run directories that have summary.json.",
    )
    return parser.parse_args()


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    value = value.replace(",", "").replace("$", "")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"(?<=\d)[.;:]+$", "", value)
    if re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".", 1)[0]
    return value


def answers_equal(answer: str | None, ground_truth: str | None) -> bool:
    return answer is not None and normalize_answer(answer) == normalize_answer(ground_truth)


def extract_flexible_numeric(solution_str: str) -> str | None:
    if len(solution_str) > SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-SOLUTION_CLIP_CHARS:]
    answers = re.findall(r"(-?[0-9][0-9.,]*)", solution_str)
    for answer in reversed(answers):
        if answer not in ["", "."]:
            return answer.replace(",", "")
    return None


def extract_gsm8k_hash(solution_str: str) -> str | None:
    if len(solution_str) > SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-SOLUTION_CLIP_CHARS:]
    solutions = re.findall(r"####\s*(-?[$0-9.,]+)", solution_str)
    if not solutions:
        return None
    return solutions[-1].replace(",", "").replace("$", "")


def extract_boxed_solution(solution_str: str) -> str | None:
    if len(solution_str) > SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-SOLUTION_CLIP_CHARS:]
    idx = solution_str.rfind(r"\boxed{")
    if idx < 0:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    chars: list[str] = []
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


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    if num_correct <= 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (num_samples - num_correct - i) / (num_samples - i)
    return 1.0 - product


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_summary(path: Path) -> dict[str, Any] | None:
    summary_path = path.parent / "summary.json"
    if not summary_path.exists():
        return None
    data = load_json(summary_path)
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def read_run_config(path: Path) -> dict[str, Any]:
    config_path = path.parent / "run_config.json"
    if not config_path.exists():
        return {}
    data = load_json(config_path)
    return data if isinstance(data, dict) else {}


def pass_ks_from_metadata(
    summary: dict[str, Any] | None,
    run_config: dict[str, Any],
    fallback: str | None,
) -> list[int]:
    if fallback:
        return sorted({int(part.strip()) for part in fallback.split(",") if part.strip()})
    if summary and isinstance(summary.get("metrics"), dict):
        ks = []
        for key in summary["metrics"]:
            match = PASS_AT_PATTERN.match(key)
            if match:
                ks.append(int(match.group(1)))
        if ks:
            return sorted(set(ks))
    configured = run_config.get("pass_at_k")
    if isinstance(configured, str):
        return sorted({int(part.strip()) for part in configured.split(",") if part.strip()})
    return [1, 8, 32]


def infer_group(root: Path, predictions_path: Path) -> str:
    try:
        return predictions_path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return ""


def infer_step(checkpoint: str) -> str:
    match = re.search(r"(?:^|_)stage\d+_(\d+)(?:_|$)", checkpoint)
    if match:
        return match.group(1)
    if checkpoint.endswith("_main"):
        return "main"
    match = re.search(r"(?:step|global_step)[_-]?(\d+)", checkpoint)
    if match:
        return match.group(1)
    return ""


def original_metrics_from_summary(summary: dict[str, Any] | None, pass_ks: list[int]) -> dict[str, float | str]:
    metrics = summary.get("metrics", {}) if summary else {}
    result: dict[str, float | str] = {}
    for k in pass_ks:
        value = metrics.get(f"pass@{k}") if isinstance(metrics, dict) else None
        result[f"original_pass@{k}"] = value if value is not None else ""
    return result


def format_correct_for_prompt(
    prompt_style: str,
    hash_correct: bool,
    boxed_correct: bool,
) -> bool:
    if prompt_style in {"boxed", "validation_chat"}:
        return boxed_correct
    return hash_correct


def rescore_predictions(
    root: Path,
    predictions_path: Path,
    pass_ks_arg: str | None,
    write_jsonl: bool,
    write_summaries: bool,
    recovered_f,
    max_recovered_per_run: int,
) -> dict[str, Any]:
    summary = read_summary(predictions_path)
    run_config = read_run_config(predictions_path)
    pass_ks = pass_ks_from_metadata(summary, run_config, pass_ks_arg)

    corrected_jsonl_path = predictions_path.with_name(
        predictions_path.name.replace(".jsonl", ".rescored.jsonl")
    )
    out_f = corrected_jsonl_path.open("w", encoding="utf-8") if write_jsonl else None

    totals = {k: 0.0 for k in pass_ks}
    original_totals = {k: 0.0 for k in pass_ks}
    num_examples = 0
    total_samples = 0
    original_correct_samples = 0
    corrected_correct_samples = 0
    recovered_samples = 0
    regressed_samples = 0
    hash_correct_samples = 0
    boxed_correct_samples = 0
    format_correct_samples = 0
    format_wrong_samples = 0
    hash_present_samples = 0
    boxed_present_samples = 0
    recovered_written = 0
    last_row: dict[str, Any] | None = None

    with predictions_path.open("r", encoding="utf-8") as in_f:
        for line in in_f:
            row = json.loads(line)
            last_row = row
            num_examples += 1
            ground_truth = str(row.get("ground_truth", ""))
            samples = row.get("samples", [])
            prompt_style = str(row.get("prompt_style") or run_config.get("prompt_style") or "")
            corrected_num_correct = 0
            original_num_correct = int(row.get("num_correct", 0) or 0)

            for sample in samples:
                completion = str(sample.get("completion") or "")
                original_extracted = sample.get("extracted_answer")
                original_correct = bool(sample.get("correct"))
                # Keep the original extraction policy fixed and only correct
                # normalization. Re-extracting from the full completion can
                # over-credit rambling outputs where an incidental final number
                # happens to match the ground truth.
                numeric_answer = str(original_extracted) if original_extracted not in [None, ""] else None
                hash_answer = extract_gsm8k_hash(completion)
                boxed_answer = extract_boxed_solution(completion)

                numeric_correct = answers_equal(numeric_answer, ground_truth)
                hash_correct = answers_equal(hash_answer, ground_truth)
                boxed_correct = answers_equal(boxed_answer, ground_truth)
                expected_format_correct = format_correct_for_prompt(
                    prompt_style,
                    hash_correct,
                    boxed_correct,
                )
                format_wrong = numeric_correct and not expected_format_correct

                sample["original_correct"] = original_correct
                sample["original_extracted_answer"] = original_extracted
                sample["corrected_extracted_answer"] = numeric_answer
                sample["numeric_correct"] = numeric_correct
                sample["hash_extracted_answer"] = hash_answer
                sample["hash_format_correct"] = hash_correct
                sample["boxed_extracted_answer"] = boxed_answer
                sample["boxed_format_correct"] = boxed_correct
                sample["expected_format_correct"] = expected_format_correct
                sample["format_wrong"] = format_wrong
                sample["correct"] = numeric_correct
                sample["extracted_answer"] = numeric_answer

                total_samples += 1
                original_correct_samples += int(original_correct)
                corrected_correct_samples += int(numeric_correct)
                recovered_samples += int(numeric_correct and not original_correct)
                regressed_samples += int(original_correct and not numeric_correct)
                hash_present_samples += int(hash_answer is not None)
                boxed_present_samples += int(boxed_answer is not None)
                hash_correct_samples += int(hash_correct)
                boxed_correct_samples += int(boxed_correct)
                format_correct_samples += int(expected_format_correct)
                format_wrong_samples += int(format_wrong)
                corrected_num_correct += int(numeric_correct)

                if numeric_correct and not original_correct and recovered_written < max_recovered_per_run:
                    recovered_f.write(
                        json.dumps(
                            {
                                "group": infer_group(root, predictions_path),
                                "checkpoint": row.get("checkpoint", predictions_path.parent.parent.name),
                                "index": row.get("index"),
                                "ground_truth": ground_truth,
                                "original_extracted_answer": original_extracted,
                                "corrected_extracted_answer": numeric_answer,
                                "hash_extracted_answer": hash_answer,
                                "boxed_extracted_answer": boxed_answer,
                                "completion": completion,
                                "predictions_path": str(predictions_path),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    recovered_written += 1

            corrected_pass_at = {
                f"pass@{k}": estimate_pass_at_k(len(samples), corrected_num_correct, k)
                for k in pass_ks
            }
            original_pass_at = {
                f"pass@{k}": estimate_pass_at_k(len(samples), original_num_correct, k)
                for k in pass_ks
            }
            row["original_num_correct"] = original_num_correct
            row["corrected_num_correct"] = corrected_num_correct
            row["original_pass_at"] = original_pass_at
            row["corrected_pass_at"] = corrected_pass_at
            row["pass_at"] = corrected_pass_at

            for k in pass_ks:
                totals[k] += corrected_pass_at[f"pass@{k}"]
                original_totals[k] += original_pass_at[f"pass@{k}"]

            if out_f is not None:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if out_f is not None:
        out_f.close()

    if last_row is None:
        raise ValueError(f"No rows found in {predictions_path}")

    checkpoint = str(last_row.get("checkpoint") or predictions_path.parent.parent.name)
    checkpoint_path = str(last_row.get("checkpoint_path") or "")
    prompt_mode = str(last_row.get("prompt_mode") or run_config.get("prompt_mode") or "")
    prompt_style = str(last_row.get("prompt_style") or run_config.get("prompt_style") or "")
    answer_format = str(last_row.get("answer_format") or "")
    group = infer_group(root, predictions_path)
    num_samples = int(last_row.get("num_samples") or (total_samples // max(num_examples, 1)))

    corrected_metrics = {f"pass@{k}": totals[k] / num_examples for k in pass_ks}
    recomputed_original_metrics = {f"pass@{k}": original_totals[k] / num_examples for k in pass_ks}
    summary_row: dict[str, Any] = {
        "group": group,
        "checkpoint": checkpoint,
        "step": infer_step(checkpoint),
        "checkpoint_path": checkpoint_path,
        "task": "gsm8k",
        "prompt_mode": prompt_mode,
        "prompt_style": prompt_style,
        "answer_format": answer_format,
        "num_examples": num_examples,
        "num_samples": num_samples,
        "total_samples": total_samples,
        "original_correct_samples": original_correct_samples,
        "corrected_correct_samples": corrected_correct_samples,
        "sample_accuracy": corrected_correct_samples / total_samples if total_samples else 0.0,
        "original_sample_accuracy": original_correct_samples / total_samples if total_samples else 0.0,
        "recovered_samples": recovered_samples,
        "regressed_samples": regressed_samples,
        "format_wrong_samples": format_wrong_samples,
        "format_wrong_rate": format_wrong_samples / total_samples if total_samples else 0.0,
        "format_correct_samples": format_correct_samples,
        "format_correct_rate": format_correct_samples / total_samples if total_samples else 0.0,
        "hash_present_samples": hash_present_samples,
        "hash_present_rate": hash_present_samples / total_samples if total_samples else 0.0,
        "hash_correct_samples": hash_correct_samples,
        "hash_correct_rate": hash_correct_samples / total_samples if total_samples else 0.0,
        "boxed_present_samples": boxed_present_samples,
        "boxed_present_rate": boxed_present_samples / total_samples if total_samples else 0.0,
        "boxed_correct_samples": boxed_correct_samples,
        "boxed_correct_rate": boxed_correct_samples / total_samples if total_samples else 0.0,
        "predictions_path": str(predictions_path),
        "rescored_predictions_path": str(corrected_jsonl_path) if write_jsonl else "",
    }
    summary_row.update(original_metrics_from_summary(summary, pass_ks))
    for k in pass_ks:
        summary_row[f"recomputed_original_pass@{k}"] = recomputed_original_metrics[f"pass@{k}"]
        summary_row[f"corrected_pass@{k}"] = corrected_metrics[f"pass@{k}"]
        original = summary_row.get(f"original_pass@{k}")
        if isinstance(original, int | float):
            summary_row[f"delta_pass@{k}"] = corrected_metrics[f"pass@{k}"] - original
        else:
            summary_row[f"delta_pass@{k}"] = (
                corrected_metrics[f"pass@{k}"] - recomputed_original_metrics[f"pass@{k}"]
            )

    if write_summaries:
        rescored_summary = dict(summary or {})
        rescored_summary["original_metrics"] = summary.get("metrics", {}) if summary else {}
        rescored_summary["recomputed_original_metrics"] = recomputed_original_metrics
        rescored_summary["metrics"] = corrected_metrics
        rescored_summary["rescoring"] = {
            key: summary_row[key]
            for key in [
                "total_samples",
                "original_correct_samples",
                "corrected_correct_samples",
                "sample_accuracy",
                "original_sample_accuracy",
                "recovered_samples",
                "regressed_samples",
                "format_wrong_samples",
                "format_wrong_rate",
                "format_correct_samples",
                "format_correct_rate",
                "hash_present_samples",
                "hash_present_rate",
                "hash_correct_samples",
                "hash_correct_rate",
                "boxed_present_samples",
                "boxed_present_rate",
                "boxed_correct_samples",
                "boxed_correct_rate",
            ]
        }
        rescored_summary["predictions_path"] = str(predictions_path)
        rescored_summary["rescored_predictions_path"] = (
            str(corrected_jsonl_path) if write_jsonl else ""
        )
        dump_json(predictions_path.parent / "summary.rescored.json", [rescored_summary])
        write_csv(predictions_path.parent / "summary.rescored.csv", [summary_row])

    return summary_row


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


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    prediction_paths = sorted(
        path
        for path in root.rglob("*__gsm8k.jsonl")
        if not path.name.endswith(".rescored.jsonl")
    )
    if args.require_summary:
        prediction_paths = [
            path for path in prediction_paths if (path.parent / "summary.json").exists()
        ]
    if not prediction_paths:
        raise SystemExit(f"No *__gsm8k.jsonl files found under {root}")

    comparison_rows: list[dict[str, Any]] = []
    recovered_path = root / args.recovered_name
    with recovered_path.open("w", encoding="utf-8") as recovered_f:
        for predictions_path in prediction_paths:
            comparison_rows.append(
                rescore_predictions(
                    root=root,
                    predictions_path=predictions_path,
                    pass_ks_arg=args.pass_at_k,
                    write_jsonl=args.write_jsonl,
                    write_summaries=args.write_summaries,
                    recovered_f=recovered_f,
                    max_recovered_per_run=args.max_recovered_per_run,
                )
            )

    comparison_path = root / args.comparison_name
    write_csv(comparison_path, comparison_rows)
    print(f"rescored {len(comparison_rows)} prediction files")
    print(f"wrote {comparison_path}")
    print(f"wrote {recovered_path}")


if __name__ == "__main__":
    main()
