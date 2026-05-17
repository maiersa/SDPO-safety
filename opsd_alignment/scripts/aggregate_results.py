"""Aggregate alignment JSONL records into grouped summary statistics."""

from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path
from opsd_alignment.src.gradients import summarize_alignment
from opsd_alignment.src.storage import read_jsonl, write_jsonl


DEFAULT_GROUP_KEYS = (
    "checkpoint",
    "teacher_context",
    "distillation_objective",
    "source",
    "difficulty",
    "student_rollout_correct",
    "selection_reason",
)

SUMMARY_FIELD_ORDER = (
    "checkpoint",
    "teacher_context",
    "distillation_objective",
    "source",
    "difficulty",
    "student_rollout_correct",
    "selection_reason",
    "total_records",
    "count_nodes",
    "invalid_alignment_count",
    "invalid_alignment_fraction",
    "mean_alignment",
    "median_alignment",
    "std_alignment",
    "standard_error",
    "fraction_positive_alignment",
    "mean_student_teacher_kl",
    "mean_student_entropy",
    "mean_baseline_success",
    "mean_branch_success",
    "mean_student_success_rate",
    "mean_num_candidates",
)


def aggregate_records(
    records: list[dict[str, Any]],
    group_keys: Iterable[str] = DEFAULT_GROUP_KEYS,
) -> list[dict[str, Any]]:
    group_keys = tuple(group_keys)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(key) for key in group_keys)].append(record)

    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = {name: value for name, value in zip(group_keys, key, strict=True)}
        alignment_values = [record.get("alignment") for record in group]
        summary.update(summarize_alignment(alignment_values))
        total_records = len(group)
        valid_count = int(summary["count_nodes"])
        summary["total_records"] = total_records
        summary["invalid_alignment_count"] = total_records - valid_count
        summary["invalid_alignment_fraction"] = (total_records - valid_count) / total_records if total_records else None
        summary["mean_student_teacher_kl"] = _mean(record.get("student_teacher_kl") for record in group)
        summary["mean_student_entropy"] = _mean(record.get("student_entropy") for record in group)
        summary["mean_baseline_success"] = _mean(record.get("baseline_success") for record in group)
        summary["mean_branch_success"] = _mean(record.get("mean_branch_success") for record in group)
        summary["mean_student_success_rate"] = _mean(_bool_to_float(record.get("student_rollout_correct")) for record in group)
        summary["mean_num_candidates"] = _mean(record.get("num_candidates") for record in group)
        summaries.append(summary)
    return summaries


def load_alignment_records(
    config: dict[str, Any],
    *,
    alignment_file: str | None = None,
    alignment_glob: str | None = None,
    model_name: str | None = None,
    teacher_context: str | None = None,
) -> list[dict[str, Any]]:
    paths = _resolve_alignment_paths(config, alignment_file, alignment_glob)
    records = []
    for path in paths:
        records.extend(read_jsonl(path))
    if model_name is not None:
        records = [record for record in records if record.get("checkpoint") == model_name]
    if teacher_context is not None:
        records = [record for record in records if record.get("teacher_context") == teacher_context]
    return records


def output_summary_paths(
    config: dict[str, Any],
    *,
    output_jsonl: str | None = None,
    output_csv: str | None = None,
) -> tuple[Path, Path]:
    jsonl_path = Path(output_jsonl) if output_jsonl else output_path(config, "summaries", "alignment_summary.jsonl")
    csv_path = Path(output_csv) if output_csv else output_path(config, "summaries", "alignment_summary.csv")
    return jsonl_path, csv_path


def write_summary_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _ordered_fields(summaries)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)


def _resolve_alignment_paths(
    config: dict[str, Any],
    alignment_file: str | None,
    alignment_glob: str | None,
) -> list[Path]:
    if alignment_file is not None and alignment_glob is not None:
        raise ValueError("Use either --alignment-file or --alignment-glob, not both")
    if alignment_file is not None:
        paths = [Path(alignment_file)]
    elif alignment_glob is not None:
        paths = [Path(path) for path in sorted(glob.glob(alignment_glob))]
    else:
        default_path = output_path(config, "alignments", "gradient_alignments.jsonl")
        if default_path.exists():
            paths = [default_path]
        else:
            shard_pattern = output_path(config, "alignments", "gradient_alignments.shard*-of-*.jsonl")
            paths = sorted(shard_pattern.parent.glob(shard_pattern.name))
    if not paths:
        raise FileNotFoundError("No gradient alignment files found")
    return paths


def _ordered_fields(summaries: list[dict[str, Any]]) -> list[str]:
    fields = list(SUMMARY_FIELD_ORDER)
    for summary in summaries:
        for key in summary:
            if key not in fields:
                fields.append(key)
    return fields


def _mean(values) -> float | None:
    valid = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            valid.append(value)
    return sum(valid) / len(valid) if valid else None


def _bool_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return None


def _parse_group_keys(value: str | None) -> tuple[str, ...]:
    if value is None or value == "default":
        return DEFAULT_GROUP_KEYS
    if value == "checkpoint_teacher":
        return ("checkpoint", "teacher_context", "distillation_objective")
    if value == "checkpoint":
        return ("checkpoint", "distillation_objective")
    keys = tuple(part.strip() for part in value.split(",") if part.strip())
    if not keys:
        raise ValueError("At least one group key is required")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--alignment-file", help="Read one gradient alignment JSONL file.")
    parser.add_argument("--alignment-glob", help="Read gradient alignment JSONL files matching this glob.")
    parser.add_argument("--model-name", help="Only aggregate one model/checkpoint.")
    parser.add_argument("--teacher-context", help="Only aggregate one teacher context.")
    parser.add_argument(
        "--group-by",
        default="default",
        help="default, checkpoint_teacher, checkpoint, or comma-separated alignment record fields.",
    )
    parser.add_argument("--output-jsonl", help="Override JSONL summary output path.")
    parser.add_argument("--output-csv", help="Override CSV summary output path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    jsonl_path, csv_path = output_summary_paths(config, output_jsonl=args.output_jsonl, output_csv=args.output_csv)
    if (jsonl_path.exists() or csv_path.exists()) and not args.overwrite:
        print(f"Skipping existing summary output: {jsonl_path} / {csv_path}")
        return

    records = load_alignment_records(
        config,
        alignment_file=args.alignment_file,
        alignment_glob=args.alignment_glob,
        model_name=args.model_name,
        teacher_context=args.teacher_context,
    )
    summaries = aggregate_records(records, group_keys=_parse_group_keys(args.group_by))
    write_jsonl(jsonl_path, summaries)
    write_summary_csv(csv_path, summaries)
    print(f"Wrote {len(summaries)} summary rows to {jsonl_path} and {csv_path}")


if __name__ == "__main__":
    main()
