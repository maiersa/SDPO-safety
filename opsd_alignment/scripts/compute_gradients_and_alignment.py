"""Compute ideal/distillation gradients and cosine alignment from cached JSONL artifacts."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path
from opsd_alignment.src.gradients import alignment, distillation_gradient, ideal_gradient
from opsd_alignment.src.storage import read_jsonl, write_jsonl


def compute_alignment_records(
    config: dict[str, Any],
    *,
    distribution_file: str | None = None,
    distribution_glob: str | None = None,
    branch_file: str | None = None,
    branch_glob: str | None = None,
    model_name: str | None = None,
    teacher_context: str | None = None,
    objective: str | None = None,
    jsd_alpha: float | None = None,
    skip_incomplete: bool = False,
) -> list[dict[str, Any]]:
    diagnostic = config["diagnostic"]
    objective = objective or diagnostic.get("distillation_objective", "forward_kl")
    jsd_alpha = float(jsd_alpha if jsd_alpha is not None else diagnostic.get("jsd_alpha", 0.5))
    min_gradient_norm = float(diagnostic.get("min_gradient_norm", 1e-8))

    distribution_records = _load_distribution_records(config, distribution_file, distribution_glob)
    if model_name is not None:
        distribution_records = [record for record in distribution_records if record["checkpoint"] == model_name]
    if teacher_context is not None:
        distribution_records = [record for record in distribution_records if record.get("teacher_context") == teacher_context]

    branch_records = _load_branch_records(config, branch_file, branch_glob)
    branch_success = _branch_success_by_key(branch_records)

    alignment_records = []
    for record in distribution_records:
        p_success = []
        missing = []
        for candidate_token_id in record["candidate_token_ids"]:
            key = _branch_key(record["checkpoint"], record["node_id"], int(candidate_token_id))
            if key not in branch_success:
                missing.append(int(candidate_token_id))
            else:
                p_success.append(branch_success[key]["p_success"])

        if missing:
            if skip_incomplete:
                continue
            raise KeyError(
                f"Missing branch success for checkpoint={record['checkpoint']!r}, node_id={record['node_id']!r}, "
                f"candidate_token_ids={missing}"
            )

        alignment_records.append(
            compute_alignment_record(
                record,
                p_success=p_success,
                branch_success=branch_success,
                objective=objective,
                jsd_alpha=jsd_alpha,
                min_gradient_norm=min_gradient_norm,
            )
        )
    return alignment_records


def compute_alignment_record(
    record: dict[str, Any],
    *,
    p_success: list[float],
    branch_success: dict[tuple[str, str, int], dict[str, Any]],
    objective: str,
    jsd_alpha: float,
    min_gradient_norm: float,
) -> dict[str, Any]:
    p_student = record["p_student"]
    p_teacher = record["p_teacher"]
    g_ideal = ideal_gradient(p_student, p_success)
    g_opsd = distillation_gradient(p_student, p_teacher, objective=objective, jsd_alpha=jsd_alpha)
    candidate_branch_records = [
        branch_success[_branch_key(record["checkpoint"], record["node_id"], int(candidate_token_id))]
        for candidate_token_id in record["candidate_token_ids"]
    ]
    baseline_success = sum(float(p) * float(success) for p, success in zip(p_student, p_success, strict=True))
    align = alignment(g_ideal, g_opsd, min_gradient_norm=min_gradient_norm)

    return {
        "question_id": record["question_id"],
        "source": record.get("source"),
        "difficulty": record.get("difficulty"),
        "checkpoint": record["checkpoint"],
        "rollout_id": record["rollout_id"],
        "node_id": record["node_id"],
        "token_position": record["token_position"],
        "teacher_context": record["teacher_context"],
        "selection_reason": record.get("selection_reason"),
        "student_rollout_correct": record.get("student_rollout_correct"),
        "student_entropy": record.get("student_entropy"),
        "student_teacher_kl": record.get("student_teacher_kl"),
        "selection_gkd_magnitude": record.get("selection_gkd_magnitude"),
        "selection_policy": record.get("selection_policy"),
        "candidate_token_ids": record["candidate_token_ids"],
        "candidate_tokens": _merge_candidate_success(record.get("candidate_tokens", []), p_success, candidate_branch_records),
        "p_student": p_student,
        "p_teacher": p_teacher,
        "p_success": p_success,
        "baseline_success": baseline_success,
        "g_ideal": g_ideal,
        "g_opsd": g_opsd,
        "alignment": align,
        "alignment_is_valid": align is not None,
        "distillation_objective": objective,
        "jsd_alpha": jsd_alpha if objective == "jsd" else None,
        "min_gradient_norm": min_gradient_norm,
        "num_candidates": len(record["candidate_token_ids"]),
        "mean_branch_success": sum(p_success) / len(p_success) if p_success else None,
        "num_forced_rollouts_per_candidate": [branch.get("num_forced_rollouts") for branch in candidate_branch_records],
        "prefix_text": record.get("prefix_text"),
        "question": record.get("question"),
        "answer": record.get("answer"),
    }


def output_alignment_path(config: dict[str, Any], *, output_file: str | None = None) -> Path:
    if output_file is not None:
        return Path(output_file)
    return output_path(config, "alignments", "gradient_alignments.jsonl")


def _load_distribution_records(
    config: dict[str, Any],
    distribution_file: str | None,
    distribution_glob: str | None,
) -> list[dict[str, Any]]:
    paths = _resolve_paths(
        config=config,
        default_parts=("distributions", "teacher_student_distributions.jsonl"),
        shard_parts=("distributions", "teacher_student_distributions.shard*-of-*.jsonl"),
        explicit_file=distribution_file,
        explicit_glob=distribution_glob,
        missing_message="No teacher/student distribution files found",
    )
    return _read_many_jsonl(paths)


def _load_branch_records(
    config: dict[str, Any],
    branch_file: str | None,
    branch_glob: str | None,
) -> list[dict[str, Any]]:
    paths = _resolve_paths(
        config=config,
        default_parts=("branches", "branch_success.jsonl"),
        shard_parts=("branches", "branch_success.shard*-of-*.jsonl"),
        explicit_file=branch_file,
        explicit_glob=branch_glob,
        missing_message="No branch success files found",
    )
    return _read_many_jsonl(paths)


def _resolve_paths(
    *,
    config: dict[str, Any],
    default_parts: tuple[str, str],
    shard_parts: tuple[str, str],
    explicit_file: str | None,
    explicit_glob: str | None,
    missing_message: str,
) -> list[Path]:
    if explicit_file is not None and explicit_glob is not None:
        raise ValueError("Use either an explicit file or glob, not both")
    if explicit_file is not None:
        paths = [Path(explicit_file)]
    elif explicit_glob is not None:
        paths = [Path(path) for path in sorted(glob.glob(explicit_glob))]
    else:
        default_path = output_path(config, *default_parts)
        if default_path.exists():
            paths = [default_path]
        else:
            shard_pattern = output_path(config, *shard_parts)
            paths = sorted(shard_pattern.parent.glob(shard_pattern.name))
    if not paths:
        raise FileNotFoundError(missing_message)
    return paths


def _read_many_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records


def _branch_success_by_key(branch_records: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    by_key = {}
    for record in branch_records:
        key = _branch_key(record["checkpoint"], record["node_id"], int(record["candidate_token_id"]))
        if key in by_key:
            raise ValueError(f"Duplicate branch success record for key={key}")
        by_key[key] = record
    return by_key


def _branch_key(checkpoint: str, node_id: str, candidate_token_id: int) -> tuple[str, str, int]:
    return checkpoint, node_id, int(candidate_token_id)


def _merge_candidate_success(
    candidate_tokens: list[dict[str, Any]],
    p_success: list[float],
    branch_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = []
    for index, branch in enumerate(branch_records):
        candidate = dict(candidate_tokens[index]) if index < len(candidate_tokens) else {}
        candidate.update(
            {
                "p_success": p_success[index],
                "num_correct_continuations": branch.get("num_correct_continuations"),
                "num_forced_rollouts": branch.get("num_forced_rollouts"),
            }
        )
        merged.append(candidate)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--distribution-file", help="Read one teacher/student distribution JSONL file.")
    parser.add_argument("--distribution-glob", help="Read distribution JSONL files matching this glob.")
    parser.add_argument("--branch-file", help="Read one branch success JSONL file.")
    parser.add_argument("--branch-glob", help="Read branch success JSONL files matching this glob.")
    parser.add_argument("--model-name", help="Only compute alignments for one model from the config.")
    parser.add_argument("--teacher-context", help="Only compute alignments for one teacher context.")
    parser.add_argument("--objective", choices=["forward_kl", "reverse_kl", "jsd"], help="Override diagnostic.distillation_objective.")
    parser.add_argument("--jsd-alpha", type=float, help="Override diagnostic.jsd_alpha for JSD.")
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip records with missing branch-success candidates.")
    parser.add_argument("--output-file", help="Override output path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = output_alignment_path(config, output_file=args.output_file)
    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {out_path}")
        return

    records = compute_alignment_records(
        config,
        distribution_file=args.distribution_file,
        distribution_glob=args.distribution_glob,
        branch_file=args.branch_file,
        branch_glob=args.branch_glob,
        model_name=args.model_name,
        teacher_context=args.teacher_context,
        objective=args.objective,
        jsd_alpha=args.jsd_alpha,
        skip_incomplete=args.skip_incomplete,
    )
    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} gradient alignment records to {out_path}")


if __name__ == "__main__":
    main()
