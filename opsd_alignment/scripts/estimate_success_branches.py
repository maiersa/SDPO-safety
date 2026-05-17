"""Estimate downstream student success after forcing candidate next tokens."""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path
from opsd_alignment.src.grading import grade_answer
from opsd_alignment.src.models import build_model_runner
from opsd_alignment.src.storage import read_jsonl, write_jsonl


def estimate_success_records(
    config: dict[str, Any],
    *,
    model_name: str | None = None,
    distribution_file: str | None = None,
    distribution_glob: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    device: str = "auto",
    torch_dtype: str = "auto",
) -> list[dict[str, Any]]:
    _validate_shard_args(shard_index, num_shards)
    diagnostic = config["diagnostic"]
    generation_cfg = config["generation"]
    forced_rollouts_per_candidate = int(diagnostic.get("forced_rollouts_per_candidate", 4))
    base_seed = int(config.get("seed", 0))

    distribution_records = _load_distribution_records(config, distribution_file, distribution_glob)
    if model_name is not None:
        distribution_records = [record for record in distribution_records if record["checkpoint"] == model_name]

    tasks = _unique_candidate_tasks(distribution_records)
    sharded_tasks = [task for task_index, task in enumerate(tasks) if task_index % num_shards == shard_index]

    tasks_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in sharded_tasks:
        tasks_by_checkpoint[task["checkpoint"]].append(task)

    model_configs = {model["name"]: model for model in config["models"]}
    records: list[dict[str, Any]] = []

    for checkpoint, checkpoint_tasks in tasks_by_checkpoint.items():
        if checkpoint not in model_configs:
            raise ValueError(f"No model config found for checkpoint {checkpoint!r}")
        model_cfg = model_configs[checkpoint]
        runner = build_model_runner(model_cfg, device=device, torch_dtype=torch_dtype)
        max_new_tokens = int(
            diagnostic.get(
                "branch_max_new_tokens",
                model_cfg.get("max_new_tokens", generation_cfg.get("max_new_tokens", 512)),
            )
        )
        temperature = float(diagnostic.get("branch_temperature", generation_cfg.get("temperature", 0.7)))
        top_p = float(diagnostic.get("branch_top_p", generation_cfg.get("top_p", 0.95)))

        for task in checkpoint_tasks:
            forced_prefix_token_ids = list(task["prefix_token_ids"]) + [int(task["candidate_token_id"])]
            forced_token_text = runner.decode([int(task["candidate_token_id"])])
            forced_rollouts = []
            correct_count = 0
            for rollout_index in range(forced_rollouts_per_candidate):
                seed = base_seed + 50_000_000 + int(task["task_index"]) * 1_000 + rollout_index
                generation = runner.continue_from_tokens(
                    forced_prefix_token_ids,
                    seed=seed,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                branch_text = forced_token_text + generation.text
                grade = grade_answer(branch_text, str(task["answer"]), source=str(task.get("source") or "gsm8k"))
                correct_count += int(grade.is_correct)
                forced_rollouts.append(
                    {
                        "rollout_index": rollout_index,
                        "seed": seed,
                        "forced_token_text": forced_token_text,
                        "continuation_text": generation.text,
                        "branch_text": branch_text,
                        "continuation_token_ids": generation.token_ids,
                        "parsed_answer": grade.raw_answer,
                        "normalized_answer": grade.normalized_answer,
                        "normalized_ground_truth": grade.normalized_ground_truth,
                        "is_correct": grade.is_correct,
                        "invalid_parse": grade.invalid_parse,
                    }
                )

            records.append(
                {
                    "question_id": task["question_id"],
                    "source": task.get("source"),
                    "difficulty": task.get("difficulty"),
                    "checkpoint": checkpoint,
                    "rollout_id": task["rollout_id"],
                    "node_id": task["node_id"],
                    "token_position": task["token_position"],
                    "candidate_token_id": task["candidate_token_id"],
                    "candidate_token_str": forced_token_text,
                    "teacher_contexts_seen": task["teacher_contexts_seen"],
                    "prefix_token_ids": task["prefix_token_ids"],
                    "forced_prefix_token_ids": forced_prefix_token_ids,
                    "forced_rollouts": forced_rollouts,
                    "num_correct_continuations": correct_count,
                    "num_forced_rollouts": forced_rollouts_per_candidate,
                    "p_success": correct_count / forced_rollouts_per_candidate if forced_rollouts_per_candidate else None,
                    "branch_generation_config": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                    },
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "task_index": task["task_index"],
                    "answer": task.get("answer"),
                }
            )
    return records


def output_branch_path(config: dict[str, Any], *, shard_index: int = 0, num_shards: int = 1, output_file: str | None = None) -> Path:
    if output_file is not None:
        return Path(output_file)
    if num_shards == 1:
        return output_path(config, "branches", "branch_success.jsonl")
    return output_path(config, "branches", f"branch_success.shard{shard_index:05d}-of-{num_shards:05d}.jsonl")


def _load_distribution_records(
    config: dict[str, Any],
    distribution_file: str | None,
    distribution_glob: str | None,
) -> list[dict[str, Any]]:
    if distribution_file is not None and distribution_glob is not None:
        raise ValueError("Use either --distribution-file or --distribution-glob, not both")

    if distribution_file is not None:
        paths = [Path(distribution_file)]
    elif distribution_glob is not None:
        paths = [Path(path) for path in sorted(glob.glob(distribution_glob))]
    else:
        default_path = output_path(config, "distributions", "teacher_student_distributions.jsonl")
        if default_path.exists():
            paths = [default_path]
        else:
            shard_glob = output_path(config, "distributions", "teacher_student_distributions.shard*-of-*.jsonl")
            paths = sorted(shard_glob.parent.glob(shard_glob.name))

    if not paths:
        raise FileNotFoundError("No teacher/student distribution files found")

    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records


def _unique_candidate_tasks(distribution_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in distribution_records:
        for candidate_token_id in record["candidate_token_ids"]:
            key = (record["checkpoint"], record["node_id"], int(candidate_token_id))
            if key not in tasks_by_key:
                tasks_by_key[key] = {
                    "checkpoint": record["checkpoint"],
                    "question_id": record["question_id"],
                    "source": record.get("source"),
                    "difficulty": record.get("difficulty"),
                    "rollout_id": record["rollout_id"],
                    "node_id": record["node_id"],
                    "token_position": record["token_position"],
                    "candidate_token_id": int(candidate_token_id),
                    "teacher_contexts_seen": [],
                    "prefix_token_ids": record["prefix_token_ids"],
                    "answer": record.get("answer"),
                }
            context = record.get("teacher_context")
            if context is not None and context not in tasks_by_key[key]["teacher_contexts_seen"]:
                tasks_by_key[key]["teacher_contexts_seen"].append(context)

    tasks = list(tasks_by_key.values())
    for task_index, task in enumerate(tasks):
        task["task_index"] = task_index
    return tasks


def _validate_shard_args(shard_index: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--model-name", help="Only estimate branches for one model from the config.")
    parser.add_argument("--distribution-file", help="Read one teacher/student distribution JSONL file.")
    parser.add_argument("--distribution-glob", help="Read distribution JSONL files matching this glob.")
    parser.add_argument("--device", default="auto", help="HF device, e.g. auto, cuda:0, cpu.")
    parser.add_argument("--torch-dtype", default="auto", help="auto, bf16, fp16, or fp32.")
    parser.add_argument("--shard-index", type=int, default=0, help="This worker's deterministic shard index.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of deterministic shards.")
    parser.add_argument("--output-file", help="Override output path. Useful for scheduler-provided scratch paths.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = output_branch_path(
        config,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        output_file=args.output_file,
    )
    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {out_path}")
        return

    records = estimate_success_records(
        config,
        model_name=args.model_name,
        distribution_file=args.distribution_file,
        distribution_glob=args.distribution_glob,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} branch success records to {out_path}")


if __name__ == "__main__":
    main()
