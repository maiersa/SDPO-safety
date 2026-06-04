"""Compute candidate-set student and teacher distributions at selected nodes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path
from opsd_alignment.src.candidate_selection import CandidateToken, union_topk_candidates
from opsd_alignment.src.gradients import entropy, renormalize_logprobs, student_teacher_kl
from opsd_alignment.src.models import build_model_runner
from opsd_alignment.src.prompts import build_teacher_prompt
from opsd_alignment.src.storage import read_jsonl, write_jsonl


def compute_distribution_records(
    config: dict[str, Any],
    *,
    model_name: str | None = None,
    teacher_context: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    device: str = "auto",
    torch_dtype: str = "auto",
) -> list[dict[str, Any]]:
    _validate_shard_args(shard_index, num_shards)
    diagnostic = config["diagnostic"]
    top_k_student = int(diagnostic.get("top_k_student", 5))
    top_k_teacher = int(diagnostic.get("top_k_teacher", 5))
    distillation_objective = diagnostic.get("distillation_objective", "forward_kl")
    jsd_alpha = float(diagnostic.get("jsd_alpha", 0.5))

    nodes = list(read_jsonl(output_path(config, "nodes", "selected_nodes.jsonl")))
    if model_name is not None:
        nodes = [node for node in nodes if node["checkpoint"] == model_name]

    teacher_contexts = [teacher_context] if teacher_context is not None else list(config.get("teacher_contexts") or [])
    if not teacher_contexts:
        raise ValueError("No teacher contexts configured or provided")

    tasks = [
        {"node": node, "teacher_context": context}
        for node in nodes
        for context in teacher_contexts
    ]
    sharded_tasks = [task for task_index, task in enumerate(tasks) if task_index % num_shards == shard_index]

    tasks_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in sharded_tasks:
        tasks_by_checkpoint[task["node"]["checkpoint"]].append(task)

    model_configs = {model["name"]: model for model in config["models"]}
    records: list[dict[str, Any]] = []

    for checkpoint, checkpoint_tasks in tasks_by_checkpoint.items():
        if checkpoint not in model_configs:
            raise ValueError(f"No model config found for checkpoint {checkpoint!r}")
        runner = build_model_runner(model_configs[checkpoint], device=device, torch_dtype=torch_dtype)
        student_top_cache: dict[str, tuple[list[int], list[float]]] = {}

        for task in checkpoint_tasks:
            node = task["node"]
            context = task["teacher_context"]
            node_id = node["node_id"]
            student_prefix_token_ids = list(node["prefix_token_ids"])
            prefix_text = str(node.get("prefix_text") or "")

            if node_id not in student_top_cache:
                student_top = runner.topk_next_token_distribution(student_prefix_token_ids, top_k_student)
                student_top_cache[node_id] = (student_top.token_ids, student_top.logprobs)
            student_top_token_ids, student_top_logprobs = student_top_cache[node_id]

            if context == "control":
                teacher_prompt = _student_prompt_from_node(node) + prefix_text
                teacher_prefix_token_ids = student_prefix_token_ids
            else:
                teacher_prompt = build_teacher_prompt(
                    context,
                    question=str(node["question"]),
                    answer=str(node["answer"]),
                    reference_solution=str(node.get("reference_solution") or ""),
                    student_prefix=prefix_text,
                    student_prompt=node.get("student_prompt") or _student_prompt_from_node(node),
                )
                teacher_prefix_token_ids = runner.encode(teacher_prompt, add_special_tokens=True)
            teacher_top = runner.topk_next_token_distribution(teacher_prefix_token_ids, top_k_teacher)

            candidates = union_topk_candidates(
                student_top_token_ids,
                teacher_top.token_ids,
                student_top_logprobs,
                teacher_top.logprobs,
            )
            candidate_token_ids = [candidate.token_id for candidate in candidates]
            student_logprobs = runner.next_token_logprobs(student_prefix_token_ids, candidate_token_ids)
            teacher_logprobs = runner.next_token_logprobs(teacher_prefix_token_ids, candidate_token_ids)
            p_student = renormalize_logprobs(student_logprobs)
            p_teacher = renormalize_logprobs(teacher_logprobs)

            candidate_records = _candidate_records(
                runner,
                candidates,
                student_logprobs=student_logprobs,
                teacher_logprobs=teacher_logprobs,
                p_student=p_student,
                p_teacher=p_teacher,
            )
            records.append(
                {
                    "question_id": node["question_id"],
                    "source": node.get("source"),
                    "difficulty": node.get("difficulty"),
                    "checkpoint": checkpoint,
                    "rollout_id": node["rollout_id"],
                    "node_id": node_id,
                    "token_position": node["token_position"],
                    "teacher_context": context,
                    "selection_reason": node.get("selection_reason"),
                    "student_rollout_correct": node.get("student_rollout_correct"),
                    "student_entropy": entropy(p_student),
                    "student_teacher_kl": student_teacher_kl(p_student, p_teacher),
                    "selection_student_entropy": node.get("student_entropy"),
                    "selection_student_teacher_kl": node.get("student_teacher_kl"),
                    "selection_gkd_magnitude": node.get("gkd_magnitude"),
                    "selection_policy": node.get("selection_policy"),
                    "prefix_token_ids": student_prefix_token_ids,
                    "prefix_text": prefix_text,
                    "teacher_prompt": teacher_prompt,
                    "teacher_prefix_token_ids": teacher_prefix_token_ids,
                    "candidate_token_ids": candidate_token_ids,
                    "candidate_tokens": candidate_records,
                    "student_logprobs": student_logprobs,
                    "teacher_logprobs": teacher_logprobs,
                    "p_student": p_student,
                    "p_teacher": p_teacher,
                    "top_k_student": top_k_student,
                    "top_k_teacher": top_k_teacher,
                    "distillation_objective": distillation_objective,
                    "jsd_alpha": jsd_alpha if distillation_objective == "jsd" else None,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "question": node.get("question"),
                    "answer": node.get("answer"),
                    "reference_solution": node.get("reference_solution"),
                }
            )
    return records


def output_distribution_path(config: dict[str, Any], *, shard_index: int = 0, num_shards: int = 1, output_file: str | None = None) -> Path:
    if output_file is not None:
        return Path(output_file)
    if num_shards == 1:
        return output_path(config, "distributions", "teacher_student_distributions.jsonl")
    return output_path(
        config,
        "distributions",
        f"teacher_student_distributions.shard{shard_index:05d}-of-{num_shards:05d}.jsonl",
    )


def _candidate_records(
    runner,
    candidates: list[CandidateToken],
    *,
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    p_student: list[float],
    p_teacher: list[float],
) -> list[dict[str, Any]]:
    records = []
    for index, candidate in enumerate(candidates):
        records.append(
            {
                "token_id": candidate.token_id,
                "token_str": runner.decode([candidate.token_id]),
                "student_logprob": student_logprobs[index],
                "teacher_logprob": teacher_logprobs[index],
                "p_student": p_student[index],
                "p_teacher": p_teacher[index],
                "in_student_topk": candidate.in_student_topk,
                "in_teacher_topk": candidate.in_teacher_topk,
            }
        )
    return records


def _student_prompt_from_node(node: dict[str, Any]) -> str:
    question = str(node.get("question") or "")
    return f"Question:\n{question}\n\nSolve the problem step by step and give the final answer."


def _validate_shard_args(shard_index: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--model-name", help="Only compute distributions for one model from the config.")
    parser.add_argument("--teacher-context", help="Only compute one teacher context from the config.")
    parser.add_argument("--device", default="auto", help="HF device, e.g. auto, cuda:0, cpu.")
    parser.add_argument("--torch-dtype", default="auto", help="auto, bf16, fp16, or fp32.")
    parser.add_argument("--shard-index", type=int, default=0, help="This worker's deterministic shard index.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of deterministic shards.")
    parser.add_argument("--output-file", help="Override output path. Useful for scheduler-provided scratch paths.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = output_distribution_path(
        config,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        output_file=args.output_file,
    )
    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {out_path}")
        return

    records = compute_distribution_records(
        config,
        model_name=args.model_name,
        teacher_context=args.teacher_context,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} teacher/student distribution records to {out_path}")


if __name__ == "__main__":
    main()
