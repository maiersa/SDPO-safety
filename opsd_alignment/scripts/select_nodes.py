"""Select diagnostic token positions from student rollouts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path
from opsd_alignment.src.candidate_selection import union_topk_candidates
from opsd_alignment.src.gradients import entropy, renormalize_logprobs, student_teacher_kl
from opsd_alignment.src.models import build_model_runner
from opsd_alignment.src.node_selection import NodeScore, select_diagnostic_nodes
from opsd_alignment.src.prompts import build_teacher_prompt
from opsd_alignment.src.storage import read_jsonl, write_jsonl


def select_node_records(
    config: dict[str, Any],
    *,
    teacher_context: str | None = None,
    model_name: str | None = None,
    max_positions_per_rollout: int | None = None,
    device: str = "auto",
    torch_dtype: str = "auto",
) -> list[dict[str, Any]]:
    diagnostic = config["diagnostic"]
    teacher_context = teacher_context or _default_teacher_context(config)
    rollout_path = output_path(config, "rollouts", "student_rollouts.jsonl")
    rollouts = list(read_jsonl(rollout_path))
    if model_name is not None:
        rollouts = [rollout for rollout in rollouts if rollout["checkpoint"] == model_name]

    rollouts_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        rollouts_by_checkpoint[rollout["checkpoint"]].append(rollout)

    model_configs = {model["name"]: model for model in config["models"]}
    selected_records: list[dict[str, Any]] = []

    for checkpoint, checkpoint_rollouts in rollouts_by_checkpoint.items():
        if checkpoint not in model_configs:
            raise ValueError(f"No model config found for checkpoint {checkpoint!r}")
        runner = build_model_runner(model_configs[checkpoint], device=device, torch_dtype=torch_dtype)

        for rollout in checkpoint_rollouts:
            generated_token_ids = list(rollout["generated_token_ids"])
            prompt_token_ids = list(rollout["prompt_token_ids"])
            max_position = len(generated_token_ids)
            if max_positions_per_rollout is not None:
                max_position = min(max_position, int(max_positions_per_rollout))

            scored_nodes: list[NodeScore] = []
            score_metadata: dict[int, dict[str, Any]] = {}
            for token_position in range(max_position):
                student_prefix_token_ids = prompt_token_ids + generated_token_ids[:token_position]
                student_prefix_text = runner.decode(generated_token_ids[:token_position])
                if teacher_context == "control":
                    teacher_prompt = rollout["prompt"] + student_prefix_text
                    teacher_prefix_token_ids = student_prefix_token_ids
                else:
                    teacher_prompt = build_teacher_prompt(
                        teacher_context,
                        question=rollout["question"],
                        answer=str(rollout["answer"]),
                        reference_solution=str(rollout.get("reference_solution") or ""),
                        student_prefix=student_prefix_text,
                        student_prompt=rollout["prompt"],
                    )
                    teacher_prefix_token_ids = runner.encode(teacher_prompt, add_special_tokens=True)

                student_top = runner.topk_next_token_distribution(
                    student_prefix_token_ids,
                    int(diagnostic.get("top_k_student", 5)),
                )
                teacher_top = runner.topk_next_token_distribution(
                    teacher_prefix_token_ids,
                    int(diagnostic.get("top_k_teacher", 5)),
                )
                candidates = union_topk_candidates(
                    student_top.token_ids,
                    teacher_top.token_ids,
                    student_top.logprobs,
                    teacher_top.logprobs,
                )
                candidate_token_ids = [candidate.token_id for candidate in candidates]
                student_logprobs = runner.next_token_logprobs(student_prefix_token_ids, candidate_token_ids)
                teacher_logprobs = runner.next_token_logprobs(teacher_prefix_token_ids, candidate_token_ids)
                p_student = renormalize_logprobs(student_logprobs)
                p_teacher = renormalize_logprobs(teacher_logprobs)
                score = NodeScore(
                    token_position=token_position,
                    student_entropy=entropy(p_student),
                    student_teacher_kl=student_teacher_kl(p_student, p_teacher),
                    special_token=generated_token_ids[token_position] in _special_token_ids(runner),
                    after_final_answer=_after_final_answer_marker(runner.decode(generated_token_ids[:token_position])),
                )
                scored_nodes.append(score)
                score_metadata[token_position] = {
                    "candidate_token_ids": candidate_token_ids,
                    "p_student": p_student,
                    "p_teacher": p_teacher,
                    "student_prefix_token_ids": student_prefix_token_ids,
                    "student_prefix_text": student_prefix_text,
                    "teacher_prefix_token_ids": teacher_prefix_token_ids,
                    "teacher_prompt": teacher_prompt,
                }

            selected = select_diagnostic_nodes(
                scored_nodes,
                nodes_per_rollout=int(diagnostic.get("nodes_per_rollout", 3)),
            )
            for selected_index, (score, reason) in enumerate(selected):
                metadata = score_metadata[score.token_position]
                selected_records.append(
                    {
                        "question_id": rollout["question_id"],
                        "source": rollout.get("source"),
                        "difficulty": rollout.get("difficulty"),
                        "checkpoint": checkpoint,
                        "rollout_id": rollout["rollout_id"],
                        "node_id": f"{rollout['rollout_id']}:node{selected_index}",
                        "token_position": score.token_position,
                        "prefix_token_ids": metadata["student_prefix_token_ids"],
                        "prefix_text": metadata["student_prefix_text"],
                        "selection_reason": reason,
                        "teacher_context_for_selection": teacher_context,
                        "teacher_prompt_for_selection": metadata["teacher_prompt"],
                        "teacher_prefix_token_ids_for_selection": metadata["teacher_prefix_token_ids"],
                        "student_entropy": score.student_entropy,
                        "student_teacher_kl": score.student_teacher_kl,
                        "student_rollout_correct": rollout["is_correct"],
                        "candidate_token_ids_for_selection": metadata["candidate_token_ids"],
                        "p_student_for_selection": metadata["p_student"],
                        "p_teacher_for_selection": metadata["p_teacher"],
                        "question": rollout["question"],
                        "answer": rollout["answer"],
                        "reference_solution": rollout.get("reference_solution"),
                    }
                )
    return selected_records


def _default_teacher_context(config: dict[str, Any]) -> str:
    contexts = config.get("teacher_contexts") or ["full_solution"]
    if "full_solution" in contexts:
        return "full_solution"
    return contexts[0]


def _special_token_ids(runner) -> set[int]:
    tokenizer = getattr(runner, "tokenizer", None)
    if tokenizer is None:
        return set()
    return {token for token in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id] if token is not None}


def _after_final_answer_marker(text: str) -> bool:
    lowered = text.lower()
    return "####" in text or "\\boxed" in text or "final answer" in lowered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--teacher-context", help="Teacher context used for KL-based node selection.")
    parser.add_argument("--model-name", help="Only select nodes for one model from the config.")
    parser.add_argument("--max-positions-per-rollout", type=int, help="Debug cap for scanned generated positions.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = output_path(config, "nodes", "selected_nodes.jsonl")
    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {out_path}")
        return

    records = select_node_records(
        config,
        teacher_context=args.teacher_context,
        model_name=args.model_name,
        max_positions_per_rollout=args.max_positions_per_rollout,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} selected nodes to {out_path}")


if __name__ == "__main__":
    main()
