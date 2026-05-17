"""Generate and grade student rollouts for the OPSD alignment diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path
from opsd_alignment.src.grading import grade_answer
from opsd_alignment.src.models import build_model_runner
from opsd_alignment.src.prompts import build_student_prompt
from opsd_alignment.src.storage import read_jsonl, write_jsonl


def generate_rollout_records(
    config: dict[str, Any],
    *,
    model_name: str | None = None,
    question_limit: int | None = None,
    device: str = "auto",
    torch_dtype: str = "auto",
) -> list[dict[str, Any]]:
    questions = list(read_jsonl(config["paths"]["questions"]))
    diagnostic = config.get("diagnostic", {})
    if question_limit is None:
        question_limit = diagnostic.get("questions_limit")
    if question_limit is not None:
        questions = questions[: int(question_limit)]

    generation_cfg = config["generation"]
    base_seed = int(config.get("seed", 0))
    records: list[dict[str, Any]] = []

    models = config["models"]
    if model_name is not None:
        models = [model for model in models if model["name"] == model_name]
        if not models:
            raise ValueError(f"No model named {model_name!r} in config")

    for model_index, model_cfg in enumerate(models):
        runner = build_model_runner(model_cfg, device=device, torch_dtype=torch_dtype)
        checkpoint = model_cfg["name"]
        model_max_new_tokens = int(model_cfg.get("max_new_tokens", generation_cfg.get("max_new_tokens", 512)))
        num_rollouts = int(generation_cfg.get("student_rollouts_per_question", 1))

        for question_index, question in enumerate(questions):
            prompt = build_student_prompt(question["question"])
            prompt_token_ids = runner.encode(prompt, add_special_tokens=True)
            for rollout_index in range(num_rollouts):
                seed = base_seed + model_index * 1_000_000 + question_index * 1_000 + rollout_index
                generation = runner.generate(
                    prompt,
                    seed=seed,
                    max_new_tokens=model_max_new_tokens,
                    temperature=float(generation_cfg.get("temperature", 0.7)),
                    top_p=float(generation_cfg.get("top_p", 0.95)),
                )
                grade = grade_answer(generation.text, str(question["answer"]), source=str(question.get("source", "gsm8k")))
                rollout_id = f"{checkpoint}:{question['id']}:{rollout_index}"
                records.append(
                    {
                        "question_id": question["id"],
                        "source": question.get("source"),
                        "difficulty": question.get("difficulty"),
                        "checkpoint": checkpoint,
                        "model_path": model_cfg.get("path"),
                        "rollout_id": rollout_id,
                        "rollout_index": rollout_index,
                        "seed": seed,
                        "prompt": prompt,
                        "prompt_token_ids": prompt_token_ids,
                        "generated_text": generation.text,
                        "generated_token_ids": generation.token_ids,
                        "parsed_answer": grade.raw_answer,
                        "normalized_answer": grade.normalized_answer,
                        "normalized_ground_truth": grade.normalized_ground_truth,
                        "is_correct": grade.is_correct,
                        "invalid_parse": grade.invalid_parse,
                        "num_generated_tokens": len(generation.token_ids),
                        "generation_config": {
                            "temperature": float(generation_cfg.get("temperature", 0.7)),
                            "top_p": float(generation_cfg.get("top_p", 0.95)),
                            "max_new_tokens": model_max_new_tokens,
                        },
                        "question": question["question"],
                        "answer": question["answer"],
                        "reference_solution": question.get("reference_solution"),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--model-name", help="Only run one model from the config.")
    parser.add_argument("--question-limit", type=int, help="Override diagnostic.questions_limit.")
    parser.add_argument("--device", default="auto", help="HF device, e.g. auto, cuda:0, cpu.")
    parser.add_argument("--torch-dtype", default="auto", help="auto, bf16, fp16, or fp32.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = output_path(config, "rollouts", "student_rollouts.jsonl")
    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {out_path}")
        return

    records = generate_rollout_records(
        config,
        model_name=args.model_name,
        question_limit=args.question_limit,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    write_jsonl(out_path, records)
    print(f"Wrote {len(records)} student rollouts to {out_path}")


if __name__ == "__main__":
    main()
