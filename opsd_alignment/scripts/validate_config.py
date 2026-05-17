"""Validate an OPSD alignment config before launching GPU jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path
from opsd_alignment.src.models import resolve_model_path


REQUIRED_TOP_LEVEL_KEYS = ("models", "teacher_contexts", "generation", "diagnostic", "paths")


def validate_config(config: dict[str, Any], *, check_model_paths: bool = True) -> list[str]:
    issues: list[str] = []
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in config:
            issues.append(f"Missing top-level key: {key}")

    if "paths" in config:
        questions = Path(config["paths"].get("questions", ""))
        if not questions.exists():
            issues.append(f"Question file does not exist: {questions}")
        output_dir = output_path(config)
        if output_dir.exists() and not output_dir.is_dir():
            issues.append(f"Output path exists but is not a directory: {output_dir}")
        existing_parent = output_dir.parent
        while not existing_parent.exists() and existing_parent != existing_parent.parent:
            existing_parent = existing_parent.parent
        if not existing_parent.exists():
            issues.append(f"No existing ancestor directory for output path: {output_dir}")

    models = config.get("models") or []
    if not isinstance(models, list) or not models:
        issues.append("models must be a non-empty list")
    else:
        names = set()
        for index, model in enumerate(models):
            name = model.get("name") if isinstance(model, dict) else None
            if not name:
                issues.append(f"models[{index}] is missing name")
                continue
            if name in names:
                issues.append(f"Duplicate model name: {name}")
            names.add(name)
            if check_model_paths:
                try:
                    resolve_model_path(model)
                except Exception as exc:  # noqa: BLE001 - this is a validator, report any path issue.
                    issues.append(str(exc))

    contexts = config.get("teacher_contexts") or []
    valid_contexts = {"control", "answer_only", "full_solution"}
    for context in contexts:
        if context not in valid_contexts:
            issues.append(f"Unknown teacher context: {context}")

    diagnostic = config.get("diagnostic") or {}
    objective = diagnostic.get("distillation_objective", "forward_kl")
    if objective not in {"forward_kl", "reverse_kl", "jsd"}:
        issues.append(f"Unsupported distillation_objective: {objective}")
    if objective == "jsd":
        alpha = float(diagnostic.get("jsd_alpha", 0.5))
        if not 0.0 < alpha < 1.0:
            issues.append("jsd_alpha must be in (0, 1)")

    for int_key in ("nodes_per_rollout", "top_k_student", "top_k_teacher", "forced_rollouts_per_candidate"):
        if int_key in diagnostic and int(diagnostic[int_key]) <= 0:
            issues.append(f"diagnostic.{int_key} must be positive")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--skip-model-paths", action="store_true", help="Do not require checkpoint paths to resolve/exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    issues = validate_config(config, check_model_paths=not args.skip_model_paths)
    if issues:
        print("Config validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print(f"Config OK: {args.config}")


if __name__ == "__main__":
    main()
