"""Plot OPSD gradient-alignment results."""

from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path
from opsd_alignment.src.storage import read_jsonl


PLOT_FILENAMES = {
    "mean_by_checkpoint_context": "mean_alignment_by_checkpoint_context.png",
    "alignment_by_correctness": "alignment_by_correctness.png",
    "alignment_vs_kl": "alignment_vs_kl.png",
    "alignment_vs_success_rate": "alignment_vs_student_success_rate.png",
    "alignment_distribution": "alignment_distribution.png",
}


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
    records = [record for record in records if _valid_number(record.get("alignment"))]
    if model_name is not None:
        records = [record for record in records if record.get("checkpoint") == model_name]
    if teacher_context is not None:
        records = [record for record in records if record.get("teacher_context") == teacher_context]
    return records


def generate_plots(config: dict[str, Any], records: list[dict[str, Any]], output_dir: str | Path | None = None) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) if output_dir is not None else output_path(config, "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mean_by_checkpoint_context": output_dir / PLOT_FILENAMES["mean_by_checkpoint_context"],
        "alignment_by_correctness": output_dir / PLOT_FILENAMES["alignment_by_correctness"],
        "alignment_vs_kl": output_dir / PLOT_FILENAMES["alignment_vs_kl"],
        "alignment_vs_success_rate": output_dir / PLOT_FILENAMES["alignment_vs_success_rate"],
        "alignment_distribution": output_dir / PLOT_FILENAMES["alignment_distribution"],
    }

    if not records:
        _write_placeholder_plots(plt, paths, "No valid alignment records.\nCosine is undefined when g_ideal or g_opsd has near-zero norm.")
        (output_dir / "NO_VALID_ALIGNMENTS.txt").write_text(
            "No valid alignment records. This can happen in tiny smoke runs when all forced branches have identical success, "
            "making g_ideal the zero vector. Increase questions/nodes/forced rollouts for a more informative run.\n",
            encoding="utf-8",
        )
        return paths

    checkpoint_order = _checkpoint_order(config, records)
    context_order = _teacher_context_order(config, records)

    _plot_mean_by_checkpoint_context(plt, records, checkpoint_order, context_order, paths["mean_by_checkpoint_context"])
    _plot_alignment_by_correctness(plt, records, checkpoint_order, paths["alignment_by_correctness"])
    _plot_alignment_vs_kl(plt, records, checkpoint_order, paths["alignment_vs_kl"])
    _plot_alignment_vs_success_rate(plt, records, checkpoint_order, paths["alignment_vs_success_rate"])
    _plot_alignment_distribution(plt, records, checkpoint_order, context_order, paths["alignment_distribution"])
    return paths


def _plot_mean_by_checkpoint_context(plt, records, checkpoint_order, context_order, path: Path) -> None:
    grouped = _group_values(records, ("checkpoint", "teacher_context"), "alignment")
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(context_order), 1)
    x_positions = list(range(len(checkpoint_order)))
    for context_index, context in enumerate(context_order):
        means = []
        errors = []
        for checkpoint in checkpoint_order:
            values = grouped.get((checkpoint, context), [])
            mean, stderr = _mean_stderr(values)
            means.append(mean if mean is not None else 0.0)
            errors.append(stderr if stderr is not None else 0.0)
        offsets = [x + (context_index - (len(context_order) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, means, width=width, yerr=errors, capsize=3, label=str(context))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(checkpoint_order, rotation=20, ha="right")
    ax.set_ylabel("Mean cosine alignment")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Mean alignment by checkpoint and teacher context")
    ax.legend(title="Teacher context")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_alignment_by_correctness(plt, records, checkpoint_order, path: Path) -> None:
    grouped = _group_values(records, ("checkpoint", "student_rollout_correct"), "alignment")
    labels = [False, True]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x_positions = list(range(len(checkpoint_order)))
    for label_index, label in enumerate(labels):
        means = []
        errors = []
        for checkpoint in checkpoint_order:
            values = grouped.get((checkpoint, label), [])
            mean, stderr = _mean_stderr(values)
            means.append(mean if mean is not None else 0.0)
            errors.append(stderr if stderr is not None else 0.0)
        offsets = [x + (label_index - 0.5) * width for x in x_positions]
        ax.bar(offsets, means, width=width, yerr=errors, capsize=3, label="correct" if label else "incorrect")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(checkpoint_order, rotation=20, ha="right")
    ax.set_ylabel("Mean cosine alignment")
    ax.set_xlabel("Checkpoint")
    ax.set_title("Alignment by rollout correctness")
    ax.legend(title="Student rollout")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_alignment_vs_kl(plt, records, checkpoint_order, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for checkpoint in checkpoint_order:
        points = [record for record in records if record.get("checkpoint") == checkpoint and _valid_number(record.get("student_teacher_kl"))]
        if not points:
            continue
        ax.scatter(
            [float(record["student_teacher_kl"]) for record in points],
            [float(record["alignment"]) for record in points],
            alpha=0.65,
            s=24,
            label=str(checkpoint),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("KL(student || teacher)")
    ax.set_ylabel("Cosine alignment")
    ax.set_title("Alignment vs teacher-student KL")
    ax.legend(title="Checkpoint", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_alignment_vs_success_rate(plt, records, checkpoint_order, path: Path) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("checkpoint")].append(record)
    fig, ax = plt.subplots(figsize=(7, 5))
    for checkpoint in checkpoint_order:
        group = grouped.get(checkpoint, [])
        if not group:
            continue
        success_rate = _mean([_bool_to_float(record.get("student_rollout_correct")) for record in group])
        mean_alignment = _mean([record.get("alignment") for record in group])
        if success_rate is None or mean_alignment is None:
            continue
        ax.scatter([success_rate], [mean_alignment], s=70)
        ax.annotate(str(checkpoint), (success_rate, mean_alignment), xytext=(5, 4), textcoords="offset points", fontsize="small")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Empirical student rollout success rate")
    ax.set_ylabel("Mean cosine alignment")
    ax.set_title("Alignment vs student success rate")
    ax.set_xlim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_placeholder_plots(plt, paths: dict[str, Path], message: str) -> None:
    for title, path in paths.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_title(title.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def _plot_alignment_distribution(plt, records, checkpoint_order, context_order, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    data = []
    for checkpoint in checkpoint_order:
        for context in context_order:
            values = [float(record["alignment"]) for record in records if record.get("checkpoint") == checkpoint and record.get("teacher_context") == context]
            if values:
                labels.append(f"{checkpoint}\n{context}")
                data.append(values)
    if data:
        ax.violinplot(data, showmeans=True, showextrema=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cosine alignment")
    ax.set_title("Distribution of alignment values")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _resolve_alignment_paths(config: dict[str, Any], alignment_file: str | None, alignment_glob: str | None) -> list[Path]:
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


def _group_values(records: list[dict[str, Any]], keys: tuple[str, ...], value_key: str) -> dict[tuple[Any, ...], list[float]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        value = record.get(value_key)
        if _valid_number(value):
            grouped[tuple(record.get(key) for key in keys)].append(float(value))
    return grouped


def _checkpoint_order(config: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    configured = [model["name"] for model in config.get("models", []) if "name" in model]
    observed = [str(record.get("checkpoint")) for record in records if record.get("checkpoint") is not None]
    return _ordered_unique(configured + observed)


def _teacher_context_order(config: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    configured = [str(context) for context in config.get("teacher_contexts", [])]
    observed = [str(record.get("teacher_context")) for record in records if record.get("teacher_context") is not None]
    return _ordered_unique(configured + observed)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _mean_stderr(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance) / math.sqrt(len(values))


def _mean(values) -> float | None:
    valid = [float(value) for value in values if _valid_number(value)]
    return sum(valid) / len(valid) if valid else None


def _valid_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--alignment-file", help="Read one gradient alignment JSONL file.")
    parser.add_argument("--alignment-glob", help="Read gradient alignment JSONL files matching this glob.")
    parser.add_argument("--model-name", help="Only plot one model/checkpoint.")
    parser.add_argument("--teacher-context", help="Only plot one teacher context.")
    parser.add_argument("--output-dir", help="Override plot output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for workflow symmetry; plots are always regenerated.")
    args = parser.parse_args()

    config = load_config(args.config)
    records = load_alignment_records(
        config,
        alignment_file=args.alignment_file,
        alignment_glob=args.alignment_glob,
        model_name=args.model_name,
        teacher_context=args.teacher_context,
    )
    paths = generate_plots(config, records, output_dir=args.output_dir)
    print("Wrote plots:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
