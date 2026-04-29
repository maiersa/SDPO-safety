#!/usr/bin/env python3
"""Plot pass@k curves/bars from pretraining benchmark summary CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        action="append",
        required=True,
        help="summary.csv or combined_summary.csv from scripts/eval_pretrain_benchmarks.py. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults next to the first CSV.")
    parser.add_argument("--title", default="Pretraining Benchmark Comparison")
    parser.add_argument("--metrics", default=None, help="Comma-separated metrics to plot, e.g. pass@1,pass@8,pass@32.")
    parser.add_argument("--group-by", choices=["checkpoint", "task"], default="checkpoint")
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def metric_columns(rows: list[dict[str, str]], requested: str | None) -> list[str]:
    if requested:
        return [metric.strip() for metric in requested.split(",") if metric.strip()]
    metrics = sorted(
        {key for row in rows for key in row if key.startswith("pass@")},
        key=lambda value: int(value.split("@", 1)[1]),
    )
    if not metrics:
        raise SystemExit("No pass@k columns found in the supplied CSV files.")
    return metrics


def label_for(row: dict[str, str]) -> str:
    label = row.get("checkpoint", "checkpoint")
    mode = row.get("prompt_mode")
    fewshot = row.get("num_fewshot")
    if mode and fewshot not in (None, ""):
        label = f"{label} ({mode}, {fewshot}-shot)"
    return label


def maybe_step(label: str) -> int | None:
    matches = re.findall(r"(?:step_|step)(\d+)", label)
    if not matches:
        return None
    return int(matches[-1])


def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv)
    if not rows:
        raise SystemExit("No rows found in the supplied CSV files.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it in this environment and rerun.") from exc

    metrics = metric_columns(rows, args.metrics)
    output_dir = (args.output_dir or args.summary_csv[0].parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
    axes = axes[0]

    labels = [label_for(row) for row in rows]
    steps = [maybe_step(label) for label in labels]
    use_line = args.group_by == "checkpoint" and all(step is not None for step in steps) and len(set(steps)) > 1

    for axis, metric in zip(axes, metrics, strict=True):
        values = [float(row[metric]) for row in rows]
        if use_line:
            ordered = sorted(zip(steps, values, labels), key=lambda item: item[0])
            x_values = [item[0] for item in ordered]
            y_values = [item[1] for item in ordered]
            axis.plot(x_values, y_values, marker="o", linewidth=2.2)
            for step, value, label in ordered:
                axis.annotate(label.split(" ", 1)[0], (step, value), textcoords="offset points", xytext=(0, 8), ha="center")
            axis.set_xlabel("Checkpoint step")
        else:
            axis.bar(labels, values, color="#3569a8")
            axis.tick_params(axis="x", rotation=25)
            axis.set_xlabel(args.group_by.replace("_", " ").title())
        axis.set_title(metric)
        axis.set_ylabel("Estimated pass rate")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(args.title, fontsize=15)
    fig.tight_layout()
    png_path = output_dir / "pretrain_benchmark_comparison.png"
    pdf_path = output_dir / "pretrain_benchmark_comparison.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[done] wrote {png_path}")
    print(f"[done] wrote {pdf_path}")


if __name__ == "__main__":
    main()
