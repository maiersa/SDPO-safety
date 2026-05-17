"""Merge sharded JSONL artifacts from OPSD alignment cluster runs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

from opsd_alignment.scripts.common import add_config_arg, load_config, output_path


ARTIFACT_PRESETS = {
    "distributions": (
        ("distributions", "teacher_student_distributions.shard*-of-*.jsonl"),
        ("distributions", "teacher_student_distributions.jsonl"),
    ),
    "branches": (
        ("branches", "branch_success.shard*-of-*.jsonl"),
        ("branches", "branch_success.jsonl"),
    ),
    "alignments": (
        ("alignments", "gradient_alignments.shard*-of-*.jsonl"),
        ("alignments", "gradient_alignments.jsonl"),
    ),
    "summaries": (
        ("summaries", "alignment_summary.shard*-of-*.jsonl"),
        ("summaries", "alignment_summary.jsonl"),
    ),
}


def merge_jsonl_files(input_paths: Iterable[str | Path], output_path_: str | Path, *, overwrite: bool = False) -> int:
    input_paths = [Path(path) for path in input_paths]
    if not input_paths:
        raise FileNotFoundError("No input JSONL files matched")

    output_path_ = Path(output_path_)
    if output_path_.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path_}")

    output_path_.parent.mkdir(parents=True, exist_ok=True)
    num_records = 0
    with output_path_.open("w", encoding="utf-8") as output_handle:
        for input_path in input_paths:
            with input_path.open("r", encoding="utf-8") as input_handle:
                for line in input_handle:
                    if not line.strip():
                        continue
                    output_handle.write(line if line.endswith(chr(10)) else line + chr(10))
                    num_records += 1
    return num_records


def resolve_merge_paths(
    config: dict,
    *,
    artifact: str | None = None,
    input_glob: str | None = None,
    output_file: str | None = None,
) -> tuple[list[Path], Path]:
    if artifact is not None and input_glob is not None:
        raise ValueError("Use either --artifact or --input-glob, not both")
    if artifact is None and input_glob is None:
        raise ValueError("Provide --artifact or --input-glob")

    if artifact is not None:
        if artifact not in ARTIFACT_PRESETS:
            raise ValueError(f"Unknown artifact {artifact!r}; expected one of {sorted(ARTIFACT_PRESETS)}")
        input_parts, output_parts = ARTIFACT_PRESETS[artifact]
        pattern = output_path(config, *input_parts)
        input_paths = sorted(pattern.parent.glob(pattern.name))
        default_output = output_path(config, *output_parts)
    else:
        input_paths = [Path(path) for path in sorted(glob.glob(str(input_glob)))]
        default_output = Path(output_file) if output_file else None

    if output_file is not None:
        resolved_output = Path(output_file)
    elif default_output is not None:
        resolved_output = default_output
    else:
        raise ValueError("--output-file is required when using --input-glob")

    return input_paths, resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arg(parser)
    parser.add_argument("--artifact", choices=sorted(ARTIFACT_PRESETS), help="Known OPSD artifact type to merge.")
    parser.add_argument("--input-glob", help="Custom shard glob to merge.")
    parser.add_argument("--output-file", help="Override merged JSONL output path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    input_paths, out_path = resolve_merge_paths(
        config,
        artifact=args.artifact,
        input_glob=args.input_glob,
        output_file=args.output_file,
    )
    num_records = merge_jsonl_files(input_paths, out_path, overwrite=args.overwrite)
    print(f"Merged {len(input_paths)} files / {num_records} records into {out_path}")


if __name__ == "__main__":
    main()
