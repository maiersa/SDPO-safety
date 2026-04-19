#!/usr/bin/env python
"""Download selected OLMo checkpoints or revisions from Hugging Face.

Example:
  python scripts/download_stage1_olmo_checkpoints.py \
    --model-repo allenai/Olmo-3-1125-32B \
    --output-root /dlabscratch1/$USER/checkpoints/olmo-stage1

The script snapshots each requested revision into:
  <output_root>/<revision>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_STAGE1_REVISIONS = [
    "stage1-step1000",
    "stage1-step4000",
    "stage1-step16000",
    "stage1-step64000",
    "stage1-step128000",
    "stage1-step256000",
    "stage1-step656000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-repo",
        required=True,
        help="Hugging Face repo id, e.g. allenai/Olmo-3-1125-32B",
    )
    parser.add_argument(
        "--output-root",
        default=f"/dlabscratch1/{os.environ.get('USER', 'user')}/checkpoints/olmo-stage1",
        help="Directory where each revision will be materialized as its own HF model folder.",
    )
    parser.add_argument(
        "--default-stage1-revisions",
        action="store_true",
        help="Use the built-in Stage 1 sweep revisions when no --revision values are provided.",
    )
    parser.add_argument(
        "--revision",
        action="append",
        dest="revisions",
        default=None,
        help="Specific revision to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--destination-name",
        default=None,
        help=(
            "Optional subdirectory name to use instead of the raw revision name. "
            "Only valid when downloading exactly one revision."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target directory already looks populated.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HF token. If omitted, huggingface_hub uses the local login/token environment.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        default=None,
        help="Optional allow pattern forwarded to snapshot_download. Pass multiple times if needed.",
    )
    return parser.parse_args()


def looks_downloaded(path: Path) -> bool:
    required = ["config.json"]
    return path.is_dir() and all((path / name).exists() for name in required)


def main() -> None:
    args = parse_args()
    if args.revisions:
        revisions = args.revisions
    elif args.default_stage1_revisions:
        revisions = DEFAULT_STAGE1_REVISIONS
    else:
        revisions = ["main"]

    if args.destination_name and len(revisions) != 1:
        raise SystemExit("--destination-name can only be used with exactly one revision.")

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Model repo:   {args.model_repo}")
    print(f"Output root:  {output_root}")
    print(f"Revisions:    {', '.join(revisions)}")
    if args.cache_dir:
        print(f"Cache dir:    {args.cache_dir}")
    if args.allow_patterns:
        print(f"Allow only:   {args.allow_patterns}")
    print()

    for revision in revisions:
        target_name = args.destination_name or revision
        target_dir = output_root / target_name
        if looks_downloaded(target_dir) and not args.force:
            print(f"[{revision}] Found existing model directory at {target_dir}; skipping.")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{revision}] Downloading into {target_dir} ...")
        snapshot_download(
            repo_id=args.model_repo,
            revision=revision,
            local_dir=str(target_dir),
            cache_dir=args.cache_dir,
            token=args.token,
            allow_patterns=args.allow_patterns,
            resume_download=True,
        )
        print(f"[{revision}] Done.")

    print()
    print("Finished downloading checkpoints.")


if __name__ == "__main__":
    main()
