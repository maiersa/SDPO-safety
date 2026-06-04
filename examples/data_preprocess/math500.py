#!/usr/bin/env python3
"""Preprocess HuggingFaceH4/MATH-500 to the local benchmark parquet schema."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import datasets

from verl.utils.hdfs_io import copy, makedirs


DATA_SOURCE = "HuggingFaceH4/MATH-500"
INSTRUCTION = r"Let's think step by step and output the final answer within \boxed{}."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--local_save_dir", default="datasets/math500")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None)
    args = parser.parse_args()

    local_save_dir = args.local_dir or args.local_save_dir
    out_dir = Path(os.path.expanduser(local_save_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = args.local_dataset_path or DATA_SOURCE
    dataset = datasets.load_dataset(dataset_id)
    test_dataset = dataset["test"]

    def process_fn(example: dict, idx: int) -> dict:
        question = str(example["problem"]).strip()
        answer = str(example["answer"]).strip()
        return {
            "level": example.get("level"),
            "type": example.get("subject"),
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": f"{question} {INSTRUCTION}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "test",
                "index": idx,
                "unique_id": example.get("unique_id"),
                "subject": example.get("subject"),
                "level": example.get("level"),
                "solution": example.get("solution"),
            },
        }

    test_dataset = test_dataset.map(function=process_fn, with_indices=True)
    test_dataset.to_parquet(str(out_dir / "test.parquet"))

    with (out_dir / "test_example.json").open("w", encoding="utf-8") as f:
        json.dump(test_dataset[0], f, indent=2, ensure_ascii=False)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=str(out_dir), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
