# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess OpenMathInstruct to verl parquet format.

The output mirrors the GSM8K/MATH preprocessing scripts:
  - train.parquet
  - test.parquet

Default behavior targets NVIDIA's OpenMathInstruct-1 release because it
already provides train/validation splits. If a dataset only has a train split,
this script can deterministically carve out a validation split.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


DEFAULT_DATASET = "nvidia/OpenMathInstruct-1"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_VAL_SPLIT = "validation"
DEFAULT_VAL_SIZE = 5000


def _get_field(example, candidates):
    for key in candidates:
        if key in example and example[key] is not None:
            return example[key]
    return None


def _build_splits(raw_dataset, train_split, val_split, val_size, seed):
    if train_split not in raw_dataset:
        raise ValueError(f"Missing train split '{train_split}' in dataset. Available: {list(raw_dataset.keys())}")

    train_dataset = raw_dataset[train_split]

    if val_split in raw_dataset:
        val_dataset = raw_dataset[val_split]
    elif "test" in raw_dataset:
        val_dataset = raw_dataset["test"]
    else:
        if val_size <= 0 or val_size >= len(train_dataset):
            raise ValueError(
                f"Cannot carve validation split of size {val_size} from train split with {len(train_dataset)} rows."
            )
        split_dataset = train_dataset.train_test_split(test_size=val_size, seed=seed, shuffle=True)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]

    return train_dataset, val_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated alias for --local_save_dir.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="Load dataset from a local path instead of HF hub.")
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET, help="HF dataset repo id.")
    parser.add_argument("--train_split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--val_split", default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--filter_incorrect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows marked correct when an is_correct field exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/openmathinstruct",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
        data_source = args.local_dataset_path
    else:
        dataset = datasets.load_dataset(args.dataset_name)
        data_source = args.dataset_name

    train_dataset, test_dataset = _build_splits(
        raw_dataset=dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        val_size=args.val_size,
        seed=args.seed,
    )

    if args.filter_incorrect:
        if "is_correct" in train_dataset.column_names:
            train_dataset = train_dataset.filter(lambda example: bool(example["is_correct"]))
        if "is_correct" in test_dataset.column_names:
            test_dataset = test_dataset.filter(lambda example: bool(example["is_correct"]))

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = _get_field(example, ["problem", "question"])
            if question_raw is None:
                raise ValueError(f"Could not find a problem/question field in example keys: {list(example.keys())}")

            answer_raw = _get_field(example, ["generated_solution", "solution", "predicted_answer"])
            if answer_raw is None:
                raise ValueError(f"Could not find a solution field in example keys: {list(example.keys())}")

            ground_truth = _get_field(example, ["expected_answer", "answer"])
            if ground_truth is None:
                raise ValueError(f"Could not find a ground-truth answer field in example keys: {list(example.keys())}")

            prompt = question_raw + " " + instruction_following
            source_name = _get_field(example, ["dataset", "problem_source", "source"]) or "openmathinstruct"

            data = {
                # Route OpenMath through the math scorer in verl.
                "data_source": "math",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "source_dataset": source_name,
                    "raw_data_source": data_source,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    with open(os.path.join(local_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "dataset_name": args.dataset_name,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "val_size": args.val_size,
                "seed": args.seed,
                "filter_incorrect": args.filter_incorrect,
                "train_rows": len(train_dataset),
                "test_rows": len(test_dataset),
            },
            f,
            indent=2,
        )

    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2)
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(test_dataset[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
