#!/bin/bash

# Prepare BeaverTails safety dataset for GRPO/SDPO training
# Downloads PKU-Alignment/BeaverTails and extracts safety harm categories

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="$PROJECT_ROOT/datasets"
BEAVERTAILS_DIR="$DATASETS_DIR/beavertails_safety"

mkdir -p "$BEAVERTAILS_DIR"

# Cleanup legacy artifact from older script versions.
rm -f "$BEAVERTAILS_DIR/raw.jsonl"

echo "Preparing BeaverTails safety dataset for constitutional alignment..."
echo ""

# Create Python script to download and filter BeaverTails
BEAVERTAILS_DIR="$BEAVERTAILS_DIR" python3 << 'EOF'
import json
import os
import random
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets library not found. Install with: pip install datasets")
    exit(1)

beavertails_dir = Path(os.environ.get("BEAVERTAILS_DIR", "datasets/beavertails_safety"))
beavertails_dir.mkdir(parents=True, exist_ok=True)

print("Downloading BeaverTails dataset from HuggingFace...")
try:
    train_split_name = "330k_train"
    test_split_name = "330k_test"
    ds_train = load_dataset("PKU-Alignment/BeaverTails", split=train_split_name)
    ds_test = load_dataset("PKU-Alignment/BeaverTails", split=test_split_name)
    print(
        f"✓ Loaded BeaverTails splits: "
        f"{train_split_name}={len(ds_train)}, {test_split_name}={len(ds_test)}"
    )
except Exception as e:
    print(f"✗ Failed to load BeaverTails: {e}")
    print("  Fallback: Create placeholder BeaverTails data")
    # Create minimal placeholder
    ds_train = None
    ds_test = None

def first_nonempty(ex, candidates, default=""):
    for key in candidates:
        val = ex.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return default


def normalize_record(example, idx):
    category = first_nonempty(example, ["category", "harm_category", "subcategory"], "unknown")
    prompt = first_nonempty(example, ["prompt", "question", "instruction", "input"])
    answer = first_nonempty(
        example,
        ["response", "answer", "chosen", "answer_0", "safe_response", "output"],
        "I cannot help with that.",
    )

    if not prompt:
        return None

    return {
        "prompt": prompt,
        "idx": idx,
        "dataset": "beavertails_safety",
        "system": None,
        "answer": answer,
        "tests": None,
        "kind": "safety",
        "elo": 1500,
        "description": f"BeaverTails safety category: {category}",
    }


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_target_safety_category(example, safety_categories):
    category = example.get("category")
    if isinstance(category, str):
        lowered = category.lower()
        return any(target in lowered for target in safety_categories)
    if isinstance(category, dict):
        for key, value in category.items():
            if not value:
                continue
            lowered_key = str(key).lower()
            if any(target in lowered_key for target in safety_categories):
                return True
        return False
    if isinstance(category, list):
        for item in category:
            lowered_item = str(item).lower()
            if any(target in lowered_item for target in safety_categories):
                return True
        return False
    return False


def keep_safe_response_if_available(example):
    if "is_safe" in example:
        return bool(example.get("is_safe", False))
    return True


if ds_train is not None and ds_test is not None:
    # Filter for safety-relevant harm categories.
    safety_categories = {
        "violence",
        "incitement",
        "aiding_abetting",
        "aiding_and_abetting",
        "privacy_violation",
    }

    print(f"Filtering for safety categories: {safety_categories}")
    category_column = "category" if "category" in ds_train.column_names else None
    if category_column is None:
        print("⚠ No 'category' column found; using full dataset without category filtering")
        filtered_train = ds_train
        filtered_test = ds_test
    else:
        filtered_train = ds_train.filter(lambda x: has_target_safety_category(x, safety_categories))
        filtered_test = ds_test.filter(lambda x: has_target_safety_category(x, safety_categories))

    if "is_safe" in filtered_train.column_names:
        safe_train = filtered_train.filter(keep_safe_response_if_available)
        if len(safe_train) > 0:
            filtered_train = safe_train
        else:
            print("⚠ 'is_safe' filtering removed all train rows; using category-only train data")
    if "is_safe" in filtered_test.column_names:
        safe_test = filtered_test.filter(keep_safe_response_if_available)
        if len(safe_test) > 0:
            filtered_test = safe_test
        else:
            print("⚠ 'is_safe' filtering removed all test rows; using category-only test data")

    print(
        "✓ Filtered safety-relevant examples: "
        f"train={len(filtered_train)}, test={len(filtered_test)}"
    )

    train_rows = []
    for i, item in enumerate(filtered_train):
        row = normalize_record(item, i)
        if row is not None:
            train_rows.append(row)

    test_rows = []
    for i, item in enumerate(filtered_test):
        row = normalize_record(item, i)
        if row is not None:
            test_rows.append(row)

    if not train_rows or not test_rows:
        print("✗ No usable records found after normalization; falling back to placeholders")
        ds_train = None
        ds_test = None

if ds_train is None or ds_test is None:
    print("Creating placeholder dataset structure...")
    train_rows = []
    for i in range(10):
        train_rows.append(
            {
                "prompt": "Is it legal to access someone's private email?",
                "idx": i,
                "dataset": "beavertails_safety",
                "system": None,
                "answer": "I cannot help with that.",
                "tests": None,
                "kind": "safety",
                "elo": 1500,
                "description": "BeaverTails safety category: privacy_violation",
            }
        )
    test_rows = train_rows[-1:]

# Deterministically shuffle each split.
rng = random.Random(42)
rng.shuffle(train_rows)
rng.shuffle(test_rows)

train_path = beavertails_dir / "train.json"
test_path = beavertails_dir / "test.json"
sample_path = beavertails_dir / "sample_10.json"

write_jsonl(train_path, train_rows)
write_jsonl(test_path, test_rows)

with open(sample_path, "w") as f:
    json.dump(train_rows[:10], f, indent=2)

print(f"✓ Wrote {len(train_rows)} records to {train_path}")
print(f"✓ Wrote {len(test_rows)} records to {test_path}")
print(f"✓ Saved sample (first 10 train rows) to {sample_path}")

print("\n✓ BeaverTails preparation complete")
print(f"  Next step: Run data preprocessing with:")
print(f"    python data/preprocess.py --data_source {beavertails_dir}")

EOF

echo ""
echo "================================================================"
echo "Converting to parquet format (required for training)..."
echo "================================================================"

python data/preprocess.py --data_source "$BEAVERTAILS_DIR"

echo ""
echo "✓ BeaverTails safety dataset ready at: $BEAVERTAILS_DIR"
echo ""
echo "Files:"
ls -lh "$BEAVERTAILS_DIR"/ 2>/dev/null || echo "  (files will appear after preprocessing completes)"
