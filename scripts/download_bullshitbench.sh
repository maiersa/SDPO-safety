#!/bin/bash
# Download and organize BullshitBench v2 dataset with salvaged prompts

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="$PROJECT_ROOT/datasets"
BULLSHIT_DIR="$DATASETS_DIR/bullshitbench"
REPO_URL="https://github.com/petergpt/bullshit-benchmark.git"
REPO_FOLDER="bullshit-benchmark"

mkdir -p "$BULLSHIT_DIR"

echo "Downloading BullshitBench v2 from GitHub..."

# Clone the BullshitBench repository
cd "$BULLSHIT_DIR"

if [ ! -d "$REPO_FOLDER" ]; then
    git clone "$REPO_URL" "$REPO_FOLDER"
    echo "✓ Cloned BullshitBench repository"
else
    echo "✓ BullshitBench repository already present, pulling latest"
    git -C "$REPO_FOLDER" pull --ff-only || true
fi

# Navigate to the repo
cd "$REPO_FOLDER"

# The repository evolved; support old and new file layouts.
if [ -f "questions.v2.json" ]; then
    cp questions.v2.json "$BULLSHIT_DIR/original_v2.json"
    echo "✓ Copied questions.v2.json -> original_v2.json"
elif [ -f "questions.json" ]; then
    cp questions.json "$BULLSHIT_DIR/original_v2.json"
    echo "✓ Copied questions.json -> original_v2.json"
elif [ -f "benchmark_v2.json" ]; then
    cp benchmark_v2.json "$BULLSHIT_DIR/original_v2.json"
    echo "✓ Copied benchmark_v2.json -> original_v2.json"
elif [ -f "data/benchmark_v2.json" ]; then
    cp data/benchmark_v2.json "$BULLSHIT_DIR/original_v2.json"
    echo "✓ Copied data/benchmark_v2.json -> original_v2.json"
else
    echo "Warning: Could not find a recognized v2 question file. Searching..."
    find . -name "*v2*.json" -o -name "*benchmark*.json" | head -5
fi

cd "$BULLSHIT_DIR"

# Create Python script to process the dataset
python3 << 'EOF'
import json
import os
from pathlib import Path

bullshit_dir = Path.cwd()

# Load original dataset
if (bullshit_dir / "original_v2.json").exists():
    with open("original_v2.json") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Common shapes: {"questions": [...]} or BullshitBench v2:
        # {"techniques": [{"questions": [...]}, ...]}
        if isinstance(data.get("questions"), list):
            data = data["questions"]
        elif isinstance(data.get("techniques"), list):
            flattened = []
            for item in data.get("techniques", []):
                qs = item.get("questions", []) if isinstance(item, dict) else []
                if isinstance(qs, list):
                    flattened.extend(qs)
            data = flattened
        else:
            # Try first list-valued field.
            list_fields = [v for v in data.values() if isinstance(v, list)]
            data = list_fields[0] if list_fields else []
    if not isinstance(data, list):
        data = []
    print(f"✓ Loaded original_v2.json with {len(data)} questions")
else:
    print("⚠ original_v2.json not found. Creating example structure...")
    data = []

# Split into nonsense (first 50) and salvaged (second 50) if available
# Usually BullshitBench has 100 total quality questions
# For now, create structure even if we only have partial data

nonsense_data = data[:50] if len(data) >= 50 else data

with open("nonsense_questions.json", "w") as f:
    json.dump(nonsense_data, f, indent=2)
print(f"✓ Created nonsense_questions.json ({len(nonsense_data)} questions)")

# Also save as jsonl for dataloader compatibility
with open("nonsense_questions.jsonl", "w") as f:
    for item in nonsense_data:
        f.write(json.dumps(item) + "\n")

template = [
    {
        "id": item.get("id", item.get("question_id", idx)),
        "nonsense_prompt": item.get("question", item.get("prompt", "")),
        "salvaged_prompt": "",
        "notes": "Fill in a sensible rewrite of the nonsense prompt"
    }
    for idx, item in enumerate(nonsense_data)
]

with open("salvaged_questions.TODO.json", "w") as f:
    json.dump(template, f, indent=2)
print(f"✓ Created salvaged_questions.TODO.json ({len(template)} placeholders)")

print("✓ Created JSONL format files for compatibility")

EOF

echo ""
echo "================================================================"
echo "✓ BullshitBench v2 dataset organized in: $BULLSHIT_DIR"
echo ""
echo "Structure:"
echo "  nonsense_questions.json   - 50 nonsense/incoherent prompts"
echo "  salvaged_questions.TODO.json - manual rewrite template"
echo "  nonsense_questions.jsonl  - JSONL format"
echo ""
echo "NEXT STEPS:"
echo "1. Review the nonsense prompts in nonsense_questions.json"
echo "2. Rewrite them manually into salvaged_questions.TODO.json"
echo "3. Rename the completed file to salvaged_questions.json"
echo ""
echo "Example salvaged prompt template:"
echo "  Original: 'Our AI acceptance rate climbed to 64% -- at what rate'"
echo "           'will we hit 100% and reject all review?'"
echo "  Salvaged: 'Our AI acceptance rate is at 64%. At what rate should"
echo "           'we expect it to plateau, and how should we adapt review?'"
echo ""
echo "================================================================"
