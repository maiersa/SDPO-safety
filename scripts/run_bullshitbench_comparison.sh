#!/usr/bin/env bash
set -euo pipefail

MODELS_JSON="${MODELS_JSON:-scripts/bullshitbench_models.example.json}"
OUT_DIR="${OUT_DIR:-outputs/bullshitbench/$(date -u +%Y%m%d_%H%M%S)}"

python scripts/eval_bullshitbench_models.py \
  --models-json "${MODELS_JSON}" \
  --output-dir "${OUT_DIR}" \
  "$@"

python scripts/plot_bullshitbench_results.py \
  --summary-csv "${OUT_DIR}/comparison_summary.csv"

echo "BullshitBench comparison saved to ${OUT_DIR}"
