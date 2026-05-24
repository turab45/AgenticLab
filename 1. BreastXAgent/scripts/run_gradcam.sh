#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-data/breast_density_classification}"
PREDICTIONS_CSV="${PREDICTIONS_CSV:-outputs/predictions.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gradcam}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/gradcam_results.csv}"
DEVICE="${DEVICE:-auto}"
MPLCONFIGDIR="${MPLCONFIGDIR:-.cache/matplotlib}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

PYTHONPATH=src python3 -m breastxagent.cli gradcam \
  --repo-dir "$REPO_DIR" \
  --predictions-csv "$PREDICTIONS_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --output-csv "$OUTPUT_CSV" \
  --device "$DEVICE"
