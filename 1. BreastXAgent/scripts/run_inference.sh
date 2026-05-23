#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-data/breast_density_classification}"
IMAGE_DIR="${IMAGE_DIR:-}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/predictions.csv}"
DEVICE="${DEVICE:-auto}"

if [[ -n "$IMAGE_DIR" ]]; then
  PYTHONPATH=src python3 -m breastxagent.cli infer \
    --repo-dir "$REPO_DIR" \
    --image-dir "$IMAGE_DIR" \
    --output-csv "$OUTPUT_CSV" \
    --device "$DEVICE"
else
  PYTHONPATH=src python3 -m breastxagent.cli infer \
    --repo-dir "$REPO_DIR" \
    --output-csv "$OUTPUT_CSV" \
    --device "$DEVICE"
fi

