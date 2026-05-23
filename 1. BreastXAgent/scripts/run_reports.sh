#!/usr/bin/env bash
set -euo pipefail

GRADCAM_CSV="${GRADCAM_CSV:-outputs/gradcam_results.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/agent_reports.csv}"

PYTHONPATH=src python3 -m breastxagent.cli report \
  --gradcam-csv "$GRADCAM_CSV" \
  --output-csv "$OUTPUT_CSV"

