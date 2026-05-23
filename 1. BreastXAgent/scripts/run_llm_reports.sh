#!/usr/bin/env bash
set -euo pipefail

GRADCAM_CSV="${GRADCAM_CSV:-outputs/gradcam_results.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/llm_agent_reports.csv}"
HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

PYTHONPATH=src python3 -m breastxagent.cli report \
  --gradcam-csv "$GRADCAM_CSV" \
  --output-csv "$OUTPUT_CSV" \
  --use-llm \
  --hf-model "$HF_MODEL"

