#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${1:-data/breast_density_classification}"

PYTHONPATH=src python3 -m breastxagent.cli download \
  --local-dir "$LOCAL_DIR"

