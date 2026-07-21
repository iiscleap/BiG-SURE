#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/consolidate_multilingual_results.py" \
  --results_dir "${TASK_ROOT}/results" \
  --output_dir "${TASK_ROOT}/results/consolidated" \
  --save_raw
