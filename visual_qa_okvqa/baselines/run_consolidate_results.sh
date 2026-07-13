#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p outputs/consolidated
"${PYTHON_BIN}" consolidate_vqa_results.py
