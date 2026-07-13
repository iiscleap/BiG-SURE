#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT="${BIGSURE_ROOT}/data/visual_qa/okvqa"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for PKL in outputs/responses/vanilla/*/validation_generations.pkl; do
  RUN_NAME="$(basename "$(dirname "${PKL}")")"
  "${PYTHON_BIN}" evaluation/compute_vqa_accuracies.py \
    --vanilla_run_dir "$(dirname "${PKL}")" \
    --predictions "${PKL}" \
    --metadata "${DATA_ROOT}/metadata.csv" \
    --dataset okvqa \
    --output "outputs/responses/vanilla/${RUN_NAME}/vqa_accuracy.json"
done
