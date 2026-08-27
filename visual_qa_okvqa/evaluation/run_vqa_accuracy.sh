#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT="${BIGSURE_ROOT}/data/visual_qa/okvqa"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

shopt -s nullglob
PKLS=(outputs/responses/vanilla/*_seed10/validation_generations.pkl)
[[ "${#PKLS[@]}" -gt 0 ]] || {
  echo "No vanilla generations found under outputs/responses/vanilla/. Run inference/run_vanilla_generation.sh first." >&2
  exit 1
}

for PKL in "${PKLS[@]}"; do
  RUN_NAME="$(basename "$(dirname "${PKL}")")"
  "${PYTHON_BIN}" evaluation/calc_vqa_accuracy.py \
    --input "${PKL}" \
    --dataset okvqa \
    --output "outputs/responses/vanilla/${RUN_NAME}/vqa_accuracy.json"
  "${PYTHON_BIN}" evaluation/validate_okvqa_artifacts.py \
    --vanilla "${PKL}" \
    --accuracy "outputs/responses/vanilla/${RUN_NAME}/vqa_accuracy.json" \
    --expected-examples 200 \
    --high-temp 10 \
    --low-temp 3
done
