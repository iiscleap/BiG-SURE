#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_FILE="${RUNS_FILE:-${ROOT}/config/runs.tsv}"
WANDB_BASE_DIR="${WANDB_BASE_DIR:-${ROOT}/outputs/wandb}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
mkdir -p outputs/strawman_ablations
[[ -f "${RUNS_FILE}" ]] || { echo "Missing run manifest: ${RUNS_FILE}"; exit 1; }
[[ "$(wc -l < "${RUNS_FILE}")" -gt 1 ]] || { echo "No runs in ${RUNS_FILE}; run both generation scripts first."; exit 1; }

tail -n +2 "${RUNS_FILE}" | while IFS=$'\t' read -r dataset model seed vanilla_run rephrased_run; do
  [[ -n "${dataset}" && -n "${vanilla_run}" && -n "${rephrased_run}" ]] || continue
  entailments="outputs/entailments/${dataset}_${model}_seed${seed}_rephrased.npz"
  [[ -f "${entailments}" ]] || { echo "Skipping ${dataset}/${model}/seed${seed}: missing ${entailments}"; continue; }
  "${PYTHON_BIN}" snne/compute_strawman_baselines.py \
    --run_dir "${rephrased_run}" \
    --vanilla_run_dir "${vanilla_run}" \
    --wandb_base_dir "${WANDB_BASE_DIR}" \
    --entailments_file "${entailments}" \
    --output_dir "outputs/strawman_ablations/${dataset}_${model}_seed${seed}" \
    --score_mode entail_prob \
    --combine min \
    --weighting_scheme entropy_confidence \
    --k_low_t 3 \
    --subsample_high_t 10 \
    --dataset "${dataset}" \
    --model_name "${model}" \
    --metric squad \
    --seed "${seed}"
done
