#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"
export PYTHONPATH="${BIGSURE_ROOT}/text_qa:${SCRIPT_DIR}:${PYTHONPATH:-}"

EXP_BASE="${TASK_ROOT}/outputs"
ENTAILMENTS_BASE="${TASK_ROOT}/outputs/entailments"
RESULTS_BASE="${TASK_ROOT}/results/spectral_energy"
DATASETS=("sciq")
MODELS=("apertus" "aya")
SEEDS=(10)
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${ENTAILMENTS_BASE}" "${RESULTS_BASE}"

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      VANILLA_JSON="${EXP_BASE}/${DATASET}/inference/${MODEL}_${DATASET}_infer_vanilla_seed${SEED}/generate_with_accuracy.json"
      REPHRASED_JSON="${EXP_BASE}/${DATASET}/inference/${MODEL}_${DATASET}_infer_rephrased_sampling_seed${SEED}/generate.json"
      ENTAILMENTS_FILE="${ENTAILMENTS_BASE}/${DATASET}_${MODEL}_rephrased_sampling_seed${SEED}.npz"

      if [[ ! -f "${VANILLA_JSON}" || ! -f "${REPHRASED_JSON}" ]]; then
        echo "[skip] missing evaluated vanilla or rephrased sampling json for ${MODEL} ${DATASET} seed${SEED}"
        continue
      fi

      if [[ ! -f "${ENTAILMENTS_FILE}" ]]; then
        echo "[skip] missing entailments: ${ENTAILMENTS_FILE}"
        echo "       run baselines/run_precompute_multilingual.sh first"
        continue
      fi

      "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_spectral_energy_multilingual.py" \
        --vanilla_file "${VANILLA_JSON}" \
        --sampling_file "${REPHRASED_JSON}" \
        --entailments_file "${ENTAILMENTS_FILE}" \
        --output_dir "${RESULTS_BASE}/${DATASET}_${MODEL}_seed${SEED}_gemini_entail_prob_min" \
        --dataset "${DATASET}" \
        --score_mode entail_prob \
        --combine min \
        --metric gemini \
        --subsample_high_t 10 \
        --subsample_seed 123
    done
  done
done
