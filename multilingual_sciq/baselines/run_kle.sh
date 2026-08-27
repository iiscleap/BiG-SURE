#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"
export PYTHONPATH="${BIGSURE_ROOT}/text_qa:${SCRIPT_DIR}:${PYTHONPATH:-}"

EXP_BASE="${TASK_ROOT}/outputs"
RESULTS_BASE="${TASK_ROOT}/results/kle"
FILTER_JSON="${BIGSURE_ROOT}/data/multilingual_sciq/sciq/sciq_rephrased_300.json"
LANGUAGES=(en zh ja fr)
DATASETS=("sciq")
MODELS=("apertus" "aya")
SEEDS=(10)
PYTHON_BIN="${PYTHON_BIN:-python}"

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      VANILLA_JSON="${EXP_BASE}/${DATASET}/inference/${MODEL}_${DATASET}_infer_vanilla_seed${SEED}/generate_with_accuracy.json"
      SAMPLING_JSON="${EXP_BASE}/${DATASET}/inference/${MODEL}_${DATASET}_infer_sampling_seed${SEED}/generate.json"

      if [[ ! -f "${VANILLA_JSON}" || ! -f "${SAMPLING_JSON}" ]]; then
        echo "[skip] missing vanilla/eval or sampling json for ${MODEL} ${DATASET} seed${SEED}"
        continue
      fi

      "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_multilingual_kle.py" \
        --vanilla_json "${VANILLA_JSON}" \
        --sampling_json "${SAMPLING_JSON}" \
        --output_dir "${RESULTS_BASE}/${DATASET}_${MODEL}_seed${SEED}_gemini" \
        --languages "${LANGUAGES[@]}" \
        --dataset "${DATASET}" \
        --model_name "${MODEL}" \
        --random_seed "${SEED}" \
        --metric gemini \
        --filter_ids_from "${FILTER_JSON}"
    done
  done
done
