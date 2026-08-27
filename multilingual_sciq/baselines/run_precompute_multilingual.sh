#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"
export PYTHONPATH="${BIGSURE_ROOT}/text_qa:${SCRIPT_DIR}:${PYTHONPATH:-}"

EXP_BASE="${TASK_ROOT}/outputs"
ENTAILMENTS_BASE="${TASK_ROOT}/outputs/entailments"
DATASETS=("sciq")
MODELS=("apertus" "aya")
SEEDS=(10)

K_LOW_T=3
SUBSAMPLE_HIGH_T=50
BATCH_SIZE=1024
MODEL_NAME="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${ENTAILMENTS_BASE}"

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      VANILLA_JSON="${EXP_BASE}/${DATASET}/inference/${MODEL}_${DATASET}_infer_vanilla_seed${SEED}/generate_with_accuracy.json"
      if [[ ! -f "${VANILLA_JSON}" ]]; then
        VANILLA_JSON="${EXP_BASE}/${DATASET}/inference/${MODEL}_${DATASET}_infer_vanilla_seed${SEED}/generate.json"
      fi

      REPHRASED_JSON="${EXP_BASE}/${DATASET}/inference/${MODEL}_${DATASET}_infer_rephrased_sampling_seed${SEED}/generate.json"
      ENTAILMENTS_FILE="${ENTAILMENTS_BASE}/${DATASET}_${MODEL}_rephrased_sampling_seed${SEED}.npz"

      if [[ ! -f "${VANILLA_JSON}" || ! -f "${REPHRASED_JSON}" ]]; then
        echo "[skip] missing vanilla or rephrased sampling json for ${MODEL} ${DATASET} seed${SEED}"
        continue
      fi

      if [[ -f "${ENTAILMENTS_FILE}" ]] && "${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_entailments_multilingual.py" --check_file "${ENTAILMENTS_FILE}"; then
        echo "[exists] ${ENTAILMENTS_FILE}"
        continue
      fi

      if [[ -f "${ENTAILMENTS_FILE}" ]]; then
        echo "[stale] recomputing legacy entailment archive ${ENTAILMENTS_FILE}"
      fi

      "${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_entailments_multilingual.py" \
        --vanilla_file "${VANILLA_JSON}" \
        --sampling_file "${REPHRASED_JSON}" \
        --output_file "${ENTAILMENTS_FILE}" \
        --mode sampling \
        --k_low_t "${K_LOW_T}" \
        --subsample_high_t "${SUBSAMPLE_HIGH_T}" \
        --batch_size "${BATCH_SIZE}" \
        --model_name "${MODEL_NAME}"
    done
  done
done
