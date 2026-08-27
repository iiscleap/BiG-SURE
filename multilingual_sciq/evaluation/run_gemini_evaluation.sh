#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"

EXP_BASE="${TASK_ROOT}/outputs"
FILTER_JSON="${BIGSURE_ROOT}/data/multilingual_sciq/sciq/sciq_rephrased_300.json"
LANGUAGES=(en zh ja fr)
DATASETS=("sciq")
MODELS=("apertus" "aya")
SEEDS=(10)
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${GOOGLE_API_KEY:?Set GOOGLE_API_KEY before running Gemini evaluation.}"

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_DIR="${EXP_BASE}/${DATASET}/inference/${MODEL}_${DATASET}_infer_vanilla_seed${SEED}"
      INPUT_JSON="${RUN_DIR}/generate.json"
      OUTPUT_JSON="${RUN_DIR}/generate_with_accuracy.json"

      if [[ ! -f "${INPUT_JSON}" ]]; then
        echo "[skip] missing ${INPUT_JSON}"
        continue
      fi

      "${PYTHON_BIN}" "${SCRIPT_DIR}/gemini_evaluate_generations.py" \
        --input_json "${INPUT_JSON}" \
        --output_json "${OUTPUT_JSON}" \
        --dataset "${DATASET}" \
        --model "${MODEL}" \
        --languages "${LANGUAGES[@]}" \
        --filter_ids_from "${FILTER_JSON}"
    done
  done
done
