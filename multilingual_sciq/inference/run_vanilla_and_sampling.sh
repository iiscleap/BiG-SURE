#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"

DATA_DIR="${BIGSURE_ROOT}/data/multilingual_sciq/{}"
REPHRASED_DIR="${BIGSURE_ROOT}/data/multilingual_sciq/sciq"
PROMPT_DIR="${TASK_ROOT}/prompt"
FILTER_JSON="${REPHRASED_DIR}/sciq_rephrased_300.json"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_DIR="${WANDB_DIR:-${TASK_ROOT}/outputs/wandb}"

DATASETS=("sciq")
MODELS=("apertus" "aya")
MODES=("vanilla" "sampling")
SEEDS=(10)

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for MODE in "${MODES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        if [[ "${MODE}" == "vanilla" ]]; then
          SUFFIX="infer_vanilla_seed${SEED}"
        else
          SUFFIX="infer_sampling_seed${SEED}"
        fi

        "${PYTHON_BIN}" "${SCRIPT_DIR}/inference.py" \
          --model_name "${MODEL}" \
          --dataset "${DATASET}" \
          --inference_mode "${MODE}" \
          --suffix "${SUFFIX}" \
          --seed "${SEED}" \
          --data_dir "${DATA_DIR}" \
          --rephrased_data_dir "${REPHRASED_DIR}" \
          --prompt_dir "${PROMPT_DIR}" \
          --wandb_dir "${WANDB_DIR}" \
          --filter_ids_from "${FILTER_JSON}" \
          --continue_generate True
      done
    done
  done
done
