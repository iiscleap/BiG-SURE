#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"

DATA_DIR="${BIGSURE_ROOT}/data/multilingual_sciq/{}"
REPHRASED_DIR="${BIGSURE_ROOT}/data/multilingual_sciq/sciq"
PROMPT_DIR="${TASK_ROOT}/prompt"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_DIR="${WANDB_DIR:-${TASK_ROOT}/outputs/wandb}"

DATASETS=("sciq")
MODELS=("apertus" "aya")
SEEDS=(10 20 30 40 50)
K_SAMPLES=10

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      "${PYTHON_BIN}" "${SCRIPT_DIR}/inference.py" \
        --model_name "${MODEL}" \
        --dataset "${DATASET}" \
        --inference_mode "rephrased_sampling" \
        --suffix "infer_rephrased_sampling_seed${SEED}" \
        --seed "${SEED}" \
        --k "${K_SAMPLES}" \
        --data_dir "${DATA_DIR}" \
        --rephrased_data_dir "${REPHRASED_DIR}" \
        --prompt_dir "${PROMPT_DIR}" \
        --wandb_dir "${WANDB_DIR}" \
        --continue_generate True
    done
  done
done
