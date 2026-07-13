#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODELS=("llava-v1.6-mistral-7b-hf" "Pixtral-12B-2409" "Qwen3-VL-8B-Instruct")
MODEL_DIRS=("llava-v1.6-mistral-7b-hf" "Pixtral-12B-2409" "Qwen3-VL-8B-Instruct")
SEEDS=(10 20 30 40 50)

for IDX in "${!MODELS[@]}"; do
  MODEL="${MODELS[$IDX]}"
  MODEL_DIR="${MODEL_DIRS[$IDX]}"
  for SEED in "${SEEDS[@]}"; do
    "${PYTHON_BIN}" snne/compute_blackbox_semantic_entropy.py \
      --dataset okvqa \
      --num_generations 10 \
      --model_name "${MODEL}" \
      --data_path "outputs/responses/vanilla/${MODEL_DIR}_seed${SEED}" \
      --metric vqa_acc \
      --metric_threshold 0.5 \
      --suffix "_vqa_acc_thr0.5_seed${SEED}"
  done
done
