#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIGSURE_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT="${BIGSURE_ROOT}/data/visual_qa/okvqa"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
# LLaVA/Qwen3-VL use snne. For Pixtral, temporarily leave only Pixtral in
# MODELS and activate the snne_pixtral environment first.
PYTHON_BIN="${PYTHON_BIN:-python}"
export WANDB_PROJECT="${WANDB_PROJECT:-bigsure-okvqa}"
WANDB_DIR="${WANDB_DIR:-${ROOT}/outputs/wandb}"

MODELS=(
  "llava-hf/llava-v1.6-mistral-7b-hf"
  "mistralai/Pixtral-12B-2409"
  "Qwen/Qwen3-VL-8B-Instruct"
)
SEEDS=(10 20 30 40 50)

NUM_GENERATIONS=10
TEMPERATURE=1.0
MIN_P=0.0
CHECKPOINT_INTERVAL=500

mkdir -p logs

for MODEL in "${MODELS[@]}"; do
  MODEL_SAFE="${MODEL//\//_}"
  MODEL_DIR="${MODEL##*/}"
  for SEED in "${SEEDS[@]}"; do
    OUTPUT="outputs/responses/perturbed/${MODEL_DIR}_seed${SEED}/validation_generations.pkl"
    "${PYTHON_BIN}" snne/generate_okvqa_rephrased_perturbed.py \
      --model_name "${MODEL}" \
      --input_csv "${DATA_ROOT}/rephrased_perturbations.csv" \
      --vqa_image_dir "${DATA_ROOT}/images_perturbed" \
      --output "${OUTPUT}" \
      --wandb_dir "${WANDB_DIR}" \
      --num_generations "${NUM_GENERATIONS}" \
      --temperature "${TEMPERATURE}" \
      --min_p "${MIN_P}" \
      --reset_seed \
      --random_seed "${SEED}" \
      --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
      --experiment_lot "okvqa_perturbed" \
      2>&1 | tee "logs/${MODEL_SAFE}_okvqa_perturbed_seed${SEED}.log"
  done
done
