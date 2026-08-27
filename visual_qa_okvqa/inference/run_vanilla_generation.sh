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
SEEDS=(10)

NUM_GENERATIONS=10
NUM_LOW_TEMP_GENERATIONS=3
TEMPERATURE=1.0
LOW_TEMP=0.1
NUM_SAMPLES=200
MIN_P=0.0

mkdir -p logs

for MODEL in "${MODELS[@]}"; do
  MODEL_SAFE="${MODEL//\//_}"
  MODEL_DIR="${MODEL##*/}"
  for SEED in "${SEEDS[@]}"; do
    OUTPUT="outputs/responses/vanilla/${MODEL_DIR}_seed${SEED}/validation_generations.pkl"
    "${PYTHON_BIN}" snne/generate_okvqa_answers.py \
      --model_name "${MODEL}" \
      --input_csv "${DATA_ROOT}/metadata.csv" \
      --vqa_image_dir "${DATA_ROOT}/images" \
      --output "${OUTPUT}" \
      --wandb_dir "${WANDB_DIR}" \
      --num_generations "${NUM_GENERATIONS}" \
      --num_low_temp_generations "${NUM_LOW_TEMP_GENERATIONS}" \
      --num_samples "${NUM_SAMPLES}" \
      --temperature "${TEMPERATURE}" \
      --low_temp "${LOW_TEMP}" \
      --min_p "${MIN_P}" \
      --reset_seed \
      --random_seed "${SEED}" \
      --experiment_lot "okvqa_vanilla" \
      --metric "vqa_acc" \
      2>&1 | tee "logs/${MODEL_SAFE}_okvqa_vanilla_seed${SEED}.log"
    "${PYTHON_BIN}" evaluation/validate_okvqa_artifacts.py \
      --vanilla "${OUTPUT}" \
      --expected-examples "${NUM_SAMPLES}" \
      --high-temp "${NUM_GENERATIONS}" \
      --low-temp "${NUM_LOW_TEMP_GENERATIONS}"
  done
done
