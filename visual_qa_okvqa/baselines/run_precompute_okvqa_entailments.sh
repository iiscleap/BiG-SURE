#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODELS=("llava-v1.6-mistral-7b-hf" "Pixtral-12B-2409" "Qwen3-VL-8B-Instruct")
SEEDS=(10 20 30 40 50)
SUBSAMPLE_HIGH_T=350

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    VANILLA_PKL="outputs/responses/vanilla/${MODEL}_seed${SEED}/validation_generations.pkl"
    PERTURBED_PKL="outputs/responses/perturbed/${MODEL}_seed${SEED}/validation_generations.pkl"
    OUT="outputs/entailments/${MODEL}_seed${SEED}_perturbed.npz"

    if [[ ! -f "${VANILLA_PKL}" || ! -f "${PERTURBED_PKL}" ]]; then
      echo "Skipping ${MODEL} seed ${SEED}; missing vanilla or perturbed pkl"
      continue
    fi

    "${PYTHON_BIN}" snne/precompute_entailments_vqa.py \
      --vanilla_pkl "${VANILLA_PKL}" \
      --rephrased_pkl "${PERTURBED_PKL}" \
      --output_file "${OUT}" \
      --mode perturbed \
      --k_low_t 3 \
      --subsample_high_t "${SUBSAMPLE_HIGH_T}" \
      --fp16
  done
done
