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
COMPLETED=0

mkdir -p outputs/entailments

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    VANILLA_PKL="outputs/responses/vanilla/${MODEL}_seed${SEED}/validation_generations.pkl"
    PERTURBED_PKL="outputs/responses/perturbed/${MODEL}_seed${SEED}/validation_generations.pkl"
    ACCURACY_JSON="outputs/responses/vanilla/${MODEL}_seed${SEED}/vqa_accuracy.json"
    OUT="outputs/entailments/${MODEL}_seed${SEED}_perturbed.npz"

    if [[ ! -f "${VANILLA_PKL}" || ! -f "${PERTURBED_PKL}" || ! -f "${ACCURACY_JSON}" ]]; then
      echo "[skip] ${MODEL}/seed${SEED}: missing vanilla, perturbed, or accuracy artifact" >&2
      continue
    fi

    "${PYTHON_BIN}" snne/precompute_entailments_vqa.py \
      --vanilla_pkl "${VANILLA_PKL}" \
      --accuracy_file "${ACCURACY_JSON}" \
      --rephrased_pkl "${PERTURBED_PKL}" \
      --output_file "${OUT}" \
      --mode perturbed \
      --k_low_t 3 \
      --subsample_high_t "${SUBSAMPLE_HIGH_T}" \
      --fp16
    COMPLETED=$((COMPLETED + 1))
  done
done

[[ "${COMPLETED}" -gt 0 ]] || {
  echo "No complete OKVQA generation/evaluation pairs were found." >&2
  exit 1
}
