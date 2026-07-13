#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODELS=("llava-v1.6-mistral-7b-hf" "Pixtral-12B-2409" "Qwen3-VL-8B-Instruct")
SEEDS=(10 20 30 40 50)
SUBSAMPLE_HIGH_T=10
SUBSAMPLE_SEED=123

short_model() {
  case "$1" in
    llava-v1.6-mistral-7b-hf) echo "llava7b" ;;
    Pixtral-12B-2409) echo "pixtral12b" ;;
    Qwen3-VL-8B-Instruct) echo "qwen8b" ;;
    *) echo "$1" ;;
  esac
}

OUT_BASE="outputs/spectral_energy"
mkdir -p "${OUT_BASE}"

for MODEL in "${MODELS[@]}"; do
  SHORT="$(short_model "${MODEL}")"
  for SEED in "${SEEDS[@]}"; do
    ENTAIL="outputs/entailments/${MODEL}_seed${SEED}_perturbed.npz"
    VANILLA="outputs/responses/vanilla/${MODEL}_seed${SEED}/validation_generations.pkl"
    if [[ ! -f "${ENTAIL}" || ! -f "${VANILLA}" ]]; then
      echo "Skipping ${MODEL} seed ${SEED}; missing entailments or vanilla pkl"
      continue
    fi

    "${PYTHON_BIN}" snne/compute_spectral_energy_weighted_okvqa.py \
      --entailments_file "${ENTAIL}" \
      --vanilla_pkl "${VANILLA}" \
      --metric vqa_acc \
      --metric_threshold 0.5 \
      --output_dir "${OUT_BASE}/okvqa_${SHORT}_seed${SEED}_ss${SUBSAMPLE_SEED}_entail_prob_min" \
      --similarity deberta \
      --score_mode entail_prob \
      --combine min \
      --weighting_scheme entropy_confidence \
      --divergence_measure js \
      --sim_threshold 0.5 \
      --subsample_high_t "${SUBSAMPLE_HIGH_T}" \
      --subsample_seed "${SUBSAMPLE_SEED}" \
      --model_name "${MODEL}"
  done
done
