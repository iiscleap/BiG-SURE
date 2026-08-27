#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIGSURE_ROOT="$(cd "${ROOT}/.." && pwd)"
DATA_ROOT="${BIGSURE_ROOT}/data/text_qa"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(10)
DATASETS=(trivia_qa svamp)
MODELS=(
  Meta-Llama-3.1-8B-Instruct
  Llama-3.3-70B-Instruct-4bit
  Qwen2.5-72B-Instruct-4bit
  Qwen2.5-7B-Instruct
  gemma-3-12b-it
  gemma-3-27b-it
)

num_samples_for_dataset() {
  case "$1" in
    trivia_qa) echo 400 ;;
    svamp) echo 300 ;;
  esac
}

subset_csv_for_dataset() {
  case "$1" in
    trivia_qa) echo "${DATA_ROOT}/triviaqa/llama_validation_trivia_qa.csv" ;;
    svamp) echo "${DATA_ROOT}/svamp/llama_validation_svamp.csv" ;;
  esac
}

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export SCRATCH_DIR="${SCRATCH_DIR:-${ROOT}/outputs}"
export BIGSURE_RUN_MANIFEST="${BIGSURE_RUN_MANIFEST:-${ROOT}/config/runs.tsv}"
export WANDB_PROJECT="${WANDB_PROJECT:-bigsure-text-qa}"
WANDB_DIR="${WANDB_DIR:-${ROOT}/outputs/wandb}"
mkdir -p logs/vanilla

for dataset in "${DATASETS[@]}"; do
  num_samples="$(num_samples_for_dataset "${dataset}")"
  subset_csv="$(subset_csv_for_dataset "${dataset}")"
  [[ -f "${subset_csv}" ]] || { echo "Missing canonical subset: ${subset_csv}"; exit 1; }
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      "${PYTHON_BIN}" snne/generate_answers.py \
        --model_name "${model}" \
        --dataset "${dataset}" \
        --num_samples "${num_samples}" \
        --subset_csv "${subset_csv}" \
        --wandb_dir "${WANDB_DIR}" \
        --metric squad \
        --num_generations 10 \
        --n_low_temp 3 \
        --low_temp 0.1 \
        --temperature 1.0 \
        --min_p 0.0 \
        --no-compute_snn \
        --no-compute_wsnn \
        --reset_seed \
        --random_seed "${seed}" \
        --data_seed 10 \
        --suffix "_seed${seed}" \
        2>&1 | tee "logs/vanilla/${dataset}_${model}_seed${seed}.log"
    done
  done
done
