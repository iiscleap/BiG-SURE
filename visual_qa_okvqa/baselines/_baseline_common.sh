#!/usr/bin/env bash

baseline_setup() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  ROOT="$(cd "${script_dir}/.." && pwd)"
  PYTHON_BIN="${PYTHON_BIN:-python}"

  cd "${ROOT}"
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
  export SCRATCH_DIR="${SCRATCH_DIR:-${ROOT}/outputs}"
  export WANDB_DIR="${WANDB_DIR:-${ROOT}/outputs/wandb}"
  export WANDB_PROJECT="${WANDB_PROJECT:-bigsure-okvqa}"

  MODELS=("llava-v1.6-mistral-7b-hf" "Pixtral-12B-2409" "Qwen3-VL-8B-Instruct")
  SEEDS=(10 20 30 40 50)
}

run_okvqa_baseline() {
  local entrypoint="$1"
  local completed=0

  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local data_path="outputs/responses/vanilla/${model}_seed${seed}"
      local generations="${data_path}/validation_generations.pkl"
      local accuracy="${data_path}/vqa_accuracy.json"
      if [[ ! -f "${generations}" ]]; then
        echo "[skip] ${model}/seed${seed}: missing ${generations}" >&2
        continue
      fi
      if [[ ! -f "${accuracy}" ]]; then
        echo "[skip] ${model}/seed${seed}: missing ${accuracy}; run evaluation/run_vqa_accuracy.sh" >&2
        continue
      fi

      echo "[run] ${entrypoint} ${model}/seed${seed}"
      "${PYTHON_BIN}" "snne/${entrypoint}" \
        --dataset okvqa \
        --num_generations 10 \
        --model_name "${model}" \
        --data_path "${data_path}" \
        --vqa_json "${accuracy}" \
        --metric vqa_acc \
        --metric_threshold 0.5 \
        --suffix "_vqa_acc_thr0.5" \
        --random_seed "${seed}"
      completed=$((completed + 1))
    done
  done

  [[ "${completed}" -gt 0 ]] || {
    echo "No evaluated vanilla generation runs were available for ${entrypoint}." >&2
    exit 1
  }
}
