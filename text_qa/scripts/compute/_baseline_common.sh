#!/usr/bin/env bash

baseline_setup() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  ROOT="$(cd "${script_dir}/../.." && pwd)"
  PYTHON_BIN="${PYTHON_BIN:-python}"
  RUNS_FILE="${RUNS_FILE:-${ROOT}/config/runs.tsv}"

  cd "${ROOT}"
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
  export SCRATCH_DIR="${SCRATCH_DIR:-${ROOT}/outputs}"
  export WANDB_DIR="${WANDB_DIR:-${ROOT}/outputs/wandb}"
  export WANDB_PROJECT="${WANDB_PROJECT:-bigsure-text-qa}"

  [[ -f "${RUNS_FILE}" ]] || { echo "Missing run manifest: ${RUNS_FILE}" >&2; exit 1; }
  [[ "$(wc -l < "${RUNS_FILE}")" -gt 1 ]] || {
    echo "No runs in ${RUNS_FILE}; run scripts/generate/generate_qa.sh first." >&2
    exit 1
  }
}

resolve_vanilla_files_dir() {
  local run_dir="$1"
  if [[ -f "${run_dir}/validation_generations.pkl" ]]; then
    printf '%s\n' "${run_dir}"
  elif [[ -f "${run_dir}/files/validation_generations.pkl" ]]; then
    printf '%s\n' "${run_dir}/files"
  else
    return 1
  fi
}

run_text_baseline() {
  local entrypoint="$1"
  shift
  local completed=0

  while IFS=$'\t' read -r dataset model seed vanilla_run _rephrased_run; do
    [[ -n "${dataset}" && -n "${vanilla_run}" ]] || continue
    local data_path
    if ! data_path="$(resolve_vanilla_files_dir "${vanilla_run}")"; then
      echo "[skip] ${dataset}/${model}/seed${seed}: no validation_generations.pkl under ${vanilla_run}" >&2
      continue
    fi

    echo "[run] ${entrypoint} ${dataset}/${model}/seed${seed}"
    "${PYTHON_BIN}" "snne/${entrypoint}" \
      --dataset "${dataset}" \
      --num_generations 10 \
      --model_name "${model}" \
      --data_path "${data_path}" \
      --metric squad \
      --metric_threshold 0.5 \
      --suffix "_squad" \
      --random_seed "${seed}" \
      "$@"
    completed=$((completed + 1))
  done < <(tail -n +2 "${RUNS_FILE}")

  [[ "${completed}" -gt 0 ]] || {
    echo "No usable vanilla generation runs were found in ${RUNS_FILE}." >&2
    exit 1
  }
}
