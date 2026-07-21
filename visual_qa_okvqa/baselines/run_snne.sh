#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_baseline_common.sh"
baseline_setup
run_okvqa_baseline compute_snne.py
