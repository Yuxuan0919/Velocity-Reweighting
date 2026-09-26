#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export TASK="${TASK:-pickscore}"

export LOGDIR="${LOGDIR:-${REPO_DIR}/logs/public/experiment_plans/step1_1_linear_scale15}"
export SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/step1_1_linear_scale15/sd35_${TASK}_h200_8gpu_step1_1_linear_scale15_kl1e-4}"
export RUN_NAME="${RUN_NAME:-sd35_${TASK}_h200_8gpu_step1_1_linear_scale15_kl1e-4}"

echo "Step 1.1: linear_target_scale=15"
exec bash "${SCRIPT_DIR}/../step1/run_sd3_h200_8gpu_step1_linear_kl1e-4.sh" \
  "$@" --reward_mapping=linear --linear_target_scale=15
