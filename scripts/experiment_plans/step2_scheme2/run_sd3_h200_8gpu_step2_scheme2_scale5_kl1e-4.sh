#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export TASK="${TASK:-pickscore}"
export AWR_VARIANCE_GATE=false

export LOGDIR="${LOGDIR:-${REPO_DIR}/logs/public/experiment_plans/step2_scheme2_scale5}"
export SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/step2_scheme2_scale5/sd35_${TASK}_h200_8gpu_step2_scheme2_scale5_kl1e-4}"
export RUN_NAME="${RUN_NAME:-sd35_${TASK}_h200_8gpu_step2_scheme2_scale5_kl1e-4}"

echo "Step 2 Scheme 2: mass_shift_transform=square_both, linear_target_scale=5, variance gate=false"
exec bash "${SCRIPT_DIR}/../step1/run_sd3_h200_8gpu_step1_linear_kl1e-4.sh" \
  "$@" --reward_mapping=linear --linear_target_scale=5 \
  --mass_shift_transform=square_both --awr_variance_gate=false
