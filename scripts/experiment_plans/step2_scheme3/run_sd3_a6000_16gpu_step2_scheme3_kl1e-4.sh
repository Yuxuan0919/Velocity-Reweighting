#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export TASK="${TASK:-pickscore}"
export AWR_VARIANCE_GATE=false
export NNODES="${NNODES:-${SENSECORE_PYTORCH_NNODES:-2}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
case "${NPROC_PER_NODE}" in
  8) ;;
  *) echo "NPROC_PER_NODE must be 8 for the A6000 2x8 layout" >&2; exit 2 ;;
esac
case "${NNODES}" in
  2) ;;
  *) echo "NNODES must be 2 for the A6000 2x8 layout" >&2; exit 2 ;;
esac
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

export LOGDIR="${LOGDIR:-${REPO_DIR}/logs/public/experiment_plans/step2_scheme3}"
export SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/step2_scheme3/sd35_${TASK}_a6000_${WORLD_SIZE}gpu_step2_scheme3_kl1e-4}"
export RUN_NAME="${RUN_NAME:-sd35_${TASK}_a6000_${WORLD_SIZE}gpu_step2_scheme3_kl1e-4}"

echo "Step 2 Scheme 3: mass_shift_transform=exp_positive, linear_target_scale=1, gamma_prime=1, epsilon=1e-6, variance gate=false"
exec bash "${SCRIPT_DIR}/../step1/run_sd3_a6000_16gpu_step1_linear_kl1e-4.sh" \
  "$@" --reward_mapping=linear --linear_target_scale=1 \
  --mass_shift_transform=exp_positive \
  --mass_shift_gamma_prime=1 --mass_shift_epsilon=1e-6 --awr_variance_gate=false
