#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export TASK="${TASK:-pickscore}"
export NNODES="${NNODES:-${SENSECORE_PYTORCH_NNODES:-2}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
case "${NPROC_PER_NODE}" in
  6|8) ;;
  *) echo "NPROC_PER_NODE must be 8, or 6 for the optional 2x6 layout" >&2; exit 2 ;;
esac
case "${NNODES}" in
  1|2) ;;
  *) echo "NNODES must be 1 or 2" >&2; exit 2 ;;
esac
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

export LOGDIR="${LOGDIR:-${REPO_DIR}/logs/public/experiment_plans/step1_1_linear_scale15}"
export SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/step1_1_linear_scale15/sd35_${TASK}_a6000_${WORLD_SIZE}gpu_step1_1_linear_scale15_kl1e-4}"
export RUN_NAME="${RUN_NAME:-sd35_${TASK}_a6000_${WORLD_SIZE}gpu_step1_1_linear_scale15_kl1e-4}"

echo "Step 1.1: linear_target_scale=15"
exec bash "${SCRIPT_DIR}/../step1/run_sd3_a6000_16gpu_step1_linear_kl1e-4.sh" \
  "$@" --reward_mapping=linear --linear_target_scale=15
