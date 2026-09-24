#!/usr/bin/env bash
set -euo pipefail

# Step 1: use the AWR training setup with f(r) = max(r, 0).
# Default: PickScore, one H200 node with 8 GPUs, KL=0.
# All other environment variables and CLI arguments go to the AWR launcher.
# Example for two 4090 nodes (set NODE_RANK=0 / 1 on each node):
# AWR_PRESET=4090_16gpu_kl1e-4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/run_sd3_step1_linear.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWR_PRESET="${AWR_PRESET:-h200_8gpu_kl0}"

case "${AWR_PRESET}" in
  h200_8gpu_kl0) LAUNCHER=run_sd3_h200_8gpu_flowawr.sh ;;
  h200_8gpu_kl1e-4) LAUNCHER=run_sd3_h200_8gpu_flowawr_kl1e-4.sh ;;
  4090_16gpu_kl0) LAUNCHER=run_sd3_4090_16gpu_flowawr.sh ;;
  4090_16gpu_kl1e-4) LAUNCHER=run_sd3_4090_16gpu_flowawr_kl1e-4.sh ;;
  *)
    echo "AWR_PRESET must be h200_8gpu_kl0, h200_8gpu_kl1e-4, 4090_16gpu_kl0, or 4090_16gpu_kl1e-4" >&2
    exit 2
    ;;
esac

export REWARD_MAPPING=linear
exec bash "${SCRIPT_DIR}/flowawr_exps/${LAUNCHER}" "$@"
