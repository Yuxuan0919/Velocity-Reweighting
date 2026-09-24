#!/usr/bin/env bash
set -euo pipefail

# Step 1 linear: H200, one node with 8 GPUs, KL=1e-4.
# Reuse the matching AWR launcher; all other environment/CLI settings pass through.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REWARD_MAPPING=linear
exec bash "${SCRIPT_DIR}/flowawr_exps/run_sd3_h200_8gpu_flowawr_kl1e-4.sh" "$@"
