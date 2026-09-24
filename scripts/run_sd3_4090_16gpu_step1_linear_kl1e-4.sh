#!/usr/bin/env bash
set -euo pipefail

# Step 1 linear: 4090, two nodes with 8 GPUs each, KL=1e-4.
# Set NODE_RANK=0 / 1 on the two nodes and use the same MASTER_ADDR.
# Reuse the matching AWR launcher; all other environment/CLI settings pass through.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REWARD_MAPPING=linear
exec bash "${SCRIPT_DIR}/flowawr_exps/run_sd3_4090_16gpu_flowawr_kl1e-4.sh" "$@"
