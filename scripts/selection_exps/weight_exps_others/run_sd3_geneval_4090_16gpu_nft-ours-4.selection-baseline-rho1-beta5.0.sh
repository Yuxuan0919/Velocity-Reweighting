#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_TASK="geneval"
export MASS_SHIFT_SCHEME="C"
export MASS_SHIFT_RHO="1.0"
export EXPERIMENT_TAG="baseline-rho1"
export EXPERIMENT_BETA="5.0"
export MASTER_PORT="${MASTER_PORT:-29534}"
exec "${SCRIPT_DIR}/_run_sd3_4090_16gpu_nft-ours-4.selection-common.sh" "$@"
