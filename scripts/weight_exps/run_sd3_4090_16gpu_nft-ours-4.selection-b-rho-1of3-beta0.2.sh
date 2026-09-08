#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MASS_SHIFT_SCHEME="B"
export MASS_SHIFT_RHO="0.3333333333333333"
export EXPERIMENT_TAG="scheme-b-rho-1of3"
export EXPERIMENT_BETA="0.2"
exec "${SCRIPT_DIR}/_run_sd3_4090_16gpu_nft-ours-4.selection-common.sh" "$@"
