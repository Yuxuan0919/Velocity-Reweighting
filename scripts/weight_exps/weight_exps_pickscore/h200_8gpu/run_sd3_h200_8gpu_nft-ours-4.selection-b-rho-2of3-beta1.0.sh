#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MASS_SHIFT_SCHEME="B"
export MASS_SHIFT_RHO="0.6666666666666666"
export EXPERIMENT_TAG="scheme-b-rho-2of3"
export EXPERIMENT_BETA="1.0"
exec "${SCRIPT_DIR}/_run_sd3_h200_8gpu_nft-ours-4.selection-soft-common.sh" "$@"
