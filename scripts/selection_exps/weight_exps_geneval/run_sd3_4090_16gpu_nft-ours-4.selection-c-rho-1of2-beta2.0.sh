#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MASS_SHIFT_SCHEME="C"
export MASS_SHIFT_RHO="0.5"
export EXPERIMENT_TAG="scheme-c-rho-1of2"
export EXPERIMENT_BETA="2.0"
exec "${SCRIPT_DIR}/_run_sd3_4090_16gpu_nft-ours-4.selection-common.sh" "$@"
