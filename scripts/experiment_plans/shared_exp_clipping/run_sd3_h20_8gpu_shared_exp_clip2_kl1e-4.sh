#!/usr/bin/env bash
set -euo pipefail

# E2: fixed PickScore experiment on H20 8 GPUs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_run_sd3_h20_8gpu_shared_exp_clipping.sh" 2 "$@"
