#!/usr/bin/env bash
set -euo pipefail

# E3: fixed PickScore experiment on A6000 16 GPUs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/_run_sd3_a6000_16gpu_shared_exp_clipping.sh" 3 "$@"
