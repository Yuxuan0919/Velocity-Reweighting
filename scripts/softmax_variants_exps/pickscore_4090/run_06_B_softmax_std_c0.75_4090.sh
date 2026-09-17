#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEIGHT_VARIANT="B"
export WEIGHT_C="0.75"
export WEIGHT_P="1.0"
export EXPERIMENT_TAG="run06-B-softmax_std_c0.75"
exec "${SCRIPT_DIR}/_run_sd3_4090_pickscore_common.sh" "$@"
