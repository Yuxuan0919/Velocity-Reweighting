#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEIGHT_VARIANT="D"
export WEIGHT_C="0.01"
export WEIGHT_P="1.0"
export EXPERIMENT_TAG="run10-D-sigmoid_c0.01"
exec "${SCRIPT_DIR}/_run_sd3_4090_pickscore_common.sh" "$@"
