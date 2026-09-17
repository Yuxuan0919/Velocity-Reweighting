#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEIGHT_VARIANT="F"
export WEIGHT_C="0.5"
export WEIGHT_P="1.0"
export EXPERIMENT_TAG="run15-F-softmax_l1_c0.5"
exec "${SCRIPT_DIR}/_run_sd3_4090_pickscore_common.sh" "$@"
