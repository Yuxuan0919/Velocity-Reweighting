#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEIGHT_VARIANT="A"
export WEIGHT_C="0.05"
export WEIGHT_P="1.0"
export EXPERIMENT_TAG="run02-A-near_max_c0.05"
exec "${SCRIPT_DIR}/_run_sd3_4090_pickscore_common.sh" "$@"
