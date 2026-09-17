#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEIGHT_VARIANT="A"
export WEIGHT_C="0.25"
export WEIGHT_P="1.0"
export EXPERIMENT_TAG="run04-A-near_max_c0.25"
exec "${SCRIPT_DIR}/_run_sd3_4090_pickscore_common.sh" "$@"
