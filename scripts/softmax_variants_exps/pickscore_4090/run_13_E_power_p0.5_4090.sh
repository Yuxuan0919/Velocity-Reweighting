#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEIGHT_VARIANT="E"
export WEIGHT_C="1.0"
export WEIGHT_P="0.5"
export EXPERIMENT_TAG="run13-E-power_p0.5"
exec "${SCRIPT_DIR}/_run_sd3_4090_pickscore_common.sh" "$@"
