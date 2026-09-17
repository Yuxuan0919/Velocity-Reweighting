#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WEIGHT_VARIANT="E"
export WEIGHT_C="1.0"
export WEIGHT_P="2.0"
export EXPERIMENT_TAG="run14-E-power_p2.0"
exec "${SCRIPT_DIR}/_run_sd3_4090_pickscore_common.sh" "$@"
