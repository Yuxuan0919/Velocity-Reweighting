#!/usr/bin/env bash
set -euo pipefail

# Run two isolated copies of the same local RL harness and require their
# TensorBoard reward series to match bit-for-bit.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-699983977898.dkr.ecr.us-east-1.amazonaws.com/ds-core-cn-omra-trainer@sha256:531d95485182ec361c3ee5d1a307ddfe931798e81db55725c02048d46e693b5f}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/determinism_pickscore_validation_runs}"
VALIDATION_ID="${VALIDATION_ID:?Set VALIDATION_ID to a fresh identifier}"
NUM_STEPS="${NUM_STEPS:-100}"
VALIDATION_ROOT="${OUTPUT_ROOT}/${VALIDATION_ID}"

case "${VALIDATION_ID}" in
  *[!A-Za-z0-9._-]*|'')
    echo "VALIDATION_ID must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
    ;;
esac
if [[ ! "${NUM_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ -e "${VALIDATION_ROOT}" || -L "${VALIDATION_ROOT}" ]]; then
  echo "Refusing to reuse validation directory: ${VALIDATION_ROOT}" >&2
  exit 2
fi

for run_id in run-a run-b; do
  IMAGE="${IMAGE}" \
  OUTPUT_ROOT="${VALIDATION_ROOT}" \
  RUN_ID="${run_id}" \
  NUM_EPOCHS="${NUM_STEPS}" \
    "${REPO_DIR}/scripts/run_local_determinism_pickscore.sh"
done

docker run --rm --network=none \
  --volume "${REPO_DIR}:${REPO_DIR}" \
  --workdir "${REPO_DIR}" \
  --entrypoint python \
  "${IMAGE}" \
  scripts/compare_tensorboard_rewards.py \
  "${VALIDATION_ROOT}/run-a" \
  "${VALIDATION_ROOT}/run-b" \
  --expected-steps "${NUM_STEPS}"
