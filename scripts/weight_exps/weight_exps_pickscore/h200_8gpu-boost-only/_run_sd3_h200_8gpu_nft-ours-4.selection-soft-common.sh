#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"
SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"

REWARD_CKPTS="${REPO_DIR}/reward_ckpts"
TRAIN_SCRIPT="scripts/train_nft_sd3_ours-4.singleloss-alpha-selection-soft-record-boost-only.py"

MASS_SHIFT_SCHEME="${MASS_SHIFT_SCHEME:?MASS_SHIFT_SCHEME must be set to B or C}"
MASS_SHIFT_RHO="${MASS_SHIFT_RHO:?MASS_SHIFT_RHO must be set}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:?EXPERIMENT_TAG must be set}"
EXPERIMENT_BETA="${EXPERIMENT_BETA:?EXPERIMENT_BETA must be set}"

case "${MASS_SHIFT_SCHEME}" in
  B|C) ;;
  *)
    echo "MASS_SHIFT_SCHEME must be B or C" >&2
    exit 2
    ;;
esac

NPROC_PER_NODE=8
PER_DEVICE_BATCH=9
GRADIENT_ACCUMULATION_STEPS=16
EFFECTIVE_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH * GRADIENT_ACCUMULATION_STEPS))
if [[ "${EFFECTIVE_BATCH}" -ne 1152 ]]; then
  echo "Invalid effective batch: ${EFFECTIVE_BATCH} (expected 1152)" >&2
  exit 2
fi

LOGDIR="${LOGDIR:-${REPO_DIR}/logs/weight_exps/weight_exps_pickscore/h200_8gpu}"
SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/weight_exps/weight_exps_pickscore/h200_8gpu/sd35_pickscore_h200_8gpu_nft_ours-4selection-soft-boost-only-${EXPERIMENT_TAG}-KL1e-4-beta${EXPERIMENT_BETA}-fulltime}"
RUN_NAME="${RUN_NAME:-sd35_pickscore_h200_8gpu_nft_ours-4selection-soft-boost-only-${EXPERIMENT_TAG}-KL1e-4-beta${EXPERIMENT_BETA}-fulltime}"

mkdir -p "${LOGDIR}" "${SAVE_DIR}" "${REPO_DIR}/.cache"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29519}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

export REWARD_CKPTS_DIR="${REWARD_CKPTS}"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"

echo "Launching PickScore H200 soft ${EXPERIMENT_TAG}: 1x${NPROC_PER_NODE} GPUs, per-device batch=${PER_DEVICE_BATCH}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, effective batch=${EFFECTIVE_BATCH}, scheme=${MASS_SHIFT_SCHEME}, rho=${MASS_SHIFT_RHO}, beta=${EXPERIMENT_BETA}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" \
  --config=config/nft.py:sd3_pickscore \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOGDIR}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.train_batch_size="${PER_DEVICE_BATCH}" \
  --config.sample.num_batches_per_epoch="${GRADIENT_ACCUMULATION_STEPS}" \
  --config.train.batch_size="${PER_DEVICE_BATCH}" \
  --config.train.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --config.train.mass_shift_scheme="${MASS_SHIFT_SCHEME}" \
  --config.train.mass_shift_rho="${MASS_SHIFT_RHO}" \
  --config.beta="${EXPERIMENT_BETA}" \
  --config.train.beta=0.0001 \
  --config.train.timestep_fraction=1.0 \
  --config.train.trajectory_alpha_prediction=old_prediction \
  "$@"
