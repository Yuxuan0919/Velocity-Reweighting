#!/usr/bin/env bash
set -euo pipefail

# Single-node H200 adaptation; preserve 48 prompt groups x 24 images per round.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"
SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"
TRAIN_SCRIPT="scripts/alpha_exps/train_nft_sd3_ours-1.singleloss-alpha-vloss.py"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-6}"
case "${PER_DEVICE_BATCH}" in
  ''|*[!0-9]*) echo "PER_DEVICE_BATCH must be a positive integer" >&2; exit 2 ;;
esac
PER_DEVICE_BATCH=$((10#${PER_DEVICE_BATCH}))
if ((PER_DEVICE_BATCH < 1)); then
  echo "PER_DEVICE_BATCH must be a positive integer" >&2
  exit 2
fi

GLOBAL_MICRO_BATCH=$((8 * PER_DEVICE_BATCH))
if ((GLOBAL_MICRO_BATCH % 24 != 0 || 1152 % GLOBAL_MICRO_BATCH != 0)); then
  echo "8 * PER_DEVICE_BATCH must preserve 24-image groups and divide 1152" >&2
  exit 2
fi
NUM_BATCHES_PER_ROUND=$((1152 / GLOBAL_MICRO_BATCH))
GRADIENT_ACCUMULATION_STEPS="${NUM_BATCHES_PER_ROUND}"
EFFECTIVE_BATCH=$((GLOBAL_MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS))

RUN_INDEX="${RUN_INDEX:-1}"
GPU_TYPE="${GPU_TYPE:-H200}"
RUN_NAME="${RUN_NAME:-sd35_pickscore_nft_ours-1singleloss-vpred-alpha-old-KL1e-4-beta1.0-fulltime_r${RUN_INDEX}_${GPU_TYPE}}"
LOGDIR="${LOGDIR:-${REPO_DIR}/logs/alpha_exps}"
SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/alpha_exps/${RUN_NAME}}"
mkdir -p "${LOGDIR}" "${SAVE_DIR}" "${REPO_DIR}/.cache"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export REWARD_CKPTS_DIR="${REPO_DIR}/reward_ckpts"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"

echo "PickScore vpred alpha-old: H200 GPUs=8, per-device batch=${PER_DEVICE_BATCH}, rollout batches=${NUM_BATCHES_PER_ROUND}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, effective batch=${EFFECTIVE_BATCH}"

torchrun --standalone --nnodes=1 --nproc_per_node=8 "${TRAIN_SCRIPT}" \
  --config=config/nft.py:sd3_pickscore \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOGDIR}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.train_batch_size="${PER_DEVICE_BATCH}" \
  --config.sample.test_batch_size="${PER_DEVICE_BATCH}" \
  --config.sample.num_batches_per_epoch="${NUM_BATCHES_PER_ROUND}" \
  --config.train.batch_size="${PER_DEVICE_BATCH}" \
  --config.train.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --config.beta=1.0 \
  --config.train.beta=0.0001 \
  --config.train.timestep_fraction=1.0 \
  --config.train.trajectory_alpha_prediction=old_prediction \
  "$@"
