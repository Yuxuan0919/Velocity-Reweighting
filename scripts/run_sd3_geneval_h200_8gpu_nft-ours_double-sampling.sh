#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"

SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"

OPENCLIP_CKPT="${REPO_DIR}/reward_ckpts/geneval/openclip/ViT-L-14-state_dict.pt"
REWARD_CKPTS="${REPO_DIR}/reward_ckpts"

LOGDIR="${REPO_DIR}/logs"
SAVE_DIR="${REPO_DIR}/outputs/nft_sd3_geneval_nft_ours-KL1e-4-double-fixv-all"
RUN_NAME="sd35_geneval_h200_8gpu_nft_ours-KL1e-4-onlyt-double-sampling-fixv-all"

NPROC_PER_NODE=8

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

export GENEVAL_OPENCLIP_PATH="${OPENCLIP_CKPT}"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/train_nft_sd3_ours_double-sampling.py \
  --config=config/nft.py:sd3_geneval \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOGDIR}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.beta=1 \
  --config.train.beta=0.0001 \
  --config.train.correction_mean_mode=all
