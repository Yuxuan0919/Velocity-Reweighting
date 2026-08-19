#!/usr/bin/env bash
set -euo pipefail

# Experiment A path retained for comparison (this script originally lived directly under scripts/):
# REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"

SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"

OPENCLIP_CKPT="${REPO_DIR}/reward_ckpts/geneval/openclip/ViT-L-14-state_dict.pt"
REWARD_CKPTS="${REPO_DIR}/reward_ckpts"

LOGDIR="${REPO_DIR}/logs"
NUM_EPOCHS="${NUM_EPOCHS:-600}"
TRANSITION_ITERATION="${TRANSITION_ITERATION:-100}"
# Experiment A output names retained for comparison:
# SAVE_DIR="${REPO_DIR}/outputs/nft_sd3_geneval_nft_ours-KL1e-4"
# RUN_NAME="sd35_geneval_h200_8gpu_nft_ours-KL1e-4"
SAVE_DIR="${REPO_DIR}/outputs/nft_sd3_geneval_nft_ours-weight-b-dynamic-asymmetric-M${NUM_EPOCHS}-T${TRANSITION_ITERATION}-KL1e-2-beta1，0"
RUN_NAME="sd35_geneval_h200_8gpu_nft_ours-weight-b-dynamic-asymmetric-M${NUM_EPOCHS}-T${TRANSITION_ITERATION}-KL1e-2-beta1.0"

NPROC_PER_NODE=8

mkdir -p "${LOGDIR}" "${SAVE_DIR}" "${REPO_DIR}/.cache"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Experiment A port retained for comparison:
# export MASTER_PORT="${MASTER_PORT:-29519}"
export MASTER_PORT="${MASTER_PORT:-29524}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

export GENEVAL_OPENCLIP_PATH="${OPENCLIP_CKPT}"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"

# Experiment A entry point retained for comparison:
# torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/train_nft_sd3_ours.py \  # original continuation disabled
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/weight_exps/train_nft_sd3_ours_weight_b_dynamic_asymmetric.py \
  --config=config/nft.py:sd3_geneval \
  --config.num_epochs="${NUM_EPOCHS}" \
  --dynamic_asymmetric_transition_iteration="${TRANSITION_ITERATION}" \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOGDIR}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.beta=1.0 \
  --config.train.beta=0.01
