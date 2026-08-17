#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"

SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"

REWARD_CKPTS="${REPO_DIR}/reward_ckpts"

LOGDIR="${REPO_DIR}/logs"
SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/nft_sd3_multi_reward_nft_ours}"
RUN_NAME="${RUN_NAME:-sd35_multi_reward_h200_8gpu_nft_ours}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p "${LOGDIR}" "${SAVE_DIR}" "${REPO_DIR}/.cache"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ROOT/CONDA_ENV (or PYTHON_BIN) to the configured DiffusionNFT environment." >&2
  exit 1
fi

for required_path in \
  "${SD3_MODEL}/model_index.json" \
  "${REWARD_CKPTS}/open_clip_pytorch_model.bin" \
  "${REWARD_CKPTS}/HPS_v2.1_compressed.pt" \
  "${REWARD_CKPTS}/pickscore/processor/config.json" \
  "${REWARD_CKPTS}/pickscore/processor/tokenizer.json" \
  "${REWARD_CKPTS}/pickscore/model/config.json" \
  "${REWARD_CKPTS}/pickscore/model/model.safetensors" \
  "${REWARD_CKPTS}/clipscore/clip-vit-large-patch14/config.json" \
  "${REWARD_CKPTS}/clipscore/clip-vit-large-patch14/preprocessor_config.json" \
  "${REWARD_CKPTS}/clipscore/clip-vit-large-patch14/model.safetensors"; do
  if [[ ! -s "${required_path}" ]]; then
    echo "Required model or reward checkpoint is missing: ${required_path}" >&2
    exit 1
  fi
done

cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29519}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"
export HPS_ROOT="${REPO_DIR}/.cache/hpsv2"
export XDG_CACHE_HOME="${REPO_DIR}/.cache"
export REWARD_CKPTS_DIR="${REWARD_CKPTS}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

"${PYTHON_BIN}" -c "import hpsv2.src.open_clip; import flow_grpo.rewards"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "Multi-reward preflight passed (PickScore + HPSv2 + CLIPScore)."
  exit 0
fi

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" scripts/train_nft_sd3_ours.py \
  --config=config/nft.py:sd3_multi_reward \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOGDIR}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}"
