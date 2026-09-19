#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"

SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"
REWARD_CKPTS="${REPO_DIR}/reward_ckpts"

LOGDIR="${REPO_DIR}/logs"
PLATFORM_NNODES="${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-}}"
PLATFORM_NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-${SLURM_NODEID:-}}}"
NNODES="${NNODES:-${PLATFORM_NNODES:-2}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
NODE_RANK="${NODE_RANK:-${PLATFORM_NODE_RANK}}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-6}"

case "${NPROC_PER_NODE}" in
  6)
    DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    ;;
  8)
    DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    ;;
  *)
    echo "NPROC_PER_NODE must be 8, or 6 for the optional 2x6 12-GPU layout" >&2
    exit 2
    ;;
esac

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
case "${WORLD_SIZE}" in
  8|12|16) ;;
  *)
    echo "Total WORLD_SIZE must be one of: 8, 12, 16" >&2
    exit 2
    ;;
esac

if [[ -z "${NODE_RANK}" ]] && ((NNODES == 1)); then
  NODE_RANK=0
fi
case "${NODE_RANK}" in
  ''|*[!0-9]*)
    echo "Unable to determine NODE_RANK for multi-node training." >&2
    echo "Expected NODE_RANK, SENSECORE_PYTORCH_NODE_RANK, RANK, or SLURM_NODEID." >&2
    exit 2
    ;;
esac
if ((NODE_RANK < 0 || NODE_RANK >= NNODES)); then
  echo "NODE_RANK=${NODE_RANK} is outside the valid range 0..$((NNODES - 1))" >&2
  exit 2
fi

case "${PER_DEVICE_BATCH}" in
  ''|*[!0-9]*)
    echo "PER_DEVICE_BATCH must be an integer from 1 to 6" >&2
    exit 2
    ;;
esac
if ((PER_DEVICE_BATCH < 1 || PER_DEVICE_BATCH > 6)); then
  echo "PER_DEVICE_BATCH must be in the validated 1..6 range" >&2
  exit 2
fi

GLOBAL_MICRO_BATCH=$((WORLD_SIZE * PER_DEVICE_BATCH))
if ((GLOBAL_MICRO_BATCH % 24 != 0 || 1152 % GLOBAL_MICRO_BATCH != 0)); then
  echo "WORLD_SIZE=${WORLD_SIZE}, PER_DEVICE_BATCH=${PER_DEVICE_BATCH} cannot preserve both 24-image prompt groups and effective batch 1152" >&2
  exit 2
fi

GRADIENT_ACCUMULATION_STEPS=$((1152 / GLOBAL_MICRO_BATCH))
EFFECTIVE_BATCH=$((GLOBAL_MICRO_BATCH * GRADIENT_ACCUMULATION_STEPS))
if [[ "${EFFECTIVE_BATCH}" -ne 1152 ]]; then
  echo "Invalid effective batch: ${EFFECTIVE_BATCH} (expected 1152)" >&2
  exit 2
fi

SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/nft_sd3_multi_reward_4090_${WORLD_SIZE}gpu_nft_ours-KL1e-4-beta1.0-STAGE1}"
RUN_NAME="${RUN_NAME:-sd35_multi_reward_4090_${WORLD_SIZE}gpu_nft_ours-KL1e-4-beta1.0-STAGE1}"

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29519}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if ((NNODES > 1)) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "Set MASTER_ADDR to the hostname or IP address of node rank 0 for multi-node training" >&2
  exit 2
fi

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

if ((NNODES == 1)); then
  TORCHRUN_DISTRIBUTED_ARGS=(--standalone --nnodes=1)
else
  TORCHRUN_DISTRIBUTED_ARGS=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi

echo "Distributed platform env: SENSECORE_PYTORCH_NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-unset}, RANK=${RANK:-unset}, SENSECORE_PYTORCH_NNODES=${SENSECORE_PYTORCH_NNODES:-unset}, platform WORLD_SIZE=${PLATFORM_NNODES:-unset}"
echo "Launching node ${NODE_RANK}/${NNODES}: ${NNODES}x${NPROC_PER_NODE}=${WORLD_SIZE} GPUs, per-device batch=${PER_DEVICE_BATCH}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, effective batch=${EFFECTIVE_BATCH}"

"${PYTHON_BIN}" -m torch.distributed.run "${TORCHRUN_DISTRIBUTED_ARGS[@]}" --nproc_per_node="${NPROC_PER_NODE}" scripts/train_nft_sd3_ours.py \
  --config=config/nft.py:sd3_multi_reward \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOGDIR}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.train_batch_size="${PER_DEVICE_BATCH}" \
  --config.sample.test_batch_size="${PER_DEVICE_BATCH}" \
  --config.sample.num_batches_per_epoch="${GRADIENT_ACCUMULATION_STEPS}" \
  --config.train.batch_size="${PER_DEVICE_BATCH}" \
  --config.train.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --config.beta=1.0 \
  --config.train.beta=0.0001 \
  --config.sample.num_steps=10
