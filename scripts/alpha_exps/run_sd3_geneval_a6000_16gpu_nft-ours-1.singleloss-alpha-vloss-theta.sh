#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"
SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"

OPENCLIP_CKPT="${REPO_DIR}/reward_ckpts/geneval/openclip/ViT-L-14-state_dict.pt"
TRAIN_SCRIPT="scripts/alpha_exps/train_nft_sd3_ours-1.singleloss-alpha-vloss.py"

LOGDIR="${LOGDIR:-${REPO_DIR}/logs/alpha_exps}"
PLATFORM_NNODES="${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-}}"
PLATFORM_NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-${SLURM_NODEID:-}}}"
NNODES="${NNODES:-${PLATFORM_NNODES:-2}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
NODE_RANK="${NODE_RANK:-${PLATFORM_NODE_RANK}}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-6}"

if ((NNODES != 2 || NPROC_PER_NODE != 8)); then
  echo "A6000 16-GPU training requires NNODES=2 and NPROC_PER_NODE=8" >&2
  exit 2
fi
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

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

RUN_INDEX="${RUN_INDEX:-1}"
GPU_TYPE="${GPU_TYPE:-A6000}"
RUN_NAME="${RUN_NAME:-sd35_geneval_nft_ours-1singleloss-vpred-alpha-theta-KL1e-4-beta1.0-fulltime_r${RUN_INDEX}_${GPU_TYPE}}"
SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/alpha_exps/${RUN_NAME}}"

mkdir -p "${LOGDIR}" "${SAVE_DIR}" "${REPO_DIR}/.cache"

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"
cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29532}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if ((NNODES > 1)) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "Set MASTER_ADDR to the hostname or IP address of node rank 0 for multi-node training" >&2
  exit 2
fi

export GENEVAL_OPENCLIP_PATH="${OPENCLIP_CKPT}"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"

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
echo "Launching Geneval vloss alpha-theta on node ${NODE_RANK}/${NNODES}: ${NNODES}x${NPROC_PER_NODE}=${WORLD_SIZE} GPUs, per-device batch=${PER_DEVICE_BATCH}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, effective batch=${EFFECTIVE_BATCH}"

torchrun "${TORCHRUN_DISTRIBUTED_ARGS[@]}" --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" \
  --config=config/nft.py:sd3_geneval \
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
  --config.train.timestep_fraction=1.0 \
  --config.train.trajectory_alpha_prediction=forward_prediction
