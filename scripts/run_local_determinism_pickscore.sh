#!/usr/bin/env bash
set -euo pipefail

# Local, single-node reproducibility harness. The training configuration is
# intentionally small (one 24-image prompt group per optimizer step) so two
# 100-step runs can be compared on a devbox without changing the RL path.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-699983977898.dkr.ecr.us-east-1.amazonaws.com/ds-core-cn-omra-trainer@sha256:531d95485182ec361c3ee5d1a307ddfe931798e81db55725c02048d46e693b5f}"
HF_CACHE="${HF_CACHE:-/home/coder/.cache/huggingface}"
SD3_MODEL="${SD3_MODEL:-${HF_CACHE}/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80}"
REWARD_CKPTS_DIR="${REWARD_CKPTS_DIR:-${REPO_DIR}/reward_ckpts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/determinism_pickscore}"
RUN_ID="${RUN_ID:?Set RUN_ID to a fresh identifier, for example run-a or run-b}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

case "${RUN_ID}" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
    ;;
esac
for numeric_value in "${NUM_EPOCHS}" "${NPROC_PER_NODE}" "${PER_DEVICE_BATCH}"; do
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_EPOCHS, NPROC_PER_NODE, and PER_DEVICE_BATCH must be positive integers" >&2
    exit 2
  fi
done
if ((NPROC_PER_NODE * PER_DEVICE_BATCH != 24)); then
  echo "The local harness requires exactly one 24-image prompt group per step" >&2
  exit 2
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
SAVE_DIR="${RUN_DIR}/checkpoints"
LOGDIR="${RUN_DIR}/tensorboard"
STDOUT_LOG="${RUN_DIR}/train.log"
CACHE_DIR="${RUN_DIR}/cache"
if [[ -e "${RUN_DIR}" || -L "${RUN_DIR}" ]]; then
  echo "Refusing to reuse RUN_ID=${RUN_ID}; choose a fresh RUN_ID" >&2
  exit 2
fi
mkdir -p "${SAVE_DIR}" "${LOGDIR}" "${CACHE_DIR}/transformers" "${CACHE_DIR}/diffusers"

for required_path in \
  "${SD3_MODEL}/model_index.json" \
  "${REWARD_CKPTS_DIR}/pickscore/model/model.safetensors" \
  "${REWARD_CKPTS_DIR}/pickscore/processor/config.json"; do
  if [[ ! -s "${required_path}" ]]; then
    echo "Missing required local asset: ${required_path}" >&2
    exit 2
  fi
done

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NVLS_ENABLE=0
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=1
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=lo
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TOKENIZERS_PARALLELISM=false

docker run --rm --gpus all --ipc=host \
  --network=none \
  --shm-size=128g \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --env PYTHONHASHSEED \
  --env CUBLAS_WORKSPACE_CONFIG \
  --env NCCL_ALGO \
  --env NCCL_PROTO \
  --env NCCL_NVLS_ENABLE \
  --env NCCL_MIN_NCHANNELS \
  --env NCCL_MAX_NCHANNELS \
  --env NCCL_NET \
  --env NCCL_NET_PLUGIN \
  --env NCCL_SOCKET_IFNAME \
  --env TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
  --env TOKENIZERS_PARALLELISM \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HOME="${HF_CACHE}" \
  --env HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub" \
  --env TRANSFORMERS_CACHE="${CACHE_DIR}/transformers" \
  --env DIFFUSERS_CACHE="${CACHE_DIR}/diffusers" \
  --env REWARD_CKPTS_DIR="${REWARD_CKPTS_DIR}" \
  --volume "${REPO_DIR}:${REPO_DIR}" \
  --volume "${HF_CACHE}:${HF_CACHE}:ro" \
  --workdir "${REPO_DIR}" \
  --entrypoint bash \
  "${IMAGE}" -lc "
    set -euo pipefail
    export PYTHONPATH='${REPO_DIR}'
    torchrun --nnodes=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port=29531 \
      --nproc_per_node='${NPROC_PER_NODE}' \
      scripts/train_nft_sd3_ours-1.singleloss-alpha.py \
      --config=config/nft.py:sd3_pickscore \
      --config.pretrained.model='${SD3_MODEL}' \
      --config.logdir='${LOGDIR}' \
      --config.save_dir='${SAVE_DIR}' \
      --config.run_name='determinism-pickscore-${RUN_ID}' \
      --config.num_epochs='${NUM_EPOCHS}' \
      --config.debug=True \
      --config.strict_determinism=True \
      --config.sample.train_batch_size='${PER_DEVICE_BATCH}' \
      --config.sample.test_batch_size='${PER_DEVICE_BATCH}' \
      --config.sample.num_batches_per_epoch=1 \
      --config.train.batch_size='${PER_DEVICE_BATCH}' \
      --config.train.gradient_accumulation_steps=1 \
      --config.beta=1.0 \
      --config.train.beta=0.0001 \
      --config.train.timestep_fraction=1.0 \
      --config.train.trajectory_alpha_prediction=old_prediction \
      2>&1 | tee '${STDOUT_LOG}'
  "
