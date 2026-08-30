#!/usr/bin/env bash
set -euo pipefail

# Run a fresh, local 8-GPU PickScore trajectory for bitwise reproducibility
# checks. The pinned image and offline assets keep the software and model inputs
# fixed; RUN_ID must be unique so an old checkpoint can never be resumed.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-699983977898.dkr.ecr.us-east-1.amazonaws.com/ds-core-cn-omra-trainer@sha256:531d95485182ec361c3ee5d1a307ddfe931798e81db55725c02048d46e693b5f}"
HF_CACHE="${HF_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}"
SD3_MODEL="${SD3_MODEL:-${HF_CACHE}/hub/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80}"
REWARD_CKPTS_DIR="${REWARD_CKPTS_DIR:-${REPO_DIR}/reward_ckpts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/determinism_pickscore}"
RUN_ID="${RUN_ID:?Set RUN_ID to a fresh identifier, for example run-a or run-b}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-3}"
NUM_BATCHES_PER_EPOCH="${NUM_BATCHES_PER_EPOCH:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${NUM_BATCHES_PER_EPOCH}}"
STRICT_DETERMINISM="${STRICT_DETERMINISM:-true}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29531}"

case "${RUN_ID}" in
  *[!A-Za-z0-9._-]*|'')
    echo "RUN_ID must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
    ;;
esac

for numeric_value in \
  "${NUM_EPOCHS}" \
  "${NPROC_PER_NODE}" \
  "${PER_DEVICE_BATCH}" \
  "${NUM_BATCHES_PER_EPOCH}" \
  "${GRADIENT_ACCUMULATION_STEPS}" \
  "${MASTER_PORT}"; do
  if [[ ! "${numeric_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Epoch, process, batch, accumulation, and port values must be positive integers" >&2
    exit 2
  fi
done

if ((MASTER_PORT > 65535)); then
  echo "MASTER_PORT must be at most 65535" >&2
  exit 2
fi
if ((NPROC_PER_NODE * PER_DEVICE_BATCH != 24)); then
  echo "The harness requires exactly one 24-image prompt group per rollout batch" >&2
  exit 2
fi
if ((NUM_BATCHES_PER_EPOCH % GRADIENT_ACCUMULATION_STEPS != 0)); then
  echo "NUM_BATCHES_PER_EPOCH must be divisible by GRADIENT_ACCUMULATION_STEPS" >&2
  exit 2
fi
if [[ "${STRICT_DETERMINISM}" != "true" && "${STRICT_DETERMINISM}" != "false" ]]; then
  echo "STRICT_DETERMINISM must be true or false" >&2
  exit 2
fi

for required_path in \
  "${SD3_MODEL}/model_index.json" \
  "${REWARD_CKPTS_DIR}/pickscore/model/model.safetensors" \
  "${REWARD_CKPTS_DIR}/pickscore/processor/config.json"; do
  if [[ ! -s "${required_path}" ]]; then
    echo "Missing required local asset: ${required_path}" >&2
    exit 2
  fi
done

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

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'image=%s\n' "${IMAGE}"
  printf 'num_epochs=%s\n' "${NUM_EPOCHS}"
  printf 'num_batches_per_epoch=%s\n' "${NUM_BATCHES_PER_EPOCH}"
  printf 'gradient_accumulation_steps=%s\n' "${GRADIENT_ACCUMULATION_STEPS}"
  printf 'strict_determinism=%s\n' "${STRICT_DETERMINISM}"
  sha256sum \
    "${REPO_DIR}/config/base.py" \
    "${REPO_DIR}/config/nft.py" \
    "${REPO_DIR}/scripts/train_nft_sd3_ours-1.singleloss-alpha.py" \
    "${BASH_SOURCE[0]}"
} >"${RUN_DIR}/run_metadata.txt"

# The image defaults to OFI/EFA networking, which is unavailable on the local
# devbox. Socket/loopback changes transport only; algorithm/protocol/channel
# selection remains at the image/PyTorch defaults used by the minimal profile.
export PYTHONHASHSEED=42
export NCCL_NET=Socket
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=lo
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TOKENIZERS_PARALLELISM=false

docker run --rm --gpus all --ipc=host \
  --network=none \
  --hostname localhost \
  --shm-size=128g \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --env PYTHONHASHSEED \
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
  --volume "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  --volume "${REWARD_CKPTS_DIR}:${REWARD_CKPTS_DIR}:ro" \
  --volume "${HF_CACHE}:${HF_CACHE}:ro" \
  --volume "${SD3_MODEL}:${SD3_MODEL}:ro" \
  --env VR_DET_REPO_DIR="${REPO_DIR}" \
  --env VR_DET_MASTER_PORT="${MASTER_PORT}" \
  --env VR_DET_NPROC_PER_NODE="${NPROC_PER_NODE}" \
  --env VR_DET_SD3_MODEL="${SD3_MODEL}" \
  --env VR_DET_LOGDIR="${LOGDIR}" \
  --env VR_DET_SAVE_DIR="${SAVE_DIR}" \
  --env VR_DET_RUN_ID="${RUN_ID}" \
  --env VR_DET_NUM_EPOCHS="${NUM_EPOCHS}" \
  --env VR_DET_STRICT_DETERMINISM="${STRICT_DETERMINISM}" \
  --env VR_DET_PER_DEVICE_BATCH="${PER_DEVICE_BATCH}" \
  --env VR_DET_NUM_BATCHES_PER_EPOCH="${NUM_BATCHES_PER_EPOCH}" \
  --env VR_DET_GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
  --env VR_DET_STDOUT_LOG="${STDOUT_LOG}" \
  --workdir "${REPO_DIR}" \
  --entrypoint bash \
  "${IMAGE}" -lc '
    set -euo pipefail
    export PYTHONPATH="/app/runtime/compat_py312:${VR_DET_REPO_DIR}"
    torchrun --nnodes=1 --node_rank=0 \
      --master_addr=127.0.0.1 --master_port="${VR_DET_MASTER_PORT}" \
      --nproc_per_node="${VR_DET_NPROC_PER_NODE}" \
      scripts/train_nft_sd3_ours-1.singleloss-alpha.py \
      --config=config/nft.py:sd3_pickscore \
      --config.pretrained.model="${VR_DET_SD3_MODEL}" \
      --config.logdir="${VR_DET_LOGDIR}" \
      --config.save_dir="${VR_DET_SAVE_DIR}" \
      --config.run_name="determinism-pickscore-${VR_DET_RUN_ID}" \
      --config.num_epochs="${VR_DET_NUM_EPOCHS}" \
      --config.debug=True \
      --config.strict_determinism="${VR_DET_STRICT_DETERMINISM}" \
      --config.sample.train_batch_size="${VR_DET_PER_DEVICE_BATCH}" \
      --config.sample.test_batch_size="${VR_DET_PER_DEVICE_BATCH}" \
      --config.sample.num_batches_per_epoch="${VR_DET_NUM_BATCHES_PER_EPOCH}" \
      --config.train.batch_size="${VR_DET_PER_DEVICE_BATCH}" \
      --config.train.gradient_accumulation_steps="${VR_DET_GRADIENT_ACCUMULATION_STEPS}" \
      --config.beta=1.0 \
      --config.train.beta=0.0001 \
      --config.train.timestep_fraction=1.0 \
      --config.train.trajectory_alpha_prediction=old_prediction \
      2>&1 | tee "${VR_DET_STDOUT_LOG}"
  '
