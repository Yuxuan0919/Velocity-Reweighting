#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   DRY_RUN=1 CLIP_RUN=3 TASK=geneval bash "$0"
#   CLIP_RUN=1 TASK=pickscore bash "$0"
# For 2x8 GPUs, run on both nodes with a shared MASTER_ADDR and NODE_RANK=0/1.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"
SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"
TRAIN_SCRIPT="${REPO_DIR}/scripts/flowawr_exps/train_nft_sd3_ours-singleloss-AWR.py"

TASK="${TASK:-pickscore}"
CLIP_RUN="${CLIP_RUN:-1}"
DRY_RUN="${DRY_RUN:-0}"
case "${TASK}" in
  pickscore|geneval) OPTIMIZER_STEPS_PER_ROUND=1 ;;
  ocr) OPTIMIZER_STEPS_PER_ROUND=2 ;;
  *) echo "TASK must be pickscore, geneval, or ocr" >&2; exit 2 ;;
esac
case "${CLIP_RUN}" in
  1) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=2 ;;
  2) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=1 ;;
  3) VARIANT=two_sided; EPSILON_MINUS=0.6; EPSILON_PLUS=2 ;;
  4) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=3 ;;
  5) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=0.6 ;;
  6) VARIANT=two_sided; EPSILON_MINUS=0.3; EPSILON_PLUS=2 ;;
  7) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=5 ;;
  8) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=0.3 ;;
  U) VARIANT=unclipped; EPSILON_MINUS=1; EPSILON_PLUS=2 ;;
  *) echo "CLIP_RUN must be 1..8 (document priority) or U (unclipped control)" >&2; exit 2 ;;
esac
case "${DRY_RUN}" in
  0|1) ;;
  *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

NNODES="${NNODES:-${SENSECORE_PYTORCH_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${SLURM_NODEID:-}}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29534}"
case "${NNODES}" in
  1|2) ;;
  *) echo "NNODES must be 1 or 2 (8 GPUs per node)" >&2; exit 2 ;;
esac
if [[ -z "${NODE_RANK}" ]] && ((NNODES == 1)); then
  NODE_RANK=0
fi
case "${NODE_RANK}" in
  0|1) ;;
  *) echo "Set NODE_RANK to 0 or 1 on each node" >&2; exit 2 ;;
esac
if ((NODE_RANK >= NNODES)); then
  echo "NODE_RANK must be less than NNODES" >&2
  exit 2
fi
if ((NNODES == 1)); then
  DISTRIBUTED_ARGS=(--standalone --nnodes=1)
else
  if [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
    echo "Set MASTER_ADDR to the hostname or IP address of node rank 0" >&2
    exit 2
  fi
  DISTRIBUTED_ARGS=(--nnodes=2 --node_rank="${NODE_RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
fi

# Fixed protocol: 48 complete groups x K=24 = 1152 samples per round.
NPROC_PER_NODE=8
PER_DEVICE_BATCH=6
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
NUM_BATCHES_PER_ROUND=$((1152 / (WORLD_SIZE * PER_DEVICE_BATCH)))
GRADIENT_ACCUMULATION_STEPS=$((NUM_BATCHES_PER_ROUND / OPTIMIZER_STEPS_PER_ROUND))
RUN_NAME="${RUN_NAME:-sd35_${TASK}_${WORLD_SIZE}gpu_shared_exp_clip${CLIP_RUN}_kl1e-4}"
LOGDIR="${LOGDIR:-${REPO_DIR}/logs/shared_exp_clipping/${RUN_NAME}}"
SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/shared_exp_clipping/${RUN_NAME}}"

# The fixed matrix/protocol arguments follow optional extra flags so the run
# continues to match its named matrix row. Extra flags can set e.g. num_epochs.
COMMAND=(
  torchrun "${DISTRIBUTED_ARGS[@]}" --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_SCRIPT}"
  --config="${REPO_DIR}/config/nft.py:sd3_${TASK}"
  "$@"
  --config.pretrained.model="${SD3_MODEL}"
  --config.logdir="${LOGDIR}"
  --config.save_dir="${SAVE_DIR}"
  --config.run_name="${RUN_NAME}"
  --config.sample.train_batch_size="${PER_DEVICE_BATCH}"
  --config.sample.test_batch_size="${PER_DEVICE_BATCH}"
  --config.sample.num_image_per_prompt=24
  --config.sample.num_batches_per_epoch="${NUM_BATCHES_PER_ROUND}"
  --config.train.batch_size="${PER_DEVICE_BATCH}"
  --config.train.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
  --config.train.num_inner_epochs=1
  --config.train.timestep_fraction=1.0
  --config.train.beta=0.0001
  --awr_variance_gate=false
  --awr_gamma_floor=1e-6
  --shared_clip_variant="${VARIANT}"
  --shared_clip_epsilon_plus="${EPSILON_PLUS}"
  --shared_clip_epsilon_minus="${EPSILON_MINUS}"
)
if ((DRY_RUN)); then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

source "${CONDA_ROOT}/bin/activate" "${CONDA_ENV}"
cd "${REPO_DIR}"
mkdir -p "${LOGDIR}" "${SAVE_DIR}" "${REPO_DIR}/.cache"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export REWARD_CKPTS_DIR="${REPO_DIR}/reward_ckpts"
export GENEVAL_OPENCLIP_PATH="${REPO_DIR}/reward_ckpts/geneval/openclip/ViT-L-14-state_dict.pt"
export HF_HOME="${REPO_DIR}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export DIFFUSERS_CACHE="${HF_HOME}/diffusers"

echo "Shared exponential clipping: row=${CLIP_RUN}, variant=${VARIANT}, epsilon-= ${EPSILON_MINUS}, epsilon+= ${EPSILON_PLUS}, task=${TASK}, node=${NODE_RANK}/${NNODES}, GPUs=${WORLD_SIZE}, rollout batches=${NUM_BATCHES_PER_ROUND}, accumulation=${GRADIENT_ACCUMULATION_STEPS}"
exec "${COMMAND[@]}"
