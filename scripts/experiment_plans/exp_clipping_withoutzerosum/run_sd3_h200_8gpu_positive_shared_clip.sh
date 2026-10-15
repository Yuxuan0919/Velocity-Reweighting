#!/usr/bin/env bash
set -euo pipefail

# Positive-side shared scaling; negative shifts keep c * baseline advantage.
# Training defaults match scripts/flowawr_exps/run_sd3_h200_8gpu_flowawr_kl1e-4.sh.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"
SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"
TRAIN_SCRIPT="scripts/experiment_plans/exp_clipping_withoutzerosum/train_nft_sd3_ours-singleloss-AWR-positive-shared-clip.py"

TASK="${TASK:-pickscore}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-6}"
SHARED_CLIP_C="${SHARED_CLIP_C:-1.0}"
SHARED_CLIP_EPSILON_PLUS="${SHARED_CLIP_EPSILON_PLUS:-9.0}"
KL_COEF="${KL_COEF:-0.0001}"

case "${TASK}" in
  pickscore) OPTIMIZER_STEPS_PER_ROUND=1; DEFAULT_AWR_VARIANCE_GATE=false ;;
  geneval) OPTIMIZER_STEPS_PER_ROUND=1; DEFAULT_AWR_VARIANCE_GATE=true ;;
  ocr) OPTIMIZER_STEPS_PER_ROUND=2; DEFAULT_AWR_VARIANCE_GATE=true ;;
  *) echo "TASK must be geneval, ocr, or pickscore" >&2; exit 2 ;;
esac
AWR_VARIANCE_GATE="${AWR_VARIANCE_GATE:-${DEFAULT_AWR_VARIANCE_GATE}}"
case "${PER_DEVICE_BATCH}" in
  ''|*[!0-9]*) echo "PER_DEVICE_BATCH must be a positive integer" >&2; exit 2 ;;
esac
PER_DEVICE_BATCH=$((10#${PER_DEVICE_BATCH}))
if ((PER_DEVICE_BATCH < 1)); then
  echo "PER_DEVICE_BATCH must be a positive integer" >&2
  exit 2
fi
case "${AWR_VARIANCE_GATE}" in
  true|false) ;;
  *) echo "AWR_VARIANCE_GATE must be true or false" >&2; exit 2 ;;
esac

# 48 prompt groups x 24 images = 1152 images per rollout round.
GLOBAL_BATCH=$((8 * PER_DEVICE_BATCH))
if ((GLOBAL_BATCH % 24 != 0 || 1152 % GLOBAL_BATCH != 0)); then
  echo "8 * PER_DEVICE_BATCH must preserve 24-image groups and divide 1152" >&2
  exit 2
fi
NUM_BATCHES_PER_ROUND=$((1152 / GLOBAL_BATCH))
if ((NUM_BATCHES_PER_ROUND % OPTIMIZER_STEPS_PER_ROUND != 0)); then
  echo "Rollout batch count must be divisible by optimizer steps per round" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((NUM_BATCHES_PER_ROUND / OPTIMIZER_STEPS_PER_ROUND))

RUN_NAME="${RUN_NAME:-sd35_${TASK}_h200_8gpu_positive-shared-clip_c${SHARED_CLIP_C}_epsplus${SHARED_CLIP_EPSILON_PLUS}_kl${KL_COEF}}"
LOGDIR="${LOGDIR:-${REPO_DIR}/logs/public/experiment_plans/exp_clipping_withoutzerosum}"
SAVE_DIR="${SAVE_DIR:-${REPO_DIR}/outputs/exp_clipping_withoutzerosum/${RUN_NAME}}"

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

echo "Positive shared clipping: task=${TASK}, GPUs=8, per-device batch=${PER_DEVICE_BATCH}, rollout batches=${NUM_BATCHES_PER_ROUND}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, variance gate=${AWR_VARIANCE_GATE}, c=${SHARED_CLIP_C}, epsilon_plus=${SHARED_CLIP_EPSILON_PLUS}, KL=${KL_COEF}"

torchrun --standalone --nnodes=1 --nproc_per_node=8 "${TRAIN_SCRIPT}" \
  --config="config/nft.py:sd3_${TASK}" \
  --config.pretrained.model="${SD3_MODEL}" \
  --config.logdir="${LOGDIR}" \
  --config.save_dir="${SAVE_DIR}" \
  --config.run_name="${RUN_NAME}" \
  --config.sample.train_batch_size="${PER_DEVICE_BATCH}" \
  --config.sample.test_batch_size="${PER_DEVICE_BATCH}" \
  --config.sample.num_batches_per_epoch="${NUM_BATCHES_PER_ROUND}" \
  --config.train.batch_size="${PER_DEVICE_BATCH}" \
  --config.train.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --config.train.timestep_fraction=1.0 \
  --config.train.beta="${KL_COEF}" \
  --awr_variance_gate="${AWR_VARIANCE_GATE}" \
  --shared_clip_c="${SHARED_CLIP_C}" \
  --shared_clip_epsilon_plus="${SHARED_CLIP_EPSILON_PLUS}" \
  "$@"
