#!/usr/bin/env bash
set -euo pipefail

# Internal implementation; launch an experiment through its numbered script.
CLIP_ROW="${1:-}"
if (($# == 0)); then
  echo "Launch the numbered A6000 16-GPU experiment script." >&2
  exit 2
fi
shift
case "${CLIP_ROW}" in
  1) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=2 ;;
  2) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=1 ;;
  3) VARIANT=two_sided; EPSILON_MINUS=0.6; EPSILON_PLUS=2 ;;
  4) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=3 ;;
  5) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=0.6 ;;
  6) VARIANT=two_sided; EPSILON_MINUS=0.3; EPSILON_PLUS=2 ;;
  7) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=5 ;;
  8) VARIANT=upper; EPSILON_MINUS=1; EPSILON_PLUS=0.3 ;;
  U) VARIANT=unclipped; EPSILON_MINUS=1; EPSILON_PLUS=2 ;;
  *) echo "Unknown shared-clipping experiment: ${CLIP_ROW}" >&2; exit 2 ;;
esac

PREVIEW_ONLY=0
EXTRA_CONFIG_ARGS=()
for argument in "$@"; do
  case "${argument}" in
    --dry-run) PREVIEW_ONLY=1 ;;
    --config.*=*) EXTRA_CONFIG_ARGS+=("${argument}") ;;
    *)
      echo "Unsupported argument: ${argument}. Use --dry-run or --config.FIELD=VALUE." >&2
      exit 2
      ;;
  esac
done

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-DiffusionNFT}"
SD3_MODEL="${SD3_MODEL:-${REPO_DIR}/pretrained_models/sd3.5-medium}"
TRAIN_SCRIPT="${REPO_DIR}/scripts/flowawr_exps/train_nft_sd3_ours-singleloss-AWR.py"
RUN_NAME="sd35_pickscore_a6000_16gpu_shared_exp_clip${CLIP_ROW}_kl1e-4"
LOGDIR="${REPO_DIR}/logs/shared_exp_clipping/${RUN_NAME}"
SAVE_DIR="${REPO_DIR}/outputs/shared_exp_clipping/${RUN_NAME}"

# The A6000 platform supplies rendezvous information to every job replica.
# Hardware topology is fixed here, with no experiment-time layout selection.
PLATFORM_NODE_RANK="${SENSECORE_PYTORCH_NODE_RANK:-${NODE_RANK:-${RANK:-${SLURM_NODEID:-}}}}"
PLATFORM_MASTER_ADDR="${MASTER_ADDR:-}"
PLATFORM_MASTER_PORT="${MASTER_PORT:-29534}"
if ((PREVIEW_ONLY)); then
  PLATFORM_NODE_RANK="${PLATFORM_NODE_RANK:-<platform-node-rank>}"
  PLATFORM_MASTER_ADDR="${PLATFORM_MASTER_ADDR:-<platform-master-address>}"
else
  case "${PLATFORM_NODE_RANK}" in
    0|1) ;;
    *)
      echo "This launcher requires an A6000 16-GPU platform job with a valid platform-provided rank (0 or 1)." >&2
      exit 2
      ;;
  esac
  case "${PLATFORM_MASTER_ADDR}" in
    ''|localhost|127.*|::1)
      echo "The A6000 16-GPU platform job must supply a shared MASTER_ADDR reachable by all GPUs." >&2
      exit 2
      ;;
  esac
  if [[ "${SENSECORE_PYTORCH_NNODES:-2}" != 2 || "${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}" != 8 ]]; then
    echo "The platform allocation does not match the fixed A6000 16-GPU experiment." >&2
    exit 2
  fi
fi

# 16 GPUs x batch 6 x 12 rollout batches = 1152 samples = 48 groups x K=24.
# Accumulation 12 across all 10 timesteps gives one optimizer update per round.
# Protocol flags follow optional overrides to keep each experiment reproducible.
COMMAND=(
  torchrun --nnodes=2 --nproc_per_node=8
  --node_rank="${PLATFORM_NODE_RANK}"
  --master_addr="${PLATFORM_MASTER_ADDR}" --master_port="${PLATFORM_MASTER_PORT}"
  "${TRAIN_SCRIPT}"
  --config="${REPO_DIR}/config/nft.py:sd3_pickscore"
  "${EXTRA_CONFIG_ARGS[@]}"
  --config.pretrained.model="${SD3_MODEL}"
  --config.logdir="${LOGDIR}"
  --config.save_dir="${SAVE_DIR}"
  --config.run_name="${RUN_NAME}"
  --config.sample.train_batch_size=6
  --config.sample.test_batch_size=6
  --config.sample.num_image_per_prompt=24
  --config.sample.num_batches_per_epoch=12
  --config.sample.num_steps=10
  --config.sample.eval_num_steps=40
  --config.train.batch_size=6
  --config.train.gradient_accumulation_steps=12
  --config.train.num_inner_epochs=1
  --config.train.timestep_fraction=1.0
  --config.train.beta=0.0001
  --awr_variance_gate=false
  --awr_gamma_floor=1e-6
  --shared_clip_variant="${VARIANT}"
  --shared_clip_epsilon_plus="${EPSILON_PLUS}"
  --shared_clip_epsilon_minus="${EPSILON_MINUS}"
)
if ((PREVIEW_ONLY)); then
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

echo "Shared exponential clipping: E${CLIP_ROW}, PickScore, A6000 16 GPUs, variant=${VARIANT}, epsilon-=${EPSILON_MINUS}, epsilon+=${EPSILON_PLUS}"
exec "${COMMAND[@]}"
