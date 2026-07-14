#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REWARD_CKPTS_DIR="${REPO_DIR}/reward_ckpts"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
export REPO_DIR

mkdir -p "${REWARD_CKPTS_DIR}"
mkdir -p "${REWARD_CKPTS_DIR}/geneval/openclip"

download_file() {
  local url="$1"
  local dst="$2"

  if [[ -f "${dst}" && "${FORCE_DOWNLOAD}" != "1" ]]; then
    echo "Skip existing file: ${dst}"
    return 0
  fi

  rm -f "${dst}"

  if command -v curl >/dev/null 2>&1; then
    echo "Downloading ${dst##*/} with curl"
    curl -L --fail --retry 3 --retry-delay 2 -o "${dst}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    echo "Downloading ${dst##*/} with wget"
    wget -O "${dst}" "${url}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

download_file \
  "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth" \
  "${REWARD_CKPTS_DIR}/sac+logos+ava1-l14-linearMSE.pth"

download_file \
  "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth" \
  "${REWARD_CKPTS_DIR}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"

download_file \
  "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin" \
  "${REWARD_CKPTS_DIR}/open_clip_pytorch_model.bin"

download_file \
  "https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt" \
  "${REWARD_CKPTS_DIR}/HPS_v2.1_compressed.pt"

GENEVAL_OPENCLIP_DST="${REWARD_CKPTS_DIR}/geneval/openclip/ViT-L-14-state_dict.pt"
if [[ ! -f "${GENEVAL_OPENCLIP_DST}" || "${FORCE_DOWNLOAD}" == "1" ]]; then
  rm -f "${GENEVAL_OPENCLIP_DST}"
  echo "Generating ${GENEVAL_OPENCLIP_DST##*/} from open_clip pretrained=openai"
  python - <<'PY'
import os
import torch
import open_clip

repo_dir = os.environ["REPO_DIR"]
dst = os.path.join(repo_dir, "reward_ckpts", "geneval", "openclip", "ViT-L-14-state_dict.pt")

model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
torch.save(model.state_dict(), dst)
print(f"Saved {dst}")
PY
else
  echo "Skip existing file: ${GENEVAL_OPENCLIP_DST}"
fi

echo
echo "Finished populating ${REWARD_CKPTS_DIR}"
find "${REWARD_CKPTS_DIR}" -maxdepth 3 -type f | sort
