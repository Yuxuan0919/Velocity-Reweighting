# DiffusionNFT 合作者快速使用说明

本文档用于指导合作者在当前项目中完成以下流程：

1. 创建并配置虚拟环境
2. 下载 `SD3.5 Medium` 权重
3. 下载 reward checkpoints
4. 运行 `ad_normalized` / `ad_unnormalized` 两个 8 卡训练脚本

下文统一用 `<repo_root>` 表示仓库根目录，也就是当前 `DiffusionNFT` 项目的根目录。

```bash
cd <repo_root>
```

可能需要手动修改 `scripts/run_sd3_geneval_h200_8gpu_ad_normalized.sh` 和 `scripts/run_sd3_geneval_h200_8gpu_ad_unnormalized.sh` 中绝对路径
- `CONDA_ROOT`
- `CONDA_ENV`
- `SD3_MODEL`

例如：

```bash
REPO_DIR=/your/path/DiffusionNFT \
CONDA_ROOT=/your/path/miniconda3 \
SD3_MODEL=/your/path/models/sd3.5-medium \
bash scripts/run_sd3_geneval_h200_8gpu_ad_normalized.sh
```

## 1. 创建虚拟环境

建议使用 `python=3.10.16`：

```bash
conda create -n DiffusionNFT python=3.10.16 -y
conda activate DiffusionNFT
```

安装 PyTorch 和项目本身：

```bash
cd <repo_root>

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

## 2. 安装 GenEval 相关依赖

本项目这两条训练脚本使用的是 `sd3_geneval` 配置，因此还需要安装 GenEval 相关依赖。

```bash
pip install -U openmim
mim install mmengine

git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout 1.x
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v
cd ..

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 2.x
pip install -e . -v
cd ..

pip install open-clip-torch clip-benchmark
```

如果 `huggingface-cli` 不存在，可以补装：

```bash
pip install -U huggingface_hub
```

## 3. 下载 SD3.5 Medium 权重

训练脚本默认读取本地模型目录：

```bash
<repo_root>/pretrained_models/sd3.5-medium
```

请先在 Hugging Face 上确认你有权限访问 `stabilityai/stable-diffusion-3.5-medium`，然后执行：

```bash
huggingface-cli login

huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
  --local-dir ./pretrained_models/sd3.5-medium
```

下载完成后，确认该目录存在。

## 4. 下载 reward checkpoints

在已经激活 `DiffusionNFT` 环境后执行：

```bash
bash scripts/download_reward_ckpts.sh
```

脚本会自动下载并生成：

- `reward_ckpts/sac+logos+ava1-l14-linearMSE.pth`
- `reward_ckpts/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth`
- `reward_ckpts/open_clip_pytorch_model.bin`
- `reward_ckpts/HPS_v2.1_compressed.pt`
- `reward_ckpts/geneval/openclip/ViT-L-14-state_dict.pt`

其中 `geneval/openclip/ViT-L-14-state_dict.pt` 会被训练脚本通过环境变量 `GENEVAL_OPENCLIP_PATH` 使用。

## 5. 运行训练脚本

### 5.1 normalized 版本

```bash
cd <repo_root>
bash scripts/run_sd3_geneval_h200_8gpu_ad_normalized.sh
```

### 5.2 unnormalized 版本

```bash
cd <repo_root>
bash scripts/run_sd3_geneval_h200_8gpu_ad_unnormalized.sh
```

## 6. 可选：指定 `BETA_A` / `BETA_D`

两个脚本都支持从环境变量读取：

- `BETA_A`
- `BETA_D`

默认值都是 `1`。例如：

```bash
BETA_A=1 BETA_D=1 bash scripts/run_sd3_geneval_h200_8gpu_ad_normalized.sh

BETA_A=1 BETA_D=1 bash scripts/run_sd3_geneval_h200_8gpu_ad_unnormalized.sh
```

也可以和路径覆盖一起用：

```bash
REPO_DIR=/your/path/DiffusionNFT \
CONDA_ROOT=/your/path/miniconda3 \
SD3_MODEL=/your/path/models/sd3.5-medium \
BETA_A=1 BETA_D=1 \
bash scripts/run_sd3_geneval_h200_8gpu_ad_unnormalized.sh
```

## 7. 输出位置

### normalized

- 日志：`logs/`
- 输出目录：`outputs/nft_sd3_geneval_ad_normalized_betaa${BETA_A}_betad${BETA_D}`

### unnormalized

- 日志：`logs/`
- 输出目录：`outputs/nft_sd3_geneval_ad_unnormalized_betaa${BETA_A}_betad${BETA_D}`

## 8. 常见检查项

如果脚本启动失败，优先检查以下内容：

1. `conda` 路径是否和脚本中的 `CONDA_ROOT` 一致
2. `SD3_MODEL` 本地目录是否存在
3. `reward_ckpts/geneval/openclip/ViT-L-14-state_dict.pt` 是否生成成功
4. 当前机器是否有 8 张可用 GPU，并且 `CUDA_VISIBLE_DEVICES` 设置正确
5. `mmcv` / `mmdetection` / `open_clip` 是否安装在当前环境里
