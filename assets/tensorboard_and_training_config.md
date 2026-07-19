# DiffusionNFT GenEval 训练日志与配置说明

本文档整理当前 GenEval 训练 run 的 TensorBoard 记录项含义，以及该 run 实际写入 TensorBoard 的训练配置参数。

数据来源：

```text
/inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/logs/sd3_geneval_h200_8gpu_2026.06.17_17.13.22/events.out.tfevents.1781716402.yuxuan-dev14--d2a88dd4fecf-vyxypl5bbr.1759658.0
```

相关代码：

```text
scripts/train_nft_sd3.py
flow_grpo/rewards.py
flow_grpo/gen_eval.py
config/base.py
config/nft.py
scripts/run_sd3_geneval_h200_8gpu.sh
```

注意：本文档中的“实际 run 配置”来自上述 TensorBoard event 文件。该 run 使用的基础模型路径是 `pretrained_models/sd3-medium`，当时它指向的是 SD3 Medium。后续启动脚本已修正为 `pretrained_models/sd3.5-medium`，指向本地 SD3.5 Medium，见文末“当前启动脚本修正”。

## TensorBoard Tags 总览

该 event 文件记录了三类 TensorBoard 内容：

```text
scalars:
  eval_reward/accuracy
  eval_reward/strict_accuracy
  eval_reward/single_object_strict_accuracy
  eval_reward/two_object_strict_accuracy
  eval_reward/counting_strict_accuracy
  eval_reward/colors_strict_accuracy
  eval_reward/position_strict_accuracy
  eval_reward/color_attr_strict_accuracy
  eval_reward/single_object_accuracy
  eval_reward/two_object_accuracy
  eval_reward/counting_accuracy
  eval_reward/colors_accuracy
  eval_reward/position_accuracy
  eval_reward/color_attr_accuracy
  eval_reward/geneval
  eval_reward/avg
  epoch
  reward/accuracy
  reward/geneval
  reward/avg
  stats/group_size
  stats/trained_prompt_num
  stats/zero_std_ratio
  stats/reward_std_mean
  stats/mean_reward_100
  stats/mean_reward_75
  stats/mean_reward_50
  stats/mean_reward_25
  stats/mean_reward_10
  step
  gradient_update_times
  inner_epoch
  kl_div
  kl_div_loss
  old_deviate
  old_deviate_max
  old_kl_div
  policy_loss
  total_loss
  unweighted_policy_loss
  x0_norm
  x0_norm_max

images:
  eval/images
  train/images

tensors/text:
  config/text_summary
  eval/images_captions/text_summary
  train/images_captions/text_summary
```

## Eval Reward 指标

`eval_reward/*` 来自 `eval_fn()`。训练每隔 `eval_freq` 个 epoch 触发一次评估。当前配置中 `eval_freq=10`，因此 eval scalar 记录在 step `0, 10, 20, ...`。

评估阶段使用 test split：

```text
dataset/geneval/test_metadata.jsonl
```

评估采样设置：

```text
num_inference_steps = config.sample.eval_num_steps = 40
guidance_scale      = config.sample.guidance_scale = 1.0
deterministic       = True
solver              = "flow"
resolution          = 512
```

评估 reward 调用：

```python
reward_fn(images, prompts, prompt_metadata, only_strict=False)
```

也就是说 eval 会同时计算 GenEval 的非 strict accuracy、strict accuracy、软分数 `score`，以及各 task group 的分项结果。

### `eval_reward/accuracy`

GenEval 非 strict 总体 accuracy。

来源：`flow_grpo/gen_eval.py` 中 `evaluate()` 的 `correct` 结果。

含义：对所有 GenEval test prompts 求平均，判断图像是否满足 metadata 中的 object/count/color/position 等要求。非 strict 逻辑对部分数量/包含关系更宽松，和原始 GenEval evaluation 的普通 accuracy 更接近。

范围：

```text
0.0 到 1.0，越高越好
```

### `eval_reward/strict_accuracy`

GenEval strict 总体 accuracy。

来源：`flow_grpo/gen_eval.py` 中 `evaluate_reward()` 的 `strict_correct` 结果。

含义：更严格地要求目标对象数量、颜色、位置等匹配。对 counting 等任务，strict 通常比非 strict 更难。

范围：

```text
0.0 到 1.0，越高越好
```

### `eval_reward/single_object_strict_accuracy`

GenEval `single_object` 子集上的 strict accuracy。

含义：单对象生成任务是否严格成功。

### `eval_reward/two_object_strict_accuracy`

GenEval `two_object` 子集上的 strict accuracy。

含义：两个对象同时出现的任务是否严格成功。

### `eval_reward/counting_strict_accuracy`

GenEval `counting` 子集上的 strict accuracy。

含义：数量要求是否严格满足。该项通常比较敏感，因为 detector 的 confidence threshold 对 counting 使用更高阈值：

```python
COUNTING_THRESHOLD = 0.9
```

### `eval_reward/colors_strict_accuracy`

GenEval `colors` 子集上的 strict accuracy。

含义：颜色类任务是否严格成功。颜色判断会先用 Mask2Former 找对象，再用 CLIP/OpenCLIP 对 crop 做颜色分类。

### `eval_reward/position_strict_accuracy`

GenEval `position` 子集上的 strict accuracy。

含义：相对位置关系任务是否严格成功，例如 left/right/above/below。

### `eval_reward/color_attr_strict_accuracy`

GenEval `color_attr` 子集上的 strict accuracy。

含义：对象与颜色属性绑定是否严格成功，例如“红色汽车”和“蓝色杯子”这种属性绑定。

### `eval_reward/single_object_accuracy`

GenEval `single_object` 子集上的非 strict accuracy。

### `eval_reward/two_object_accuracy`

GenEval `two_object` 子集上的非 strict accuracy。

### `eval_reward/counting_accuracy`

GenEval `counting` 子集上的非 strict accuracy。

### `eval_reward/colors_accuracy`

GenEval `colors` 子集上的非 strict accuracy。

### `eval_reward/position_accuracy`

GenEval `position` 子集上的非 strict accuracy。

### `eval_reward/color_attr_accuracy`

GenEval `color_attr` 子集上的非 strict accuracy。

### `eval_reward/geneval`

GenEval reward 的软分数均值。

来源：`flow_grpo/gen_eval.py` 中 `evaluate_reward()` 返回的 `score`。

该分数不是 0/1 accuracy，而是对 metadata 条件的部分满足程度做平均。例如：

```text
对象数量接近目标数量会得到部分分数
颜色/位置满足会增加分数
完全不满足则趋近 0
```

范围通常在：

```text
0.0 到 1.0，越高越好
```

在当前 `reward_fn` 只包含 `geneval: 1.0` 时，它也是训练主要使用的 reward。

### `eval_reward/avg`

多 reward 加权后的平均 eval reward。

来源：`flow_grpo/rewards.py` 的 `multi_score()`。

当前配置：

```yaml
reward_fn:
  geneval: 1.0
```

因此：

```text
eval_reward/avg == eval_reward/geneval
```

如果未来使用 multi-reward，例如 PickScore + HPSv2 + CLIPScore，则 `avg` 是各 reward 按权重合成后的结果。

## Train Reward 指标

`reward/*` 来自训练采样阶段。每个 epoch 中，当前 policy 先采样一批图像，然后对这些训练样本计算 reward，再基于 reward 计算 advantage 并训练。

训练阶段 reward 调用：

```python
reward_fn(images, prompts, prompt_metadata, only_strict=True)
```

这点与 eval 不同。训练时为了省计算，只需要 strict reward 和软 reward，不计算非 strict accuracy，因此某些 accuracy 项在训练日志中可能恒为 0 或不记录。

### `reward/accuracy`

训练采样 batch 上的非 strict accuracy 均值。

当前训练调用 `only_strict=True`，`flow_grpo/gen_eval.py` 中会跳过非 strict `evaluate()`，所以该项在当前 run 中恒为 0，不应作为训练效果判断依据。

判断训练采样质量应主要看：

```text
reward/geneval
reward/avg
stats/mean_reward_*
eval_reward/*
```

### `reward/geneval`

训练采样 batch 上的 GenEval 软 reward 均值。

来源：`evaluate_reward()` 的 `score`。

当前单 reward 配置下，这是训练最重要的即时 reward 指标。

### `reward/avg`

训练采样 batch 上的多 reward 加权均值。

当前：

```text
reward/avg == reward/geneval
```

## Per-Prompt 统计指标

这些指标来自 `PerPromptStatTracker`。DiffusionNFT/Flow-GRPO 在同一个 prompt 下生成多张图，使用同 prompt 组内的 reward 分布来估计 advantage。

当前配置：

```text
per_prompt_stat_tracking = true
sample.num_image_per_prompt = 24
sample.global_std = true
```

### `stats/group_size`

stat tracker 中每个 prompt 分组的平均样本数。

理想情况下，和 `sample.num_image_per_prompt=24` 相关。但由于分布式采样、重复 prompt、tracker 内部缓存/清空策略等，实际记录是 tracker 统计出来的平均组大小。

用途：确认每个 prompt 是否有足够多样本用于组内 advantage 计算。

### `stats/trained_prompt_num`

当前统计窗口中参与训练的 prompt 数量。

用途：确认本轮 epoch 中有多少 distinct prompts 被用于 reward normalization / advantage 估计。

### `stats/zero_std_ratio`

reward 标准差为 0 的 prompt 组比例。

计算逻辑在 `calculate_zero_std_ratio()` 中：按 prompt 分组，计算同 prompt 下 reward 的标准差，如果标准差接近 0，则计入 zero-std。

含义：

```text
接近 0：同 prompt 下样本 reward 有差异，advantage 信号较丰富
接近 1：同 prompt 下样本 reward 几乎全一样，advantage 信号消失
```

训练后期如果该项长期接近 1，通常说明：

```text
1. 图像质量/语义全部崩坏，reward 全低
2. 或者 reward 全高但没有区分度
3. 或者 sampler 输出缺乏多样性
```

结合当前 run，后期 `reward/geneval` 接近 0 且 `zero_std_ratio` 接近 1，说明是 reward 全低导致信号塌陷。

### `stats/reward_std_mean`

各 prompt 分组内 reward 标准差的平均值。

含义：

```text
越高：同 prompt 下样本质量差异越明显，advantage 区分度更强
越低：同 prompt 下 reward 趋同，训练信号变弱
```

### `stats/mean_reward_100`

每个 prompt 分组中 top 100% 样本的 reward 均值。

由于 top 100% 等价于全部样本，因此该项近似于训练 batch 的平均 reward。

### `stats/mean_reward_75`

每个 prompt 分组中 top 75% 样本的 reward 均值。

用途：观察较好样本的平均质量是否提升。

### `stats/mean_reward_50`

每个 prompt 分组中 top 50% 样本的 reward 均值。

用途：观察每个 prompt 下中上质量样本的趋势。

### `stats/mean_reward_25`

每个 prompt 分组中 top 25% 样本的 reward 均值。

用途：观察较优样本是否仍在变好。若 `reward/avg` 下降但 `mean_reward_25` 上升，说明模型仍能生成少量好样本；若两者都下降，说明整体和优质尾部都在退化。

### `stats/mean_reward_10`

每个 prompt 分组中 top 10% 样本的 reward 均值。

用途：观察 best-of-group 的上界。如果该项也快速下降，说明模型采样质量上界也在塌陷。

## 训练进度指标

### `epoch`

外层训练 epoch 编号。

在本代码中，一个 epoch 包含：

```text
1. 用当前 old policy 采样一轮图像
2. 计算 reward
3. 基于这些样本做一轮训练
4. 更新 old adapter
```

注意：TensorBoard 中 `epoch` 可能在同一个 global step 下出现两次，因为 reward 日志和 loss 日志都记录了 `epoch`。

### `step`

global training step。

代码中每完成一次 optimizer step 后：

```python
global_step += 1
```

当前配置下，每个 epoch 通常产生 1 个 optimizer step，因此 `step` 和 `epoch` 基本同步。

### `inner_epoch`

内层训练 epoch 编号。

当前配置：

```text
train.num_inner_epochs = 1
```

所以该项通常为 0。

### `gradient_update_times`

当前 outer epoch 内已经执行的 optimizer step 数量。

当前配置下通常为 1，因为每个 outer epoch 采样一批数据后只完成一次有效参数更新。

## Loss 与稳定性指标

这些指标来自训练阶段的每次 optimizer step。它们是当前判断训练是否崩溃最关键的指标。

### `x0_norm`

干净 latent `x0` 的均方范数：

```python
torch.mean(x0 ** 2)
```

这里的 `x0` 是采样过程最后得到的 clean latent：

```python
latents_clean = latents[:, -1]
```

含义：

```text
反映当前 policy 生成 clean latent 的幅值是否正常。
```

如果 `x0_norm` 持续快速增大，通常说明生成 latent 分布正在漂移或爆炸，后续图像容易出现高频 pattern、棋盘纹、纯噪声等。

### `x0_norm_max`

干净 latent `x0` 的最大平方值：

```python
torch.max(x0 ** 2)
```

含义：比 `x0_norm` 更敏感，用于捕捉局部 latent 爆点。

如果该值从几十涨到上千，通常已经是明显数值或生成分布崩坏信号。

### `old_deviate`

当前 train adapter 输出与 old adapter 输出的均方差：

```python
mean((forward_prediction - old_prediction) ** 2)
```

其中：

```text
forward_prediction: 当前正在训练的 adapter 输出
old_prediction:    old adapter 输出，用作采样 policy / 旧策略
```

含义：当前策略和旧策略的偏离程度。

### `old_deviate_max`

当前 train adapter 输出与 old adapter 输出的最大平方差：

```python
max((forward_prediction - old_prediction) ** 2)
```

用途：监控局部异常偏移。若该项出现尖峰，说明某些位置/样本/时间步的 policy 更新非常激进。

### `kl_div`

当前 train adapter 与 reference model 的均方差监控项：

```python
mean((forward_prediction - ref_forward_prediction) ** 2)
```

其中 `ref_forward_prediction` 是禁用 LoRA adapter 后的 base model 输出。

注意：代码中命名为 `kl_div`，但实现上是 velocity prediction 的 MSE 距离，不是概率分布解析 KL。

含义：当前可训练 LoRA policy 偏离原始 base model 的程度。

### `kl_div_loss`

加入 loss 的 KL/MSE 正则项。

代码：

```python
kl_div_loss = mean((forward_prediction - ref_forward_prediction) ** 2)
loss += config.train.beta * kl_div_loss
```

当前配置：

```text
train.beta = 0.0001
```

因此虽然 `kl_div_loss` 会被记录为原始 MSE 数值，但实际加到总 loss 中时乘了 `0.0001`。

### `old_kl_div`

old adapter 与 reference model 的均方差：

```python
mean((old_prediction - ref_forward_prediction) ** 2)
```

含义：旧策略相对 base model 的偏离程度。

DiffusionNFT 每个 epoch 后会用 EMA-like 方式更新 old adapter：

```python
tgt = tgt * decay + src * (1 - decay)
```

因此 `old_kl_div` 可以反映采样 policy 是否已经被训练 policy 带偏。

### `unweighted_policy_loss`

未乘 `adv_clip_max` 前的 policy loss 均值：

```python
ori_policy_loss.mean()
```

代码中：

```python
ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
```

注意这里的 `config.beta` 是顶层 `beta`，不是 `train.beta`。

当前配置：

```text
config.beta = 1.0
```

### `policy_loss`

实际 policy loss：

```python
policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()
```

当前：

```text
train.adv_clip_max = 5
```

所以通常：

```text
policy_loss ≈ 5 * unweighted_policy_loss
```

### `total_loss`

最终用于 backward 的 loss：

```python
total_loss = policy_loss + train.beta * kl_div_loss
```

当前 `train.beta=0.0001`，所以大多数情况下 `total_loss` 主要由 `policy_loss` 主导。

如果 `total_loss` 随 `x0_norm`、`kl_div` 一起持续上升，通常说明 policy 更新导致 latent/prediction 分布失控。

## Image 与 Text 日志

### `eval/images`

评估阶段记录的图像 grid。

来源：`eval_fn()` 中最后一个 eval batch 的 `images`。

用途：

```text
1. 直接观察 eval 图像质量
2. 辅助判断 reward 下降是 evaluator 问题还是生成图像真的退化
3. 观察是否出现棋盘纹、纯色、噪声、重复 pattern
```

### `eval/images_captions/text_summary`

`eval/images` 对应的文本 caption。

内容包括 prompt 以及各 eval reward 分数。

### `train/images`

训练采样阶段记录的图像 grid。

来源：每 10 个 epoch 在主进程记录最后一个 sampling batch 的图像。

用途：观察训练采样 policy 的输出质量。

### `train/images_captions/text_summary`

`train/images` 对应的文本 caption。

内容包括 prompt 以及 `avg` reward。

### `config/text_summary`

训练启动时写入 TensorBoard 的完整 config 文本。

这是确认实际 run 参数的最可靠来源，因为它记录的是训练脚本运行时真正接收到的 config，而不是后续修改过的脚本或配置文件。

## 当前 Event Run 的实际配置

下面是该 TensorBoard event 文件中记录的完整 config。

```yaml
allow_tf32: true
base_model: sd3
beta: 1.0
dataset: /inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/dataset/geneval
debug: false
decay_type: 1
eval_freq: 10
logdir: /inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/logs
mixed_precision: fp16
num_epochs: 100000
per_prompt_stat_tracking: true
pretrained:
  model: /inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/pretrained_models/sd3-medium
  revision: ''
prompt_fn: geneval
prompt_fn_kwargs: {}
resolution: 512
resume_from: ''
reward_fn:
  geneval: 1.0
run_name: sd3_geneval_h200_8gpu_2026.06.17_17.13.22
sample:
  deterministic: true
  eval_num_steps: 40
  global_std: true
  guidance_scale: 1.0
  noise_level: 0.7
  num_batches_per_epoch: 16
  num_image_per_prompt: 24
  num_steps: 10
  solver: dpm2
  test_batch_size: 14
  train_batch_size: 9
save_dir: /inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/outputs/nft_sd3_geneval
save_freq: 30
seed: 42
train:
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-08
  adam_weight_decay: 0.0001
  adv_clip_max: 5
  adv_mode: all
  batch_size: 9
  beta: 0.0001
  ema: true
  gradient_accumulation_steps: 16
  learning_rate: 0.0003
  lora_path: null
  max_grad_norm: 1.0
  num_inner_epochs: 1
  timestep_fraction: 0.99
use_lora: true
```

## 配置参数逐项解释

### General

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `allow_tf32` | `true` | 允许 Ampere/Hopper GPU 使用 TF32 matmul/cudnn，加速训练。 |
| `base_model` | `sd3` | 代码内部模型类型分支名称。这里表示使用 SD3/SD3.5 pipeline 族。 |
| `debug` | `false` | debug 模式。为 true 时会跳过 eval/save 等逻辑。 |
| `seed` | `42` | 随机种子。代码会按 rank 派生不同进程种子。 |
| `logdir` | `/inspire/.../DiffusionNFT/logs` | TensorBoard 日志根目录。 |
| `run_name` | `sd3_geneval_h200_8gpu_2026.06.17_17.13.22` | 本次 run 名称。实际日志目录为 `logdir/run_name`。 |
| `num_epochs` | `100000` | 外层 online RL epoch 数上限。 |
| `save_freq` | `30` | 每 30 个 epoch 保存一次 checkpoint。 |
| `eval_freq` | `10` | 每 10 个 epoch 做一次 eval。 |
| `mixed_precision` | `fp16` | 混合精度类型。H200 上也可考虑对照 `bf16` 做稳定性测试。 |
| `resume_from` | `''` | 空字符串表示从头训练。 |
| `use_lora` | `true` | 只训练 LoRA adapter，不全量训练 transformer。 |
| `save_dir` | `/inspire/.../DiffusionNFT/outputs/nft_sd3_geneval` | checkpoint 保存根目录。 |

### Pretrained Model

| 参数 | 当前 event run 值 | 含义 |
|---|---|---|
| `pretrained.model` | `/inspire/.../DiffusionNFT/pretrained_models/sd3-medium` | `StableDiffusion3Pipeline.from_pretrained()` 加载的基础模型路径。该 event run 中实际是 SD3 Medium。 |
| `pretrained.revision` | `''` | HF revision。当前未使用。 |

重要：官方 DiffusionNFT `sd3_geneval` 默认是：

```python
config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
```

当前启动脚本已修正为：

```text
/inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/pretrained_models/sd3.5-medium
-> /inspire/qb-ilm/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5
```

### Dataset / Prompt / Reward

| 参数 | 当前值 | 含义 |
|---|---|---|
| `dataset` | `/inspire/.../DiffusionNFT/dataset/geneval` | GenEval metadata 数据目录。 |
| `prompt_fn` | `geneval` | 使用 `GenevalPromptDataset`。 |
| `prompt_fn_kwargs` | `{}` | prompt 函数额外参数。当前为空。 |
| `reward_fn.geneval` | `1.0` | 使用 GenEval reward，权重为 1.0。 |

当前为 single reward：

```text
avg reward = geneval reward
```

### Sampling

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `sample.train_batch_size` | `9` | 每张 GPU 训练采样阶段的 prompt batch size。 |
| `sample.test_batch_size` | `14` | 每张 GPU eval 阶段 batch size。 |
| `sample.num_batches_per_epoch` | `16` | 每个 outer epoch 采样 batch 数。 |
| `sample.num_image_per_prompt` | `24` | 每个 prompt 生成图像数量，用于 per-prompt reward/advantage 估计。 |
| `sample.num_steps` | `10` | 训练采样阶段 inference steps。 |
| `sample.eval_num_steps` | `40` | eval 阶段 inference steps。 |
| `sample.guidance_scale` | `1.0` | CFG scale。1.0 表示无 CFG。 |
| `sample.noise_level` | `0.7` | 采样时传入 `pipeline_with_logprob()` 的 noise level。 |
| `sample.deterministic` | `true` | 训练采样使用 deterministic solver 设置。 |
| `sample.solver` | `dpm2` | 训练采样 solver。eval 固定使用 `"flow"`。 |
| `sample.global_std` | `true` | per-prompt stat tracker 使用 global std 相关设置。 |

在 8 GPU 下：

```text
每个 epoch 采样图像数 = train_batch_size * world_size * num_batches_per_epoch
                   = 9 * 8 * 16
                   = 1152

每个 epoch prompt 组数 = 1152 / 24
                 = 48
```

### Training

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `train.batch_size` | `9` | 每张 GPU 训练 micro-batch 的样本数。 |
| `train.gradient_accumulation_steps` | `16` | 每个 epoch 中跨 batch 的梯度累积步数。代码中还会乘以训练 timesteps。 |
| `train.learning_rate` | `0.0003` | AdamW 学习率。 |
| `train.adam_beta1` | `0.9` | AdamW beta1。 |
| `train.adam_beta2` | `0.999` | AdamW beta2。 |
| `train.adam_weight_decay` | `0.0001` | AdamW weight decay。 |
| `train.adam_epsilon` | `1.0e-08` | AdamW epsilon。 |
| `train.max_grad_norm` | `1.0` | 梯度裁剪阈值。 |
| `train.num_inner_epochs` | `1` | 每个采样 epoch 之后，对采样数据训练几轮。 |
| `train.adv_clip_max` | `5` | advantage clip 范围为 `[-5, 5]`。 |
| `train.adv_mode` | `all` | 同时使用正负 advantage。代码仅对若干特殊模式做处理，`all` 表示保持原始 clipped advantage。 |
| `train.timestep_fraction` | `0.99` | 训练使用采样 timesteps 的比例。当前 `num_steps=10`，因此 `int(10 * 0.99)=9` 个训练 timestep。 |
| `train.beta` | `0.0001` | reference model KL/MSE 正则项权重。 |
| `train.lora_path` | `null` | 从头初始化 LoRA，不加载已有 LoRA。 |
| `train.ema` | `true` | 使用 EMA wrapper，并在 eval/save 时切换到 EMA 参数。 |

有效 backward 累积步数：

```text
num_train_timesteps = int(sample.num_steps * train.timestep_fraction)
                    = int(10 * 0.99)
                    = 9

effective_grad_accum_steps = train.gradient_accumulation_steps * num_train_timesteps
                           = 16 * 9
                           = 144
```

### Top-Level DiffusionNFT 参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `beta` | `1.0` | DiffusionNFT policy loss 中 positive/implicit-negative prediction 混合公式的顶层 beta。注意这不是 `train.beta`。 |
| `decay_type` | `1` | old adapter 更新 decay schedule 类型。 |
| `resolution` | `512` | 训练和评估图像分辨率。 |
| `per_prompt_stat_tracking` | `true` | 使用 per-prompt reward normalization / advantage tracking。 |

`decay_type=1` 对应代码：

```python
flat = 0
uprate = 0.001
uphold = 0.5
decay = min(step * 0.001, 0.5)
```

old adapter 更新：

```python
old = old * decay + current * (1 - decay)
```

含义：训练早期 old adapter 更快跟随 current adapter，后期最多以 `0.5` 的保留比例平滑更新。

## 当前启动脚本修正

当前脚本：

```text
scripts/run_sd3_geneval_h200_8gpu.sh
```

已修正为使用 SD3.5 Medium：

```bash
SD3_MODEL="${REPO_DIR}/pretrained_models/sd3.5-medium"

ln -sfn "/inspire/qb-ilm/project/chineseculture/public/yuxuan/base_models/Diffusion/SD3.5" \
  "${SD3_MODEL}"
```

当前脚本启动时实际覆盖：

```bash
--config=config/nft.py:sd3_geneval
--config.pretrained.model=/inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/pretrained_models/sd3.5-medium
--config.logdir=/inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/logs
--config.save_dir=/inspire/qb-ilm/project/chineseculture/public/yuxuan/DiffusionNFT/outputs/nft_sd3_geneval
--config.run_name=sd35_geneval_h200_8gpu
```

其余训练超参数仍来自官方 `config/nft.py:sd3_geneval`。

## 读 TensorBoard 时的建议观察顺序

建议按以下顺序判断训练是否健康：

1. 看 `eval_reward/geneval` 和 `eval_reward/accuracy` 是否持续提升或至少不快速归零。
2. 看 `reward/geneval` 和 `reward/avg` 是否在训练采样中维持合理水平。
3. 看 `x0_norm` 和 `x0_norm_max` 是否快速爆炸。
4. 看 `kl_div`、`old_kl_div` 是否出现大尖峰。
5. 看 `stats/zero_std_ratio` 是否长期接近 1。
6. 最后对照 `eval/images` 和 `train/images`，确认 reward 曲线变化是否对应真实视觉质量变化。

对于当前 event run，后期 `x0_norm/x0_norm_max` 爆炸、`zero_std_ratio` 接近 1、`eval_reward/*` 归零，并且图像出现棋盘纹，说明是生成分布真实塌陷，而不是单纯 TensorBoard 或 reward logging 错误。
