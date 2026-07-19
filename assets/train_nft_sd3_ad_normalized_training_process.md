# `train_nft_sd3_ad_normalized.py` 训练流程说明

本文档结合当前代码实现，解释 `scripts/train_nft_sd3_ad_normalized.py` 的完整训练流程，并重点梳理：

- 训练是如何从 prompt 走到 loss 的
- 当前代码里各种 `batch` 的具体含义
- 当前实现里 `M_B` 到底对应哪一批样本

本文档描述的是当前仓库中的实际实现，不是纯论文伪代码。

## 1. 相关文件

- 主训练脚本：`scripts/train_nft_sd3_ad_normalized.py`
- 配置入口：`config/nft.py`
- 基础训练配置：`config/base.py`
- 当前 normalized target 文档：`assets/normalized_target.tex`

## 2. 先看默认配置

以 `config/nft.py` 中默认的 SD3 8 卡配置为例：

- `config.sample.num_steps = 10`
- `config.train.timestep_fraction = 0.99`
- `config.sample.num_image_per_prompt = 24`
- `config.sample.train_batch_size = 9`
- `config.train.batch_size = 9`
- `config.train.gradient_accumulation_steps = 16`
- `config.beta_a = 1.0`
- `config.beta_d = 1.0`
- `config.train.beta = 1e-4`

对应代码位置：

- `config/nft.py` 第 18-45 行、第 60-67 行
- `config/base.py` 第 75-92 行

由这些默认值可以推出一些很重要的数字：

1. `num_train_timesteps = int(10 * 0.99) = 9`
2. 每次采样时，全局样本数是 `world_size * sample.train_batch_size = 8 * 9 = 72`
3. 因为 `num_image_per_prompt = 24`，所以每次全局采样 batch 中有 `72 / 24 = 3` 个唯一 prompt
4. `sample.num_batches_per_epoch = 16`，所以每个 outer epoch 全局一共采样 `72 * 16 = 1152` 张图
5. 每卡在一个 epoch 内累计 `9 * 16 = 144` 个训练样本
6. `effective_grad_accum_steps = gradient_accumulation_steps * num_train_timesteps = 16 * 9 = 144`

这意味着在默认配置下，`num_inner_epochs = 1` 时，一个 outer epoch 通常只会做一次真正的 `optimizer.step()`。

## 3. 三个策略角色

脚本里同时维护了三个“策略角色”：

1. `default` adapter
   当前正在训练的 policy，也就是 `v_theta`
2. `old` adapter
   用来采样数据的旧策略，也就是 `v_old`
3. `reference` model
   关闭 LoRA adapter 后的 base model，用来做 reference regularization，也就是 `v_ref`

对应代码位置：

- `scripts/train_nft_sd3_ad_normalized.py` 第 434-460 行

代码里不是复制三份完整 SD3，而是在同一个 base transformer 上挂了多个 LoRA adapter：

- `default`：可训练
- `old`：作为旧策略
- `disable_adapter()`：作为 reference

## 4. 数据集和采样器

### 4.1 数据集

脚本根据 `config.prompt_fn` 选择两种数据集：

- `TextPromptDataset`：读取 `train.txt` / `test.txt`
- `GenevalPromptDataset`：读取 `train_metadata.jsonl` / `test_metadata.jsonl`

对应代码位置：

- `scripts/train_nft_sd3_ad_normalized.py` 第 78-114 行
- 第 477-485 行

### 4.2 `DistributedKRepeatSampler`

训练采样器是 `DistributedKRepeatSampler`，它的作用不是普通地随机抽 prompt，而是：

1. 先决定这次全局 batch 需要多少个样本：`total_samples = num_replicas * batch_size`
2. 再根据 `k = num_image_per_prompt` 计算需要多少个唯一 prompt：`m = total_samples / k`
3. 抽取 `m` 个唯一 prompt
4. 把每个 prompt 重复 `k` 次
5. 再打乱并切分到各张卡

对应代码位置：

- `scripts/train_nft_sd3_ad_normalized.py` 第 117-151 行

在默认 8 卡配置下：

- `num_replicas = 8`
- `batch_size = 9`
- `k = 24`
- `total_samples = 72`
- `m = 72 / 24 = 3`

所以每次采样时，全局一共只会抽 3 个唯一 prompt，但每个 prompt 会生成 24 张图。这样做是为了后续做 per-prompt reward normalization。

## 5. 初始化阶段

`main()` 的前半部分完成了这些事：

1. 初始化分布式环境
2. 设置随机种子
3. 根据 `mixed_precision` 决定 AMP / `GradScaler`
4. 加载 SD3 pipeline
5. 冻结 VAE 和 text encoders
6. 构造 LoRA adapter 和 DDP
7. 创建 optimizer
8. 构造 train/test dataloader
9. 创建 reward function、EMA、TensorBoard writer

对应代码位置：

- `scripts/train_nft_sd3_ad_normalized.py` 第 370-588 行

一个重要细节：

- 训练开始前，会把 `default` adapter 的参数复制给 `old` adapter

对应代码位置：

- 第 591-595 行

这意味着第一个 epoch 采样时，`old` 和 `current` 一开始是相同的。

## 6. Outer Epoch 的整体结构

每个 outer epoch 分成两个大阶段：

1. `SAMPLING`
2. `TRAINING`

对应代码位置：

- `scripts/train_nft_sd3_ad_normalized.py` 第 597-1062 行

高层伪流程可以写成：

```text
for epoch:
    用 old policy 采样一整轮数据
    计算 reward 和 advantage
    用 current policy 在 forward noising 后的样本上训练
    用 soft update 更新 old policy
```

## 7. Sampling 阶段

### 7.1 一个 epoch 内会采样多少次

脚本会循环 `config.sample.num_batches_per_epoch` 次，每次都采一个全局 sampling batch：

- 代码位置：第 605-610 行

默认配置下，这个值是 16。

### 7.2 每次采样的具体步骤

每次采样循环中，代码会做：

1. 取出一批 prompt：`prompts, prompt_metadata = next(train_iter)`
2. 算文本 embedding：`prompt_embeds`, `pooled_prompt_embeds`
3. 必要时先做 eval / save checkpoint
4. 切换到 `old` adapter
5. 用 `pipeline_with_logprob()` 采样图片和 latent 轨迹
6. 保存最后一步 clean latent：`latents[:, -1]`
7. 保存 scheduler 的 `timesteps`
8. 异步提交 reward 计算
9. 把这些信息暂存到 `samples_data_list`

对应代码位置：

- `scripts/train_nft_sd3_ad_normalized.py` 第 615-693 行

这里最关键的是第 656-675 行：采样时明确使用的是 `old` adapter，而不是当前正在训练的 `default` adapter。

### 7.3 存下来的训练样本长什么样

每次采样后，脚本存下来的字段包括：

- `prompt_ids`
- `prompt_embeds`
- `pooled_prompt_embeds`
- `timesteps`
- `next_timesteps`
- `latents_clean`
- `rewards_future`

其中：

- `latents_clean = latents[:, -1]`
- `timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1)`

也就是说，训练时并不保存完整 latent trajectory 作为监督目标，只保存最终 clean latent 和采样时用过的 timestep 序列。

注意：

- 当前脚本里 `next_timesteps` 被保存了，但在当前 loss 里实际上没有用到

## 8. Reward 和 Advantage 阶段

### 8.1 等待 reward 返回

采样全部结束后，脚本会遍历 `samples_data_list`，等待每个异步 reward future 完成：

- 代码位置：第 695-700 行

训练阶段用的是：

- `reward_fn(..., only_strict=True)`

而评估阶段用的是：

- `eval_reward_fn(..., only_strict=False)`

### 8.2 把所有采样 batch 拼起来

一个 epoch 内 16 次采样的结果会在本卡上拼接成 `collated_samples`：

- 代码位置：第 702-710 行

默认配置下：

- 单卡每次采样 9 个样本
- 一个 epoch 采样 16 次
- 所以本卡 `collated_samples` 中大约有 `9 * 16 = 144` 个样本

### 8.3 reward 扩展到 timestep 维

当前实现中，一张图只有一个最终 reward，但训练 loss 是按 timestep 循环计算的。所以代码会把每张图的平均 reward 复制到每个训练 timestep：

```python
collated_samples["rewards"]["avg"] = (
    collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
)
```

对应代码位置：

- 第 723-725 行

注意：

- 这里复制的是 `num_train_timesteps`
- 默认是 9，不是 `sample.num_steps = 10`

### 8.4 计算 advantage

脚本会先把各卡上的 reward gather 到所有进程：

- 代码位置：第 727-730 行

然后用 `PerPromptStatTracker` 做 per-prompt normalization：

- 代码位置：第 746-752 行

这一步的逻辑是：

1. gather 全局 `prompt_ids`
2. decode 回 prompt 文本
3. 对同一 prompt 下的多张图 reward 做统计归一化
4. 得到 `advantages`

最后再把全局 `advantages` reshape 回每张卡：

- 代码位置：第 776-784 行

因此当前 `advantages` 的语义是：

- 针对同一 prompt 的相对好坏分数
- 不是全局绝对 reward

## 9. Training 阶段前的数据重排

### 9.1 样本维 shuffle

进入训练阶段后，代码先对本卡的 `filtered_samples` 按样本维做一次随机打乱：

- 代码位置：第 809-811 行

### 9.2 timestep 维独立 shuffle

然后对每个样本自己的 timestep 序列再独立打乱：

- 代码位置：第 813-819 行

这一步非常关键，因为它意味着：

- 不同样本在相同的 `j_idx` 位置上，未必对应相同的真实离散 timestep
- 所以后续一个 `M_B` 内部的样本，并不是“同一个真实 timestep 的样本集合”

### 9.3 `num_train_timesteps = 9` 的真实含义

虽然采样时 `sample.num_steps = 10`，但训练时：

```python
num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
```

默认是 `int(10 * 0.99) = 9`。

对应代码位置：

- 第 584 行

当前实现不是先裁掉某个固定 timestep，而是：

1. 每个样本先把 10 个 timestep 独立打乱
2. 然后训练循环只走 `j_idx = 0..8`

对应代码位置：

- 第 856-863 行

所以每个样本在每个 inner epoch 中都会随机丢掉 1 个 timestep。

## 10. 各种 batch 的概念

这是最容易混淆的部分。

### 10.1 `sampling batch per device`

这是单卡 dataloader 每次采样拿到的 prompt 数：

- 大小：`config.sample.train_batch_size`
- 默认：9

### 10.2 `global sampling batch`

这是一次采样时，所有卡合起来的总样本数：

- 大小：`world_size * config.sample.train_batch_size`
- 默认：`8 * 9 = 72`

### 10.3 `prompt group`

这是 sampler 保证的重复组：

- 每个唯一 prompt 会重复 `config.sample.num_image_per_prompt = 24` 次
- 默认每次全局 sampling batch 中有 3 个 prompt group

### 10.4 `collated_samples`

这是一个 outer epoch 内，本卡采样得到的全部训练样本池：

- 默认每卡大小：`9 * 16 = 144`

### 10.5 `train_sample_batch`

这是训练阶段切出来的单卡 micro-batch：

- `training_batch_size = total_batch_size_filtered // num_batches`
- 默认：`144 // 16 = 9`

对应代码位置：

- 第 821-830 行

### 10.6 `distributed micro-batch`

这是所有卡在同一个训练子步上共同参与的一批样本：

- 大小：`world_size * current_micro_batch_size`
- 默认：`8 * 9 = 72`

### 10.7 当前 `M_B` 对应的 batch

当前代码里的 `M_B` 是定义在：

- 当前 `train_sample_batch`
- 当前 `j_idx`
- 当前所有 GPU 汇总后的 distributed micro-batch

对应代码位置：

- 第 942-949 行

也就是说，一个 `M_B` 不是：

- 不是整轮 epoch 的所有样本
- 不是一个 prompt group 的 24 个样本
- 不是同一真实 timestep 的纯净分组

它是：

- 当前训练子步里参与这次 loss 计算的那一批样本的总 mass

## 11. 单个训练子步在做什么

假设我们固定一个 `train_sample_batch` 和一个 `j_idx`，当前代码做的事如下。

### 11.1 取 clean latent 和当前 timestep

```python
x0 = train_sample_batch["latents_clean"]
t = train_sample_batch["timesteps"][:, j_idx] / 1000.0
```

对应代码位置：

- 第 865-869 行

### 11.2 forward noising

```python
xt = (1 - t) * x0 + t * noise
```

对应代码位置：

- 第 871-873 行

这一步把最终 clean latent `x0` 重新加噪到 `xt`。

### 11.3 计算三个 velocity prediction

在同一个 `xt` 上，分别算：

1. `old_prediction`
2. `forward_prediction`
3. `ref_forward_prediction`

对应代码位置：

- 第 875-909 行

### 11.4 把 advantage 变成 soft add/delete score

代码先裁剪 advantage：

```python
advantages_clip = clamp(advantages, -A, A)
```

然后映射到：

```python
r_plus = clamp((advantages_clip / A) / 2 + 0.5, 0, 1)
r_minus = 1 - r_plus
```

对应代码位置：

- 第 913-935 行

这里 `A = config.train.adv_clip_max`，默认是 5。

### 11.5 构造 signed editing mass

```python
signed_edit_mass = beta_add * r_plus - beta_del * r_minus
```

对应代码位置：

- 第 942 行

默认 `beta_add = beta_del = 1` 时：

```text
signed_edit_mass = r_plus - (1 - r_plus) = 2 * r_plus - 1
```

### 11.6 当前代码里的 `M_B`

当前代码不是直接用 `1 + sum_i w_i`，而是：

1. 先在当前卡上求 `signed_edit_mass.sum()`
2. 再统计当前卡样本数 `signed_edit_mass.numel()`
3. 用 `all_reduce` 汇总全局 sum 和全局 count
4. 得到全局 `batch_signed_mass_mean`
5. 最后定义

```python
batch_total_mass = 1.0 + batch_signed_mass_mean
```

对应代码位置：

- 第 942-949 行

所以**当前代码中的 `M_B` 实际上是**

```text
M_B = 1 + 当前 distributed micro-batch 上的平均 signed mass
```

而不是严格的

```text
1 + 当前 batch 上的 signed mass 总和
```

这是当前实现非常重要的一点。

### 11.7 构造 normalized target

先得到每个样本的归一化系数：

```python
normalized_edit_coeff = signed_edit_mass / batch_total_mass
```

然后在 `x` 空间构造目标：

```python
x0_old_prediction = xt - t * old_prediction
x0_prediction = xt - t * forward_prediction
x0_add_delete_target = x0_old_prediction + normalized_edit_coeff * (x0 - x0_old_prediction)
```

对应代码位置：

- 第 954-964 行

这可以理解为：

- 先从 `old` policy 的 `x0` 预测出发
- 再按当前样本的 normalized mass 沿着 `(x0 - x0_old_prediction)` 方向编辑 target

### 11.8 计算 policy loss

当前实现使用 stop-gradient 的自适应归一化项：

```python
weight_factor = sg(mean(abs(x0_prediction - x0_add_delete_target)))
policy_loss_i = mean((x0_prediction - x0_add_delete_target)^2 / weight_factor)
policy_loss = mean(policy_loss_i * adv_clip_max)
```

对应代码位置：

- 第 965-976 行

注意这里：

- `policy_loss` 在第 971-973 行先是逐样本 loss 向量
- 第 976 行才在 batch 上取均值，并乘回外层 `adv_clip_max`

### 11.9 加 reference regularization

代码另外加了一个 reference anchor：

```python
kl_div_loss = mean((forward_prediction - ref_forward_prediction)^2)
loss = policy_loss + config.train.beta * kl_div_loss
```

对应代码位置：

- 第 987-993 行

这里 `config.train.beta` 默认是 `1e-4`，所以这项通常比较弱。

## 12. 梯度累计和参数更新

### 12.1 为什么一个 epoch 只更新一次

脚本每个 timestep 子步都会 `backward()`，但每次都会先除以：

```python
effective_grad_accum_steps = gradient_accumulation_steps * num_train_timesteps
```

对应代码位置：

- 第 803-808 行
- 第 1003-1008 行

默认值是：

- `gradient_accumulation_steps = 16`
- `num_train_timesteps = 9`
- 所以 `effective_grad_accum_steps = 144`

而当前一个 inner epoch 内，正好会有：

- 16 个 `train_sample_batch`
- 每个 batch 9 个 `j_idx`
- 总共 `16 * 9 = 144` 次 backward

所以默认配置下，一个 inner epoch 结束时恰好触发一次：

- `optimizer.step()`
- `scaler.update()`
- `optimizer.zero_grad()`

对应代码位置：

- 第 1014-1045 行

### 12.2 日志和 EMA

每次真正 `optimizer.step()` 后，代码会：

1. 汇总 `loss_terms`
2. `all_reduce` 做多卡平均
3. 写 TensorBoard
4. `global_step += 1`
5. 可选地更新 EMA

对应代码位置：

- 第 1027-1052 行

## 13. `old` policy 的 soft update

一个 outer epoch 的训练结束后，代码不会直接把 `old` 变成 `current`，而是做 soft update：

```python
old = decay * old + (1 - decay) * current
```

对应代码位置：

- 第 1057-1062 行

其中 `decay` 由 `return_decay(global_step, config.decay_type)` 给出：

- `decay_type = 1` 时：`decay = min(step * 0.001, 0.5)`

对应代码位置：

- 第 168-188 行

所以 `old` policy 会逐渐跟上 `current` policy，但始终保持一定滞后。

## 14. 一个 outer epoch 的默认数值流程

仍然以默认 8 卡配置为例，一个 outer epoch 可以理解成：

1. 采样 16 次
2. 每次全局采样 72 张图
3. 每个全局采样 batch 含 3 个 prompt group，每组 24 张图
4. 整个 epoch 全局总共得到 1152 张图
5. 每卡拿到 144 个训练样本
6. 每个样本有 10 个原始 scheduler timesteps
7. 每个 inner epoch 随机丢掉其中 1 个 timestep，只训练 9 个位置
8. 每卡把 144 个样本切成 16 个 `train_sample_batch`
9. 每个 `train_sample_batch` 大小是 9
10. 每个 `train_sample_batch` 上再循环 9 个 `j_idx`
11. 每个 `j_idx` 上，跨 8 卡一共形成 72 个样本参与当前的 distributed micro-batch
12. 当前代码的 `M_B` 就是在这 72 个样本上统计出来的平均 signed mass 归一化量
13. 一个 inner epoch 共做 144 次 backward
14. 最终做 1 次 `optimizer.step()`

## 15. 对 `M_B` 的一句话总结

当前代码里的 `M_B` 不是论文里抽象意义上的“任意一个 batch total mass”，而是更具体的：

```text
在当前训练子步中，
跨所有 GPU 的 distributed micro-batch 上，
按当前 j_idx 取到的样本集合，
先求 signed mass 的全局平均，
再加 1 得到的归一化总质量。
```

当前实现里它的公式是：

```text
M_B = 1 + mean_i[w_i]
```

而不是：

```text
M_B = 1 + sum_i[w_i]
```

这一点如果后续要和文档算法做完全一致化，需要进一步修改实现。

## 16. 读代码时最容易误解的几点

1. 采样 policy 不是当前正在训练的 `default`，而是 `old`
2. `collated_samples["rewards"]["avg"]` 会被复制到 timestep 维，但 reward 本身不是逐 timestep 重新算的
3. `j_idx` 不是“全 batch 共享的真实 timestep 编号”，因为每个样本的 timestep 序列先被独立打乱了
4. 当前 `M_B` 用的是当前 distributed micro-batch 上的平均 signed mass，而不是总和
5. 默认配置下，一个 outer epoch 通常只做一次 optimizer 更新
