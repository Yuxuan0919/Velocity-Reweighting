# RL 轨迹确定性：算子消融、根因与最小修复

本文记录 2026-08-29 至 2026-08-30 在本地 8×H200 devbox 上完成的 RL
确定性实验。目标不是让两次训练“曲线接近”，而是让同一个配置的两个 fresh
run 在 100 个 optimizer step 内产生 bitwise 相同的 TensorBoard 聚合 reward，
并进一步比较所有记录到 TensorBoard 的 float32 scalar。

完整的机器可读算子消融矩阵见
[`docs/rl_determinism_results.tsv`](rl_determinism_results.tsv)。

## 结论摘要

消融将本次观测到的 run-to-run 分叉高置信定位到训练阶段的 fused
scaled-dot-product attention（SDPA）backward/update 路径，而不是异步
PickScore 打分。

在本次硬件和软件栈上，经过验证的最小修复是：

```python
torch.use_deterministic_algorithms(True, warn_only=False)
```

不需要为了得到当前验收范围内的重复性而同时：

- 把 reward 改成同步执行；
- 关闭 TF32；
- 强制 math SDPA；
- 强制 AdamW 或 `clip_grad_norm_` 使用 single-tensor 路径；
- 固定 NCCL algorithm、protocol 或 channel 数；
- 设置 `CUBLAS_WORKSPACE_CONFIG`；
- 单独设置 `torch.backends.cudnn.deterministic`。

这些选项中有些会改变数值轨迹，但“轨迹不同”不等于“同一配置重复运行不一致”。

最强的反转对照是：

1. default SDPA 下，其他保守设置全部打开，唯独关闭 PyTorch deterministic
   guard，A/B 在 step 1 分叉；
2. default SDPA 下，其他设置全部恢复默认，唯独保留这个 guard，A/B 完全一致；
3. 第二组配置在实验 worktree 中延长到两个 fresh 100-step run 后，40 个
   scalar tags、4,100 个 scalar events 全部 bitwise identical。

## 分支和代码范围

发布分支从 pull 后的最新 `origin/master` 创建，基线提交为：

```text
fc88f16e2385fc062a87cb6ea90d8fa34deac17c
```

正式 100-step profile 20 早于这个发布分支，运行在派生自
`strict-determinism-20260829@84d19c8` 的隔离实验 worktree 中；其精确运行时
文件由下文 SHA-256 标识。也就是说，40 tags / 4,100 events 应归因于那套实验
instrumentation，不能表述为 `fc88f16` 发布源码已经完成了同样 tag 数的 100-step
验收。最新发布源码另做了精确 3-step A/B 迁移 smoke，结果见下文。

正式训练代码只增加一个 opt-in 配置和一条全局 PyTorch guard：

- `config/base.py`: `strict_determinism = False`；
- `scripts/train_nft_sd3_ours-1.singleloss-alpha.py`: 当开关为 true 时调用
  `torch.use_deterministic_algorithms(True, warn_only=False)`。

默认行为保持不变。实验还提交了本地 runner、A/B validator 和全 scalar
bitwise comparator。没有修改 Docker、JobSet、训练镜像或其他实验脚本。

## 问题定义和验收标准

这里的 step 指一次 optimizer/global step，不是 diffusion timestep。

正式验收要求：

1. 两个 run 使用相同代码、镜像、模型、reward checkpoint、配置和 GPU 拓扑；
2. 两个 run 使用不同且从未存在过的输出目录，禁止自动 resume；
3. 每个 run 完成 100 个 optimizer step；
4. TensorBoard 中聚合后的 `reward/pickscore` 和 `reward/avg` 必须分别包含连续的
   step 0–99；
5. 两个 run 所有 TensorBoard scalar 的 tag、事件顺序、step 和序列化 float32
   bits 必须相同；
6. 已记录的 scalars 中不允许 NaN 或 Inf。

比较时不能用容差，也不能直接比较 event 文件 SHA。event 文件包含 wall time、
hostname 和 run identity，因此即使 scalar 完全相同，原始文件 SHA 也会不同。

## 实验环境

- GPU：8× NVIDIA H200；
- driver：`590.48.01`；
- PyTorch：`2.13.0a0+8145d630e8.nv26.06`；
- CUDA build：`13.3`；
- cuDNN：`9.23.0`；
- Docker image：
  `699983977898.dkr.ecr.us-east-1.amazonaws.com/ds-core-cn-omra-trainer@sha256:531d95485182ec361c3ee5d1a307ddfe931798e81db55725c02048d46e693b5f`；
- workload：SD3.5 Medium、LoRA、PickScore、8-rank single-node DDP；
- seed：42；
- 每个 rollout batch：8 ranks × 3 images = 24 images；
- 正式验证：每 epoch 一个 rollout batch和一次 optimizer update。

本地 devbox 没有 EFA，因此 runner 只把 NCCL transport 覆盖为 Socket/loopback；
NCCL 的数值 algorithm、protocol 和 channel 仍保持默认选择。

## 实验设计

实验分为五层。

### 1. 复现原问题

先在缩小后的本地 workload 中把所有确定性控制项保持默认，运行两个 fresh
3-step pair。两次 step 0 的 40/40 scalar 完全一致，但 step 1 首次观测到分叉，
28/40 tags 不同。这证明问题稳定且能用很短的 pair 捕获。

### 2. 单项和负控制消融

从保守 strict control 开始，分别恢复异步 reward、默认 AdamW、默认 gradient
clipping、TF32、NCCL auto、cuDNN flags 和 CUBLAS 设置，检查每个 profile 内部的
A/B 是否一致。

### 3. SDPA 和 PyTorch guard 的反转对照

- all-off + math SDPA：PASS；
- all-off + default SDPA：step 1 FAIL；
- default SDPA + 其他 guard 全开但关闭 PyTorch guard：step 1 FAIL；
- default SDPA + 只保留 PyTorch guard：PASS。

这两组对照分别隔离 SDPA backend 和 generic deterministic guard。

### 4. Dispatcher trace 和 transformer-only backend pin

全局强制 Flash 或 cuDNN 会影响 VAE 等其他 attention shape，其中部分 shape
不支持对应 backend。因此后续实验只在训练 transformer 的三个 forward 上 pin
backend，保持 rollout 和 VAE 的选择不变。

首个训练 microbatch 的 profiler trace 显示：

- default、无 guard：cuDNN SDPA forward 111 次、backward 36 次；
- default、有 guard：Flash SDPA forward 111 次、backward 36 次。

然后分别执行：

- transformer cuDNN、无 guard：step 1 FAIL；
- transformer Flash、无 guard：step 1 FAIL；
- transformer Flash、有 guard：PASS。

因此不能把结论简化成“cuDNN 坏、Flash 好”。关键是 fused SDPA backward
与 deterministic-algorithm guard 的交互。

### 5. 100-step 正式验收

把“default SDPA + 只保留 PyTorch guard”的配置延长到 100 step，顺序运行两个
fresh run。

结果：

- 40 个 scalar tags；
- 4,100 个 scalar events；
- 所有 `(tag, event order, step, float32 bits)` 一致；
- canonical all-scalar SHA-256：
  `362beda457df721e9b2c663e464e9aa7b9c002bdf2aaa6185cc472091c694ac5`；
- `reward/pickscore` SHA-256：
  `ef343c5550d3bdedc3e1c3efa21e7469d5244f72bfb09a4217f030acf0d9707c`；
- `reward/pickscore` step 0：`0.7528820037841797`，bits `3f40bce0`；
- `reward/pickscore` step 99：`0.9097382426261902`，bits `3f68e49b`；
- run A/B wall time：516.009 s / 512.220 s；
- 无 NaN、Inf 或 fatal runtime error。

这 40 个 tags 包含消融版本临时加入的 3 个 `target_importance` 诊断项。最新
`master` 没有这三个日志；发布分支的 validator 不硬编码 tag 数，而是要求两个 run
实际出现的全部 tags 和全部 events 完全一致。最小发布代码在最新基线上追加执行的
3-step smoke 得到 37 tags / 114 events bitwise identical；它用于验证代码迁移和
runner，100-step 的正式结论仍来自上述带完整诊断 instrumentation 的 pair。

## 完整实验矩阵摘要

| ID | Profile | A/B 结果 | 首次分叉 | 解释 |
|---:|---|---|---:|---|
| 00 | strict control | PASS | — | math SDPA + 全套保守 guard |
| 01 | sync reward, 2 batches | PASS | — | 真实 overlap 的同步控制组 |
| 02 | async reward, 2 batches | PASS | — | 与 01 跨 profile 也完全相同 |
| 03 | default SDPA + guards | PASS | — | 有 PyTorch guard 时稳定 |
| 04 | global Flash only | RUN_ERROR | — | VAE FP32 attention 不支持，不是确定性失败 |
| 05 | efficient SDPA + guards | PASS | — | 有 guard 时稳定 |
| 06 | global cuDNN only | RUN_ERROR | — | 某些 SD3 attention shape 不支持 |
| 07 | default AdamW | PASS | — | 改变轨迹，但同配置可重复 |
| 08 | default clip | PASS | — | 无可见 scalar 影响 |
| 09 | TF32 on | PASS | — | 改变轨迹，但同配置可重复 |
| 10 | math, no PyTorch guard | PASS | — | math 路径本身稳定 |
| 11 | no explicit cuDNN guard | PASS | — | math 下无影响 |
| 12 | NCCL auto | PASS | — | 改变轨迹，但本机同拓扑可重复 |
| 13 | math, no PyTorch/CUBLAS guards | PASS | — | 两者在 math 下均不需要 |
| 14 | minimal math | PASS | — | 只强制 math 即可通过短测 |
| 15 | original baseline | FAIL | 1 | 第一次 update 后复现分叉 |
| 16 | minimal default | FAIL | 1 | 与 14 只差 math → default |
| 17 | default, remove only PyTorch guard | FAIL | 1 | 其他 guard 无法替代它 |
| 18 | default, retain only PyTorch guard | PASS | — | 最小 profile 的 3-step 验证 |
| 19 | minimal efficient | FAIL | 1 | efficient 无 guard 也会分叉 |
| 20 | profile 18, 100 steps | PASS | — | 正式验收，40 tags/4,100 values |
| 21 | backend trace | TRACE | — | no guard→cuDNN；guard→Flash |
| 22 | transformer cuDNN, no guard | FAIL | 1 | 30/40 tags 分叉 |
| 23 | transformer Flash, no guard | FAIL | 1 | 30/40 tags 分叉 |
| 24 | transformer Flash + guard | PASS | — | pinned Flash 也需要 guard |

## 为什么证据指向 fused SDPA backward

Attention 的核心是：

```text
P = softmax(QKᵀ / sqrt(d))
O = PV
```

backward 需要计算 `dQ`、`dK` 和 `dV`。fused CUDA kernel 会把序列切成很多
tile，由多个 warp/CTA 并行产生 partial gradient。高速实现可能让多个 block
向同一个 FP32 accumulation buffer 做原子累加，或者以未固定的顺序合并 partial
results。

原子操作保证更新不会丢失，但不保证不同 block 每次以相同顺序到达。浮点加法
不满足结合律：

```text
(a + b) + c != a + (b + c)
```

因此相同输入可以产生末几位不同的 gradient。这不是随机数种子问题：seed 能固定
prompt、noise 和 sampler 使用的 RNG，却不能固定 GPU block scheduling 或浮点归约
顺序。`sample.deterministic=True` 也只约束 diffusion sampler，不约束训练 backward。

PyTorch 的 SDPA dispatch 和 Flash backward 实现会读取全局 deterministic
algorithms 设置。当前栈里，严格 guard 会排除被标记为不确定的 cuDNN candidate，
随后选择 Flash；同时要求 Flash backward 采用 deterministic reduction。相关的
上游实现可参考：

- [SDPA backend priority and deterministic filtering](https://github.com/pytorch/pytorch/blob/8145d630e8/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- [Flash attention backward deterministic flag](https://github.com/pytorch/pytorch/blob/8145d630e8/aten/src/ATen/native/transformers/cuda/attention_backward.cu)
- [PyTorch reproducibility notes](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md#cuda-scaled-dot-product-attention)

cuDNN kernel 内部是闭源的，因此“cuDNN 中具体哪一条 atomic 指令首先产生不同
bit”不是本实验直接证明的事实。实验直接观测到的是：step 0 已记录 scalars
一致、step 1 在第一次 backward/update 后分叉，且 PASS/FAIL 随 fused SDPA/全局
guard 翻转。结合 dispatcher trace 和 PyTorch 的 deterministic-backward 实现，
证据高置信指向 fused SDPA backward，但没有直接捕获全局第一颗不同的 bit。

## 为什么 step 0 相同、step 1 才分叉

每个 epoch 的执行顺序是：

```text
old adapter rollout
  → reward
  → advantage/target/loss forward
  → backward + DDP gradient reduction
  → unscale/clip/AdamW
  → current 参数更新
  → current 混入 old adapter
  → 下一轮 rollout
```

step 0 的聚合 reward 和 loss scalars 都在第一次 optimizer update 之前形成。即使它们
最终在 update 后才写入 TensorBoard，被写入的 tensor 已经 detach，因此不会显示
backward 中刚产生的 gradient 差异。

机制上，如果第一次 fused-attention backward 产生 `g + epsilon_A` 和
`g + epsilon_B`，DDP、gradient clipping 和 AdamW 就会传递这些差异并写入
参数/optimizer state。epoch 末，更新后的 current adapter 被混入 old adapter；
首步 decay 配置让 old 几乎立即继承 current。下一轮 rollout 使用可能已经不同的
policy，因此 step 1 的聚合 reward、`x0_norm`、loss 和 importance diagnostics
同时分叉。实验没有 hash step 0 图片或 latent，所以这里描述的是与观测一致的机制，
不是对未记录 tensor 逐 bit 相等的声明。

这给出了很窄的因果边界，但由于实验没有在第一次 update 内记录逐阶段 tensor hash，
尚不能指出 backward、DDP 后 gradient、clip 后 gradient 或 AdamW 后参数中哪一个 tensor
的哪一个元素最先不同。

## RL 闭环如何放大末位误差

一次很小的 gradient 差异会先进入 AdamW 的参数、一阶状态和二阶状态。下一轮
diffusion rollout 又会在多个 denoising step 反复使用不同参数，因此 latent 和图片差异
逐步积累。

随后 reward 进入组内归一化。当前 `global_std=True` 时，分子使用本 epoch 的
per-prompt mean，分母使用本 epoch 全部 reward 的 global std：

```text
A_{p,i} = (r_{p,i} - mean_p(r)) / (std_global(r) + 1e-4)
W_{p,i} = 1 + clip(A_{p,i}, -5, 5) / 5
```

一个样本 reward 改变会影响对应 prompt 的 mean 和全局 std，因此可能改变同 prompt
乃至本 epoch 其他 prompt 的 advantage。正式 single-batch、single-prompt harness
才退化为原来的 `mean(r)` / `std(r)` 简式。训练 target 和 trajectory scaling 又依赖：

```text
correction = beta * (W - 1)
target = old_x + correction * (x0 - old_x)
alpha = 1 / (mean(abs(x0 - old_x)) + 1e-5)
```

其中 `alpha` 不直接依赖 reward 或 `W`。放大由两条路径在 loss/gradient 处汇合。

最终形成正反馈：

```text
gradient 末位差
  → 参数/optimizer state 差
  → diffusion rollout / 图片差
      ├→ reward 差 → advantage/W 差 → correction/target 差 ─┐
      └→ x0/old_x 差 → alpha 差 ────────────────────────────┤
                                                           └→ 更大的 gradient 差
```

短跑中的 PickScore 差异提供了放大例子：

- pinned cuDNN/no-guard：step 1 约 `2.05e-5`，step 2 约 `4.98e-4`，约 24 倍；
- pinned Flash/no-guard：step 1 约 `1.09e-4`，step 2 约 `1.42e-3`，约 13 倍。

这不是固定增长率，只说明 on-policy 闭环能很快把低位差异变成可见轨迹差异。

## 为什么异步 reward 不是原因

异步测试特意使用两个 rollout batches 和 accumulation=2，让第一批 reward future
在第二批 rollout 期间真实运行。同步和异步 profile：

- 各自 A/B 都 bitwise identical；
- 同步与异步跨 profile 的 40 tags / 123 values 也完全相同；
- 共同 canonical SHA-256：
  `ffb6f1710f78f71041f69561231143aebac554e05464ad3548e082b58c3769d8`。

当前实现把 future 按原始样本顺序保存，并按列表顺序调用 `.result()`；所有 reward
收齐后才 collate、all-gather 和训练。因此没有完成顺序驱动的更新、样本重排或
stale-policy update。异步只改变 wall time，不改变这条 PickScore 数值路径。

这个负结论只覆盖当前 PickScore scorer 和当前并发实现；不能自动推广到内部消费共享
RNG、修改共享状态或线程不安全的其他 scorer。

## 哪些设置只改变轨迹而不制造随机性

- TF32、默认 AdamW 和 NCCL auto 在部分 profile 中改变了相对 strict control 的数值；
- 但每个 profile 自己的 A/B 仍完全一致；
- `clip_grad_norm_` foreach、显式 cuDNN flags 和 CUBLAS workspace 在相应测试中甚至
  没有改变已记录的 scalar trajectory。

所以判断确定性时必须比较“同一个 profile 的两个 fresh run”，不能拿不同 kernel 的
数值不相等当作随机性证据。

## 最小修复

配置默认关闭：

```python
config.strict_determinism = False
```

需要确定性时设置：

```bash
--config.strict_determinism=True
```

训练入口在任何模型 forward 之前执行：

```python
torch.use_deterministic_algorithms(True, warn_only=False)
```

`warn_only=False` 是有意设计：将来如果引入没有 deterministic implementation 的
operator，训练应立即失败，而不是只打一条 warning 后静默失去复现性。

## 本地复现

前置条件：

- 8 张可用 GPU；
- Docker 中已经缓存本文 pin 的镜像；
- SD3.5 Medium snapshot 位于 runner 默认 HF cache 路径，或通过 `SD3_MODEL` 覆盖；
  runner 会把覆盖目录只读挂载到容器中的同一路径，目录内 symlink 的外部目标仍须位于
  另一个已挂载根目录下；
- `reward_ckpts/pickscore/model` 和 `reward_ckpts/pickscore/processor` 可用。

运行两个 fresh 100-step trajectory 并比较所有 scalars：

```bash
VALIDATION_ID=determinism-100step-001 \
NUM_STEPS=100 \
scripts/validate_local_determinism_pickscore.sh
```

输出默认写入：

```text
outputs/determinism_pickscore_validation_runs/<VALIDATION_ID>/
```

runner 会拒绝复用已有目录，并在每个 run 下记录 pinned image 字符串、关键 harness
设置和源码 SHA；完整 resolved config 仍以 `train.log` 为准。

快速复现未修复 baseline 的 step-1 分叉：

```bash
OUTPUT_ROOT="$PWD/outputs/determinism-baseline-example" \
RUN_ID=run-a NUM_EPOCHS=3 STRICT_DETERMINISM=false \
scripts/run_local_determinism_pickscore.sh

OUTPUT_ROOT="$PWD/outputs/determinism-baseline-example" \
RUN_ID=run-b NUM_EPOCHS=3 STRICT_DETERMINISM=false \
scripts/run_local_determinism_pickscore.sh
```

然后在含 TensorBoard 的环境中运行：

```bash
python scripts/compare_tensorboard_all_scalars.py \
  outputs/determinism-baseline-example/run-a \
  outputs/determinism-baseline-example/run-b \
  --expected-reward-steps 3
```

预期输出为 `RESULT=DIVERGED ... first_step=1`。

## Provenance

正式 100-step pair 的运行时文件 SHA 为：

```text
c7644a59cdd47486bdef21d43cebf10374ffa50290deee71b13eeece37321bae  config/base.py
ff8320e26f1b295b84c8775dc67a978277d21b9751e5dc8947695a6340c77d54  config/nft.py
886420c3327b977d24b9bdced16c73cf7c9c377cbf80182c3a99de05bf0d9b2c  scripts/train_nft_sd3_ours-1.singleloss-alpha.py
bcd29ebf8d8192696ca44687d2fb3bb7cae21447af2dcfe703676196055f865c  scripts/run_local_determinism_pickscore.sh
```

这些运行时文件包含用于消融的可配置 hooks；profile 20 把除 PyTorch guard 之外的
所有保护恢复为默认。当前分支提交的是结论收敛后的最小生产实现，而不是把整套临时
instrumentation 留在训练路径中。

最新 `origin/master@fc88f16` 上的发布源码迁移 smoke 使用当前最小实现顺序执行
两个 fresh 3-step run，得到 37 tags / 114 events 全部 bitwise identical，canonical
SHA-256 为
`9285611d127e816ac85957ab04c2de3ba44ede6a149552fbbf50c3e4e0376cdc`。
它验证的是当前发布代码、compat loader 和 runner 的精确组合；正式 100-step 结论仍
以本节前面的实验运行时 SHA 为准。

该迁移 smoke 的运行时文件 SHA 为：

```text
afd3e677b2968393c8234e6ab403d8ca6ba48287643b9bd4418d0976418f1750  config/base.py
9da0573398f7f494eb22b28dbee808346697a18d59382b5cb6215da271da37de  config/nft.py
5ae7e97edec83267c5da1d50f411f41d4d2b178a400143dc49939090627891ea  scripts/train_nft_sd3_ours-1.singleloss-alpha.py
0aa9d3104816377f2e35b5f2b048ff5fd70fcd8392fe49ffc215d9405e247c82  scripts/run_local_determinism_pickscore.sh
```

backend profiler 和 transformer-only pin 使用了临时诊断 instrumentation。它们的
结果、源码 hash 和 profile metadata 已审计，但临时代码在采集后移除，未提交到本
分支，以免污染正式训练逻辑。原始 TensorBoard、stdout 和本地模型文件也没有提交；
本文和 TSV 只保留结果摘要与 digest。

## 结论边界

- 100-step 的 bitwise 结论覆盖 TensorBoard 序列化的 float32 聚合 scalars；没有比较
  24 个 per-sample reward、图片、latent 或训练中间 tensor；debug run 没有保存
  checkpoint，因此也没有比较最终参数或 optimizer tensor hash；
- 结论覆盖当前 H200×8、单机 DDP、PyTorch/CUDA/cuDNN 版本、attention shapes 和
  PickScore workload；
- 多数单项消融只跑 3 step，因为原问题稳定地在 step 1 出现；只有最终最小配置完成了
  100-step A/B；
- 切换硬件、PyTorch/CUDA/cuDNN、shape、多节点拓扑或 reward scorer 后，应重新执行
  100-step gate；
- 实验高置信定位到 fused SDPA backward，但没有记录第一处不同的 gradient/parameter
  tensor hash，因此不声称已经定位到某条具体 CUDA atomic 指令；
- `torch.use_deterministic_algorithms` 不保证跨硬件或跨库版本得到同一组 bits，它保证的
  是受支持 operator 在同一执行栈中选择 deterministic algorithm，或者 fail fast。
