# Alpha-vloss 与 AWR：代码差异对比

比较基准：本地 `master`，提交 `8f58c82`。按两个文件的实际执行代码逐段比较，不把注释中的备选公式当作实现。

- **Alpha-vloss**：[train_nft_sd3_ours-1.singleloss-alpha-vloss.py](../scripts/alpha_exps/train_nft_sd3_ours-1.singleloss-alpha-vloss.py)
- **AWR baseline**：[train_nft_sd3_ours-singleloss-AWR.py](../scripts/flowawr_exps/train_nft_sd3_ours-singleloss-AWR.py)
- 辅助核对：[PerPromptStatTracker](../flow_grpo/stat_tracking.py)、[基础配置](../config/base.py)、[NFT 配置](../config/nft.py)。配置可由启动参数覆盖，不能仅凭 Python 文件推断某次实验的最终配置。

**结论：两者不是仅有 alpha 开关区别。reward 到系数的映射、MSE 外层加权、old-policy 更新调度、EMA checkpoint 语义均不同；直接比较 reward 曲线不能把差异单独归因于 alpha。**

## 1. 实际训练公式

统一记号：$v_\theta$ 为当前模型预测，$v_o$ 为 old adapter 预测，$v_{\rm ref}$ 为关闭 adapter 后的预训练模型预测；$u_i=\epsilon_i-x_i^0$，$\Delta_i=u_i-v_o$。$\operatorname{MSE}_{\rm feat}$ 表示对单样本的 latent 特征维度取均值。

两个不同的 beta 必须区分：$\beta_{\rm cov}=\texttt{config.beta}$ 是 Alpha 的目标修正强度；$\lambda_{\rm KL}=\texttt{config.train.beta}$ 是两者的 KL 正则系数。

### Alpha-vloss：线性 reward 映射 + alpha/time 加权 MSE

先由 `PerPromptStatTracker.update` 计算：

$$
a_i=\frac{R_i-\mu_{p_i}}{s+10^{-4}},\qquad
\hat a_i=\frac{\operatorname{clip}(a_i,-C,C)}{C},\qquad W_i=1+\hat a_i,
$$

其中 $C=\texttt{adv_clip_max}$。$\mu_{p_i}$ 按 decoded prompt 文本统计；`sample.global_std=True` 时 $s$ 是本轮全局 reward 标准差，否则使用对应 prompt 的标准差。以下公式取默认 `adv_mode="all"`，其他模式见下一节。

代码构造：

$$
v_i^*=v_o+\beta_{\rm cov}(W_i-1)\Delta_i,
\qquad
\alpha_i=\frac{1}{\operatorname{mean}_{\rm feat}|u_i-v_{\rm base}|+10^{-5}},
$$

$$
\mathcal L_{\rm Alpha}
=C\,\operatorname{mean}_i\left[
\alpha_i t_i\operatorname{MSE}_{\rm feat}(v_\theta-v_i^*)
\right]
+\lambda_{\rm KL}\operatorname{mean}_i\operatorname{MSE}_{\rm feat}(v_\theta-v_{\rm ref}).
$$

- $t_i=\texttt{timesteps}_i/1000$。
- `trajectory_alpha_prediction` 选择 $v_{\rm base}=v_o$ 或 $v_\theta$；基础配置默认 `old_prediction`。
- **alpha 在 MSE 外，不在目标修正内部；其实现没有组内均值归一化。**
- `alpha` 和目标均在 `torch.no_grad()` 内构造；即使 alpha 取当前预测，也不对 alpha 反传。
- **实际代码没有在 MSE 外乘 $W_i$。** `importance_weight` 仅通过 $\beta_{\rm cov}(W_i-1)$ 影响目标。
- 虽然计算了 $\bar Z_g=1+\beta_{\rm cov}\operatorname{mean}_{i\in g}(W_i-1)$，但 `prompt_normalizer` 只被读取和记录日志，**没有除进目标或 loss**。

定位：Alpha `compute_reinforced_flow_weights` 第 306 行；目标和 loss 第 1189–1271 行。

### AWR：指数 reward 映射 + 普通 MSE

直接使用原始 reward，先取本轮全部 GPU 样本的标准差作为温度：

$$
\gamma=\max(\operatorname{std}_{\rm rollout}(R),\gamma_{\min}),\qquad
A_i^{\rm raw}=\frac{\exp(R_i/\gamma)}{\operatorname{mean}_{j\in g}\exp(R_j/\gamma)}-1.
$$

实现中先减组内最大 logit 再求指数，以保持数值稳定。只有启用 flag 且 reward 配置包含 `ocr` 或 `geneval` 时，才应用 variance gate：

$$
A_i=\frac{A_i^{\rm raw}}{\max(\operatorname{std}_{\rm rollout}(A^{\rm raw}),s_{\min})};
\quad\text{否则 } A_i=A_i^{\rm raw}.
$$

$$
v_i^*=v_o+A_i\Delta_i,\qquad
\mathcal L_{\rm AWR}
=\operatorname{mean}_i\operatorname{MSE}_{\rm feat}(v_\theta-v_i^*)
+\lambda_{\rm KL}\operatorname{mean}_i\operatorname{MSE}_{\rm feat}(v_\theta-v_{\rm ref}).
$$

**没有 trajectory alpha、外乘 $t$、外乘 $C$，也不使用 `config.beta`。** 目标同样不参与反传。未启用 gate 时，组内 $A^{\rm raw}$ 之和理论上为零，范围为 $[-1,K-1]$；gate 会缩放范围。该 baseline 没有额外 clipping。

定位：AWR `compute_flowawr_advantages` 第 301 行；目标和 loss 第 1155–1173 行。

## 2. 会影响训练的差异

| 环节 | Alpha-vloss | AWR baseline | 影响 |
|---|---|---|---|
| reward 预处理 | 按 prompt 居中，再按全局或 prompt 标准差归一化 | 直接以原始 reward 构造组内指数权重 | 权重不是同一个映射 |
| 分组依据 | tracker 按 decoded prompt 文本；$\bar Z$ 按 `(rollout_batch_id, decoded prompt)` | `(rollout_batch_id, dataset index)`，强制每组恰好 $K$ 个样本 | 重复文本、跨 batch 重复 prompt 或解码截断时，统计集合可能不同 |
| tracker 的作用 | 返回值直接用于训练；关闭 tracking 会触发断言；每轮统计后 `clear()` | 仅日志，可关闭 | Alpha 依赖 tracker，AWR 不依赖其 advantage |
| clipping / 模式 | `adv_clip_max`；支持 `positive_only`、`negative_only`、`one_only`、`binary` | 不读取这些参数 | 给 AWR 设置这些参数不会产生对应裁剪 |
| 目标修正强度 | $\beta_{\rm cov}\hat a_i$ | $A_i$ | 不能用同名配置假设强度相同 |
| MSE 外层权重 | $C\alpha_i t_i$ | 1 | 即便目标一致，各样本梯度权重及相对 KL 强度仍不同 |
| 训练加噪 | $x_t=(1-t)x^0+t\epsilon$，$t=\texttt{timestep}/1000$ | 使用保存的 `scheduler.sigmas`，$x_\sigma=(1-\sigma)x^0+\sigma\epsilon$ | 当 $\sigma=t$ 时插值等价；不能无条件宣称不同。两者 MSE 外层是否乘 $t$ 的差异仍存在 |
| KL 计算精度 | 直接对模型输出相减、平方 | 显式转 FP32 后相减、平方 | 混合精度下可能有数值差异；正则参照模型相同 |
| old adapter 调度 | 每轮结束使用 `return_decay(global_step, config.decay_type)` | 每轮结束使用 $\min(0.001(\mathrm{epoch}+1),0.5)$ | 一个按 optimizer 更新数，一个按 rollout 轮数 |
| checkpoint 保存 | EMA 开启时，暂时将 EMA 参数写入 default adapter；optimizer 仍是实际训练参数的状态 | 保存实际训练参数；EMA 单独保存 | Alpha 恢复时模型与 optimizer 状态可能不对应同一参数轨迹 |

Alpha 的模式顺序是：先裁到 $[-C,C]$，再应用模式，最后除以 $C$。因此 `one_only` 的正系数为 $1/C$，`binary` 为 $\pm1/C$，并非最终直接取 1 或 ±1。

两者更新 old adapter 都采用 $\theta_o\leftarrow d\theta_o+(1-d)\theta$，但 Alpha 的 $d$ 可选：

- `decay_type=0`：$d=0$。
- `decay_type=1`：$d=\min(0.001\,\mathrm{global\_step},0.5)$。
- `decay_type=2`：前 75 个 optimizer step 为 0，之后 $d=\min(0.0075(\mathrm{global\_step}-75),0.999)$。

只有每轮恰好一个 optimizer step、计数对齐且 Alpha 使用 type 1 等条件满足时，两种调度才一致。AWR 不读取 `config.decay_type`。

## 3. 接口、数据流和日志差异

| 项目 | 具体代码差异 |
|---|---|
| 新增 flags | AWR 增加 `awr_variance_gate=True`、`awr_gamma_floor=1e-6`、`awr_variance_floor=1e-2`。PickScore-only 即使 flag 为 true 也不启用 gate |
| 启动校验 | AWR 强制 LoRA、`guidance_scale=1`、deterministic、每 prompt 至少 2 图、合法 timestep fraction 且选中至少 1 步；Alpha 主要额外校验 alpha prediction 选项。Alpha 虽无相同前置检查，训练参考分支仍不支持 full-model |
| Dataset / eval | AWR Dataset 返回 `index`，collate 返回三个对象，eval 解包时忽略 index；Alpha 返回两个对象 |
| rollout buffer | Alpha 保存 `next_timesteps`；AWR 改为 `sigmas` 并增加 `prompt_indices`。时间打乱分别同步作用于这两种字段；Alpha 的 `next_timesteps` 没有用于实际 loss |
| reward / advantage 形状 | Alpha 将 reward 在训练时间维重复，权重函数取 advantage 第一列；AWR reward 为 `[N,1]`，最终训练 advantage 为 `[N]` |
| 跨卡数据 | 两者 all-gather reward 和 batch id；AWR 增加 dataset index gather。Alpha 必须 gather/decode prompt token；AWR 仅开启 tracker 时执行后者 |
| 可恢复 checkpoint 判定 | AWR 额外要求 old adapter 和 `scaler.pt` 存在；Alpha 不要求这两项。两者加载代码仍有 old 缺失时复制 default 的 fallback，但 AWR 正常入口先受到严格校验 |
| 日志变化 | Alpha 记录 $W$、$\bar Z$、alpha、correction coefficient 的统计，以及 `x0_norm`、若干 max 和 `unweighted_policy_loss`；AWR 改为 `awr/*` 组统计、`correction_norm`、`advantage_abs_mean`、`sigma_mean`，并新增逐样本 `reward_samples/*` |
| 无效中间结果 | Alpha 的 `normalized_advantages` 返回值、`total_rollout_size`、训练 batch 内原始 `advantages` 不直接进入 loss；实际依赖的是由其生成的 $W$。`prompt_normalizer` 仅日志 |

## 4. 两者保持一致的部分

- imports、分布式初始化、seed、`DistributedKRepeatSampler` 的抽样逻辑、文本编码与 reward 调用方式一致；AWR Dataset 多返回 index 不改变 sampler 的取样逻辑。
- **两者实际 rollout 都启用 `old` adapter**，之后恢复 `default`；Alpha 中“使用 current rollout”的文字是注释掉的备选代码。
- 都以 rollout 最终 clean latent 为 $x^0$，训练时重新采 Gaussian noise；并非直接拿 rollout 中间 latent 训练。
- 都使用当前 adapter 反传、old adapter 提供目标锚点、关闭 adapter 的预训练模型提供 KL reference。
- optimizer 配置、样本/时间随机打乱框架、梯度累积、梯度裁剪、scaler 更新、指标 DDP reduction、评估 EMA 使用方式一致（相同配置前提下）。
- checkpoint 恢复主体、EMA 状态恢复、保存完成标记逻辑一致；保存参数和入口完整性校验的差异见上文。

## 5. 对实验结论的限制

如果目标是单独验证 trajectory alpha 的作用，需要固定 reward→advantage 映射、目标修正强度、时间权重、old-policy 调度和 checkpoint 起点，再仅改变 alpha。当前两个脚本同时改变多个因素，不能视为严格的单因素消融。

本报告基于完整文本 diff 和关键调用链检查；未运行 GPU 训练，也不声称两者训练效果相同。参考文档只借鉴标题、公式和问题说明的组织方式，其中 importance ratio / 双层采样的理论方案不代表这两个文件已实现。
