# Scripts：实验目录与方法说明

本目录围绕 [RL_in_Diffusion.pdf：Reinforced Flow Matching](../assets/RL_in_Diffusion.pdf) 中的推导和实验组织代码。研究主线是：先根据 reward 改变样本权重，再用权重确定 flow matching 的目标修正，最后回归该目标。

本文按当前 Python 实现说明各文件。PDF 中的理想公式、实验设计和代码中的数值保护、采样方式并不总是完全相同，相关差异在下文单独说明。

## 目录与文档的对应关系

| 目录 | 研究内容 | 对应文档 |
| --- | --- | --- |
| [alpha_exps/](alpha_exps/) | 比较已有 loss 与推导目标之间的 gap，研究 alpha 的位置、估计来源，以及 x/v 两种参数化 | 第二章 *Gap between current implemented algorithm and our derivation*，PDF 第 3–4 页；推导来自第一章 |
| [selection_exps/](selection_exps/) | 将正向更新集中到更强的正样本，比较硬选择、并列分数边界的软分配及正负两侧选择 | 第三章 3.1 *Concentrate the confident positive updates*，算法 2，PDF 第 5–7 页 |
| [softmax_variants_exps/](softmax_variants_exps/) | 从直接选择样本推广到 reward-to-weight 映射，比较 A–F 六类映射 | 第三章 3.2 *From simple selection to reward mappings*，算法 3，PDF 第 8–13 页 |
| [flowawr_exps/](flowawr_exps/) | 在我们自己的 single-loss 代码基础上复现 FlowAWR，作为方法对照 | 独立的 FlowAWR 复现实验，与文档的增量 velocity target 主线相关 |

## 共同符号与 loss

对同一个 prompt 的样本，记 clean latent 为 $x_0$，噪声为 $\epsilon$，SD3 的线性插值和条件速度为

$$
x_t=(1-t)x_0+t\epsilon,\qquad v=\epsilon-x_0.
$$

模型输出为 $v_\theta$，旧策略输出为 $v_{\mathrm{old}}$，对应的 clean prediction 为 $x_\theta=x_t-tv_\theta$、$x_{\mathrm{old}}=x_t-tv_{\mathrm{old}}$。$w$ 表示样本权重，真正控制修正方向的是 $w-1$；$\operatorname{sg}$ 表示停止梯度，误差中的 `mean` 对 latent 特征维求平均。

single-loss 基础目标是

$$
x_{\mathrm{target}}=x_{\mathrm{old}}+\beta(w-1)(x_0-x_{\mathrm{old}}),
$$

其对应的 velocity target 为

$$
v_{\mathrm{target}}=v_{\mathrm{old}}+\beta(w-1)(v-v_{\mathrm{old}}).
$$

`config.beta` 控制目标修正强度；`config.train.beta` 是 reference velocity MSE 的系数，二者不是同一个参数。下文聚焦 policy loss，各训练文件还保留各自的 reference 正则和训练流程。

## alpha_exps：第二章的 gap

这个目录主要研究**当前实现的误差加权回归，与文档推导的带 alpha 目标修正之间的差别**。

第二章式 (14)–(15) 将基础 single-loss 写为 velocity 形式。忽略数值保护和公共倍率后：

$$
L_{\mathrm{impl}}=\mathbb E\left[
\frac{t}{\operatorname{sg}[\operatorname{mean}|v-v_{\mathrm{old}}|]}
\operatorname{mean}(v_\theta-v_{\mathrm{target}})^2\right].
$$

这里 alpha 在平方误差外部。第一章式 (12) 和第二章式 (16) 则把 alpha 放在增量目标内部：

$$
v_{\mathrm{target}}^{\mathrm{derived}}
=v_{\mathrm{old}}+\mathbb E_{z\sim\pi_o}
\left[(w-1)\alpha_t^o(v_t(x_t\mid z)-v_{\mathrm{old}})\right],
\qquad
L_{\mathrm{derived}}=\mathbb E\left[t\operatorname{mean}(v_\theta-v_{\mathrm{target}}^{\mathrm{derived}})^2\right].
$$

这两种 alpha 放置方式通常会改变目标和梯度，不能仅通过改写变量名互相替代。文档中的条件期望与密度比 alpha 也需要样本估计；代码使用误差倒数作为代理量。

| Python 文件 | 实际用途与实现 |
| --- | --- |
| [train_nft_sd3_origin.py](alpha_exps/train_nft_sd3_origin.py) | DiffusionNFT 的双分支 baseline。使用 positive prediction 和 implicit negative prediction，两支分别以自身的 clean prediction 误差做自适应归一化，再按 advantage 混合。用于与我们的 single-loss 和推导目标比较。 |
| [train_nft_sd3_ours-1.singleloss-alpha.py](alpha_exps/train_nft_sd3_ours-1.singleloss-alpha.py) | x-prediction 的 single-loss 基础实现，对应第二章算法 1 的目标形式。计算 `x_target = x_old + beta * (w - 1) * (x0 - x_old)`，alpha 在 MSE 外部，分母为 clean prediction 的平均绝对误差加 epsilon。 |
| [train_nft_sd3_ours-1.singleloss-alpha-vloss.py](alpha_exps/train_nft_sd3_ours-1.singleloss-alpha-vloss.py) | 用 v-prediction 表达基础 single-loss，直接构造 `v_target`。当前代码使用 `t * mean((v_theta - v_target)^2) / (mean(abs(v - v_base)) + epsilon)`，便于直接对照第二章式 (15)。 |
| [train_nft_sd3_ours.py](alpha_exps/train_nft_sd3_ours.py) | 研究推导侧的目标修正：用 velocity 误差倒数估计 alpha，并除以当前训练 batch 的 alpha 均值；将归一化后的 alpha 放进 `v_target` 的修正项，外部保留 `t` 权重。代码还使用 prompt normalizer，修正系数为 `beta * (w / prompt_normalizer - 1)`。这是推导方向的样本实现，不是对条件期望的精确计算。 |

`trajectory_alpha_prediction` 控制 alpha 的误差基准：`old_prediction` 使用旧模型，`forward_prediction` 使用当前模型。启动脚本中的 `alpha-old` / `alpha-theta` 对应这两个选项；即使使用当前模型，alpha 和 target 的构造也在 `torch.no_grad()` 中。

**x/v 改写中的 epsilon：**令 $m=\operatorname{mean}|v-v_{\mathrm{base}}|$。x-prediction 版本严格换元后的误差权重是 $t^2/(tm+\varepsilon)$，当前 vloss 文件使用的是 $t/(m+\varepsilon)$。对 $t>0$，忽略 epsilon 时二者相同；保留同一个非零 epsilon 时并非严格相等。阅读第二章的等价公式或比较实验时，应区分理论换元和当前数值实现。

本目录启动脚本统一为 `run_sd3_{geneval,pickscore}_a6000_16gpu_*.sh`，使用双机、每机 8 卡。默认日志和模型输出分别写入 `logs/alpha_exps/`、`outputs/alpha_exps/`，运行名采用 `sd35_<任务>_nft_<实验配置>_r<轮次>_A6000`；`RUN_INDEX` 控制轮次。

## selection_exps：第三章 3.1 的强正信号选择

本目录保持基础 single-loss 的回归形式，主要修改 $w-1$ 的分配。基础信号 $\Delta$ 来自 prompt 内中心化的 reward，使用全 rollout 的 reward 标准差缩放，再裁剪并除以 `adv_clip_max`。

第三章 3.1 的核心问题是：**能否把略高于平均水平样本的正向更新，转移给更好的样本？**对应代码中的 scheme C：保留全部负信号，仅选择强正信号，再将保留的正信号除以 $\rho$。$\rho$ 是保留的信号总量比例，不是保留的样本数量比例。

| Python 文件 | 实际用途与实现 |
| --- | --- |
| [train_nft_sd3_ours-4.singleloss-alpha-selection-record.py](selection_exps/train_nft_sd3_ours-4.singleloss-alpha-selection-record.py) | 硬选择版本。按信号强度排序，选择累计绝对信号总量最接近目标比例 $\rho$ 的边界。scheme C 保留完整负侧，只集中正侧；scheme B 则对正负两侧分别选择并除以 $\rho$，作为双侧选择对照。由于选择整样本，保留的信号总量只能近似达到目标比例。 |
| [train_nft_sd3_ours-4.singleloss-alpha-selection-soft-record.py](selection_exps/train_nft_sd3_ours-4.singleloss-alpha-selection-soft-record.py) | 处理并列分数的软边界版本。按相同信号绝对值分组，边界前整组保留，边界后整组舍弃，边界组内所有样本乘同一比例，使保留信号总量在浮点精度内准确达到 $\rho$。同样支持 B/C。这里的 `soft` 指边界信号的分数分配，不是 softmax。 |

主要参数为 `config.train.mass_shift_scheme` 和 `config.train.mass_shift_rho`。$\rho=1$ 时回到不做尾部筛选的基础信号；未选中的样本取 $w-1=0$，目标回到 old prediction，仍参与回归和 reference 正则。

两个文件的 `record` 都表示记录训练 rollout：默认将每轮图片和原始 reward 按 prompt 保存到 `${SAVE_DIR}/checkpoints/training_samples/epoch_*/`，包含图片与 `rewards.json`，便于分析哪些样本被强化或弱化。可用 `--save_rollout_samples=false` 关闭。

文档 3.1 的非对称选择对应 **scheme C**；scheme B 是额外的双侧对照。B 在负侧除以 $\rho$ 后可能产生小于零的 $w$，不能把该实验的所有 `importance_weights` 都理解为严格非负的概率比。

## softmax_variants_exps：第三章 3.2 的 reward 映射

这个目录进一步研究**如何将原始 reward 映射成组内样本权重**。算法 3 和式 (22) 使用

$$
w_i=\frac{f(R_i)}{K^{-1}\sum_{j=1}^{K}f(R_j)},\qquad
\frac1K\sum_i w_i=1.
$$

训练仍通过 $w-1$ 修正 single-loss target。$w$ 是概率比，$w/K$ 才是组内样本的新概率。这里直接映射 reward，不先做 NFT 的 advantage 标准化和裁剪。

| Python 文件 | 对应实验与用途 |
| --- | --- |
| [train_nft_sd3_A_near_max.py](softmax_variants_exps/train_nft_sd3_A_near_max.py) | A：one-hot / near-max。选择满足 `R >= (1-c) * Rmax` 的样本，选中者平分全组权重；唯一最大值且 `c=0` 时成为 one-hot。研究把权重集中给最好样本是否有效。 |
| [train_nft_sd3_B_softmax_std.py](softmax_variants_exps/train_nft_sd3_B_softmax_std.py) | B：`f(R)=exp(R/(c*std_group(R)))`。通过温度系数 `c` 比较更集中或更平坦的组内权重。 |
| [train_nft_sd3_C_linear.py](softmax_variants_exps/train_nft_sd3_C_linear.py) | C：`f(R)=max(R,0)`，作为原始 reward 线性映射 baseline；没有实现文档中仅作比较的 shifted-linear 变体。 |
| [train_nft_sd3_D_sigmoid.py](softmax_variants_exps/train_nft_sd3_D_sigmoid.py) | D：`f(R)=sigmoid((R-0.5)/c)`。比较阈值附近的陡峭程度与高 reward 饱和对训练的影响。 |
| [train_nft_sd3_E_power.py](softmax_variants_exps/train_nft_sd3_E_power.py) | E：`f(R)=max(R,0)**p`，比较 `p=0.5` 的平坦映射与 `p=2` 对高 reward 的强化。 |
| [train_nft_sd3_F_softmax_l1.py](softmax_variants_exps/train_nft_sd3_F_softmax_l1.py) | F：将 B 的组内标准差替换为平均绝对偏差 `mean(abs(R-mean(R)))`，作为 softmax 温度尺度的对照。 |
| [weight_mappings.py](softmax_variants_exps/weight_mappings.py) | A–F 的公共权重计算、参数检查和数值边界处理。按 `(rollout_batch_id, prompt)` 分组，保持跨 GPU 收集后的原始样本顺序；退化组按相应规则返回中性权重 `w=1`。 |
| [test_softmax_variants.py](softmax_variants_exps/test_softmax_variants.py) | 检查六类映射的 18 个配置、并列分数与零尺度边界、分组顺序，以及训练入口和启动参数。它是验证代码，不是训练入口。 |

这些训练文件以当前 `default` adapter 采样，训练时重新均匀采样 $t\in[0,1)$，并固定使用 old prediction 计算 loss 外部的 alpha。alpha 目录中的基础 single-loss 文件目前以 `old` adapter 采样；selection 文件虽使用当前策略采样，训练仍读取 scheduler 时间步。因此跨目录比较时，除权重公式外也要核对采样与时间步设置。

详细配置和启动方式见本目录的 [README](softmax_variants_exps/README.md) 与 [experiments.json](softmax_variants_exps/experiments.json)。PDF 3.2 还描述了 **G：对正向 excess 做指数映射后恢复正向信号总量**；当前目录只有 A–F，没有对应的 G 训练文件。

## flowawr_exps：基于我们的 single-loss 代码复现 FlowAWR

[train_nft_sd3_ours-singleloss-AWR.py](flowawr_exps/train_nft_sd3_ours-singleloss-AWR.py) 是在我们自己的 single-loss 训练代码基础上实现的 FlowAWR 复现，用于方法对照。

其主要差异是将 reward 映射为组内归一化的指数权重：

$$
A_i=\frac{\exp(R_i/\gamma)}{\operatorname{mean}_{j\in\mathrm{group}}\exp(R_j/\gamma)}-1,
\qquad
v_{\mathrm{target}}=v_{\mathrm{old}}+A_i(v-v_{\mathrm{old}}).
$$

当前实现用全 rollout 的 reward 标准差确定 $\gamma$，带数值下限；回归目标使用直接的 velocity MSE，没有 alpha 目录中的误差倒数权重或显式外部 $t$ 权重。它与 softmax B 的区别还包括：B 使用每个 prompt 组自己的标准差作温度，并保留基础 single-loss 的 alpha 加权回归。

该文件沿用 LoRA、分布式训练和 checkpoint 流程，以 old adapter 做确定性 rollout，并使用 scheduler 的 sigma 构造训练插值。old policy 每轮按 `min(0.001 * iteration, 0.5)` 的保留系数更新。reference 正则仍以禁用 LoRA 后的预训练模型为基准。

代码还提供针对 Geneval/OCR 的可选 advantage 方差门控；其具体公式是本仓库的实现选择，应与 FlowAWR 复现的核心目标区分。

## scripts 根目录的 Python 入口

| Python 文件 | 用途 |
| --- | --- |
| [train_nft_sd3.py](train_nft_sd3.py) | 基础 SD3 DiffusionNFT 训练入口，采用正负双分支 loss 和 DDP 训练流程。 |
| [train_nft_sd3_origin.py](train_nft_sd3_origin.py) | 根目录的 baseline 训练版本，同样保留双分支目标，并包含 checkpoint 自动查找与恢复等训练支持；alpha 实验使用其目录内的 baseline 文件。 |
| [evaluation.py](evaluation.py) | 独立评估入口，加载模型或 LoRA checkpoint，在指定数据集上生成样本并评估 reward，可保存生成图片。 |

## history_versions

[history_versions/](history_versions/) 保存之前所有零散实验的历史代码，此处不逐文件展开。
