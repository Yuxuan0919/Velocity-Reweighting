# Importance Ratio 存在的问题：





- **针对 importance weight  $w_D(z)$  实际训练中存在问题：**

  ![image-20260813162518948](C:\Users\Yu Xuan\AppData\Roaming\Typora\typora-user-images\image-20260813162518948.png)

  1.  **在 MSE loss 之外再乘上 $W(x^0_i,c_i)$ 存在问题：$W(x_i^0,c_i)$ 和 Advantage正相关，因此对于所有负样本，会由于乘上的$W(x_i^0,c_i)$进一步让梯度变小，而正样本的梯度会被进一步放大，造成训练的不对称**



但是根据论文推导，外层的确需要这样一个两个分布之间的 importance ratio 项，分析出现问题真正的原因：



原本外面把
$$
c_t = \alpha_t^{old}(x_t,z)  \frac{\beta (W(z)-1)}{Z} (v_t-v_{old})
$$

$$
v^n(x_t,t)
=
v^o(x_t,t)
+
\mathbb E_{z_{\mathrm{corr}}}
[c_t(x_t,z_{\mathrm{corr}})]
$$

替换为单样本 pseudo-target ，使用了同一个 $z$：

$$
w_{\mathcal D}(z)
\left\|
v_\theta(x_t,t)
-v^o(x_t,t)
-c_t(x_t,z)
\right\|^2.
$$

实际上把两个作用不同的 z 合并了：

1. 外层的 $z_{\mathrm{out}}$ 用来生成 $x_t$，并提供$\pi^n$与$\pi^o$之间的 importance ratio；
2. 内层的 $z_{\mathrm{corr}}$ 用来 Monte Carlo 估计 $\mathbb E[c_t(x_t,z_{\mathrm{corr}})]$。

代回 loss，

$$
\begin{aligned}
\mathcal L(\theta)
=
\mathbb E_{
z_{\mathrm{out}}\sim\pi^o,
x_t\sim p_t(\cdot\mid z_{\mathrm{out}})
}
\Bigg[
\frac{\pi^n(z_{\mathrm{out}})}{\pi^o(z_{\mathrm{out}})}
\Big\|
v_\theta(x_t)-v^o(x_t) -
\mathbb E_{z_{\mathrm{corr}}\sim P_{\mathcal D}}
[c_t(x_t,z_{\mathrm{corr}})]
\Big\|^2
\Bigg].
\end{aligned}
$$





## 双层采样近似：内层使用同 prompt 的全部 $K$ 个样本

对外层样本 $i$，仍用 $z_i$ 生成 $x_{t,i}$，并仅由 $z_i$ 提供 loss 外层的 importance ratio。内层则把相同 prompt 下的 $K$ 个 rollout 记为 $\mathcal G_i=\{z_1,\ldots,z_K\}$，将其视为 $P_{\mathcal D}(z\mid c_i)$ 的经验分布，直接计算完整经验均值：
$$
\widehat{\mathbb E}_{z_{\mathrm{corr}}\sim P_{\mathcal D}(\cdot\mid c_i)}
[c_t(x_{t,i},z_{\mathrm{corr}})]
=\frac{1}{K}\sum_{j\in\mathcal G_i}c_t(x_{t,i},z_j).
$$

固定外层的 $x_{t,i}$，不能用每个 $z_j$ 重新生成 $x_t$。每个内层样本在同一个 $x_{t,i}$ 处 conditional velocity 可直接写为
$$
v_t(x_{t,i}\mid z_j)=\frac{x_{t,i}-z_j}{t_i}.
$$

要求 $t_i>0$；若训练 timestep 中包含严格的 $t=0$，应直接跳过该点。

因此内层 correction 的实现形式为

$$
\overline c_i
=\frac{1}{K}\sum_{j\in\mathcal G_i}
\widehat\alpha_t(x_{t,i},z_j)
\frac{\beta(W_j-1)}{\bar Z_j}
\left(
\frac{x_{t,i}-z_j}{t_i}-v^o(x_{t,i})
\right),
$$

最终 loss 为

$$
\widehat{\mathcal L}_i=
\frac{W_i}{\bar Z_i}
\left\|
v_\theta(x_{t,i})-v^o(x_{t,i})-\mathtt{sg}(\overline c_i)
\right\|^2.
$$

实现时应先对 $K$ 个 $c_t$ 求均值，再构造一次 pseudo-target。

### 合理性

$z_{\mathrm{out}}$ 决定“在哪个 $x_t$ 上训练”和外层样本的重要程度；同 prompt 的 $K$ 个 $z_{\mathrm{corr}}$ 则共同决定“这个 $x_t$ 上应该往哪个方向修正”。对于一个 $z_i$ 和对应的 $x_{t,i}$ , 它的 velocity correction 方向，不应该只是由自己单样本的gt velocity或其反方向来决定，更合理的是同一个prompt下sample出的 K 个 $z_j$， 它们作为 K 个修正方向 $\frac{x_{t,i} - z_j}{t_i}$ 的集合，它们的加权平均根据每个样本对应的 reward 来决定。



## 当前 double-sampling 代码的完整实现

本节描述的是 **scripts/train_nft_sd3_ours_double-sampling.py** 当前实际执行的算法，而不只是理论上的期望形式。

> **实验版本说明：** 当前代码已将前面理论方案中的
> $v_{i,j}^{\mathrm{cond}}=(x_{t,i}-z_j)/t_i$
> 注释保留，并实际启用有界 surrogate
> $v_{i,j}^{\mathrm{surr}}=\varepsilon_i-z_j$。
> 这样会消除 $(z_i-z_j)/t_i$ 在小 $t_i$ 时的放大，但新量不再是固定
> $x_{t,i}$ 处严格推导得到的 conditional velocity。下面记录的是这个实验版本的实际效果。

### 1. 符号、代码变量和张量形状

设一轮 rollout 中总共获得 $D$ 个 clean latent。对 prompt $c$ 采样的 $K$ 个 clean latent 记为

$$
\mathcal G(c)=\{z_1,z_2,\ldots,z_K\}.
$$

训练 micro-batch 大小记为 $B$。对 micro-batch 中第 $i$ 个 outer sample，代码中的主要量如下。

| 数学符号 | 代码变量 | 形状 | 含义 |
| --- | --- | --- | --- |
| $z_i$ | x0[i] | $[C,H,W]$ | 生成 outer $x_{t,i}$ 的 clean latent |
| $z_j$ | corr_z[i,j] | $[C,H,W]$ | 与 $z_i$ 同 prompt 的第 $j$ 个 correction latent |
| $x_{t,i}$ | xt[i] | $[C,H,W]$ | 只由 outer sample $z_i$ 和新采样的 noise 生成 |
| $v_i^o$ | old_prediction[i] | $[C,H,W]$ | old adapter 在 $(x_{t,i},c_i,t_i)$ 上的 velocity |
| $v_i^\theta$ | forward_prediction[i] | $[C,H,W]$ | 当前可训练 adapter 的 velocity |
| $v_{i,j}^{\mathrm{surr}}$ | conditional_velocity[i,j] | $[C,H,W]$ | 共享 outer noise 的有界 correction-direction surrogate $\varepsilon_i-z_j$ |
| $W_i$ | importance_weight[i] | scalar | outer sample 的 importance weight |
| $W_j$ | corr_importance_weight[i,j] | scalar | 第 $j$ 个内层 correction sample 的 weight |
| $\bar Z_i$ | prompt_normalizer[i] | scalar | outer sample 所属 prompt group 的 normalizer |
| $\bar Z_j$ | corr_prompt_normalizer[i,j] | scalar | correction sample $j$ 所属 group 的 normalizer |
| $\widehat\alpha_{i,j}$ | trajectory_alpha[i,j] | $[1,1,1]$ | 固定 $x_{t,i}$ 后，$z_j$ 的 trajectory contribution surrogate |
| $\bar c_i$ | mean_velocity_correction[i] | $[C,H,W]$ | $K$ 个子 correction 的算术平均 |

当 latent 是四维 batch tensor 时，corr_z 的完整形状为 $[B,K,C,H,W]$，trajectory_alpha 的形状为 $[B,K,1,1,1]$。

### 2. rollout 后如何得到 $W$、$\bar Z$ 和同 prompt 分组

代码先从原 NFT 流程获得每个 rollout 的 advantage $A_j$。在默认 adv_mode=all 时，实际使用的量为

$$
\begin{aligned}
A_j^{\mathrm{clip}}
&=
\operatorname{clip}(A_j,-A_{\max},A_{\max}),\\
\widehat A_j
&=
\frac{A_j^{\mathrm{clip}}}{A_{\max}},\\
W_j
&=
\max\{\epsilon,1+\widehat A_j\}.
\end{aligned}
$$

当前代码中 $\epsilon=10^{-5}$。因此默认模式下，$W_j$ 大致位于 $[10^{-5},2]$ 之间：

- advantage 为正时，$W_j>1$；
- advantage 为负时，$W_j<1$；
- advantage 为零时，$W_j=1$。

如果配置使用 positive_only、negative_only、one_only 或 binary，代码会先对 clipped advantage 做相应变换，但后续 double-sampling 计算不变。

代码使用下面的 key 对全部 GPU 上的 rollout 分组：

~~~text
(rollout_batch_id, decoded_prompt)
~~~

即使 prompt 文本相同，只要来自不同 sampling batch，也不会被合并。每个 group 必须恰好含有 $K$ 个不同 rollout，否则代码直接报错。

对 group $\mathcal G(c)$，prompt normalizer 为

$$
\bar Z_c
=
1+\beta\frac{1}{K}
\sum_{j=1}^{K}(W_j-1).
$$

同一 group 中所有 $\bar Z_j$ 实际上都等于 $\bar Z_c$。代码仍以 per-sample tensor 保存它，是为了让它和 outer sample 一起 shuffle 和 batch。这里的 $\beta$ 对应 **config.beta**，同时也出现在后面的 correction coefficient 中。

### 3. 跨 GPU 的全局 correction bank

同一 prompt 的 $K$ 个 rollout 会被 sampler 打乱，并可能分布在不同 GPU 上。进入训练后，outer sample 还会在每个 rank 上再次 shuffle。因此当前 micro-batch 本身不一定含有完整的 $K$ 个样本。

代码在每轮 rollout 结束后对 latents_clean 执行 all-gather，使每个 GPU 都保存相同的

$$
\mathrm{global\_corr\_latents}
\in
\mathbb R^{D\times C\times H\times W}.
$$

全局 $W$ 和 $\bar Z$ 也按完全相同的 rank-major 顺序保存为

~~~text
global_corr_importance_weights  # [D]
global_corr_prompt_normalizers  # [D]
~~~

对每个 outer sample $i$，correction_group_indices[i] 保存它的 $K$ 个同 prompt 样本在全局 bank 中的下标。这个索引会和 outer sample 一起被 shuffle，所以无论 outer sample 最后进入哪个 micro-batch，都可以找回完整 correction group。

全局 bank、$W$、$\bar Z$ 和 group indices 都不参与梯度计算。它们在本轮 rollout 数据的所有 inner epoch 中保持不变。

### 4. outer $x_{t,i}$ 如何生成

对第 $i$ 个 outer clean latent $z_i$，训练代码重新采样

$$
\varepsilon_i\sim\mathcal N(0,I),
$$

然后生成

$$
x_{t,i}
=
(1-t_i)z_i+t_i\varepsilon_i.
$$

$x_{t,i}$ 只由 outer sample $z_i$ 决定。内层的 $z_j$ 不会各自重新采样 noise，也不会各自生成新的 $x_t$。

随后代码在固定的 $(x_{t,i},c_i,t_i)$ 上分别计算

$$
v_i^o=v_{\theta_{\mathrm{old}}}(x_{t,i},c_i,t_i),
\qquad
v_i^\theta=v_\theta(x_{t,i},c_i,t_i).
$$

old prediction 每个 outer sample 只前向一次。内层 $K$ 个 correction 全部复用同一个 $v_i^o$。需要注意，$v_i^o$ 仍然是在真实 outer $x_{t,i}$ 上计算的，而下面的 $\varepsilon_i-z_j$ 是实验性 correction-direction surrogate；二者的差异被用于构造 correction。

### 5. 一个 $x_{t,i}$ 使用的 $K$ 个有界 velocity surrogate

对固定的 outer $x_{t,i}$，代码从全局 bank 取回

$$
\{z_1,z_2,\ldots,z_K\}=\mathcal G(c_i),
$$

然后复用生成 outer $x_{t,i}$ 时采样的同一个 noise $\varepsilon_i$，为每个 $z_j$ 构造

$$
 \boxed{
v_{i,j}^{\mathrm{surr}}
=
\varepsilon_i-z_j
}.
$$

这个量可以写成

$$
v_{i,j}^{\mathrm{surr}}
=
(\varepsilon_i-z_i)
+
(z_i-z_j).
$$

所以当前一个 outer sample 使用的 $K$ 个方向具体由以下两部分组成：

- 它们共享 outer sample 的基础 flow-matching velocity $\varepsilon_i-z_i$；
- 每个 $z_j$ 额外提供有界的跨样本偏移 $z_i-z_j$；
- 当 $j=i$ 时，$z_i-z_j=0$，所以

$$
v_{i,i}^{\mathrm{surr}}
=
\varepsilon_i-z_i,
$$

正好退化为原单样本代码中的 noise-x0。因此一个 correction group 包含一个 self direction 和同 prompt 的其他 $K-1$ 个 cross-sample direction。

与原公式相比，

$$
\underbrace{\frac{x_{t,i}-z_j}{t_i}}_{\text{原 fixed-}x_t\text{ conditional velocity}}
=
(\varepsilon_i-z_i)+\frac{z_i-z_j}{t_i},
$$

当前 surrogate 将跨样本项从 $(z_i-z_j)/t_i$ 改为 $z_i-z_j$，因此它不会在
$t_i\rightarrow0$ 时出现 $1/t_i$ 放大。不过，$\varepsilon_i-z_j$ 严格对应的是
由 $z_j$ 和 $\varepsilon_i$ 组成的另一条直线路径：

$$
x_{t,i,j}'=(1-t_i)z_j+t_i\varepsilon_i,
\qquad
\frac{\mathrm d x_{t,i,j}'}{\mathrm d t_i}
=
\varepsilon_i-z_j.
$$

当 $j\ne i$ 时，通常 $x_{t,i,j}'\ne x_{t,i}$。因此当前实现是把这条替代路径的
velocity 当作真实 outer $x_{t,i}$ 上的 correction-direction surrogate，而不是声称它是
$v_t(x_{t,i}\mid z_j)$ 的严格解析值。

### 6. $\alpha_{i,j}$ 的实际计算

对每个 $v_{i,j}^{\mathrm{surr}}$，代码先计算它与固定 old velocity $v_i^o$ 的平均绝对误差：

$$
d_{i,j}
=
\operatorname{mean}_{C,H,W}
\left|
v_{i,j}^{\mathrm{surr}}-v_i^o
\right|.
$$

再计算未归一化的 reciprocal-discrepancy score：

$$
\widetilde\alpha_{i,j}
=
\frac{1}{d_{i,j}+\epsilon}.
$$

$d_{i,j}$ 越小，说明第 $j$ 个 surrogate direction 与 old model 在该 $x_{t,i}$ 上的 velocity 越一致，因此 $\widetilde\alpha_{i,j}$ 越大。

接下来不是在整个 micro-batch 上归一化，而是对每个固定 outer $x_{t,i}$ 的 $K$ 个 candidate 单独归一化：

$$
\boxed{
\widehat\alpha_{i,j}
=
\frac{
\widetilde\alpha_{i,j}
}{
\frac{1}{K}\sum_{l=1}^{K}\widetilde\alpha_{i,l}
}
}.
$$

所以对每个 outer sample $i$ 都满足

$$
\frac{1}{K}
\sum_{j=1}^{K}
\widehat\alpha_{i,j}
=1.
$$

这里 $\widehat\alpha_{i,j}$ 的平均为 $1$、总和为 $K$，它不是总和为 $1$ 的概率。这样既保留 candidate 之间的相对比例，又不让 $\alpha$ 的整体 scale 额外改变 correction 量级；后面的 correction 本身还会沿 $K$ 维取算术平均。

所有 $d_{i,j}$、$\widetilde\alpha_{i,j}$ 和 $\widehat\alpha_{i,j}$ 都在 torch.no_grad() 和 float32 中计算，不通过 target 传播梯度。

### 7. 一个 $x_{t,i}$ 的平均 correction 包含哪些子项

第 $j$ 个 correction sample 的 reward/mass-shift coefficient 是

$$
m_j
=
\frac{\beta(W_j-1)}{\bar Z_j}.
$$

第 $j$ 个子 correction 为

$$
\boxed{
c_{i,j}
=
\widehat\alpha_{i,j}
\frac{\beta(W_j-1)}{\bar Z_j}
\left(
v_{i,j}^{\mathrm{surr}}-v_i^o
\right)
}.
$$

将 surrogate velocity 完全展开后，

$$
c_{i,j}
=
\widehat\alpha_{i,j}
\frac{\beta(W_j-1)}{\bar Z_j}
\left[
(\varepsilon_i-z_i)
+
(z_i-z_j)
-v_i^o
\right].
$$

因此，一个具体 outer $x_{t,i}$ 的平均 correction 确切包含下面 $K$ 项：

$$
\begin{aligned}
\bar c_i
=\frac{1}{K}\Bigg[&
\widehat\alpha_{i,1}
\frac{\beta(W_1-1)}{\bar Z_1}
\left(
\varepsilon_i-z_1-v_i^o
\right)\\
&+
\widehat\alpha_{i,2}
\frac{\beta(W_2-1)}{\bar Z_2}
\left(
\varepsilon_i-z_2-v_i^o
\right)\\
&+\cdots\\
&+
\widehat\alpha_{i,K}
\frac{\beta(W_K-1)}{\bar Z_K}
\left(
\varepsilon_i-z_K-v_i^o
\right)
\Bigg].
\end{aligned}
$$

这 $K$ 项中包含 $j=i$ 的 self term：

$$
c_{i,i}
=
\widehat\alpha_{i,i}
\frac{\beta(W_i-1)}{\bar Z_i}
\left[
(\varepsilon_i-z_i)-v_i^o
\right],
$$

以及其他 $K-1$ 个 cross-sample term：

$$
c_{i,j}
=
\widehat\alpha_{i,j}
\frac{\beta(W_j-1)}{\bar Z_j}
\left[
(\varepsilon_i-z_i)
+
(z_i-z_j)
-v_i^o
\right],
\qquad j\ne i.
$$

每个子项由三个因素共同决定：

1. **方向**：$v_{i,j}^{\mathrm{surr}}-v_i^o$，表示第 $j$ 条实验 surrogate trajectory 相对 old velocity 建议如何变化；
2. **trajectory contribution**：$\widehat\alpha_{i,j}>0$，表示该方向与 old velocity 的相对一致程度；
3. **reward mass shift**：$\beta(W_j-1)/\bar Z_j$，决定该方向是正向加入还是反向扣除，以及强度多大。

具体来说：

- $W_j>1$ 时，$m_j>0$，第 $j$ 个 direction 以原方向加入平均 correction；
- $W_j<1$ 时，$m_j<0$，第 $j$ 个 discrepancy direction 以反方向进入 correction；
- $W_j=1$ 时，$m_j=0$，该样本仍参与 $\alpha$ 的 group normalization，但自身对 $\bar c_i$ 的直接向量贡献为零。

reward 不参与 $\alpha$ 的计算，$\alpha$ 也不决定 correction 的正负号。$\alpha$ 根据 velocity consistency 分配相对 trajectory contribution，$W_j-1$ 根据 reward/advantage 决定质量偏好和质量转移方向。

### 8. 在 group mean correction 与 self correction 之间路由

代码首先保留完整的 $K$ 项平均 correction：

$$
\bar c_i^{\mathrm{group}}
=
\frac{1}{K}
\sum_{j=1}^{K}c_{i,j},
$$

同时利用 outer sample 的全局 bank index，在它的 correction group 中唯一找到
$j=i$ 的位置，并取出 self correction：

$$
c_i^{\mathrm{self}}=c_{i,i}.
$$

这里的 $c_{i,i}$ 使用当前代码沿完整 $K$ 个 candidate 归一化后得到的
$\widehat\alpha_{i,i}$。因此它表示“从已经计算好的 $K$ 个 $c_{i,j}$ 中只取
$j=i$ 的一项”，而不会退回旧代码中跨 micro-batch 归一化 $\alpha$ 的方式。

配置 **config.train.correction_mean_mode** 决定哪些 outer sample 使用
$\bar c_i^{\mathrm{group}}$：

| 参数值 | 使用 group mean 的 outer sample | 其余 outer sample |
| --- | --- | --- |
| all | 所有样本 | 无 |
| positive_only | 仅 $W_i>1$ 的正样本 | 使用 $c_i^{\mathrm{self}}$ |
| negative_only | 仅 $W_i<1$ 的负样本 | 使用 $c_i^{\mathrm{self}}$ |

全局默认值是 **all**，当前 Geneval double-sampling 启动脚本也显式使用
**all**，即所有 outer 样本都使用组平均 correction。如果将启动参数改为
**negative_only**，则路由为：

$$
c_i^{\mathrm{selected}}
=
\begin{cases}
\bar c_i^{\mathrm{group}},&W_i<1,\\
c_i^{\mathrm{self}},&W_i\ge1.
\end{cases}
$$

这正对应“负 outer sample 使用同 prompt 的 $K$ 项平均 correction，正 outer
sample 只使用自己的 $j=i$ correction”。当 $W_i=1$ 时，在 positive_only 和
negative_only 模式下都走 self 分支；此时 self mass-shift coefficient
$\beta(W_i-1)/\bar Z_i=0$，所以 self correction 为零。

一般地，定义路由 mask $M_i\in\{0,1\}$：

$$
M_i=
\begin{cases}
1,&\text{all},\\
\mathbb 1[W_i>1],&\text{positive\_only},\\
\mathbb 1[W_i<1],&\text{negative\_only},
\end{cases}
$$

则统一写为

$$
\boxed{
c_i^{\mathrm{selected}}
=
M_i\bar c_i^{\mathrm{group}}
+(1-M_i)c_i^{\mathrm{self}}
}.
$$

最终仍然只构造一个 target velocity：

$$
\boxed{
v_i^{\mathrm{target}}
=
v_i^o+\mathtt{sg}(c_i^{\mathrm{selected}})
}.
$$

整个选择和 target 构造都在 torch.no_grad() 中执行，因此不会通过
$c_i^{\mathrm{selected}}$ 传播梯度。

三个模式可以通过命令行覆盖：

~~~text
--config.train.correction_mean_mode=all
--config.train.correction_mean_mode=positive_only
--config.train.correction_mean_mode=negative_only
~~~

### 9. 外层 importance ratio 和最终 policy loss

对 outer sample $i$，首先计算特征维度上的平均 MSE：

$$
\ell_i^{\mathrm{target}}
=
\operatorname{mean}_{C,H,W}
\left(
v_i^\theta-v_i^{\mathrm{target}}
\right)^2.
$$

代码首先计算 group-mean 分支原本对应的 outer importance ratio：

$$
q_i^{\mathrm{group}}
=
\frac{W_i}{\bar Z_i}.
$$

现在 outer MSE 是否乘这个 importance ratio 与 correction 路由保持一致。
继续使用上一节的 group-mean mask $M_i$，实际使用的外层系数为

$$
q_i^{\mathrm{selected}}
=
M_i\frac{W_i}{\bar Z_i}
+(1-M_i).
$$

因此

$$
q_i^{\mathrm{selected}}
=
\begin{cases}
\dfrac{W_i}{\bar Z_i},
&\text{使用 group mean correction},\\[4pt]
1,
&\text{使用 self correction}.
\end{cases}
$$

也就是说，使用 self correction 时，MSE 外面不再乘 $W_i/\bar Z_i$。
单个 outer sample 对 policy loss 的贡献为

$$
\ell_i^{\mathrm{policy}}
=
t_i
q_i^{\mathrm{selected}}
\ell_i^{\mathrm{target}}.
$$

对 micro-batch 取平均并乘上 adv_clip_max 后，代码中的 policy loss 为

$$
\boxed{
\mathcal L_{\mathrm{policy}}
=
A_{\max}
\frac{1}{B}
\sum_{i=1}^{B}
t_i
q_i^{\mathrm{selected}}
\left\|
v_i^\theta-v_i^o-c_i^{\mathrm{selected}}
\right\|_{\mathrm{mean}}^2
}.
$$

correction 路由和外层 weight 路由现在是绑定的：

- group-mean 分支使用完整 $K$ 项 correction，并在 MSE 外乘
  $W_i/\bar Z_i$；
- self 分支只使用 $c_{i,i}$，MSE 外层系数固定为 $1$；
- 内层每个 $c_{i,j}$ 中的 $\beta(W_j-1)/\bar Z_j$ 不受此修改影响。

例如在 negative_only 模式中，按照当前路由定义，$W_i<1$ 的负 outer
sample 使用 group mean，因此仍乘 $W_i/\bar Z_i$；$W_i\ge1$ 的样本使用
self correction，因此不乘外层 importance ratio。这里的判断依据是“最终是否
使用 self correction”，而不是单独根据正负标签再写一套判断。

### 10. reference loss 和两个 beta 的区别

代码还保留 reference-model regularization：

$$
\mathcal L_{\mathrm{ref}}
=
\operatorname{mean}_{i,C,H,W}
\left(
v_i^\theta-v_i^{\mathrm{ref}}
\right)^2.
$$

最终在梯度累积 scale 之前的 loss 为

$$
\mathcal L
=
\mathcal L_{\mathrm{policy}}
+
\lambda_{\mathrm{ref}}
\mathcal L_{\mathrm{ref}}.
$$

代码中两个名称相似但意义不同的配置是：

- **config.beta** 对应 correction 和 $\bar Z$ 中的 $\beta$；
- **config.train.beta** 对应 $\lambda_{\mathrm{ref}}$，是 reference loss 的系数。

二者不是同一个数学量。

### 11. $t_i=0$ 时的实际效果

当前实验 surrogate $v_{i,j}^{\mathrm{surr}}=\varepsilon_i-z_j$ 不再包含 $1/t_i$，
所以代码不需要为了计算该方向而构造 safe denominator。原来的 safe_t 和
$(x_{t,i}-z_j)/t_i$ 实现已作为注释保留。

为了保持原先“$t_i=0$ 不参与 policy 训练”的行为，代码仍然计算
valid_t=(t_i>0)，并将 $t_i=0$ 样本的 mean_velocity_correction 置零。因此

$$
\bar c_i=0,
\qquad
v_i^{\mathrm{target}}=v_i^o.
$$

policy loss 外层还乘有 $t_i$，所以

$$
\ell_i^{\mathrm{policy}}=0.
$$

也就是说，$t_i=0$ 样本不产生 policy gradient，但 reference loss 仍正常计算。这在不改变 DDP backward 次数和梯度累积流程的情况下，实现了跳过 $t=0$ policy 训练的效果。

对很小但严格大于零的 $t_i$，当前 surrogate 也始终有界：

$$
v_{i,j}^{\mathrm{surr}}=\varepsilon_i-z_j,
$$

不会再出现 $(z_i-z_j)/t_i$ 的数值放大。不过 cross-sample 偏移
$z_i-z_j$ 并不会随着 $t_i\to0^+$ 自动消失；只是从原来的 $O(1/t_i)$ 变为
$O(1)$。这是本实验版本需要通过训练效果验证的行为。

### 12. 当前算法的逐步伪代码

~~~text
for each rollout/training iteration:
    1. 对每个 prompt c 生成 K 个 clean latent {z_1, ..., z_K}
    2. 计算 reward 和 NFT advantage A_j
    3. 计算 W_j = max(epsilon, 1 + clip(A_j, -A_max, A_max) / A_max)
    4. 对每个 (rollout_batch_id, prompt) group 计算
           Z_bar_c = 1 + beta * mean_j(W_j - 1)
    5. all-gather 所有 GPU 的 clean latent，建立 global correction bank
    6. 为每个 outer sample 保存同 prompt 的 K 个 bank index

    for each shuffled outer micro-batch:
        for each selected timestep:
            7. 对每个 outer sample i 采样 epsilon_i，并构造
                   x_t_i = (1 - t_i) * z_i + t_i * epsilon_i
            8. 计算 current velocity v_theta_i 和 old velocity v_old_i

            9. 从 global bank 取回同 prompt 的 {z_1, ..., z_K}
           10. 对 j = 1, ..., K 计算
                   v_surr_ij = epsilon_i - z_j
                   d_ij = mean_feat(abs(v_surr_ij - v_old_i))
                   alpha_raw_ij = 1 / (d_ij + epsilon)
           11. 对每个固定 i，沿 K 维归一化
                   alpha_ij = alpha_raw_ij / mean_j(alpha_raw_ij)
           12. 对 j = 1, ..., K 计算
                   c_ij = alpha_ij * beta * (W_j - 1) / Z_bar_j
                          * (v_surr_ij - v_old_i)
           13. 同时得到两种候选 correction
                   c_group_i = mean_j(c_ij)
                   c_self_i = c_i,i
           14. 根据 correction_mean_mode 生成 use_group_mean_i
                   all:           所有 i 都为 True
                   positive_only: W_i > 1 时为 True
                   negative_only: W_i < 1 时为 True
           15. 选择最终 correction 并构造一个 pseudo-target
                   c_selected_i = where(
                       use_group_mean_i, c_group_i, c_self_i
                   )
                   v_target_i = v_old_i + stop_gradient(c_selected_i)
           16. 根据 correction 路由选择外层 MSE 系数
                   q_outer_i = where(
                       use_group_mean_i, W_i / Z_bar_i, 1
                   )
           17. 计算 outer weighted policy loss
                   L_policy = A_max * mean_i[
                       t_i * q_outer_i
                       * mean_feat((v_theta_i - v_target_i)^2)
                   ]
           18. 加上 reference loss，再按现有流程累积和更新梯度
~~~

### 13. 当前实现达到的效果和独立性边界

原单样本实现中，outer sample $z_i$ 同时决定 $x_{t,i}$、外层 importance ratio 和全部 pseudo-target correction。现在三者的职责被拆分为：

1. $z_i$ 和 $\varepsilon_i$ 决定在哪个 $x_{t,i}$ 上训练；
2. 使用 group mean correction 时，outer $W_i/\bar Z_i$ 决定这个
   $x_{t,i}$ 的 MSE 在 loss 中有多重要；使用 self correction 时，外层
   MSE 系数固定为 $1$；
3. 同 prompt 的 $K$ 个 $z_j$ 和各自的 $W_j-1$ 先产生 group mean correction，同时 $j=i$ 产生 self correction；最终由 correction_mean_mode 和 outer $W_i$ 决定使用哪一个。

对同 prompt 的每个 outer sample $i$，内层使用的 $z_j$、$W_j$ 和 $\bar Z_j$ 集合相同；但因为 $\varepsilon_i$、$x_{t,i}$、$t_i$ 和 $v_i^o$ 不同，不同 outer sample 得到的 $v_{i,j}^{\mathrm{surr}}$、$\alpha_{i,j}$ 和 $\bar c_i$ 仍然不同。因此这不是给同 prompt 的所有 $x_t$ 强行使用同一个 correction vector，而是在每个具体 outer sample 上，使用它自己的 noise 和 old prediction 重新评估同一组 $K$ 条 surrogate direction。

最后，当前 double-sampling 是外层/内层职责上的解耦：代码始终使用同 prompt 的完整 $K$ 样本计算 $\alpha_{i,j}$ 和全部 correction 子项，其中也包含 outer sample $z_i$ 自身；随后才根据路由模式选择 $K$ 项平均或单独的 $j=i$ 项。因此它不是两套完全不重叠的 iid rollout，也不是 leave-one-out 估计。当前默认及 Geneval 启动配置都是 all，因此所有 outer sample 都使用 group mean correction，并保留 $W_i/\bar Z_i$。如果切换为 negative_only，则负 outer sample 使用完整 $K$ 项经验平均并保留 $W_i/\bar Z_i$，正/中性 outer sample 使用 self correction且不再乘外层 $W_i/\bar Z_i$。
