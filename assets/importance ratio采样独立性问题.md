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
v_t(x_{t,i}\mid z_j)=\frac{x_{t,i} - z_j}{t_i} = \epsilon_i - z_i + \frac{z_i-z_j}{t_i}.
$$

### $\widehat\alpha_t$ 的具体计算

理论上的 trajectory contribution factor 是 conditional-to-marginal density ratio：

$$
\alpha_t^o(x_t,z)
=
\frac{p_t(x_t\mid z)}{p_t^o(x_t)}.
$$

它描述给定 $x_t$ 后轨迹端点 $z$ 对该位置 velocity field 的相对贡献，并满足

$$
\mathbb E_{z\sim\pi^o}\left[\alpha_t^o(x_t,z)\right]=1.
$$

由于实际训练中无法直接计算边缘密度 $p_t^o(x_t)$，代码使用 velocity discrepancy 的倒数作为可计算的 surrogate。对每个固定的外层 $x_{t,i}$ 和组内样本 $z_j$，先定义

$$
\Delta v_{i,j}
=
v_t(x_{t,i}\mid z_j)-v^o(x_{t,i},t_i)
=
\frac{x_{t,i}-z_j}{t_i}-v^o(x_{t,i},t_i),
$$

以及 feature 维度上的平均绝对误差

$$
d_{i,j}
=
\operatorname{mean}_{\mathrm{feat}}
\left|\Delta v_{i,j}\right|.
$$

未归一化的 contribution score 为

$$
\widetilde\alpha_{i,j}
=
\frac{1}{d_{i,j}+\epsilon},
$$

其中当前实现取 $\epsilon=10^{-5}$。然后针对每一个固定的外层 $x_{t,i}$，只在其同 prompt 的 $K$ 个 correction 样本上独立归一化：

$$
\widehat\alpha_{i,j}
=
\frac{\widetilde\alpha_{i,j}}
{\frac{1}{K}\sum_{\ell\in\mathcal G_i}\widetilde\alpha_{i,\ell}}.
$$

因此每个外层样本对应的 $K$ 个系数都满足

$$
\frac{1}{K}\sum_{j\in\mathcal G_i}\widehat\alpha_{i,j}=1.
$$

直观上，某个 $z_j$ 的 conditional velocity 越接近旧模型在固定 $x_{t,i}$ 上的 velocity，该轨迹得到的相对 contribution 越大。这里的 $\widehat\alpha_{i,j}$ 只是对真实密度比 $\alpha_t^o(x_{t,i},z_j)$ 的启发式近似，并不是精确的 density-ratio estimator。discrepancy、归一化系数以及最终 correction 均使用 stop-gradient，不参与反向传播。

因此内层 correction 的实现形式为

$$
\overline c_i
=\frac{1}{K}\sum_{j\in\mathcal G_i}
\widehat\alpha_t(x_{t,i},z_j)
\frac{\beta(W_j-1)}{\bar Z_j}
\left(
\frac{x_{t,i} - z_j}{t_i} -v^o(x_{t,i})
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

$z_{\mathrm{out}}$ 决定“在哪个 $x_t$ 上训练”和外层样本的重要程度；同 prompt 的 $K$ 个 $z_{\mathrm{corr}}$ 则共同决定“这个 $x_t$ 上应该往哪个方向修正”。对于一个 $z_i$ 和对应的 $x_{t,i}$ , 它的 velocity correction 方向，不应该只是由自己单样本的gt velocity或其反方向来决定，更合理的是同一个prompt下sample出的 K 个 $z_j$， 作为 K 个修正方向的集合，它们的加权平均根据每个样本对应的 reward 来决定。
