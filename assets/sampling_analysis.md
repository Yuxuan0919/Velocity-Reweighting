## Importance ratio 的正确推导与原文分析

### 结论

`ours_idea.tex` 中一般形式的 importance ratio

$$
\frac{d\pi^n}{d\pi^o}(\mathbf z)
=
\frac{1}{Z}
\exp\left(\frac{r(\mathbf z)}{\gamma}\right)
$$

是正确的。

但在 partial rollout coverage 部分，原文定义的

$$
w_{\mathcal D}(\mathbf z_i)
=
\frac{\frac{\beta}{D}\exp(r_i/\gamma)}{Z}
$$

不是 importance ratio，而是新分布在 observed atom $\mathbf z_i$ 上的 probability mass。真正的 importance ratio 应当去掉 $\beta/D$：

$$
\boxed{
\rho_{\mathcal D}(\mathbf z_i)
=
\frac{\exp(r_i/\gamma)}{Z}
}.
$$

## 一、reward tilting 下的 importance ratio

考虑旧分布 $\pi^o$。令

$$
q(\mathbf z)
=
\exp\left(\frac{r(\mathbf z)}{\gamma}\right).
$$

reward tilting 后的新分布定义为

$$
\pi^n(d\mathbf z)
=
\frac{q(\mathbf z)}{Z}\pi^o(d\mathbf z),
$$

其中

$$
Z
=
\mathbb E_{\mathbf z\sim\pi^o}[q(\mathbf z)].
$$

因此，从旧 sampling measure $\pi^o$ 到新分布 $\pi^n$ 的 Radon--Nikodym derivative 是

$$
\boxed{
\rho(\mathbf z)
=
\frac{d\pi^n}{d\pi^o}(\mathbf z)
=
\frac{q(\mathbf z)}{Z}
}.
$$

它满足标准 importance-sampling identity：

$$
\boxed{
\mathbb E_{\mathbf z\sim\pi^n}[f(\mathbf z)]
=
\mathbb E_{\mathbf z\sim\pi^o}
\left[
\rho(\mathbf z)f(\mathbf z)
\right]
}.
$$

所以，只要外层已经写成 $\mathbf z\sim\pi^o$，loss 内部就必须乘 density ratio $d\pi^n/d\pi^o$，不能乘新分布在某个 sample 上的 probability mass。

## 二、partial rollout mixture

沿用 `ours_idea.tex` 的假设：

$$
\pi^o
=
(1-\beta)\pi_{\mathrm{bg}}
+
\beta\pi_{\mathcal D},
$$

其中 observed rollout 的经验分布为

$$
\pi_{\mathcal D}
=
\frac{1}{D}
\sum_{i=1}^D
\delta_{\mathbf z_i}.
$$

假设 background reward 已经平移为零，并记

$$
q_i
=
\exp(r_i/\gamma).
$$

那么

$$
q(\mathbf z)=1,
\qquad
\mathbf z\sim\pi_{\mathrm{bg}},
$$

且 normalization constant 为

$$
\boxed{
Z
=
(1-\beta)
+
\frac{\beta}{D}
\sum_{j=1}^Dq_j
}.
$$

下面按照原文把 background 和 observed rollout 视为不同子总体，并假设

$$
\pi_{\mathrm{bg}}(\{\mathbf z_i\})=0.
$$

## 三、background 上的 importance ratio

background 在新分布中的 measure 是

$$
\pi^n_{\mathrm{bg}}(d\mathbf z)
=
\frac{1-\beta}{Z}
\pi_{\mathrm{bg}}(d\mathbf z).
$$

旧分布在同一区域的 measure 是

$$
\pi^o_{\mathrm{bg}}(d\mathbf z)
=
(1-\beta)
\pi_{\mathrm{bg}}(d\mathbf z).
$$

因此 background 上的 density ratio 是

$$
\boxed{
\rho_{\mathrm{bg}}(\mathbf z)
=
\frac{
\pi^n_{\mathrm{bg}}(d\mathbf z)
}{
\pi^o_{\mathrm{bg}}(d\mathbf z)
}
=
\frac{1}{Z}
}.
$$

所以 `ours_idea.tex` 中

$$
w_{\mathrm{bg}}(\mathbf z)=\frac{1}{Z}
$$

作为 background 上的 importance ratio 是正确的。

## 四、observed atom 上的 probability mass 与 importance ratio

旧分布在 atom $\mathbf z_i$ 上已有的 probability mass 是

$$
\boxed{
\pi^o(\{\mathbf z_i\})
=
\frac{\beta}{D}
}.
$$

reward tilting 后，新分布在该 atom 上的 probability mass 是

$$
\boxed{
\pi^n(\{\mathbf z_i\})
=
\frac{\beta}{D}
\frac{q_i}{Z}
}.
$$

但 importance ratio 是新旧 probability mass 之比：

$$
\begin{aligned}
\rho_{\mathcal D}(\mathbf z_i)
&=
\frac{
\pi^n(\{\mathbf z_i\})
}{
\pi^o(\{\mathbf z_i\})
}\\
&=
\frac{
\frac{\beta}{D}\frac{q_i}{Z}
}{
\frac{\beta}{D}
}\\
&=
\frac{q_i}{Z}.
\end{aligned}
$$

因此

$$
\boxed{
\rho_{\mathcal D}(\mathbf z_i)
=
\frac{q_i}{Z}
}
\qquad\text{而不是}\qquad
\boxed{
w_{\mathcal D}(\mathbf z_i)
=
\frac{\beta}{D}
\frac{q_i}{Z}
}.
$$

二者的区别可以概括为：

| 对象 | 旧 probability mass | 新 probability mass | importance ratio $d\pi^n/d\pi^o$ |
| --- | ---: | ---: | ---: |
| 整个 background region | $1-\beta$ | $(1-\beta)/Z$ | $1/Z$ |
| observed atom $\mathbf z_i$ | $\beta/D$ | $\beta q_i/(DZ)$ | $q_i/Z$ |

原文的 $w_{\mathrm{bg}}$ 取自最后一列，而 $w_{\mathcal D}$ 取自第三列，却把二者都称为 importance weight。这是原文的核心错误。

## 五、无 reward 时的 sanity check

令所有 reward 都为零，则

$$
q_i=1,
\qquad
Z=1.
$$

此时 reward tilting 不改变分布：

$$
\pi^n=\pi^o.
$$

所以正确的 importance ratio 应当在所有位置都等于 $1$：

$$
\frac{d\pi^n}{d\pi^o}(\mathbf z)=1.
$$

但原文定义给出

$$
w_{\mathrm{bg}}=1,
\qquad
w_{\mathcal D}(\mathbf z_i)=\frac{\beta}{D}.
$$

这说明 $w_{\mathcal D}$ 不可能是 importance ratio。它只能表示新分布在 atom $\mathbf z_i$ 上放置的 probability mass。

## 六、原文 objective 为什么会重复计算 sampling probability

正确的 importance-sampling identity 是

$$
\mathbb E_{\mathbf z\sim\pi^n}[f(\mathbf z)]
=
\mathbb E_{\mathbf z\sim\pi^o}
\left[
\frac{d\pi^n}{d\pi^o}(\mathbf z)
f(\mathbf z)
\right].
$$

对 observed atoms，正确结果为

$$
\begin{aligned}
&\sum_{i=1}^D
\underbrace{\frac{\beta}{D}}_{\pi^o(\{\mathbf z_i\})}
\underbrace{\frac{q_i}{Z}}_{\text{importance ratio}}
f(\mathbf z_i)\\
&\qquad=
\sum_{i=1}^D
\underbrace{
\frac{\beta}{D}\frac{q_i}{Z}
}_{\pi^n(\{\mathbf z_i\})}
f(\mathbf z_i).
\end{aligned}
$$

如果外层已经声明 $\mathbf z\sim\pi^o$，却又在 loss 内乘原文定义的

$$
w_{\mathcal D}(\mathbf z_i)
=
\frac{\beta}{D}\frac{q_i}{Z},
$$

则 observed 部分变成

$$
\sum_{i=1}^D
\underbrace{\frac{\beta}{D}}_{\text{由外层 sampling 提供}}
\underbrace{
\left(
\frac{\beta}{D}\frac{q_i}{Z}
\right)
}_{\text{又乘了一次 atom mass}}
f(\mathbf z_i).
$$

即

$$
\sum_{i=1}^D
\left(\frac{\beta}{D}\right)^2
\frac{q_i}{Z}
f(\mathbf z_i).
$$

相对于正确结果，额外多出了一个 $\beta/D$。

这不只是可以忽略的全局常数，因为 background 分支没有同样的 $\beta/D$；它会改变 background 与 observed region 的相对权重。

## 七、不同 sampling measure 对应的正确 coefficient

### 情况一：从完整旧 mixture $\pi^o$ 抽样

如果 Monte Carlo 样本来自完整的 $\pi^o$，应使用

$$
\boxed{
\rho(\mathbf z)
=
\begin{cases}
1/Z,
&
\mathbf z\in\mathrm{bg},\\
q_i/Z,
&
\mathbf z=\mathbf z_i.
\end{cases}
}
$$

这里不能再乘 $\beta/D$，因为它已经包含在 sampling measure $\pi^o$ 中。

### 情况二：只从 $\pi_{\mathcal D}$ 均匀抽样，估计 observed region 对完整新分布的贡献

observed region 对新分布 expectation 的贡献是

$$
\begin{aligned}
\sum_{i=1}^D
\frac{\beta}{D}
\frac{q_i}{Z}
f(\mathbf z_i)
&=
\mathbb E_{\mathbf z_i\sim\pi_{\mathcal D}}
\left[
\frac{\beta q_i}{Z}
f(\mathbf z_i)
\right].
\end{aligned}
$$

因此，当代码使用 minibatch mean 估计 $\pi_{\mathcal D}$ expectation 时，每个样本应乘

$$
\boxed{
\frac{\beta q_i}{Z}
},
$$

而不是

$$
\frac{\beta}{D}\frac{q_i}{Z}.
$$

minibatch mean 已经提供了经验分布中的 $1/D$。在 loss coefficient 中再次保留 $1/D$，会使梯度规模随 rollout dataset size $D$ 人为减小。

这种采样方式只估计 observed contribution。完整的新分布 expectation 还包含

$$
\frac{1-\beta}{Z}
\mathbb E_{\mathbf z\sim\pi_{\mathrm{bg}}}
[f(\mathbf z)].
$$

因此，仅使用 observed rollout samples 不能无偏估计完整的 $\pi^n$ expectation。

### 情况三：只优化 observed region 内的条件分布

如果目标不是完整 $\pi^n$，而只是 reward tilting 后 observed region 内的条件分布，那么应先重新归一化：

$$
\pi_{\mathcal D}^n(\{\mathbf z_i\})
=
\frac{q_i}{\sum_jq_j}.
$$

相对于均匀经验分布 $\pi_{\mathcal D}$ 的 importance ratio 是

$$
\boxed{
\rho_{\mathcal D}^{\mathrm{cond}}(\mathbf z_i)
=
\frac{q_i}{D^{-1}\sum_jq_j}
}.
$$

此时 $\beta$ 和 background mass 都不应出现在 importance ratio 中。

## 八、probability-mass decomposition 本身是否正确

原文把 $w_{\mathcal D}$ 分解为

$$
\frac{\beta}{D}\frac{q_i}{Z}
=
\frac{\beta}{D}\frac{1}{Z}
+
\frac{\beta}{D}\frac{q_i-1}{Z}.
$$

这个代数等式作为新 atom probability mass 的分解是正确的。

但是，第二项不能单独称为完整的 probability-mass increment，因为

$$
\begin{aligned}
\pi^n(\{\mathbf z_i\})
-
\pi^o(\{\mathbf z_i\})
&=
\frac{\beta}{D}
\left(
\frac{q_i}{Z}-1
\right)\\
&=
\frac{\beta}{D}
\left[
\frac{q_i-1}{Z}
+
\left(
\frac{1}{Z}-1
\right)
\right].
\end{aligned}
$$

完整变化还包含所有 observed atoms 共享的 normalization shift $1/Z-1$。

如果把 probability-mass decomposition 转换为相对于旧 empirical measure 的 density-ratio decomposition，需要整体除以旧 atom mass $\beta/D$：

$$
\boxed{
\rho_{\mathcal D}(\mathbf z_i)
=
\frac{1}{Z}
+
\frac{q_i-1}{Z}
=
\frac{q_i}{Z}
}.
$$

因此：

- $\frac{\beta}{D}\frac{q_i}{Z}$ 回答“新分布在 atom $\mathbf z_i$ 上放了多少 probability mass”；
- $\frac{q_i}{Z}$ 回答“从旧分布采到 $\mathbf z_i$ 后，sample loss 应乘多大的 importance weight”。

## 九、对 `ours_idea.tex` importance-ratio 部分的最终判断

| 原文内容 | 判断 | 原因 |
| --- | --- | --- |
| $\pi^n=q\pi^o/Z$ | 正确 | 这是 reward tilting 的 measure-level 定义。 |
| $w_r=\pi^n/\pi^o=q/Z$ | 正确 | 这是真正的 importance ratio。 |
| $Z=(1-\beta)+\frac{\beta}{D}\sum_iq_i$ | 正确 | 条件是 background reward 已平移为零。 |
| $w_{\mathrm{bg}}=1/Z$ | 正确 | 它是 background 上的 density ratio。 |
| $w_{\mathcal D}=\frac{\beta}{D}\frac{q_i}{Z}$ | 量本身正确，名称错误 | 它是 new atom probability mass，不是 importance ratio。 |
| 把 $w_{\mathcal D}$ 放进 $\mathbf z\sim\pi^o$ 的 objective | 错误 | 外层已经包含旧 sampling probability，又乘了一次 atom mass。 |
| minibatch mean 中使用 $\beta q_i/(DZ)$ | 错误 | batch mean 已经提供 $1/D$；正确的 observed-contribution coefficient 是 $\beta q_i/Z$。 |
| $w_{\mathcal D}$ 的 baseline 加 reward component 分解 | 代数正确 | 只能解释为 new atom mass decomposition，不能称为 importance-ratio decomposition。 |

最终需要把原文中的符号分成两个明确对象：

$$
\boxed{
m_i^n
:=
\pi^n(\{\mathbf z_i\})
=
\frac{\beta}{D}\frac{q_i}{Z}
}
$$

和

$$
\boxed{
\rho_i
:=
\frac{d\pi^n}{d\pi^o}(\mathbf z_i)
=
\frac{q_i}{Z}
}.
$$

只要外层已经是从 $\pi^o$ 或 $\pi_{\mathcal D}$ 进行 sampling，就必须使用相对于该 sampling measure 的 density ratio，而不能再次乘单个 atom 的 probability mass。
