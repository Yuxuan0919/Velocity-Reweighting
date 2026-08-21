# Quality-Adaptive Quantile

## 1. Reward 归一化

设第 k 轮训练中，共采样 C 个 prompt，每个 prompt 产生 K 个 rollout 样本，总样本数为

$$
D = CK.
$$

记第 j 个 prompt 的第 i 个样本 reward 为 $r_{j,i}$。

首先要将 raw reward 归一化到 $[0,1]$ 得到 normalized reward $\tilde r_{j,i}$（本身满足的不需要做这一步）

---

## 2. 估计当前训练阶段的整体样本质量

用当前一轮全部 D 个样本的平均 normalized reward 衡量当前 policy 的整体质量：

$$
s_k
=
\frac{1}{D}
\sum_{j=1}^{C}
\sum_{i=1}^{K}
\tilde r_{j,i}.
$$

由于 $\tilde r_{j,i}\in[0,1]$，因此

$$
s_k\in[0,1].
$$

$s_k$ 越大，表示当前 policy 整体生成质量越高。

---

## 3. Adaptive Quantile

根据每一轮训练采样出的样本整体质量，动态地决定优化的 quantile level：

$$
q_k = 1-s_k.
$$

其中 $q_k$ 表示每个 rollout group 中预期需要被削弱的样本比例。

因此：

- 训练早期，$s_k$ 较小，$q_k$ 较大，只增强少量高 reward 样本；
- 训练后期，$s_k$ 较大，$q_k$ 较小，仅削弱少量低 reward 样本。

---

## 4. Baseline Reward (Per-prompt)

 分位数的值 $q_k$ 由当前训练轮次的全部采样样本总体决定，并计算每个 prompt group 中的 baseline reward 值。

对于第 j 个 prompt：

$$
c_j
=
Q_{q_k}
\left(
\tilde r_{j,1},
\ldots,
\tilde r_{j,K}
\right),
$$

 $Q_q(\cdot)$ 表示 $q$ 分位数。

---

## 5. 计算 Weight

定义每个样本相对于当前 prompt baseline 的 centered reward：

$$
a_{j,i}
=
\tilde r_{j,i}-c_j.
$$

再将其映射为最终的 mass weight。一个简单形式为

$$
W_{j,i}
=
1+
\operatorname{clip}
\left(
\frac{a_{j,i}}{Z},
-A,
A
\right),
$$

$Z$ 取 global reward variance (D个样本一起算std, NFT 实际代码实现中使用同样做法），$A$ 限制最大 mass shift。

因此：

$$
\tilde r_{j,i}>c_j
\quad\Rightarrow\quad
W_{j,i}>1,
$$

表示该样本对应的 probability mass 被增强；

$$
\tilde r_{j,i}<c_j
\quad\Rightarrow\quad
W_{j,i}<1,
$$

表示该样本对应的 probability mass 被削弱。

---

## 6. Probability 角度直观解释

由于

$$
c_j = Q_{q_k}(R_j),
$$

近似有

$$
P(\tilde r<c_j)\approx q_k,
$$

以及

$$
P(\tilde r>c_j)\approx 1-q_k.
$$

又

$$
q_k=1-s_k,
$$

因此

$$
P(\tilde r>c_j)\approx s_k.
$$

也就是说：

> **当前 policy 的平均 normalized reward 决定当前 rollout population 中大约有多大比例的样本应该获得 positive mass shift。**

例如，当

$$
s_k=0.3,
$$

则每个 prompt group 中大约 top $30\%$ 的样本被增强，bottom $70\%$ 的样本被削弱。

当模型训练到中后期

$$
s_k=0.8,
$$

每个 prompt 中大约 top $80\%$ 的样本被增强，只有 bottom $20\%$ 的样本应该被削弱。

因此，方法根据当前 policy 的整体质量自动调整正负 mass shift 的比例，而不再固定使用 group mean 将每个 prompt 内的样本强制划分为近似对称的正负更新。





##  为什么我预期优于 Group-Mean Baseline

原本的 group-mean baseline 始终以组内平均 reward 作为正负样本的分界，因此本质上无论当前模型整体生成质量高低，每个 group 中都必然同时存在被增强和被削弱的样本。这会导致训练早期一些绝对质量仍然较差、但相对高于组内均值的样本被错误增强；训练后期则可能削弱一些绝对质量已经较高、但略低于组内均值的样本。

Quality-Adaptive Quantile 不再固定正负样本的相对比例，而是利用当前 policy 的整体 reward 水平动态决定 quantile。当模型整体质量较低时，仅增强 reward 分布中的高质量长尾样本；随着模型整体质量提高，越来越多样本可以获得 positive mass shift，仅保留少量低质量样本被削弱。因此，该设计使 mass reweight 的正负比例能够随训练阶段自适应变化，相比固定的 group-mean rule，它更符合我们的根本目标：**判断一个样本在当前 policy 状态下是否值得增加或减少 probability mass，而不是强制进行组内“零和”的相对比较。**
