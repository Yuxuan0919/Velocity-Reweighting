# Reweight Design



对于 $D$ 个采样样本的 rewards $\{r_1,\dots,r_D\}$：

首先，我们指定 mass shift center $b$，它也是判断正负样本集合的标准。

之后，得到 raw mass shift：

$$
r_i-b.
$$

此时 $r_i-b$ 的范围是不可控的，取决于 $r_i$ 的分布特点以及 $b$ 的取法。我们希望将 $r_i-b$ 可控地映射到标准的 $[-1,1]$ 范围，这样后续才能在固定的输入范围内设计非线性映射方法，从而更好地选择性地增强或减弱一部分样本的 mass shift 信号。

分别定义正负两侧的最大幅度：

$$
a_+^{\max}
=
\max_{r_i>b}|r_i-b|,
$$

$$
a_-^{\max}
=
\max_{r_i<b}|r_i-b|.
$$

然后定义 normalized mass shift：

$$
\Delta_i
=
\frac{r_i-b}{a_+^{\max}},
\qquad r_i>b,
$$

$$
\Delta_i
=
\frac{r_i-b}{a_-^{\max}},
\qquad r_i<b.
$$

此时 $\Delta_i$ 会被限制在 $[-1,1]$ 范围内，并且在正负样本集合均非空时，正负两侧的最大幅度分别达到 $1$ 和 $-1$，因此不再需要进行额外的 clip 操作。

接下来，通过非线性映射对这些样本进行选择性地增强和减弱。我们希望把更多 mass 分配给明显高于 $b$ 的样本，而不是那些仅略高于 $b$ 的 borderline samples。后者并不一定是真正的高质量样本，并且随着模型优化，它们之后可能重新成为负样本。如果当前先明显增强、之后又进行削弱，这种矛盾的优化方向可能增加训练优化难度。

比如，可以使用 Double Tanh：

$$
s_i
=
\frac{1}{2}
\left[
\tanh\left(k(\Delta_i-c)\right)
+
\tanh\left(k(\Delta_i+c)\right)
\right].
$$

由于最后还会除以 $m^+$ 或 $m^-$，nonlinear mapping 的作用本质上是**重新分配固定的 mass shift magnitude budget**，而不是决定这一轮整体 mass shift 的大小。

最后，我们对映射后的 $s_i$ 进行正负两侧 mass shift magnitude 的独立归一化：

$$
m^-
=
\sum_i (-s_i)\mathbf{1}[s_i<0],
$$

$$
m^+
=
\sum_i s_i\mathbf{1}[s_i\ge 0].
$$

引入额外的 mass shift magnitude 超参数 $\beta$，定义

$$
w_i-1
=
\begin{cases}
\displaystyle
\beta\frac{s_i}{m^-},
& s_i<0,\\[8pt]
\displaystyle
\beta\frac{s_i}{m^+},
& s_i\ge 0.
\end{cases}
$$

这样我们有

$$
\sum_i(w_i-1)=0.
$$

同时，正负两侧的总 mass shift magnitude 分别固定为 $\beta$：

$$
\sum_{s_i\ge0}(w_i-1)=\beta,
\qquad
\sum_{s_i<0}(w_i-1)=-\beta.
$$

因此，不同训练轮次即使 reward 分布以及 $s_i$ 的整体幅度不同，也不会导致总的 shift signal 大幅变化。每一轮的总 mass shift magnitude 由额外的超参数 $\beta$ 显式控制，而 $s_i$ 主要负责决定这些固定的 mass shift magnitude 应该如何在样本之间分配。



#### 进一步解释 Max Normalization 和 Nonlinear Mapping 的实际数学上的作用

假设我们有两组 positive raw mass shift $r_i-b$：

$$
\{0.1,0.12,0.11,0.13,1.0\},
$$

以及

$$
\{0.1,0.12,0.11,0.13\}.
$$

对于第一组，

$$
a_+^{\max}=1,
$$

因此 normalization 后保持不变。由于组内已经存在一个显著更好的正样本 $1.0$，我们实际上并不需要充分利用 $0.1,0.12,0.11,0.13$ 这些较弱的正信号，而可以通过后续 nonlinear mapping 进一步将更多 mass 集中到这个明显更好的样本上。

对于第二组，

$$
a_+^{\max}=0.13,
$$

normalization 后近似变为

$$
\{0.77,0.92,0.85,1.0\}.
$$

此时组内并不存在一个真正突出的强正样本，因此 max normalization 会将当前这些样本在标准区间内展开，再结合 nonlinear mapping，从中选择相对更好的样本，并向它们分配更多 mass。

因此，我们实际上**需要这种对极端样本的敏感性**：当组内出现显著高或显著低的样本时，我们应该充分利用这些更明确、更可靠的优化信号；而当不存在明显的极端样本时，则利用 max normalization 和 nonlinear mapping 提高组内相对差异的分辨率。

---







## 方法探索：

整个 reweight 方法可以统一为以下三步：

1. 通过上面所述的 mass shift center $b$ 和 Max Normalization 得到 normalized linear mass shift：

$$
c_i = \Delta_i \in [-1,1].
$$

​	将 $c_i$ 按从小到大排列为
$$
c_1\le \cdots \le c_j\le 0
\le c_{j+1}\le \cdots \le c_D.
$$

2. 通过某种 mapping 或 sample selection 得到重新分配后的 signal $s_i$。

3. 对正负两侧分别进行 mass normalization，使每一侧的总 mass shift magnitude 固定为 $\beta$。

我们现在主要对中间的一步进行探索：



### Scheme A: Double Tanh Soft Selection

使用 Double Tanh 对全部样本进行连续的 soft selection：

$$
s_i
=
\frac{1}{2}
\left[
\tanh\left(k(c_i-c)\right)
+
\tanh\left(k(c_i+c)\right)
\right].
$$

靠近 $0$ 的 borderline samples 得到较小的 $|s_i|$，而靠近 $\pm1$ 的 strong samples 得到较大的 $|s_i|$。

最后仍然使用

$$
m^-
=
\sum_i(-s_i)\mathbf{1}[s_i<0],
\qquad
m^+
=
\sum_i s_i\mathbf{1}[s_i>0],
$$

以及

$$
w_i-1
=
\begin{cases}
\displaystyle
\beta\frac{s_i}{m^-},
& s_i<0,\\[8pt]
\displaystyle
\beta\frac{s_i}{m^+},
& s_i>0.
\end{cases}
$$

这一方案需要调节的超参数：

- $\beta$: 整体 mass shift 强度
- $k$: 控制 tanh 斜率， k 越大越陡
- $c$: 控制 double tanh 两个对称的中心位置



---

### Scheme B: Symmetric Hard Selection

第一种 linear alternative 直接用 **hard selection** 替代 Double Tanh：直接将靠近 $0$ 的样本 signal 置为 $0$。

为了系统地控制 hard selection 的强度，我们可以引入一个 retained mass ratio 参数：

$$
\rho \in (0,1].
$$

 $\rho$ 表示保留每一侧原始 cumulative mass 的比例。

具体来说，先定义正负两侧的原始 total magnitude：

$$
M^-
=
\left|\sum_{i=1}^{j}c_i\right|,
\qquad
M^+
=
\sum_{i=j+1}^{D}c_i.
$$

对于给定的 $\rho$，分别从 negative tail 和 positive tail 开始累计，选择最接近目标 cumulative mass 的边界：

$$
l_{\rho}
=
\arg\min_{1\le l\le j}
\left|
\left|\sum_{i=1}^{l}c_i\right|
-
\rho M^-
\right|,
$$

$$
h_{\rho}
=
\arg\min_{j+1\le h\le D}
\left|
\sum_{i=h}^{D}c_i
-
\rho M^+
\right|.
$$

然后定义

$$
s_i^{(B,\rho)}
=
\begin{cases}
c_i, & i\le l_{\rho},\\
0, & l_{\rho}<i<h_{\rho},\\
c_i, & i\ge h_{\rho}.
\end{cases}
$$

最后仍然使用之前的 fixed $\beta$ budget normalization：

$$
w_i-1
=
\begin{cases}
\displaystyle
\beta\frac{s_i}{m^-},
& s_i<0,\\[8pt]
\displaystyle
\beta\frac{s_i}{m^+},
& s_i>0.
\end{cases}
$$

可以测试三组 setting， 先固定 $\beta$ = 1：

- **B-$1/3$**：$\rho=1/3$，每一侧只保留贡献约 $1/3$ cumulative mass 的 strongest signal samples；
- **B-$1/2$**：$\rho=1/2$，每一侧保留约 $1/2$ cumulative mass；
- **B-$2/3$**：$\rho=2/3$，每一侧保留约 $2/3$ cumulative mass。

$\rho$ 越小，hard selection 越激进，训练越集中在极端样本；$\rho$ 越大，保留的中等强度样本越多。当 $\rho=1$ 时，该方案退化为 linear mapping。

因此 Scheme B 可以直接验证：**Double Tanh 的收益究竟来自连续 nonlinear mapping，还是主要来自将训练信号集中到 strong signal samples,削弱对中性样本的使用。**

---



### Scheme C: Asymmetric Hard Selection

第二种 linear alternative 采用 asymmetric 设计：**negative side 全部保留，只对 positive side 做 strong-tail selection。**

Negative side 固定为

$$
l=j,
$$

即所有 $c_i\le0$ 的 samples 都参与训练。

Positive side 同样引入 retained mass ratio $\rho$。定义

$$
M^+
=
\sum_{i=j+1}^{D}c_i,
$$

并选择

$$
h_{\rho}
=
\arg\min_{j+1\le h\le D}
\left|
\sum_{i=h}^{D}c_i
-
\rho M^+
\right|.
$$

于是

$$
s_i^{(C,\rho)}
=
\begin{cases}
c_i, & i\le j,\\
0, & j<i<h_{\rho},\\
c_i, & i\ge h_{\rho}.
\end{cases}
$$

最后仍然用相同的 fixed $\beta$ budget normalization。

测试三组 setting， 先固定 $\beta$ = 1：

- **C-$1/3$**：positive side 只保留贡献约 $1/3$ cumulative positive mass 的 strongest samples；
- **C-$1/2$**：positive side 保留约 $1/2$ cumulative positive mass；
- **C-$2/3$**：positive side 保留约 $2/3$ cumulative positive mass。

需要注意：由于最终 positive total budget 始终重新归一化到 $\beta$，更小的 $\rho$ 不会减小 positive side 的总训练强度，而是会将相同的 positive mass budget 集中到更少、更明确的 strong positive samples 上。

因此 Scheme C 可以验证：**positive samples 是否需要比 negative samples 更严格的 confidence filtering，以及这种 asymmetry 的强度应该多大。**





####  关于线性情况下 Max Normalization 的实际作用

需要明白一点：**只要后续 mapping 保持线性，那么无论是否进行 hard selection，前面的 Max Normalization 实际上都不会影响最终的 weight shift。**

对于 positive side，Max Normalization 后有

$$
c_i
=
\frac{r_i-b}{a_+^{\max}}.
$$

如果不做 selection，直接进行最终的 mass normalization，则

$$
w_i-1
=
\beta
\frac{c_i}{\sum_{k:c_k>0}c_k}
=
\beta
\frac{\frac{r_i-b}{a_+^{\max}}}
{\sum_{k:r_k>b}\frac{r_k-b}{a_+^{\max}}}
=
\beta
\frac{r_i-b}{\sum_{k:r_k>b}(r_k-b)}.
$$

因此 $a_+^{\max}$ 会被约掉；negative side 同理。

即使做 hard selection，以 positive side 为例，

$$
h_{\rho}
=
\arg\min_h
\left|
\sum_{i=h}^{D}c_i
-
\rho\sum_{i=j+1}^{D}c_i
\right|.
$$

代入 $c_i=(r_i-b)/a_+^{\max}$ 后，整个目标只多出一个共同的正比例系数 $1/a_+^{\max}$，因此 $h_{\rho}$ 不会改变。完成 selection 后，$a_+^{\max}$ 又会在最终的 fixed $\beta$ budget normalization 中被约掉。

因此 max normalization 真正产生作用的场景是 Scheme A 这类 nonlinear mapping，因为此时

$$
s_i=f\!\left(\frac{r_i-b}{a_{\pm}^{\max}}\right)
$$

中的 scale 无法再从 nonlinear mapping 中约掉。此时 Max Normalization 才真正负责把输入稳定到 $[-1,1]$，并决定不同样本落在 nonlinear mapping 的哪个区域。





#### 关于老师写的 $\gamma_l$ 和 $\gamma_h$

原始版本中引入

$$
\gamma_l
\left|
\sum_{i=1}^{l}c_i
\right|
=
\gamma_h
\left|
\sum_{i=h}^{D}c_i
\right|
$$

来重新平衡 positive 和 negative tails。

但是，如果最终仍然使用之前提出的 fixed $\beta$ buget normalization，那么 $\gamma_l$ 和 $\gamma_h$ 会被归一化直接约掉。例如 negative side：

$$
\beta
\frac{\gamma_l c_i}
{\sum_{k:s_k<0}|\gamma_l c_k|}
=
\beta
\frac{c_i}
{\sum_{k:s_k<0}|c_k|}.
$$

positive side 同理。因此不需要额外引入 $\gamma_l$ 和 $\gamma_h$。







## Ablation 组合

#### 1. Total mass shift magnitude $\beta$ Ablation:

首先应当在所有 hard selection 实验之前，先进行一组 **Linear-1 / No Selection**。这一组固定
$$
\rho=1,
$$

即不删除任何 positive 或 negative samples，此时 Scheme B 和 Scheme C 完全退化为同一个 linear formulation：

$$
s_i=c_i.
$$

它主要用于验证 **mass shift center $b$ + linear weighting + fixed mass budget normalization** 这一基础 formulation 本身能否达到什么水平。

在这一组中先不引入 selection，只测试不同的 total mass shift magnitude $\beta$：

- **Linear-1, $\beta=0.5$**
- **Linear-1, $\beta=1$**
- **Linear-1, $\beta=5$**
- **Linear-1, $\beta=10$**

如果其中能够找到与 baseline 接近或更好的 $\beta$，再固定合适的 $\beta$，继续比较 selection 方案，从而把 **基础 linear formulation 的影响** 和 **hard selection 本身的影响** 分开。



#### 2. Hard Selection Ablation

| Scheme           | Retained Cumulative Mass | Mapping   | Final Budget |
| ---------------- | ------------------------ | --------- | ------------ |
| B-$1/3$          | $1/3$ per side           | Linear    | $\pm\beta$   |
| B-$1/2$          | $1/2$ per side           | Linear    | $\pm\beta$   |
| B-$2/3$          | $2/3$ per side           | Linear    | $\pm\beta$   |
| C-$1/3$          | Positive $1/3$           | Linear    | $\pm\beta$   |
| C-$1/2$          | Positive $1/2$           | Linear    | $\pm\beta$   |
| C-$2/3$          | Positive $2/3$           | Linear    | $\pm\beta$   |
| A (待定实验方案) |                          | Nonlinear |              |

这组实验主要两个目的：

1. **B ：hard tail selection 的强度是否重要？** 通过 $1/3 \rightarrow 1/2 \rightarrow 2/3$ 可以观察训练是否更偏好极端样本，还是需要保留更多中等强度信号。
2. **B vs. C：positive / negative 是否应该 asymmetric selection？** 在相同 $\rho$ 下比较 B 和 C，可以直接判断是否有必要保留全部 negative samples，而只过滤 positive borderline samples。



所有实验保持：

- KL coefficient = 1e-4
- Timestep-fraction = 1.0

以及得到 $w_i - 1$ 后的完整训练 formulation (原本的 $\beta$ 在这里已经写到 $w_i - 1$ 里面了）:
$$
\mathbf{x}_i^{\text{target}} = \mathbf{x}_i^{\text{old}} + (w_i - 1)(\mathbf{x}_i-\mathbf{x}_i^{\text{old}})
$$

$$
\mathcal{L}_{\text{policy}} = \frac{1}{B}\sum_{i=1}^B\left[\frac{||\mathbf{x}_i^{\theta}-\mathbf{x}_i^{\text{target}}||^2}{\operatorname{sg}[\operatorname{mean}|\mathbf{x}_i-\mathbf{x}_i^{\text{old}}|}  \right]
$$

