# 2-Branch 与 Single-Branch 的推导比较

## 目的

本文推导当前 `our idea` 的：

- 2-branch 版本
- single-branch 版本

在**不考虑 `weight_factor` 归一化项**时，二者是否等价。

这里也暂时忽略：

- `adv_clip_max`
- batch mean
- KL 项

只分析 policy loss 的主体形式。

## 统一记号

定义 `x` 空间中的几个量：

$$
x_\theta = x_t - t v_\theta
$$

$$
x_{\mathrm{old}} = x_t - t v_{\mathrm{old}}
$$

$$
u = x_0 - x_{\mathrm{old}}
$$

$$
d = x_\theta - x_{\mathrm{old}}
$$

其中：

- $u$ 表示真实样本相对 old policy 的方向
- $d$ 表示当前模型相对 old policy 的移动方向

## 一、2-Branch 版本

当前 2-branch 版 `our idea` 的核心形式是：

$$
v^+ = \beta r\, v_\theta + (1-\beta r)\, v_{\mathrm{old}}
$$

$$
v^- = (1+\beta(1-r))v_{\mathrm{old}} - \beta(1-r) v_\theta
$$

并使用

$$
L_{2\text{branch}}
=
\frac{1}{\beta}
\left(
\|x^+ - x_0\|^2
+
\|x^- - x_0\|^2
\right)
$$

其中 $x^+$、$x^-$ 分别是对应 velocity 的 `x` 空间预测。

### 正分支

由于系数和为 1，正分支在 `x` 空间中可写成：

$$
x^+ = \beta r\, x_\theta + (1-\beta r)\, x_{\mathrm{old}}
$$

即

$$
x^+ = x_{\mathrm{old}} + \beta r\, d
$$

因此

$$
L_+ = \|x^+ - x_0\|^2 = \|\beta r\, d - u\|^2
$$

### 负分支

负分支在 `x` 空间中可写成：

$$
x^- = x_{\mathrm{old}} - \beta(1-r)\, d
$$

因此

$$
L_- = \|x^- - x_0\|^2 = \|-\beta(1-r)\, d - u\|^2
$$

### 合并

所以 2-branch 的主体 loss 为

$$
L_{2\text{branch}}
=
\frac{1}{\beta}
\left(
\|\beta r\, d - u\|^2
+
\|-\beta(1-r)\, d - u\|^2
\right)
$$

展开得到：

$$
L_{2\text{branch}}
=
\beta\big(r^2 + (1-r)^2\big)\|d\|^2
- 2(2r-1)d^\top u
+ \frac{2}{\beta}\|u\|^2
$$

### 写成单目标形式

对上式完成平方：

$$
L_{2\text{branch}}
=
\beta\big(r^2 + (1-r)^2\big)
\left\|
d -
\frac{2r-1}{\beta\big(r^2 + (1-r)^2\big)}u
\right\|^2
+ \mathrm{const}
$$

因此，2-branch 等价于一个单目标回归，其有效 target 系数为

$$
c_{2\text{branch}}
=
\frac{2r-1}{\beta\big(r^2 + (1-r)^2\big)}
$$

也就是说：

$$
d \to c_{2\text{branch}}\, u
$$

或者写回 `x` 空间：

$$
x_\theta
\to
x_{\mathrm{old}}
+
\frac{2r-1}{\beta\big(r^2 + (1-r)^2\big)}
(x_0 - x_{\mathrm{old}})
$$

## 二、Single-Branch 版本

single-branch 版本使用单一 add-delete target：

$$
x^{\mathrm{target}}
=
x_{\mathrm{old}}
+
\beta(2r-1)(x_0 - x_{\mathrm{old}})
$$

即

$$
x^{\mathrm{target}} = x_{\mathrm{old}} + \beta(2r-1)u
$$

loss 主体为：

$$
L_{1\text{branch}}
=
\frac{1}{\beta}
\|x_\theta - x^{\mathrm{target}}\|^2
$$

即

$$
L_{1\text{branch}}
=
\frac{1}{\beta}
\|d - \beta(2r-1)u\|^2
$$

展开得到：

$$
L_{1\text{branch}}
=
\frac{1}{\beta}\|d\|^2
- 2(2r-1)d^\top u
+ \beta(2r-1)^2\|u\|^2
$$

因此它对应的有效 target 系数为

$$
c_{1\text{branch}} = \beta(2r-1)
$$

也就是说：

$$
d \to c_{1\text{branch}}\, u
$$

## 三、二者是否等价

比较两个系数：

$$
c_{2\text{branch}}
=
\frac{2r-1}{\beta\big(r^2 + (1-r)^2\big)}
$$

$$
c_{1\text{branch}}
=
\beta(2r-1)
$$

若二者相等，则需要满足：

$$
\beta(2r-1)
=
\frac{2r-1}{\beta\big(r^2 + (1-r)^2\big)}
$$

整理得到：

$$
(2r-1)
\left[
\beta^2\big(r^2 + (1-r)^2\big) - 1
\right]
= 0
$$

因此只有两类情况会相等：

### 情况 1

$$
2r-1 = 0
\quad \Longrightarrow \quad
r = 0.5
$$

### 情况 2

$$
\beta^2\big(r^2 + (1-r)^2\big) = 1
$$

## 四、默认 $\beta=1$ 时的情况

当 $\beta=1$ 时：

$$
c_{2\text{branch}}
=
\frac{2r-1}{r^2 + (1-r)^2}
$$

$$
c_{1\text{branch}}
=
2r-1
$$

若二者相等，则要求：

$$
\frac{2r-1}{r^2 + (1-r)^2} = 2r-1
$$

因此只有：

$$
r = 0.5
$$

或者

$$
r^2 + (1-r)^2 = 1
$$

后者只在

$$
r = 0 \quad \text{或} \quad r = 1
$$

时成立。

所以在默认 $\beta=1$ 下，2-branch 和 single-branch 只在

$$
r \in \{0,\; 0.5,\; 1\}
$$

时相同，其余情况下都不同。

## 五、直观解释

当 $\beta=1$ 时：

$$
c_{2\text{branch}}
=
\frac{2r-1}{r^2 + (1-r)^2}
$$

而

$$
r^2 + (1-r)^2 \in [0.5, 1]
$$

因此对于大多数中间区域的 $r$，有：

$$
|c_{2\text{branch}}| \ge |c_{1\text{branch}}|
$$

而且通常严格大于。

这意味着：

- 2-branch 更激进
- single-branch 更保守

例如，当 $r=0.75$ 时：

$$
c_{1\text{branch}} = 2r-1 = 0.5
$$

$$
c_{2\text{branch}}
=
\frac{0.5}{0.75^2 + 0.25^2}
=
\frac{0.5}{0.625}
= 0.8
$$

当 $r=0.25$ 时：

$$
c_{1\text{branch}} = -0.5
$$

$$
c_{2\text{branch}} = -0.8
$$

所以 2-branch 会把中间 reward 区域的样本推得更远，single-branch 则更温和。

## 六、另一种理解

把两个 loss 展开后对比：

### 2-branch

$$
L_{2\text{branch}}
=
\beta\big(r^2 + (1-r)^2\big)\|d\|^2
- 2(2r-1)d^\top u
+ \frac{2}{\beta}\|u\|^2
$$

### single-branch

$$
L_{1\text{branch}}
=
\frac{1}{\beta}\|d\|^2
- 2(2r-1)d^\top u
+ \beta(2r-1)^2\|u\|^2
$$

可以看到：

- 两者的交叉项
  $$
  -2(2r-1)d^\top u
  $$
  是一样的
- 真正不同的是 $\|d\|^2$ 的系数

对 $\beta=1$ 而言：

- 2-branch 的 $\|d\|^2$ 系数是
  $$
  r^2 + (1-r)^2
  $$
- single-branch 的 $\|d\|^2$ 系数是
  $$
  1
  $$

因为

$$
r^2 + (1-r)^2 \le 1
$$

所以 2-branch 对“偏离 old policy”的惩罚更弱，因此会得到更大的有效 target 系数。

## 结论

在**不考虑 `weight_factor` 归一化项**的情况下：

- 2-branch `our idea`
- single-branch `our idea`

**一般不等价。**

更具体地说：

- 2-branch 的有效 target 系数是
  $$
  \frac{2r-1}{\beta(r^2+(1-r)^2)}
  $$
- single-branch 的有效 target 系数是
  $$
  \beta(2r-1)
  $$

在默认 $\beta=1$ 时，它们只在

$$
r = 0,\; 0.5,\; 1
$$

时相同，其余情况下都不同，而且 2-branch 通常更激进。
