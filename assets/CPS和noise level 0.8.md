# CPS 与 `noise_level=0.8` 的含义





- **CPS（Coefficients-Preserving Sampling）的作用：在 Flow Matching 的确定性采样轨迹中引入可控随机性，同时保持对应时间步原有的噪声系数。**



SD3 的 Flow Matching 推理通常使用确定性的 ODE 轨迹。同一个 prompt 和相同的初始状态会得到确定性的下一步状态，因此难以直接将每一步解释为能够产生多个不同结果的随机策略。

Flow-GRPO 需要针对同一个 prompt 采样一组不同的图像，并利用组内 reward 的相对差异计算 advantage。为此，Flow-GRPO-Fast 会先沿 ODE 进行确定性采样，在随机选中的少数中间步骤注入噪声、产生不同的轨迹分支，之后再继续使用 ODE 完成生成。CPS 就是这些随机步骤所采用的一种采样方式。



## Flow Matching 中的端点估计

按照 SD3 使用的 Rectified Flow 表示，当前状态可以写为

$$
x_\sigma=(1-\sigma)x_0+\sigma x_1,
$$

其中：

1. $x_0$ 表示干净图像端点；
2. $x_1$ 表示高斯噪声端点；
3. $\sigma$ 表示当前状态的噪声强度；
4. 模型预测的 velocity 满足 $v_\theta\approx x_1-x_0$。

根据当前状态 $x_\sigma$ 和模型输出 $v_\theta$，可以分别估计干净图像端点和噪声端点：

$$
\widehat{x}_0=x_\sigma-\sigma v_\theta,
$$

$$
\widehat{x}_1=x_\sigma+(1-\sigma)v_\theta.
$$

Flow-GRPO 的官方实现对应为

```python
pred_original_sample = sample - sigma * model_output
noise_estimate = sample + model_output * (1 - sigma)
```



## CPS 的更新公式

令

$$
\eta=\mathtt{noise\_level},
\qquad
\phi=\frac{\pi}{2}\eta.
$$

当下一个时间步的噪声强度为 $\sigma_{\mathrm{prev}}$ 时，CPS 的更新可以写成

$$
\begin{aligned}
x_{\mathrm{prev}}
=
&(1-\sigma_{\mathrm{prev}})\widehat{x}_0\\
&+\sigma_{\mathrm{prev}}\cos(\phi)\widehat{x}_1\\
&+\sigma_{\mathrm{prev}}\sin(\phi)\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\end{aligned}
$$

其中噪声部分被拆成两项：

1. $\sigma_{\mathrm{prev}}\cos(\phi)\widehat{x}_1$：保留当前轨迹所预测的原噪声成分；
2. $\sigma_{\mathrm{prev}}\sin(\phi)\epsilon$：重新采样并注入的新随机噪声。

官方代码中的等价实现为

```python
std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)

prev_sample_mean = (
    pred_original_sample * (1 - sigma_prev)
    + noise_estimate
    * torch.sqrt(sigma_prev**2 - std_dev_t**2)
)

prev_sample = prev_sample_mean + std_dev_t * variance_noise
```

当 $\mathtt{noise\_level}\in[0,1]$ 时，

$$
\sqrt{\sigma_{\mathrm{prev}}^2-\mathtt{std\_dev}_t^2}
=
\sigma_{\mathrm{prev}}\cos\left(\frac{\pi}{2}\eta\right),
$$

因此上述代码与 CPS 更新公式一致。



## 为什么称为 Coefficients-Preserving

CPS 中旧噪声与新噪声的平方系数之和为

$$
\begin{aligned}
&\left[\sigma_{\mathrm{prev}}\cos(\phi)\right]^2
+\left[\sigma_{\mathrm{prev}}\sin(\phi)\right]^2\\
&=\sigma_{\mathrm{prev}}^2
\left(\cos^2(\phi)+\sin^2(\phi)\right)\\
&=\sigma_{\mathrm{prev}}^2.
\end{aligned}
$$

因此，CPS 改变的是噪声成分的来源，而不是随意增大该时间步的总噪声尺度：一部分原轨迹噪声被新的随机噪声替换，但二者合成后的平方系数仍与 scheduler 为该时间步指定的噪声系数一致。

这也是 CPS 与直接在确定性更新结果上额外叠加高斯噪声的重要区别。后者可能使实际噪声尺度超过目标时间步应有的尺度，而 CPS 尽量在提供随机探索的同时保持 Flow Matching 轨迹的系数平衡。



## `noise_level=0.8` 的具体含义

当

$$
\eta=0.8
$$

时，对应角度为

$$
\phi=0.8\times\frac{\pi}{2}=0.4\pi=72^\circ.
$$

因此

$$
\sin(72^\circ)\approx0.9511,
\qquad
\cos(72^\circ)\approx0.3090.
$$

CPS 更新变成

$$
\begin{aligned}
x_{\mathrm{prev}}
=
&(1-\sigma_{\mathrm{prev}})\widehat{x}_0\\
&+0.3090\,\sigma_{\mathrm{prev}}\widehat{x}_1\\
&+0.9511\,\sigma_{\mathrm{prev}}\epsilon.
\end{aligned}
$$

从幅度系数看，新随机噪声的系数为 $0.9511\sigma_{\mathrm{prev}}$，而保留的预测噪声系数为 $0.3090\sigma_{\mathrm{prev}}$。从方差或噪声能量看：

$$
0.3090^2\approx0.0955,
$$

$$
0.9511^2\approx0.9045.
$$

也就是说：

1. 约 $9.55\%$ 的噪声能量来自当前轨迹预测的原噪声；
2. 约 $90.45\%$ 的噪声能量来自重新采样的高斯噪声；
3. 两者之和仍为该时间步原本的总噪声能量。

因此，`noise_level=0.8` **并不表示直接加入标准差为 $0.8$ 的高斯噪声，也不严格表示加入 $80\%$ 的新噪声**。它是 CPS 中的角度插值参数，实际对应约 $90.45\%$ 的新噪声能量。



## 不同 `noise_level` 的直观比较

| `noise_level` | 保留旧噪声的幅度系数 | 新噪声的幅度系数 | 直观含义 |
|---:|---:|---:|---|
| $0$ | $1.000$ | $0.000$ | 不注入新噪声，退化为确定性更新 |
| $0.2$ | $0.951$ | $0.309$ | 轻微随机扰动 |
| $0.5$ | $0.707$ | $0.707$ | 旧噪声和新噪声等幅混合 |
| $0.8$ | $0.309$ | $0.951$ | 强随机探索，大部分噪声被重新采样 |
| $1.0$ | $0.000$ | $1.000$ | 完全重新采样噪声成分 |

表中的数值是幅度系数。如果比较方差或能量占比，需要对幅度系数取平方。



## 对 GRPO 训练的影响

当 `noise_level` 较小时，同一个 prompt 产生的组内轨迹会比较相似，从而可能出现

$$
r_1\approx r_2\approx\cdots\approx r_K.
$$

此时组内 reward 标准差较小，归一化 advantage 可能缺少有效的相对差异，策略梯度信号也会偏弱。

增大 `noise_level` 会提高同一 prompt 下轨迹和最终图像的多样性，使 GRPO 更容易根据 reward 区分较好与较差的采样结果。`noise_level=0.8` 属于较强的探索设置，它接近于在选中的 CPS 步骤中重新采样噪声方向。

但探索强度并不是越高越好。过强随机性仍可能使生成结果偏离原来的高质量轨迹，并增加 reward 方差。CPS 只能保证噪声系数的平衡，不能保证任意任务、任意采样步数和任意 reward 下的 `0.8` 都是最优值。



## 与 `sde_window_size=3` 的关系

对于配置

```python
config.sample.noise_level = 0.8
config.sample.sde_window_size = 3
config.sample.sde_type = "cps"
```

`noise_level=0.8` 通常并不是在所有 denoising steps 上生效，而是在被选择的 CPS/SDE window 中生效。整体采样过程可以概括为

$$
\mathrm{ODE}
\longrightarrow
\underbrace{
\mathrm{CPS}\longrightarrow\mathrm{CPS}\longrightarrow\mathrm{CPS}
}_{\mathtt{sde\_window\_size}=3}
\longrightarrow
\mathrm{ODE}.
$$

窗口内的 CPS 步骤产生随机轨迹分支，窗口外继续使用确定性 ODE。这样可以把随机探索和训练重点限制在少数步骤内，避免整条采样轨迹都受到强随机噪声影响。



## 实现上的注意事项

1. `noise_level` 在 CPS 中应按 $[0,1]$ 范围内的插值参数理解，而不是普通 SDE 的直接噪声标准差。
2. `noise_level=0.8` 只决定单个 CPS 步骤中新旧噪声的混合比例，整体随机程度还同时取决于 `sde_window_size`、`sde_window_range`、采样总步数以及 CPS window 的位置。
3. 这些配置必须配合实际读取 `sde_type` 和 SDE window 参数的 Flow-GRPO-Fast 训练入口使用。如果训练脚本没有消费这些字段，仅在 config 中设置它们不会产生实际效果。
4. 官方配置中的 `0.8` 是经验上表现较好的默认值，不是理论上对所有训练任务都最优的固定常数。



## 参考

1. [Flow-GRPO 官方仓库与说明](https://github.com/yifan123/flow_grpo/blob/main/README.md)
2. [Flow-GRPO 官方配置](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
3. [Flow-GRPO 中 SD3 的 CPS 实现](https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py)
4. [Flow-GRPO: Training Flow Matching Models via Online Reinforcement Learning](https://arxiv.org/abs/2505.05470)
5. [Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching](https://arxiv.org/abs/2509.05952)
