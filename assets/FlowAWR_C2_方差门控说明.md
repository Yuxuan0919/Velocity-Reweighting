# FlowAWR 附录 C.2：自适应方差门控

[论文附录 C.2](https://arxiv.org/html/2606.30376) 这句话的意思是：在 OCR、GenEval 这类规则奖励任务中，**用当前整批样本的优势标准差，动态调节优势对训练目标的影响**，避免奖励波动大时更新过猛，也避免后期标准差接近零时除法失控。

对应到 [`compute_flowawr_advantages`](../scripts/train_nft_sd3_ours-singleloss-AWR.py#L301-L352)，先按每个 prompt 的 $G$ 张图计算论文式 (11) 的组内相对优势：

$$
A_i=\frac{\exp(R_i/\gamma)}{\frac{1}{G}\sum_{j=1}^{G}\exp(R_j/\gamma)}-1,
\qquad \gamma=\max\bigl(\operatorname{std}_{\text{整批}}(R),10^{-6}\bigr).
$$

随后，对**所有 GPU 汇总后的本轮 rollout** 中的 $A_i$ 求标准差 $s_A$；仅在 OCR/GenEval 且开启 `awr_variance_gate` 时，使用

$$
\widetilde A_i=\frac{A_i}{\max(s_A,10^{-2})}.
$$

这里的 $10^{-2}$ 是**分母下限**，不是把优势裁剪到 $10^{-2}$：当 $s_A=2$ 时，优势缩小一半；当 $s_A=0.001$ 时，除以 $0.01$ 而非 $0.001$，放大倍数最多为 $100$。$s_A\ge 0.01$ 时，归一化使整批优势的标准差变为 1；因此 $s_A>1$ 会压低更新幅度，$s_A<1$ 则可能放大优势。下限阻止后期小方差造成无限放大，但门控本身不会把样本置零，也不保证每个优势值都不超过 1。

最后，$\widetilde A_i$ 乘在速度修正项上，形成训练目标 $v^{\mathrm{old}}+\widetilde A_i(u_t-v^{\mathrm{old}})$（[代码](../scripts/train_nft_sd3_ours-singleloss-AWR.py#L1151-L1166)）。注意这里有**两个不同的标准差**：奖励标准差决定指数权重的温度 $\gamma$；优势标准差决定门控缩放。

**与原文的歧义：** [论文式 (11) 及消融实验](https://arxiv.org/html/2606.30376) 把默认优势写成 $A_i=G\operatorname{softmax}_i(R/\gamma)-1$。脚本保留这一形式，然后再除以 $\max(\operatorname{std}(A),0.01)$。若把附录中的“替换按组大小缩放”严格理解为：先令 $B_i=\operatorname{softmax}_i(R/\gamma)-1/G$，再用 $B_i/\max(\operatorname{std}(B),0.01)$，则两者在下限未触发时等价；下限触发时不等价。因为脚本实际为 $B_i/\max(\operatorname{std}(B),0.01/G)$，当 $G=24$ 时，对 $B$ 的有效下限约为 $4.17\times10^{-4}$。论文未给门控的明确公式，**不能仅凭这段文字断定脚本与作者实验严格一致或确实不一致**。
