# PickScore softmax variants

对应 `assets/softmax_variant/variants.tex` 的伪代码和末尾待做实验表，共 A–F 六组、18 个配置。六份 `train_nft_sd3_*.py` 均复制自 `scripts/train_nft_sd3_ours-1.singleloss-alpha.py`，替换的原代码保留为注释。公共权重公式在 `weight_mappings.py`，实验编号及启动入口在 `experiments.json`。

## 实验与入口

所有启动脚本位于 `pickscore_4090/`，均使用 `config/nft.py:sd3_pickscore`。表中 `R` 是 reward function 返回的分数：现有 PickScore scorer 已将模型相似度除以 26；这里不再对 reward 做中心化或 advantage 标准化。优先级按照run的序号，序号越小优先级越高。

| Run | 实验 | 参数 | 启动脚本 |
| --- | --- | --- | --- |
| 01 | A: near-max | c=0 | `run_01_A_near_max_c0_4090.sh` |
| 02 | A: near-max | c=0.05 | `run_02_A_near_max_c0.05_4090.sh` |
| 03 | A: near-max | c=0.1 | `run_03_A_near_max_c0.1_4090.sh` |
| 04 | A: near-max | c=0.25 | `run_04_A_near_max_c0.25_4090.sh` |
| 05 | B: softmax/std | c=0.5 | `run_05_B_softmax_std_c0.5_4090.sh` |
| 06 | B: softmax/std | c=0.75 | `run_06_B_softmax_std_c0.75_4090.sh` |
| 07 | B: softmax/std | c=1.0 | `run_07_B_softmax_std_c1.0_4090.sh` |
| 08 | B: softmax/std | c=1.5 | `run_08_B_softmax_std_c1.5_4090.sh` |
| 09 | C: linear | f(R)=[R]+ | `run_09_C_linear_4090.sh` |
| 10 | D: sigmoid | c=0.01 | `run_10_D_sigmoid_c0.01_4090.sh` |
| 11 | D: sigmoid | c=0.1 | `run_11_D_sigmoid_c0.1_4090.sh` |
| 12 | D: sigmoid | c=0.3 | `run_12_D_sigmoid_c0.3_4090.sh` |
| 13 | E: power | p=0.5 | `run_13_E_power_p0.5_4090.sh` |
| 14 | E: power | p=2.0 | `run_14_E_power_p2.0_4090.sh` |
| 15 | F: softmax/L1，可选 | c=0.5 | `run_15_F_softmax_l1_c0.5_4090.sh` |
| 16 | F: softmax/L1，可选 | c=0.75 | `run_16_F_softmax_l1_c0.75_4090.sh` |
| 17 | F: softmax/L1，可选 | c=1.0 | `run_17_F_softmax_l1_c1.0_4090.sh` |
| 18 | F: softmax/L1，可选 | c=1.5 | `run_18_F_softmax_l1_c1.5_4090.sh` |

## 公式与数值边界

所有分组均使用跨 GPU 收集后的 `(rollout_batch_id, prompt)`，不会把不同 rollout 中重复出现的 prompt 合成一个大组。每组要求恰好 `G=24` 个样本，结果按原始 rank 顺序分发回各 GPU；组大小不符会报错，以免静默产生错误权重。

每组 `mean(w)=1`，B–F 使用 `w=f(R)/mean_group(f(R))`：

- A：`S={i: R_i >= (1-c)Rmax}`，选中样本 `w=G/|S|`，其余为 0；`Rmax<=0` 时整组 `w=1`。
- B：`f(R)=exp(R/(c*std_group(R)))`，std 使用总体标准差 `ddof=0`。
- C：`f(R)=max(R,0)`。不实现 shifted-linear 的待做实验。
- D：`f(R)=sigmoid((R-0.5)/c)`，以 log-sigmoid 计算归一化，避免小 c 或负 reward 引起整组数值下溢。
- E：`f(R)=max(R,0)**p`；先截断负 reward，再计算幂。
- F：`f(R)=exp(R/(c*mean_group(abs(R-mean_group(R)))))`；L1 scale 是平均绝对偏差。

正文未定义的零分母边界采用显式回退：B/F 的 reward 完全相同，或 C/E 的 reward 全部不大于 0 时，整组返回 `w=1`。B/F 不给非零 scale 添加固定下限；指数映射通过减最大值等代数等价方式稳定计算。没有额外的 w 裁剪，A 的 one-hot 权重可以达到 24。

`w=1` 使奖励引导项 `w-1=0`，但该组仍参与回归到 old prediction 的损失及 reference 正则，并非移除该组全部梯度。

## 相对基础训练脚本的修改

1. 使用 raw reward 的组内映射取代原有 advantage 标准化、裁剪和 `w=1+adv/clip`。保留 stat tracker 作为日志和 checkpoint 状态，不使用它返回的 advantage 计算权重。
2. 按伪代码，以当前 `default` adapter 采样。原先以 `old` adapter 采样的代码保留为注释。
3. 每次训练 mini-batch 用新的均匀随机 `t∈[0,1)` 覆盖时间步缓冲；插值和三个模型 forward 都使用同一组 t。沿用基础脚本每图的训练次数及梯度累积方式。
4. 强制 trajectory-alpha 使用 `old_prediction`，与伪代码的分母一致。保留基础脚本的 `1e-5` 数值保护。

目标与损失仍沿用基础代码：

```text
x_target = x_old + config.beta * (w - 1) * (x0 - x_old)
alpha = 1 / (stop_gradient(mean(abs(x0 - x_old))) + 1e-5)
loss = config.train.adv_clip_max * mean(alpha * (x_theta - x_target)^2)
       + config.train.beta * mean((v_theta - v_ref)^2)
```

这里 `config.train.adv_clip_max` 作为伪代码中的 Δ_clip 损失倍率保留，默认 5，已不用于裁剪 w。`config.beta` 为目标位移系数，启动器默认 1.0（沿用参考公共启动器）；`config.train.beta=1e-4` 为 reference 正则系数。训练日志包含 `weight_mapping` 参数和全局权重统计。

## 4090 启动

公共启动器复制自 `scripts/weight_exps/weight_exps_pickscore/_run_sd3_4090_16gpu_nft-ours-4.selection-common.sh`。默认 2 节点 × 8 GPU，每卡 batch=6，G=24，有效 batch=1152；还支持参考脚本中的单节点 8 卡和双节点 12 卡布局。不同实验有独立输出目录，沿用自动恢复完整 checkpoint 的行为。

先检查命令，不会启动训练或占用 GPU：

```bash
NNODES=1 NPROC_PER_NODE=8 DRY_RUN=1 \
  bash scripts/softmax_variants_exps/pickscore_4090/run_01_A_near_max_c0_4090.sh
```

双节点 16 卡需要在两个节点启动同一个实验，并设置不同的 `NODE_RANK`。将示例地址换成节点 0 的实际地址：

```bash
# 节点 0
NNODES=2 NPROC_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
  bash scripts/softmax_variants_exps/pickscore_4090/run_05_B_softmax_std_c0.5_4090.sh

# 节点 1
NNODES=2 NPROC_PER_NODE=8 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
  bash scripts/softmax_variants_exps/pickscore_4090/run_05_B_softmax_std_c0.5_4090.sh
```


Conda、模型路径及通信参数沿用参考启动器，也可用 `CONDA_ROOT`、`CONDA_ENV`、`SD3_MODEL`、`MASTER_PORT` 等环境变量覆盖。命令末尾可传递其他 `--config.*` 参数。所有实验应使用相同 β、训练预算、seed 和评估配置进行对比。

## 本地验证

```bash
python -m unittest discover -s scripts/softmax_variants_exps -p 'test_*.py' -v
```

验证覆盖 18 个公式配置、分组及 rank 顺序、数值边界、原始代码保留、LaTeX 表格一致性，以及所有启动器的 dry-run 路由和分布式参数。完整 GPU 训练需要实际的模型、reward checkpoint 和对应节点。
