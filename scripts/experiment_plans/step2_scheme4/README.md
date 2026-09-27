# Step 2 Scheme 4：Exponentiate Both Sides

分支：`experiment/step2-exp-both`。

对应 `jingdong_experiment_plan.tex` 的 Scheme 4。沿用原 Step 1 的
`c=1` linear baseline：`f(r)=max(r,0)`、`w=f(r)/mean_group(f(r))`、
`Delta_w=w-1`；全组 clipped reward 为零时 `Delta_w=0`。

每个 prompt group 内，分别取非空符号侧的原始幅值 `a=abs(Delta_w)`。
`S_plus` 为原始正 shift 之和，尺度按该侧围绕均值的平均绝对偏差计算：

```text
mean_a_s = mean_side(a)
MAD_s = mean_side(abs(a - mean_a_s))
gamma_s = gamma_prime / max(MAD_s, epsilon)
gamma_prime = 1, epsilon = 1e-6

Delta_w > 0: transformed_shift = S_plus * exp(gamma_plus * Delta_w)
                               / sum_positive exp(gamma_plus * Delta_w)
Delta_w < 0: transformed_shift = -min(1, tau * exp(gamma_minus * abs(Delta_w)))
Delta_w = 0: transformed_shift = 0

sum_negative min(1, tau * exp(gamma_minus * abs(Delta_w))) = S_minus

v_target = v_old + transformed_shift * (u - v_old), u = noise - x0
```

`S_minus` 是原始负 shift 绝对值之和。`gamma_minus` 根据原始负幅值独立
计算，求解 `tau` 期间保持固定。逐组保留负质量：达到上限的样本固定为
`-1`，剩余质量继续按指数比例分配。两侧总质量均保持不变，组内总 shift
仍为零，且 `1+transformed_shift >= 0`。

实现用中心化指数 `exp(gamma_s * (a - max(a)))` 避免指数溢出。
正侧归一化结果不变；负侧的 `tau` 随公共指数因子相应重标度，最终得到
相同的受限质量分配，中心化前后的 `tau` 数值不要求相同。

空符号侧跳过；单样本侧或幅值全相等的侧保持原值，全零组保持全零。

两个入口复用原 Step 1 launcher，默认 PickScore、KL=1e-4、每卡 batch=6、
48组×24图=1152图/轮、训练/评估采样10/40步、`timestep_fraction=1.0`。
H200 8卡的 rollout batches / 梯度累积为24，A6000 16卡为12；默认每轮一次
optimizer step，`lr=3e-4`、`seed=42`、`fp16`，loss 沿用 Step 1。

```bash
# A6000，默认双机各 8 卡；两节点分别设置 NODE_RANK=0 / 1
# 将 ... 替换为主节点地址
NODE_RANK=0 MASTER_ADDR=... \
  bash scripts/experiment_plans/step2_scheme4/run_sd3_a6000_16gpu_step2_scheme4_kl1e-4.sh

# H200，单机 8 卡
bash scripts/experiment_plans/step2_scheme4/run_sd3_h200_8gpu_step2_scheme4_kl1e-4.sh
```

入口继承 Step 1 的基础设施设置：`CONDA_ROOT` 默认指向
`/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3`，
`CONDA_ENV=DiffusionNFT`，`SD3_MODEL` 默认为仓库的
`pretrained_models/sd3.5-medium`。运行环境需要相应 conda 环境、模型、
`reward_ckpts/` 和任务数据；可通过上述环境变量指定实际位置。
A6000 多机运行需要各节点能访问同一 `MASTER_ADDR` 和 `MASTER_PORT`，且满足
原 launcher 的分布式网络条件；脚本不会配置网络或下载依赖。

其他普通环境变量和命令行参数继续透传，包括 `TASK`、`PER_DEVICE_BATCH`
和 `--config.seed`。A6000 沿用 `NNODES`、`NPROC_PER_NODE` 及 SenseCore
环境变量规则，输出名称使用实际 GPU 总数。入口在其他参数之后固定：

```text
--reward_mapping=linear --linear_target_scale=1
--mass_shift_transform=exp_both
--mass_shift_gamma_prime=1 --mass_shift_epsilon=1e-6
--awr_variance_gate=false
```

关闭 variance gate 可保留逐组质量；对其他任务也固定关闭。默认日志目录为
`logs/public/experiment_plans/step2_scheme4/`，checkpoint 目录为
`outputs/step2_scheme4/`，run name 包含任务、硬件、GPU数及
`step2_scheme4_kl1e-4`。仍可显式设置 `LOGDIR`、`SAVE_DIR` 和 `RUN_NAME`。
checkpoint 记录 reward mapping、target scale、transform，以及
`mass_shift_gamma_prime` 和 `mass_shift_epsilon`；这些方法参数不匹配时
拒绝续训。初始化 TensorBoard 也记录所选的 gamma_prime 和 epsilon。

TensorBoard 的 `awr/shift/` 下，`gamma_positive_mean` 和
`gamma_positive_max` 记录有效 gamma，`mad_positive_mean` 和
`mad_positive_max` 记录 MAD，`mad_floor_positive_mean` 记录 epsilon
下限触发比例；这些统计只包含正侧非空的 prompt group。
同一命名空间下，负侧也记录 `gamma_negative_mean`、`gamma_negative_max`、
`mad_negative_mean`、`mad_negative_max` 和 `mad_floor_negative_mean`，
统计范围为负侧非空的组。
`awr/shift/negative_bound_fraction` 记录负样本中达到 `-1` 的比例；同时
保留变换前后的质量和质量守恒、零和误差诊断。

核心代码为 `flow_grpo/mass_shift.py` 和共享入口
`scripts/experiment_plans/step1/train_nft_sd3_ours-singleloss-AWR.py`。
交付实验表：[Step2_scheme4_experiment_tracker.tsv](../../../assets/step2_scheme4/Step2_scheme4_experiment_tracker.tsv)。
表格使用11个有名列和 UTF-8 BOM，E1为 A6000 16卡（优先级0），E2为
H200 8卡（优先级1），初始状态均为“待开始”。“保存名称”与默认 `RUN_NAME`
一致，配置列另外列出日志和 checkpoint 目录。
