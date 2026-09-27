# Step 2 Scheme 2：Square Both Sides

分支：`experiment/step2-square-both`。

对应 `jingdong_experiment_plan.tex` 的 Step 2 Scheme 2。沿用原 Step 1 的
`c=1` linear baseline：`f(r)=max(r,0)`、`w=f(r)/mean_group(f(r))`、
`Delta_w=w-1`；全组 clipped reward 为零时 `Delta_w=0`。

在每个 prompt group 内分别处理正、负 shift。记正质量为 `S_plus`，正 shift
的平方和为 `Q_plus`，负 shift 绝对值之和为 `S_minus`：

```text
Delta_w > 0: transformed_shift = S_plus * Delta_w^2 / Q_plus
Delta_w < 0: transformed_shift = -min(1, tau * abs(Delta_w)^2)
Delta_w = 0: transformed_shift = 0

sum_negative min(1, tau * abs(Delta_w)^2) = S_minus
v_target = v_old + transformed_shift * (u - v_old), u = noise - x0
```

`tau` 逐组求解，保留原来的负质量；达到上限的样本固定为 `-1`，剩余负质量
继续按平方比例分配。正质量也保持不变，因此组内总 shift 仍为零，且
`1+transformed_shift >= 0`。全零 shift 保持全零。

两个入口复用原 Step 1 launcher，默认 PickScore、KL=1e-4、每卡 batch=6、
48组×24图=1152图/轮、训练/评估采样10/40步、`timestep_fraction=1.0`。
H200 8卡的 rollout batches / 梯度累积为24，A6000 16卡为12；默认每轮一次
optimizer step，优化器和 loss 均沿用 Step 1。

```bash
# H200，单机 8 卡
bash scripts/experiment_plans/step2_scheme2/run_sd3_h200_8gpu_step2_scheme2_kl1e-4.sh

# A6000，默认双机各 8 卡；两节点分别设置 NODE_RANK=0 / 1
# 将 ... 替换为主节点地址
NODE_RANK=0 MASTER_ADDR=... \
  bash scripts/experiment_plans/step2_scheme2/run_sd3_a6000_16gpu_step2_scheme2_kl1e-4.sh
```

需要与 Step 1 相同的 `DiffusionNFT` conda 环境、SD3.5 Medium 模型、
`reward_ckpts/` 和任务数据。`CONDA_ROOT`、`CONDA_ENV`、`SD3_MODEL`、`TASK`
及其他普通环境变量和命令行参数仍可传入。A6000 沿用 `NNODES`、
`NPROC_PER_NODE` 及 SenseCore 的规则，输出名称使用实际 GPU 总数。

入口在其他参数之后固定 `--reward_mapping=linear --linear_target_scale=1
--mass_shift_transform=square_both --awr_variance_gate=false`。负 shift 的绝对值
上限固定为1，transform 和 scale 不能在此入口覆盖。关闭 variance gate
可保留逐组质量和负侧上限；对其他任务也固定关闭。

默认日志目录为 `logs/public/experiment_plans/step2_scheme2/`，checkpoint 目录
为 `outputs/step2_scheme2/`，run name 包含任务、硬件、GPU数及
`step2_scheme2_kl1e-4`。仍可显式设置 `LOGDIR`、`SAVE_DIR` 和 `RUN_NAME`。
恢复 checkpoint 时 reward mapping、target scale 和 mass-shift transform
必须匹配，Step 1 或 Step 1.1 的 checkpoint 不能作为本实验的续训状态。

TensorBoard 的 `awr/raw_advantage_std` 保留原 Linear shift 的标准差，
`awr/advantage_std`、`awr/advantage_min` 等记录实际使用的变换后 shift。
`awr/shift/` 下记录变换前后的每组平均正、负质量，各组最大的质量守恒和
零和误差，以及负样本中达到 `-1` 的比例；这些诊断在 float64 转换为训练
使用的 float32 之前计算。目标修正的均方大小仍记录为 `correction_norm`。

交付实验表：[Step2_scheme2_experiment_tracker.tsv](../../../assets/step2_scheme2/Step2_scheme2_experiment_tracker.tsv)。
沿用15列、UTF-8 BOM格式，包含 H200 8卡和 A6000 16卡两组，初始状态为待运行。
