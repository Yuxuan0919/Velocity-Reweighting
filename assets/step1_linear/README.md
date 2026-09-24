# Step 1 linear 实验交付

[Excel 表格](Step1_linear_experiment_tracker.xlsx) 和 [TSV 表格](Step1_linear_experiment_tracker.tsv)
使用指定的 15 列。Run 01 使用 H200 8 卡，Run 02 使用 4090 16 卡，
两项的 KL 均为 1e-4，分别运行各自的启动脚本。两项均未启动，训练步数为 0，
eval reward 留空。本次不安排 KL=0 实验。
Run 编号只在这份交付表内计数。

本地 **8×A100 缩量验证已通过**：真实训练、reward、评估、参数更新及 checkpoint
验收均成功。该历史验证使用 KL=0，不代表本次 KL=1e-4 配置已完成 GPU 验证。

| Run | 启动脚本（`scripts/` 下） | 硬件 | reference/KL 系数 | AWR 对照 |
| --- | --- | --- | --- | --- |
| 01 | `run_sd3_h200_8gpu_step1_linear_kl1e-4.sh` | H200，1×8 卡 | 1e-4 | 对齐 H200/KL=1e-4 的 AWR 启动配置；repo 暂无同配置结果日志 |
| 02 | `run_sd3_4090_16gpu_step1_linear_kl1e-4.sh` | 4090，2×8 卡 | 1e-4 | 已有 `sd35_pickscore_4090_16gpu_flowawr_kl1e-4-r1`、`-r2` 日志 |

## 交付执行命令

在训练机器的 repo 根目录执行。模型、PickScore 权重及 Conda 环境沿用对应 AWR
运行环境；必要时设置 `CONDA_ROOT`、`CONDA_ENV`、`SD3_MODEL`。

```bash
# Run 01：单节点执行
bash scripts/run_sd3_h200_8gpu_step1_linear_kl1e-4.sh

# Run 02：下面两条分别在节点 0 / 1 执行
# 将 10.0.0.1 替换为节点 0 的实际地址，两个节点必须一致
NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
  bash scripts/run_sd3_4090_16gpu_step1_linear_kl1e-4.sh
NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
  bash scripts/run_sd3_4090_16gpu_step1_linear_kl1e-4.sh
```

两个入口都默认使用 PickScore、每卡 batch=6、variance gate=false，并自动设置
`REWARD_MAPPING=linear`。4090 入口默认双机各 8 卡，可从平台环境读取节点布局。
不再通过 `AWR_PRESET` 选择实验；原统一入口已移除。

每项的 checkpoint 写入 `outputs/step1_linear/<保存名称>/checkpoints/`，
TensorBoard 写入 `logs/step1_linear/<保存名称>_<时间戳>/`。表中保存名称是基础名称，
与 launcher 默认值一致。开始新实验时使用新的输出目录；已有同名 linear checkpoint
会按 AWR 原逻辑自动恢复。追加重复实验时，同时设置独立的 `RUN_NAME` 和 `SAVE_DIR`。

## 配置对齐范围

每个 linear 入口直接调用对应的 **KL=1e-4** AWR launcher 和同一 trainer，只将
`reward_mapping` 从 `exponential` 改成 `linear`，另设独立输出名称。
代码中的目标和损失为：

```text
r_plus = max(r, 0)
Delta_w = r_plus / mean_group(r_plus) - 1  # 全组为0时 Delta_w=0
u = noise - x0
v_target = v_old + Delta_w * (u - v_old)
L = MSE(v_theta, v_target) + lambda_ref * MSE(v_theta, v_ref)
```

不使用 trajectory alpha 或外部 t 权重；target 修正系数为 1。
`config.train.beta` 在此入口表示 `lambda_ref`，两项均取 1e-4。
两项均关闭 PickScore variance gate。

| 项目 | Run 01（H200/KL=1e-4 配置） | Run 02（4090/KL=1e-4 配置） |
| --- | --- | --- |
| reference/KL 系数 | 1e-4 | 1e-4 |
| 模型 / 分辨率 | SD3.5-medium / 512×512 | 同左 |
| 训练、评估数据 | `dataset/pickscore/train.txt`、`test.txt` | 同左 |
| 每轮样本 | 48 组 × 24 图 = 1152 | 同左 |
| 每卡采样 / 训练 / 评估 batch | 6 / 6 / 6 | 6 / 6 / 6 |
| 每轮 rollout batches | 24 | 12 |
| 配置中的梯度累积 | 24 | 12 |
| 实际 backward 累积（含 10 个时刻） | 240 | 120 |
| 每轮 optimizer steps | 1 | 1 |
| 学习率 / seed | 3e-4 / 42（各 rank 使用 42+rank） | 同左 |
| 优化器 / 精度 | AdamW / fp16 | 同左 |
| 采样 | old adapter，确定性 DPM2，CFG=1，10 步 | 同左 |
| 训练时间 | scheduler 的 sigma，timestep_fraction=1，10 个时刻 | 同左 |
| old policy 更新 | 每轮更新，保留系数 `min(0.001*(epoch+1), 0.5)` | 同左 |
| 评估 | EMA 参数，flow solver，40 步，每 10 轮（含初始评估） | 同左 |
| checkpoint | 每 30 轮 | 同左 |

这里的“对齐”指 repo 当前 AWR 配置和训练实现。历史运行如果额外传过 CLI 参数、
环境覆盖、不同模型/奖励权重或使用不同代码版本，需把对应设置同步给 linear。
已有 H200/KL=0 日志仅保留作历史记录，不能作为本次 H200/KL=1e-4 的
严格“只改 f”对照；要作这种比较，需要同配置的 H200 AWR/KL=1e-4 结果。

已读取四份 AWR TensorBoard 日志：均记录 `group_count=48`、`variance_gate=0`、
每轮 `gradient_update_times=1`，评估每 10 步。日志未保存完整 config/启动命令，
因此不能据此确认历史学习率、seed 等覆盖参数。Excel 的“AWR日志核对”页列出
日志中最新可见评估值；它们不代表已经结束实验的最终 reward 或训练预算。

“训练步数”填写 optimizer `global_step`；当前 eval reward 填最新
`eval_reward/pickscore`，最终 eval reward 在实验结束后按相同评估步数填写。
当前 scorer 将原始 PickScore 除以 26，已有 AWR 日志使用相同数量级（约
0.76–0.91）；表中统一填此日志尺度，不与示例中的 23.51 直接比较。
计划停止步数与所选 AWR 对照保持一致；本次没有新增训练预算或改动原
`config.num_epochs=100000` 默认值。
