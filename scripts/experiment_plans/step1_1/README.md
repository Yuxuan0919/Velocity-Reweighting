# Step 1.1：Linear target scale = 15

对应 `jingdong_experiment_plan.tex` 的 Step 1.1：保留 Step 1 的 linear 权重，
仅将目标修正乘以 `c=15`：

```text
v_target = v_old + 15 * Delta_w * (u - v_old)
```

两个入口直接调用原 Step 1 launcher，默认 PickScore、KL=1e-4；采样、batch、
优化器、variance gate 和环境设置均沿用对应的 Step 1 配置。

```bash
# H200，单机 8 卡
bash scripts/experiment_plans/step1_1/run_sd3_h200_8gpu_step1_linear_scale15_kl1e-4.sh

# A6000，默认双机各 8 卡；两节点分别设置 NODE_RANK=0 / 1
# 将 ... 替换为主节点地址
NODE_RANK=0 MASTER_ADDR=... \
  bash scripts/experiment_plans/step1_1/run_sd3_a6000_16gpu_step1_linear_scale15_kl1e-4.sh
```

其他环境变量和命令行参数原样传递，包括 `TASK`、`CONDA_ROOT`、`SD3_MODEL`、
`PER_DEVICE_BATCH` 和 `--config.seed`。A6000 保留原来的 `NNODES`、
`NPROC_PER_NODE` 及 SenseCore 环境变量规则，输出名称使用实际 GPU 总数。
入口在其他参数之后固定 `--reward_mapping=linear --linear_target_scale=15`。
原 Step 1 入口默认 `c=1`，可作为对照。

默认日志放在 `logs/public/experiment_plans/step1_1_linear_scale15/`，
checkpoint 放在 `outputs/step1_1_linear_scale15/`，run name 区分任务、硬件、
GPU 数和 KL，并包含 `step1_1_linear_scale15`。仍可通过 `LOGDIR`、`SAVE_DIR`、
`RUN_NAME` 显式覆盖；比较时保持其余参数与 Step 1 一致。

新 checkpoint 会记录 reward mapping 和 target scale。即使手动复用 `SAVE_DIR`
或 `--config.resume_from`，`c=15` 入口也会拒绝恢复旧格式或 `c=1` checkpoint；
本实验保存的匹配 `linear`、`c=15` 的 checkpoint 可正常恢复。

交付实验表：[Step1_1_linear_scale15_experiment_tracker.tsv](../../../assets/step1_1_linear_scale15/Step1_1_linear_scale15_experiment_tracker.tsv)。
沿用原 15 列格式，包含 H200 8 卡和 A6000 16 卡两组，初始状态为待运行。
