# 本轮实验交付检查

分支：`experiment/linear-square-ablation`。基于 `a9ad975eda4b208b0e9d6ec811c695ad9fd1f305`。
当前为本地待审阅版本，尚未暂存、提交或推送；原实验分支不变。

## 实验安排

- 新增9组：Linear c=5/20/30；Linear c=15后期降至7.5/5；正侧平方、负侧原值 c=5/20；正侧平方、负侧均匀 c=5/20。
- 恢复原B（Linear c=15），计为Run02，默认仍为A6000双机16卡；接受卡差异，不重复新开同配置实验。
- C2.3/C2.4继续原H200单机8卡任务，对应Run05/06；新分支提供中断后的可选续跑入口。
- 每组独立脚本，实验参数固定；9组新实验默认A6000双机各8卡，共16卡，rollout batches和梯度累积均为12。H200单机8卡仅作为显式布局选项，rollout batches和梯度累积均为24；两种布局均保持每轮1152图。
- 本轮无新增截断实验。原平方负侧的−1边界在乘c之前，乘c后的最终系数仍可低于−1。

## 修改或新增的文件

核心代码3个：

- `flow_grpo/mass_shift.py`：新增正侧平方和负侧均匀；原`square_both_sides`函数保留。
- `flow_grpo/target_scale.py`：新增c调度及checkpoint调度兼容校验。
- `scripts/experiment_plans/step1/train_nft_sd3_ours-singleloss-AWR.py`：接入映射/调度、恢复保护、可选最终评估保存及训练诊断；保留原目标、loss和梯度裁剪顺序。

交付启动目录 `scripts/experiment_plans/linear_square_ablation/` 共15个文件：

- `manifest.json`
- `launch.py`
- `README.md`
- `run_plan01_linear_scale5.sh`
- `run_plan02_linear_scale15_resume.sh`
- `run_plan03_linear_scale20.sh`
- `run_plan04_linear_scale30.sh`
- `run_plan05_square_both_scale5_resume.sh`
- `run_plan06_square_both_scale20_resume.sh`
- `run_plan07_linear_decay15to7p5.sh`
- `run_plan08_linear_decay15to5.sh`
- `run_plan09_square_positive_scale5.sh`
- `run_plan10_square_positive_scale20.sh`
- `run_plan11_square_positive_uniform_negative_scale5.sh`
- `run_plan12_square_positive_uniform_negative_scale20.sh`

表格和生成工具4个：

- `scripts/experiment_plans/build_linear_square_handoff.py`
- `assets/linear_square_ablation/Linear_Square_Experiment_Handoff_20260929.xlsx`
- `assets/linear_square_ablation/Linear_Square_Experiment_Handoff_20260929.tsv`
- `assets/linear_square_ablation/REVIEW.md`（本文件）

合计22个文件。`tests/`、本地TeX、`0929_discuss.md`及原始导出Excel不在提交清单中。

## 验证

- 34项CPU unittest通过：18项映射/调度、13项训练接入与恢复/诊断、3项旧行为回归。
- 旧Linear/Square的advantage、既有统计量、固定c下的目标/loss/梯度与基准代码数值一致。
- 两种新负侧规则保持各侧质量；测试覆盖随机组、空组、极值、负侧边界，以及400/600步衰减边界和恢复校验。
- 12个入口的shell语法通过；24组双布局dry-run通过；用mock torchrun核对24组实际转交参数与预览一致。
- 非法实验参数覆盖、必需checkpoint缺失和包含空格的路径已检查。
- Trainer实际导入和帮助flags注册通过；修改的Python语法和`git diff --check`通过。
- XLSX包含4个子表；ZIP/XML有效，主表12行18列与TSV逐单元格一致。未在Excel中人工渲染预览。
- 未启动GPU训练；多卡训练、模型/数据资源可用性仍需运行机器验证。

测试留在本地，不推送。B续跑需要运行机器上的完整checkpoint；表中历史eval步数不等于checkpoint步数。
