# Linear / Square 消融实验

代码分支：`experiment/linear-square-ablation`，当前本地待检查、提交和发布。
固定 PickScore、seed=42、fp16、lr=3e-4、KL=1e-4，训练至全局step=1000；
每10步评估、每30步保存，训练结束再做一次最终评估和完整保存。训练/评估采样
为10/40步、`timestep_fraction=1.0`、每卡batch=6、variance gate关闭。
每轮48组×24图=1152图、一次 optimizer 更新尝试。全局step包含AMP跳过的
更新尝试，实际成功更新数另记为 `successful_optimizer_updates`。新实验默认
A6000 双机各8卡，共16卡，rollout batches和梯度累积均为12；
显式选择 H200 单机8卡时均为24。

每个计划都有固定入口，无需设置方法、c或训练超参数环境变量。

| Plan | 表中编号 | 方法 / c | 默认布局 | 动作 |
| --- | --- | --- | --- | --- |
| 01 | [B.1](run_plan01_linear_scale5.sh) | none / 5 | `a6000_16gpu` | 新组 |
| 02 | [B](run_plan02_linear_scale15_resume.sh) | none / 15 | `a6000_16gpu` | 必须续跑 |
| 03 | [B.2](run_plan03_linear_scale20.sh) | none / 20 | `a6000_16gpu` | 新组 |
| 04 | [B.3](run_plan04_linear_scale30.sh) | none / 30 | `a6000_16gpu` | 新组 |
| 05 | C2.3 | square_both / 5 | `h200_8gpu` | 保留正在运行的任务 |
| 06 | C2.4 | square_both / 20 | `h200_8gpu` | 保留正在运行的任务 |
| 07 | [B.4](run_plan07_linear_decay15to7p5.sh) | none / 15→7.5 | `a6000_16gpu` | 新组 |
| 08 | [B.5](run_plan08_linear_decay15to5.sh) | none / 15→5 | `a6000_16gpu` | 新组 |
| 09 | [C1.1](run_plan09_square_positive_scale5.sh) | square_positive / 5 | `a6000_16gpu` | 新组 |
| 10 | [C1.2](run_plan10_square_positive_scale20.sh) | square_positive / 20 | `a6000_16gpu` | 新组 |
| 11 | [D1](run_plan11_square_positive_uniform_negative_scale5.sh) | square_positive_uniform_negative / 5 | `a6000_16gpu` | 新组 |
| 12 | [D2](run_plan12_square_positive_uniform_negative_scale20.sh) | square_positive_uniform_negative / 20 | `a6000_16gpu` | 新组 |
| 13 | [B.6](run_plan13_linear_fork420_scale5.sh) | none / 15→5，step420后立即切换 | `a6000_16gpu` | 从原B的checkpoint-420分叉 |

Linear 使用 `f(r)=max(r,0)`、组内均值归一化和 `Delta_w=w-1`。
`square_positive` 对正侧平方后重新归一化以保留正质量，负侧逐样本不变；
`square_positive_uniform_negative` 保留同一正侧规则，将原负质量均匀分给
原本严格负 shift 的样本。零 shift 保持为零。c乘在重分配后的整个目标修正上。
Plan07/08在全局step400之前使用c=15，400至600线性降到7.5或5，
600之后保持末值；schedule随checkpoint中的全局step续接。

## 运行与布局

先检查解析结果，dry-run只读取文件并输出JSON，不创建目录、不激活conda、
不调用base launcher或torchrun。JSON包含完整训练参数、最终torchrun参数、
输出目录和尚需设置的多机环境信息：

```bash
bash scripts/experiment_plans/linear_square_ablation/run_plan01_linear_scale5.sh --dry-run

# 默认A6000双机各8卡；两个节点分别设置NODE_RANK=0 / 1
# 将 ... 替换为主节点地址
NODE_RANK=0 MASTER_ADDR=... bash scripts/experiment_plans/linear_square_ablation/run_plan01_linear_scale5.sh

# 可选H200单机8卡，须显式指定布局
bash scripts/experiment_plans/linear_square_ablation/run_plan01_linear_scale5.sh --layout h200_8gpu
```

布局只接受 `h200_8gpu` 和 `a6000_16gpu`。新组按实验和布局使用独立
`run_name`、`outputs/linear_square_ablation/<run_name>/` 和
`logs/public/experiment_plans/linear_square_ablation/<run_name>/`。
具体名称见 [manifest.json](manifest.json) 的 `layout_variants`。

沿用Step1的基础设施：默认 `CONDA_ENV=DiffusionNFT`，`CONDA_ROOT` 为
`/inspire/qb-ilm/project/chineseculture/public/yuxuan/miniconda3`，模型默认在
仓库的 `pretrained_models/sd3.5-medium`。环境、模型、`reward_ckpts/` 和
任务数据须已可用；多机节点之间须能访问同一个 `MASTER_ADDR` / `MASTER_PORT`。
可设置 `REPO_DIR`、`CONDA_ROOT`、`CONDA_ENV`、`SD3_MODEL`；节点rank继续
沿用原launcher的 `NODE_RANK` / SenseCore / 调度器环境变量规则。

环境中的 `TASK`、`PER_DEVICE_BATCH`、`NNODES`、`NPROC_PER_NODE`、
`AWR_VARIANCE_GATE`、`RUN_NAME`、`SAVE_DIR`、`LOGDIR` 不会改变固定实验。
除 `--layout` 和 `--dry-run` 外，只接受以下显式路径参数：

- `--config.pretrained.model`：模型路径或模型ID。
- `--config.resume_from`：完整checkpoint目录的绝对路径。
- `--config.save_dir`、`--config.logdir`：显式指定本实验的存储位置。
- `--fork_from_checkpoint`：仅Plan13接受，指定原B实验的绝对路径，目录名必须为 `checkpoint-420`。

其他参数（包括方法、c、seed和 `--config.num_epochs`）会报错；总步数固定1000。
固定训练参数和方法flags位于转交给base launcher的参数末尾。

## B：必须从原Linear c=15实验续跑

Plan02默认A6000双机16卡，`require_resume=true`。默认搜索以下历史目录，
不假定目录或完整checkpoint已存在：

```text
${REPO_DIR}/outputs/step1_1_linear_scale15/sd35_pickscore_a6000_16gpu_step1_1_linear_scale15_kl1e-4
```

```bash
bash scripts/experiment_plans/linear_square_ablation/run_plan02_linear_scale15_resume.sh --dry-run

NODE_RANK=0 MASTER_ADDR=... bash scripts/experiment_plans/linear_square_ablation/run_plan02_linear_scale15_resume.sh --config.resume_from=/absolute/path/to/complete-checkpoint
```

若目录缺失会明确报错并提示传入实际checkpoint路径；目录存在但没有完整
可兼容checkpoint时由trainer报错，不能从头误跑。显式改为H200布局时仍搜索
原A6000历史目录，新的run/log名称反映实际布局。新组也可自动恢复自己目录中
的完整checkpoint；普通续跑不能跨c、shape或schedule。Plan13的显式分叉只允许
从原Linear恒定c=15、step420变更为恒定c=5。

## B.6：从Linear c=15的checkpoint-420分叉到c=5

Plan13用于检查回落前降低c是否减轻后续回撤。读取原B的完整 `checkpoint-420`，
从下一轮训练开始立即固定c=5，保持lr=3e-4、KL=1e-4和其他训练设置；
它与B.5 / Plan08从头训练并在400–600步逐渐降低c的方案不同。
全局step继续从420计数，训练至1000，最多增加580次更新尝试。
重点比较500–560步附近及之后的回撤、current–old偏差、目标修正尖峰和850/1000步分数；
已有c=15历史曲线只作为初步对照，当前无新增实验结果。

两个布局默认都从原A6000实验读取：

```text
${REPO_DIR}/outputs/step1_1_linear_scale15/sd35_pickscore_a6000_16gpu_step1_1_linear_scale15_kl1e-4/checkpoints/checkpoint-420
```

```bash
# 即使checkpoint不在当前机器上，也可核对配置；不会创建目录或启动训练
bash scripts/experiment_plans/linear_square_ablation/run_plan13_linear_fork420_scale5.sh --dry-run

# A6000双机各8卡：两个节点分别用NODE_RANK=0 / 1，主节点地址保持相同
NODE_RANK=0 MASTER_ADDR=... bash scripts/experiment_plans/linear_square_ablation/run_plan13_linear_fork420_scale5.sh --fork_from_checkpoint=/absolute/path/to/checkpoints/checkpoint-420

# 显式选择H200单机8卡；源checkpoint仍为同一个A6000 checkpoint-420
bash scripts/experiment_plans/linear_square_ablation/run_plan13_linear_fork420_scale5.sh --layout h200_8gpu --fork_from_checkpoint=/absolute/path/to/checkpoints/checkpoint-420
```

首次分叉要求源checkpoint包含 `_SUCCESS`、current/old LoRA、optimizer、
scaler、EMA、training state和prompt stat tracker等完整状态；trainer核对
step=420、原目标为Linear恒定c=15。恢复optimizer动量及其原学习率，
加载后验证lr=3e-4，不自动重置optimizer。源checkpoint只读，保存和日志使用
独立Plan13目录；显式路径也不能与源实验目录或彼此重叠。

再次执行同一入口，会优先恢复Plan13自己目录下带有相同分叉来源的完整c=5
checkpoint，即使源checkpoint已离线也可继续。也可用 `--config.resume_from`
指定此分叉的子checkpoint；此参数不能用来指定c=15的源checkpoint。
目标目录已有内容但没有完整可续跑checkpoint时会报错，防止重新从420启动。
`require_resume=true`；源/子checkpoint缺失或不完整时不得从头训练。

旧checkpoint未保存完整随机数状态，分叉会按seed=42初始化随机数流，
因此不能精确重放原实验420步之后的样本轨迹。若跨A6000/H200布局，
每轮图数和更新次数保持不变，但分布式随机数和数值差异仍会影响对照。
dry-run JSON列出源路径、源目录是否存在、子checkpoint搜索路径及剩余训练预算。

## C2.3 / C2.4：保留正在运行的任务

Plan05/06分别是原 `experiment/step2-square-both` 的H200 c=5/c=20；
不打断、不自动重启正在运行的任务。manifest保留原branch和 `original_script`。
仅在原任务意外停止，或需要从checkpoint补齐到1000步并补最终eval/save时，
使用新分支的可选入口（均 `require_resume=true`）：

- [C2.3 可选续跑](run_plan05_square_both_scale5_resume.sh)
- [C2.4 可选续跑](run_plan06_square_both_scale20_resume.sh)

两者默认仍用H2008，搜索和保存到各自历史目录：

```text
${REPO_DIR}/outputs/step2_scheme2_scale5/sd35_pickscore_h200_8gpu_step2_scheme2_scale5_kl1e-4
${REPO_DIR}/outputs/step2_scheme2_scale20/sd35_pickscore_h200_8gpu_step2_scheme2_scale20_kl1e-4
```

可先给可选入口加 `--dry-run` 核对；需要时用 `--config.resume_from` 指定
实际完整checkpoint。可选入口的run/log名称含resume及实际布局，历史保存
路径在dry-run结果和manifest中均完整展示。
