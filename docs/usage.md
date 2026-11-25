# 评测 学习与结果

以下命令均从仓库根目录执行，并使用已经生成的 `configs/portable.local.yaml`。`plan`、`check` 不请求模型；`eval`、`learn` 会调用模型并启动仿真。当前发布版本的真实运行仍待验收。

## CLI 命令

| 命令 | 用途 |
| --- | --- |
| `bole plugins [kind]` | 查看 benchmark、harness、policy adapter |
| `bole plan` | 展开任务、seed、布局和回合计划 |
| `bole check` | 检查路径、已有镜像、配置和所选 GPU |
| `bole eval` | 每回合创建独立的 Codex 会话和工作区 |
| `bole learn` | 同一任务内连续运行，保留 Codex 会话和工作区，收尾保存状态 |
| `bole memory inspect` | 只读查看已有 Codex home 的 memory 文件和索引情况 |

默认入口为 `./bin/bole`；激活 `.venv` 后也可以使用已安装的 `bole` 命令。

## 任务 seed 与回合

```bash
./bin/bole plan --resources configs/portable.local.yaml \
  --task make_toast --episodes 1 --no-memory

./bin/bole plan --resources configs/portable.local.yaml \
  --task make_toast --task hang_mugs --episodes 5 --no-memory

./bin/bole plan --resources configs/portable.local.yaml \
  --suite Precision --episodes 1 --no-memory
```

`--task`、`--seed` 和 `--gpu` 可以重复指定。`--episodes N` 表示**每个可执行任务的总回合数**，默认从原生 seed-major 序列取前 N 回合，不是每个 seed 各 N 回合。默认 seeds 为 `[0, 1, 2]`，较短前缀可能只使用 seed 0。

固定版本的完整目录包含 54 个可执行任务定义、42 个逻辑任务，原生完整计划为 6,300 回合。该数字表示计划范围，不表示已经运行或验证过全部任务。未给 `--episodes` 时使用所选任务的原生配额；首次实验应显式缩减回合数。

通过 `--mode learn` 可以只展开学习计划。查看计划中的 `coverage`、`planned_seeds` 和实例路径，确认实际选择范围。

## 单回合评测

把 `BOLE_MODEL` 设置为实际服务提供的模型 ID，导出资源配置指定的凭据变量后执行：

```bash
./bin/bole eval --resources configs/portable.local.yaml \
  --task make_toast --episodes 1 --gpu 0 \
  --model "$BOLE_MODEL" --timeout 1200 --model-requests 50 \
  --no-memory --output ./runs
```

显式 `--no-memory` 适用于先检查基本评测链路。默认 memory 开启；普通 eval 不要求事先生成 memory。

示例多 GPU 调度：

```bash
./bin/bole eval --resources configs/portable.local.yaml \
  --task make_toast --episodes 4 --gpu 0 --gpu 1 --parallel 2 \
  --model "$BOLE_MODEL" --timeout 1200 --no-memory
```

同一时刻每张所选 GPU 最多分配一个运行单元，实际并发还受 `--parallel` 和计划单元数限制。eval 的运行单元为回合，learn 的运行单元为同一任务的整条链。

## 预算与结束

| 参数 | 含义 |
| --- | --- |
| `--timeout` | 首帧交付后的墙钟预算，单位秒 |
| `--startup-timeout-s` | 启动至首帧的独立预算，默认 600 秒 |
| `--request-timeout-s` | 单次模型传输或环境请求超时，默认 120 秒 |
| `--control-steps` | 实际控制步预算 |
| `--tool-calls` | Bole 环境工具调用预算，不是全部原生 shell 命令的计数 |
| `--model-requests` | 回合执行阶段的实际模型请求尝试预算 |
| `--success-grace-steps` | 首次原生成功后继续的控制步窗口，默认 20，可设为 0 |

未配置的可选预算使用 `null`，不是 0。回合后 learn 反馈等阶段单独记录，因此 `--model-requests` 不是整条学习链的总费用上限。

原生 evaluator 在结束时提供最终成绩；曾经成功、agent 的文字声明或进程退出码都不能替代最终判分。按 `Ctrl+C` 会进入取消和收尾，请等待退出。重新运行命令会创建新的 run，目前不提供恢复同一个 run 的 CLI。

## 连续学习与镜像复用

```bash
./bin/bole learn --resources configs/portable.local.yaml \
  --task make_toast --episodes 2 --gpu 0 \
  --model "$BOLE_MODEL" --timeout 1200 --model-requests 50 \
  --save-image bole/codex-learn:make-toast-v1
```

learn 要求原生 memory 开关开启，不能传入 `--no-memory`。同一任务连续运行时保留原生会话、工作区和容器，每回合重建仿真环境。Bole 提供原生结果反馈，不规定必须生成哪个经验文件，也不强制总结或 finalize。

成功保存状态后，可以从该学习镜像开始新的 eval：

```bash
./bin/bole eval --resources configs/portable.local.yaml \
  --task make_toast --episodes 1 --gpu 0 \
  --model "$BOLE_MODEL" --timeout 1200 \
  --harness-image bole/codex-learn:make-toast-v1
```

学习镜像恢复工作区和可复用的原生持久状态，eval 创建新会话，不恢复旧对话。每回合使用独立可写副本，新增内容不写回学习镜像。同一机器上镜像必须仍存在，跨机器迁移使用 Docker 的 save/load 或自行管理的 registry。

已有同名输出标签会被拒绝。多任务自定义标签需要包含 `{task}`，例如 `--save-image 'bole/codex-learn:trial-{task}'`。异常结束可能留下带 `-partial-` 后缀的镜像，应查看清单确认保存状态。保存成功仅说明状态字节已导出，不证明生成了有效经验或提高了成绩；新容器与当前版本的完整 learn 复用仍待验收。

只读 memory 检查示例：

```bash
./bin/bole memory inspect --home /path/to/existing/codex-home
```

## Prompt shell 与 Python

`--instruction FILE` 可以替换默认 prompt。模板定义见 [`src/bole/prompts/codex.md`](../src/bole/prompts/codex.md)，实际渲染结果保存在回合 `prompt.txt`。修改正在运行的工作副本不会改写已冻结的源码或 prompt。

Codex shell 和机器人环境工具是两条能力。shell 的可写工作目录为 `/workspace`，运行时创建 `/workspace/.venv`；宿主 `.venv` 用来运行 Bole。为 Codex 安装包应使用容器内的工作区解释器，例如：

```bash
/workspace/.venv/bin/python -m pip install packaging==25.0
```

上述命令属于 Codex 容器内部，依赖网络、DNS 和包源。`harness.shell.network_access` 控制原生沙箱 shell 的网络能力；模型传输继续由宿主桥接层负责。原生 web search 当前关闭。

## 读取结果

每次运行输出一个独立的 `run_path`，典型文件如下；未到达相应阶段的回合可能缺少部分文件。

```text
runs/<run_id>/
├── config.yaml / plan.json / resources.json
├── snapshot.json / snapshot/
├── preflight.json
├── execution-config.json / execution-plan.json
├── results.json / summary.json / usage.json
├── safety_summary.json / constraint_metrics.csv
├── trajectory/events.jsonl
├── tasks/<task_id>/
│   ├── summary.json / usage.json / chain.json
│   └── episodes/<episode_id>/
│       ├── result.json / usage.json / safety.json / artifacts.json
│       ├── observations/ / trajectory/ / model-events/
│       ├── native-evaluation/ / videos/ / logs/
│       └── prompt.txt
└── learning/<task_id>/
    └── manifest.json / workspace.tar.gz / home.tar.gz
```

优先阅读 `summary.json`、`results.json` 和回合 `result.json`，再根据错误或产物索引查看细节。

- 顶层 `summary.success_rate` 使用原生分层平均，单位为百分数 0–100；顶层 `episode_micro` 的比例为 0–1。注意其他层级的 `metric_unit`。
- 主成功率为 `null` 表示计划范围存在缺失的有效原生成功判分，另报已判分子集；不要把缺失补为 0，也不要把子集成绩称为完整 benchmark 成绩。
- `completed` 表示生命周期完成，不保证任务成功。学习状态保存成功也不等于学习效果成立。
- usage 记录中的 `null` 不代表免费或零消耗。重试、反馈和后台原生请求的归因应结合相应阶段记录阅读。
- 默认视频为三路相机、5 FPS；视频记录与 agent 主动观察分别产生。使用 `--no-video` 不会关闭正常图像观察。
- 安全指标是探索性测量，非统一认证标准。`missing`、`unsupported`、`not_applicable` 不代表安全通过；非刚体、脚本驱动运动和部分碰撞归因仍有限制。

退出码通常为：0 表示命令或生命周期完成，1 表示资源检查失败或运行不完整，2 表示参数、配置或启动错误，取消通常为 130。正常任务失败仍可能退出 0，应按原生结果判断成绩。
