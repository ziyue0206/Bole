# 独立插件包示例与契约

此目录是一个可独立打包的 policy 插件，不属于 Bole 的内置注册表。`RecordedActions` 实现了完整的 `load / infer / close` 生命周期，只返回配置中明确提供的动作候选。它用于演示接入边界，不是训练模型，不是 PI0.5，也不会自动执行机器人动作。

不需要安装它就能阅读或检查 Bole。若要安装这个示例，从 Bole 仓库根目录执行以下命令；构建依赖由 pip 准备：

```bash
./.venv/bin/python -m pip install -e ./templates/plugins
bole plugins policies
```

安装后的 entry point 为 `recorded-actions = bole_recorded_actions:RecordedActions`。不必修改 Bole 源码；`bole plugins` 会同时列出内置与外部注册项。同名插件会报错，不能覆盖内置实现。

## 可执行的 policy 合同示例

示例本身只依赖 Python 标准库，可在该目录用以下命令进行本地合同检查，不启动环境或模型：

```bash
PYTHONPATH=src python - <<'PY'
from bole_recorded_actions import RecordedActions
policy = RecordedActions()
policy.load({"actions": [{"fixture_joint": [0.0]}]}, {})
candidate = policy.infer({"observation_id": "fixture-observation", "state": {}})
assert candidate["actions"] == [{"fixture_joint": [0.0]}]
assert candidate["executed"] is False
policy.close()
PY
```

`fixture_joint` 仅是合同测试字段，不是 RoboDojo 的机器人接口。真实运行的 `policy.actions` 必须由使用者按所选 benchmark 的 `capabilities.action_schema` 提供。配置方式：

```yaml
policy:
  name: recorded-actions
  actions:
    - fixture_joint: [0.0]  # 运行前替换为所选环境的有效动作
```

runner 加载 policy 后，harness 才会获得 `infer_policy` 工具。该工具以最新公开观测调用 `infer`，返回候选和观测关联，不推进物理环境。harness 决定是否接受，并通过独立的 `execute` 工具执行；所有原生校验、预算、判分和轨迹记录仍由 benchmark 与 engine 管理。推理插件不得读取 evaluator 私有字段。

## 三类注册入口

在自己的独立包 `pyproject.toml` 中注册实际已经实现的工厂或无参类：

```toml
[project.entry-points."bole.benchmarks"]
my-benchmark = "my_lab.benchmark:MyBenchmark"

[project.entry-points."bole.harnesses"]
my-harness = "my_lab.harness:MyHarness"

[project.entry-points."bole.policies"]
my-policy = "my_lab.policy:MyPolicy"
```

这些 `my_lab` 名称是接入说明，不是仓库中的占位实现。每个注册工厂无参数调用后返回适配器实例。新增 benchmark 不需要经过 RoboDojo 的 catalog 或 planner；engine 直接调用所选适配器的 `plan(config)`。插件专属配置可放在 `benchmark.options`、`harness.options` 或 `policy` 中，由该插件验证。

## Benchmark 与环境会话

`BenchmarkAdapter` 提供同步方法：

| 方法或属性 | 契约 |
| --- | --- |
| `requires_gpu` | 是否必须从 `resources.gpus` 分配独占 GPU |
| `plan(config)` | 返回 JSON 可序列化的 `schema_version: 1`、`tasks`、`episodes`、`coverage`、`protocol` |
| `check(config)` | 只读检查，返回含 `ok` 的检查报告；不启动真实 episode |
| `session(context)` | 为一个 episode 创建新环境会话 |
| `summary(plan, results)` | 提供 benchmark 原生汇总，明确 `success_rate` 单位、覆盖与分母 |

每个 episode 至少包含 `task_id`、`episode_id`、`seed`、`layout_id`、`input_asset`。ID 必须是安全路径分量，同一任务内的 episode ID 必须唯一。learn 同一任务的第一回合 `predecessor: null`，之后必须引用该任务前一回合 ID。实例字段由 benchmark 定义；不要用 RoboDojo 布局规则解释其他 benchmark。

`context` 包含 `config`、`run_id`、`task_id`、`episode_id`、`episode`、`root`、`episode_root`、`store`、`gpu`、`policy`、`input_asset`；seed、layout_id、layout_path 也提供为便捷字段。`store` 为当前回合的 `RecordStore`。`root` 是 harness 的隔离工作目录根，不是源代码根。

环境会话同步方法：

| 方法 | 返回值与边界 |
| --- | --- |
| `open()` | `instruction`、`step_limit`、`capabilities.action_schema`；完成一次新环境初始化 |
| `observe()` | 公开 `state / end_effector_poses / instruction / control_step`，可选 `vision`；不推进物理环境 |
| `delivered()` | 收到首个观测被 harness 接受的通知，开始环境端的回合时钟 |
| `execute(actions, call_id)` | `{public, private}`；公开回执可给 harness，私有边界与判分只给 engine |
| `close(reason)` | 最终 `native_result`、`artifacts`、`safety` 等；关闭和失败路径应可完成清理 |

`action_schema` 描述整个动作数组。`execute.private.actual_control_steps` 必须是非递减的实际完成控制步累计值，还可含 `termination_reason`、`first_success_control_step`、`first_success_native_result`。动作请求必须由适配器校验，不能只信赖 JSON schema。适配器在动作序列执行期间也要检查预算，返回实际执行数量；核心不将未执行动作计为已执行。

图像使用 `vision[camera].color = {encoding: "png;base64", data: "...", shape: [...]}`。最终 `native_result` 用显式 `available` 表达是否有判分，已知成功为布尔 `success`。缺失判分不能填成普通失败。RoboDojo 的 normalized score 范围为 0–1；其他 benchmark 的原生成绩单位由自己的汇总声明。

## Harness 生命周期

`open(context, tools)`、`run_episode(prompt, initial_content, on_delivered, timeout_s)`、`stop_episode(reason)`、`feedback(text)`、`export_asset(destination, metadata, partial=False)`、`close()` 都是异步方法。`bind_episode(store, tools)` 支持同步或异步，用于 learn 重新绑定本回合工具和记录器；不得偷偷建立新对话以替代连续链。

`run_episode` 只能在原生服务确实接受了首个观测后调用并等待 `on_delivered()`。返回值包含 `termination`、`finish_called`、`session_usable`。普通 final 返回 `final_response_without_finish`；预算结束不等同基础设施错误。`stop_episode` 应停止当前回合并确保原生 turn 已收敛，不能让上一回合的迟到动作进入下一回合。

learn 每任务复用一个 harness，环境每回合重建；eval 每回合新建 harness。只有 learn 在回合关闭后收到短成绩反馈。资产导出需要区分保存成功、部分链、原生 memory 生成状态以及新会话是否实际复用；历史日志不能作为恢复对话的默认入口。

若实现了 `harness.ledger`，它应提供兼容 `UsageLedger.report()` 的请求级用量与单独的 session 累计快照。learn 的 ledger 是整条会话累计值，runner 在链结束后只纳入一次，不将每回合累计快照相加。

## 冻结与验证

真实运行前，Bole 会复制插件类所属的顶层 Python 包，并将该副本加入 child runner 的导入路径。插件必须来自可复制的文件包；无文件的 namespace 包需要先提供明确包入口。插件的第三方依赖和已有运行环境保留为外部依赖，不能声称已完全封装。benchmark 若需要外部模型、场景或资源，须通过配置记录路径与版本，禁止运行中隐式下载或替换基线。

先用 CPU 合同测试验证规划、生命周期、缺失判分、public/private 隔离和真实动作计数。真实设备或 GPU 的可用性与结果需另行验证。参考现有测试：`tests/test_engine.py`、`tests/test_config_cli.py`、`tests/test_native_scoring.py`。
