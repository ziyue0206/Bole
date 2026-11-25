# 开发与插件

## 代码结构

`src/bole/config.py` 负责配置合并和校验，`cli.py` 提供入口；`engine/` 负责任务计划、生命周期和预算；`benchmarks/robodojo/` 连接原生任务与仿真 worker；`harnesses/` 连接 Codex；`runtime/` 负责代码快照、依赖和学习状态；`results/` 负责记录、视频和汇总。

当前内置 benchmark 为 `robodojo`、harness 为 `codex`，没有内置 policy。可用名字以 `./bin/bole plugins` 为准。

## CPU 契约检查

在已经建立的宿主环境中可以自行执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -v
```

常规发现模式使用 `test*.py`。部分 worker 检查需要 NumPy、Pillow 等依赖，视频检查需要宿主可用的 FFmpeg/ffprobe；缺少依赖的跳过项不是通过项。需要这些可选检查时，在隔离的开发环境安装相应依赖。

发布目录保留 `test*.py` CPU 契约检查；面向特定部署的手动探针不纳入该目录。CPU 契约检查不能替代新镜像、模型或 GPU 仿真验收。

## 插件

外部包通过 Python entry point 注册：

```toml
[project.entry-points."bole.benchmarks"]
my-benchmark = "my_package:MyBenchmark"

[project.entry-points."bole.harnesses"]
my-harness = "my_package:MyHarness"

[project.entry-points."bole.policies"]
my-policy = "my_package:MyPolicy"
```

插件安装到运行 Bole 的宿主环境，而非 Codex 工作区 venv。使用 `./bin/bole plugins` 核对发现结果，同名插件不能覆盖内置实现。

接口见 [`src/bole/api/__init__.py`](../src/bole/api/__init__.py)，可参考 [`templates/plugins/`](../templates/plugins/README.md)。该示例提供显式动作候选，不是训练好的机器人控制器。插件需要自行约束 I/O、初始化和关闭耗时，平台无法安全强杀任意不响应的 Python 线程。

## 调整资源版本

修改 [`configs/resources.lock.json`](../configs/resources.lock.json) 时，同时考虑 RoboDojo 源码、子模块、资产接口、Codex CLI、bwrap 和策略。源码目录与输出镜像标签建议使用新的版本位置，保留已完成实验的资源引用。

Codex 官方基础镜像是开发环境；Bole 使用的薄派生层还负责非 root Python、CLI、bwrap 以及可访问的 login-shell 环境。只替换镜像名称不等于这些接口已经兼容。

运行时冻结 Bole、所选 RoboDojo 源码、prompt、seccomp 和生成的机器人配置；大资产、原生工具及镜像保持外部引用。已有实验依赖的外部资源仍需保留。
