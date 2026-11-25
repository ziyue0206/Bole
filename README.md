# Bole

Bole 是面向具身 benchmark 的评测与连续学习平台。当前内置组合是 **RoboDojo + Codex**：Codex 通过环境工具获取图像和机器人状态、执行动作，Bole 负责回合编排、预算、原生判分及结果记录。

当前版本为 `0.1.0a1`。资源准备脚本和使用流程已提供；全新机器上的完整下载、镜像构建、真实模型与 GPU 仿真仍待验收。请按实验版本使用，具体范围见[版本状态](docs/release-status.md)。

## 开始使用

目标环境是启用 AppArmor 的 Linux x86_64 Docker 主机，配备符合 Isaac Sim 5.1 要求的 NVIDIA RTX GPU、驱动和 NVIDIA Container Toolkit，以及 Python 3.11+。

首次使用需要完成以下步骤：

1. clone 仓库并准备宿主依赖。
2. 创建 Bole 的 Python 环境。
3. 下载官方资产、构建仿真与 Codex 运行镜像。
4. 在宿主机加载 Codex 专用 AppArmor 策略。
5. 配置模型服务和凭据环境变量。
6. 查看计划、检查资源，再执行评测。

完整命令见[首次安装与资源准备](docs/getting-started.md)。仓库地址：[ziyue0206/Bole](https://github.com/ziyue0206/Bole)。

## 三类运行资源

| 资源 | 获取方式 |
| --- | --- |
| RoboDojo 资产 | 从官方 Hugging Face 数据集下载固定 revision 的 `Assets/**`，不下载训练数据 |
| RoboDojo 仿真镜像 | 从固定 revision 的官方源码、子模块和 Dockerfile 构建 |
| Codex 运行镜像 | 拉取 OpenAI 官方 `codex-universal` 基础镜像，下载固定版本官方 CLI 与 bwrap，再构建 Bole 薄派生层 |

版本和下载摘要保存在 [`configs/resources.lock.json`](configs/resources.lock.json)。`bole/robodojo:*`、`bole/codex:*` 是初始化脚本生成的**本地镜像标签**；本项目尚未提供这些镜像的公开 registry。

资产、镜像、缓存和模型凭据均不随源码仓库提供。

## 使用文档

- [首次安装与资源准备](docs/getting-started.md)：宿主依赖、下载、构建、AppArmor 和首次运行。
- [资源与实验配置](docs/configuration.md)：配置优先级、存储位置、模型服务、自定义镜像和缓存。
- [评测、学习与结果](docs/usage.md)：CLI、任务计划、eval、learn、学习镜像和结果读取。
- [排错](docs/troubleshooting.md)：下载、磁盘、镜像、策略、GPU 和模型连接问题。
- [开发与插件](docs/development.md)：代码结构、CPU 检查和插件入口。
- [版本状态](docs/release-status.md)：已提供内容和待验收范围。
- [第三方组件说明](THIRD_PARTY_NOTICES.md)。

## 目录

```text
Bole/
├── bin/                 # 源码 CLI 入口
├── src/bole/            # 平台、RoboDojo adapter、Codex harness、记录与结果
├── scripts/             # 资源初始化和宿主准备入口
├── configs/             # 资源版本清单和配置示例
├── containers/codex/    # 薄派生层、seccomp、AppArmor 和 shell profile
├── vendor/robodojo/     # 必要原生源码子集及第三方许可
├── prompts/             # 可选项目级 prompt
├── templates/plugins/   # policy 插件示例
├── tests/               # CPU 契约测试和显式探针
├── docs/                # 使用文档
└── licenses/            # 随源码保留的第三方许可
```

`.venv/`、`assets/`、`runtime-deps/`、`runs/` 和本地配置由用户准备或运行时生成，默认不纳入 Git。
