# 排错

先区分资源准备脚本、Bole 静态检查和真实运行。准备失败时保留配置和 `bootstrap-state.json`，按失败步骤继续；实验失败时保留整个 run 目录。

## Python 和入口

| 现象 | 检查与处理 |
| --- | --- |
| `bootstrap requires Python 3.11+` | 查看所用解释器版本，Ubuntu 22.04 默认 Python 3.10 不满足要求；使用已经准备好的 Python 3.11+ |
| 缺少 venv 或 ensurepip | 安装与实际解释器对应的 venv/pip 组件，再执行 `--steps python --apply` |
| `missing Bole Python environment` | 从仓库根目录执行 Python 准备步骤；自定义 `--venv` 后设置对应 `BOLE_PYTHON` |
| `No module named bole` | 使用 `./bin/bole`，或在正确环境重新执行 editable 安装；不要删除安装所依赖的源码目录 |
| pip 缺少构建后端 | 允许在目标虚拟环境获取 pyproject 声明的构建依赖；不要默认使用 `--no-build-isolation` |

## 下载和磁盘

| 现象 | 检查与处理 |
| --- | --- |
| `git: 'lfs' is not a git command` | 使用宿主依赖脚本安装 git-lfs，或由管理员安装；它不是 Python 包 |
| 资产文件仍为 LFS pointer | 检查 Hugging Face、Git LFS 下载及磁盘；未解析的 pointer 不可作为完整资产 |
| `Existing assets have no matching bootstrap provenance` | 若确认要复用该资产，跳过 `assets`，使用 `--assets-dir` 执行 `asset-paths,cache,config`；自行核对 revision |
| `SHA256 mismatch` | 不使用下载内容；检查网络、代理和所选版本。脚本保留 partial，不以新内容替换已存在目标 |
| `Insufficient free space` | 核对 data 目录和 DockerRootDir 各自的文件系统；修改 data 目录不能增加 Docker 数据盘空间 |
| GitHub/HF/GHCR 出现 TLS、超时或拒绝连接 | 分别检查下载进程、Git、Docker daemon 和构建网络；保留证书验证，修正证书或代理配置 |

默认资产准备只下载 `Assets/**`。不要用整个 Hugging Face 数据集的总大小估算本流程，也不要误下载训练数据。完整准备仍可能连接 Isaac 扩展和 USD 资源源站。

## 镜像和状态

| 现象 | 检查与处理 |
| --- | --- |
| `No such image` / `pull access denied` | `bole/*` 默认标签需要先在本地构建，本项目尚无对应公开 registry；运行阶段使用 `--pull=never` |
| 构建标签已存在但无匹配记录 | 检查来源，选择新的 `--harness-image` 或 `--simulator-image`；脚本不会自动覆盖未知镜像 |
| source 目录已有修改或 origin 不同 | 保留用户内容，选择新的准备目录；不使用强制 reset 来掩盖来源问题 |
| Codex 版本与策略不匹配 | 同时核对 manifest 中 CLI、bwrap、摘要和安全策略版本；只改版本号不能证明策略兼容 |
| 基础镜像变化 | 已完成的首次 pull 会记录实际 digest；需要另选版本时使用明确引用并核对准备状态和新的输出标签 |
| 配置已存在且不同 | 使用新 `--config-output`，或审阅后加 `--replace-config`；保留 `.previous` 备份 |

资源清单固定源码与工具版本，但上游 Dockerfile 中的在线包源及其可变依赖不构成逐字节可复现构建保证。

## AppArmor seccomp 与 bwrap

namespace 创建失败或 `Operation not permitted` 时，逐项检查：

1. Docker Linux 主机是否启用 AppArmor，是否已加载所配置的专用 profile。
2. 资源配置的 seccomp 文件是否存在且可由 Docker CLI 读取。
3. CLI 和 bwrap 是否匹配所选版本，容器内是否可执行。
4. 所选容器 Python 是否可由非 root 用户访问，并能创建工作区 venv。

`install_apparmor.py` 在没有 `--apply` 时只读取信息。普通用户无法读取内核 profile 列表时，`loaded: null` 表示未知，而非未加载；可由管理员通过 `aa-status` 核对。

专用 profile 的挂载权限作用于使用它的整个 Codex 容器。当前方案保持非 root、只读根目录、cap-drop 和 no-new-privileges；不要用全局禁用 AppArmor/seccomp 或 privileged 容器替代缺失的配置步骤。

## GPU 模型和首帧

| 现象 | 检查与处理 |
| --- | --- |
| `no GPU selected` | 在资源配置设置 GPU，或传入 `--gpu 0`；无 GPU 时 `check` 非就绪是预期行为 |
| `nvidia-smi` 无法通信 | 修复宿主驱动；仅存在该命令不代表 GPU 已可用 |
| 宿主可见 GPU、容器不可见 | 核对 NVIDIA Container Toolkit 与 Docker runtime 配置 |
| 缺少凭据环境变量 | 在运行 Bole 的宿主终端导出配置中 `credential_env` 对应的变量 |
| 模型返回权限、模型名或参数错误 | 选择服务实际支持的模型 ID、Responses API 和推理参数 |
| 网关代理可用但 pip 失败 | 模型代理不会自动配置 Codex shell、pip 或 Docker 下载网络 |
| 首帧超时 | 查看 simulator 日志、空缓存预热、扩展与云资产网络；确认原因后再调整 `--startup-timeout-s` |

## 结果和学习镜像

主成功率为 `null` 时，查看 native 判分覆盖和基础设施错误。视频缺失时，查看 artifacts 和编码日志；不能仅凭文件路径存在判断完整。

学习镜像恢复失败时，核对本机镜像是否存在、保存状态和原始 manifest。无需先执行 finalize。学习效果需要另做对照实验，不能从镜像保存成功直接推出。

报告问题时，提供操作系统、资源版本清单、失败步骤、已脱敏配置、具体错误和相关日志。分享 run 前自行检查 prompt、图像、模型输出和日志中是否含不宜公开的信息。
