# 首次安装与资源准备

本流程从源码仓库安装 Bole，并使用官方来源准备 RoboDojo 和 Codex。资源准备命令会下载软件和资产、构建镜像；`eval`、`learn` 才会调用模型并启动真实仿真。完整新机器安装和运行仍待验收，见[版本状态](release-status.md)。

## 1 准备主机

在运行 Docker daemon 的同一台 Linux 主机上执行 Bole。当前挂载方式使用宿主绝对路径和 host 网络，使用远程 Docker daemon、rootless Docker、Docker Desktop 或其他架构需要单独适配。

| 条件 | 要求 |
| --- | --- |
| 系统 | Linux x86_64，启用 AppArmor；以下宿主命令以 Ubuntu 24.04 为例 |
| Python | 3.11+，具备 `venv` 和 pip；Ubuntu 22.04 的默认 Python 3.10 不满足要求 |
| GPU | 符合 Isaac Sim 5.1 要求的 NVIDIA RTX GPU、驱动及足够显存 |
| Docker | 当前用户可使用 Docker Engine；已经配置 NVIDIA Container Toolkit |
| 网络 | 可访问 GitHub、Hugging Face、GHCR、Docker Hub、Python/NVIDIA 软件源及所选模型服务 |
| 资产磁盘 | 初始化脚本要求其下载数据所在磁盘至少有 50 GiB 空闲；资产本身约 40 GB |
| Docker 磁盘 | 首次仿真镜像构建前，脚本要求 Docker 数据目录所在磁盘至少有 300 GiB 空闲 |

两个磁盘要求需要分别满足；下载目录和 Docker 数据目录可能位于不同文件系统。把 `--data-dir` 放到大盘不会自动迁移 Docker 数据目录。缓存、结果、视频和并发回合还会继续消耗磁盘。

按官方说明准备 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/)、[NVIDIA 驱动和 Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。Toolkit 配置可能需要重启 Docker，会影响同机容器，应由主机管理员安排。Bole 的宿主依赖脚本不安装或重配这些组件。

显卡要求以 [Isaac Sim 5.1 官方要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)为准；A100、H100 等缺少 RT Core 的 GPU 不适用于该仿真渲染路径。

主机准备好后，可以自行检查：

```bash
python3 --version
docker info
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
docker info --format '{{.DockerRootDir}}'
df -h
```

上述 Docker 检查会启动一个临时 GPU 容器；它不是 Bole 的评测回合。

## 2 获取源码与小型宿主依赖

从 GitHub 获取源码：

```bash
git clone https://github.com/ziyue0206/Bole.git Bole
cd Bole
```

Ubuntu 24.04 可以先准备系统 Python：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
```

Ubuntu 22.04 用户须先准备 Python 3.11+ 及其对应的 venv/pip。以下所有 `python3` 命令都应替换为该兼容解释器；不要用 Python 3.10 执行 `bootstrap.py`。

查看并按需安装 git、git-lfs、curl、证书和 AppArmor 工具：

```bash
python3 scripts/install_host_dependencies.py
sudo python3 scripts/install_host_dependencies.py --apply
```

没有 `--apply` 时只显示计划。该脚本的自动安装路径支持 Debian/Ubuntu，需要管理员权限。

## 3 创建 Bole Python 环境

```bash
python3 scripts/bootstrap.py --steps python --apply
./bin/bole --help
./bin/bole plugins
```

这一步创建仓库内的 `.venv`，通过 `pip install -e .` 安装 Bole，需要访问 Python 构建依赖和包源。`bin/bole` 默认使用这份环境，不要求先激活它。

本流程支持保留源码目录的安装方式；独立 wheel 分发不属于本流程。移动或删除源码目录会影响 editable 安装及资源定位。

## 4 选择模型并预览资源准备

将下面的值替换为模型服务实际提供、且支持所需 Responses API 的模型 ID：

```bash
BOLE_MODEL='YOUR_MODEL_ID'
.venv/bin/python scripts/bootstrap.py --model "$BOLE_MODEL" --gpu 0
```

预览只读取本地信息，不拉取镜像、不下载资产。默认模型入口是 `https://api.openai.com/v1`，凭据变量名为 `OPENAI_API_KEY`。本指南始终显式指定模型 ID；代码中的模型默认值不代表你的服务账号具有相应权限。

如果使用其他服务，在接下来的**实际准备命令**中一并提供：

```text
--model-base-url https://YOUR_RESPONSES_ENDPOINT/v1
--credential-env PROVIDER_API_KEY
```

具体配置见[模型与网络](configuration.md#模型与网络)。

## 5 下载资源并构建镜像

确认磁盘空间、模型配置和 GPU 编号后执行：

```bash
.venv/bin/python scripts/bootstrap.py \
  --steps source,assets,asset-paths,codex-tools,codex-base,codex-image,simulator-image,cache,config \
  --model "$BOLE_MODEL" --gpu 0 \
  --apply
```

该命令按固定顺序准备：

| 步骤 | 行为 |
| --- | --- |
| `source` | checkout 固定 RoboDojo revision，并初始化该 revision 指向的 IsaacLab、cuRobo、XPolicyLab 子模块 |
| `assets` | 从官方 Hugging Face 仓库只下载 `Assets/**`，检查未解析的 LFS pointer |
| `asset-paths` | 从机器人模板生成独立配置，使用容器内路径，保留下载资产原样 |
| `codex-tools` | 下载固定版本官方 Codex CLI 和 bwrap，核对归档 SHA256 |
| `codex-base` | pull 官方基础镜像，记录实际 digest；后续相同准备状态复用该 digest |
| `codex-image` | 在官方基础镜像上构建 Bole 薄派生层，补齐非 root 可用的 Python、CLI 和 bwrap |
| `simulator-image` | 使用所选 RoboDojo 官方 Dockerfile 构建仿真镜像 |
| `cache` | 创建空的缓存种子目录 |
| `config` | 生成 `configs/portable.local.yaml`，记录路径、镜像、GPU 和模型入口 |

`bole/robodojo:f57e1633ffed` 和 `bole/codex:0.169.0` 是默认本地构建标签。本项目没有对应的公开镜像下载地址，直接 `docker pull bole/codex:0.169.0` 不属于此流程。

资源准备状态保存在 `runtime-deps/portable/bootstrap-state.json`。`portable.local.yaml` 采用 JSON 兼容的 YAML 格式，可以正常供 Bole 读取。第一次 Isaac 启动仍可能获取扩展或云端 USD 资产；空缓存准备完成不等于离线资源齐备。

需要使用其他磁盘、复用已有资产或拆分步骤时，先看[存储与恢复](configuration.md#存储与恢复)，并在每次相关命令中使用相同的路径参数。

## 6 在宿主机安装 AppArmor 策略

```bash
.venv/bin/python scripts/install_apparmor.py
sudo python3 scripts/install_apparmor.py --apply
```

AppArmor 策略在 **Docker 宿主机**加载，Codex 运行容器时按名称选择它。bwrap 在容器内执行；seccomp JSON 由 Docker 启动参数指定。三者位置及权限见 [Codex 容器说明](../containers/codex/README.md)。

安装器只处理名为 `bole-codex-bwrap` 的专用策略，不修改 Docker 默认 profile。已有不同内容的同名策略会要求人工审阅，再显式使用 `--replace`。安装器保存持久文件，但当前版本尚无重启后的恢复验收。

## 7 设置模型凭据

以 Bash 和默认变量名为例，交互输入凭据，避免直接写进命令历史：

```bash
read -r -s -p 'OPENAI_API_KEY: ' OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
```

若准备配置使用了 `--credential-env PROVIDER_API_KEY`，这里也必须改用 `PROVIDER_API_KEY`。仅在运行 Bole 的宿主终端提供实际值；资源配置只填写变量名。

## 8 查看计划和检查资源

```bash
./bin/bole plan \
  --resources configs/portable.local.yaml \
  --task make_toast --episodes 1 --no-memory

./bin/bole check \
  --resources configs/portable.local.yaml \
  --task make_toast --episodes 1 --gpu 0 --no-memory
```

`plan` 需要可读取的源码和布局资产，不启动容器或模型。`check` 是静态资源检查，会查询已有 Docker 镜像和所选 GPU；通过不代表真实运行已经验收。

不配置 GPU 时，`check` 返回非就绪是预期情况。逐项阅读 JSON 的 `issues`，不要把退出码 1 一律当成安装脚本损坏。

## 9 执行一个真实回合

下面的命令会启动 GPU 仿真并调用模型，可能产生费用：

```bash
./bin/bole eval \
  --resources configs/portable.local.yaml \
  --task make_toast --episodes 1 --gpu 0 \
  --model "$BOLE_MODEL" \
  --timeout 1200 --model-requests 50 --no-memory \
  --output ./runs
```

CLI 会输出新建的 `run_path`。查看结果与后续 learn 操作见[评测、学习与结果](usage.md)。上述实验命令为使用说明，当前发布快照尚未通过该真实回合验收。
