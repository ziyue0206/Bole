# 资源与实验配置

资源配置描述这台机器上的源码、资产、镜像、缓存、安全策略和模型入口；实验配置描述要运行的任务、回合、预算及记录方式。生成的 `configs/portable.local.yaml` 是资源配置，`configs/eval.example.yaml`、`configs/learn.example.yaml` 是实验示例。

## 配置优先级

从低到高依次为：

1. Bole 内置默认值。
2. `--resources FILE` 指定的资源文件；未指定时读取 `BOLE_RESOURCES`；两者都未指定时尝试仓库根目录的 `resources.yaml`。
3. `--config FILE` 指定的实验文件。
4. 实际传入的 CLI 参数。

配置文件中的相对路径以**该文件所在目录**解析；CLI 中的相对路径以当前工作目录解析。容器内的程序路径保持容器内含义。

可以显式选择资源文件，也可以在当前终端设置：

```bash
export BOLE_RESOURCES="$PWD/configs/portable.local.yaml"
./bin/bole plan --task make_toast --episodes 1 --no-memory
```

本项目的实验示例不设置 GPU、网关和模型 ID，以继承用户资源配置。自己编写实验 YAML 时，注意同名字段会覆盖资源文件。

## 资源字段

| 字段 | 含义 |
| --- | --- |
| `resources.source` | Bole 运行时读取的 RoboDojo 源码目录 |
| `source_revision` | 声明的源码 revision；完整准备流程使用 manifest 中的固定提交 |
| `assets`、`asset_revision` | 资产目录与声明的版本；声明本身不等于每次全量重新核验 |
| `asset_overrides` | 由 `asset-paths` 生成的机器人配置目录，不修改下载资产 |
| `simulator_image` | 本地仿真镜像引用 |
| `simulator_image_id` | 可选的完整 `sha256:` 镜像 ID；设置后资源检查要求匹配 |
| `simulator_root` | 仿真容器内 RoboDojo 根目录，默认 `/workspace/RoboDojo` |
| `simulator_python`、`simulator_entrypoint` | 容器 Python 命令与可选 entrypoint；默认保留上游 Conda 激活入口 |
| `simulator_kit_args` | Isaac Kit 参数字符串 |
| `simulator_env` | 仿真容器的附加环境变量映射 |
| `harness_image` | Codex 基础运行镜像或 Bole 保存的学习镜像 |
| `codex_python` | Codex 容器内用于 bootstrap 的 Python 绝对路径 |
| `codex_cli`、`codex_bwrap` | 可选的宿主可执行文件路径，作为只读挂载；薄派生镜像已内置时为 `null` |
| `codex_seccomp` | 宿主上的 seccomp JSON 文件路径 |
| `codex_apparmor` | 已在 Docker 宿主机加载的专用 AppArmor profile 名称 |
| `gpus` | 宿主 GPU 数字索引列表，例如 `[0]` 或 `[0, 1]` |
| `cache_source`、`extensions_source` | 每次环境启动时复制的缓存种子目录 |
| `cache_volumes.cache`、`cache_volumes.extensions` | 替代目录种子的已有 Docker volume；不配置时为 `null` |

目录种子优先于对应 volume。Bole 为每个环境建立可写副本，种子不会自动接收运行中新增的缓存。空种子可能导致后续回合重复预热；需要预热缓存时，单独准备可复用种子并通过配置指定。

`--simulator-image` 显式选择新镜像时，会清除继承资源文件中的 `simulator_image_id`；在自定义 YAML 中更换镜像时，也应核对或更新该 ID。

## 存储与恢复

`bootstrap.py` 的主要位置参数如下。它们是准备阶段的参数，不是 `bole eval` 参数。

| 参数 | 默认位置或用途 |
| --- | --- |
| `--manifest` | `configs/resources.lock.json` |
| `--data-dir` | `runtime-deps/portable`，保存源码、下载缓存、CLI、生成配置及准备状态 |
| `--source-dir` | 默认在 data 目录的 `robodojo-source` |
| `--assets-dir` | `assets/robodojo`；脚本将其链接到 data 目录中的资产 checkout |
| `--venv` | 仓库 `.venv`；自定义后还需用 `BOLE_PYTHON` 指向实际解释器 |
| `--config-output` | `configs/portable.local.yaml` |
| `--replace-config` | 显式替换不同的已有配置，并保留 `.previous` 副本 |
| `--simulator-image`、`--harness-image` | 自定义输出的本地镜像标签 |
| `--codex-base-image` | 覆盖官方基础镜像引用，可使用已知 digest |

例如，准备资源时改用数据盘：

```bash
.venv/bin/python scripts/bootstrap.py \
  --steps source,assets,asset-paths,codex-tools,codex-base,codex-image,simulator-image,cache,config \
  --data-dir /data/bole/runtime \
  --assets-dir /data/bole/Assets \
  --model "$BOLE_MODEL" --gpu 0 \
  --apply
```

相同参数需要在重试、重新生成配置等相关调用中保持一致。新脚本进程不会从先前终端自动恢复所有 CLI 参数；状态文件仅记录已完成的资源准备结果。

步骤可以通过 `--steps` 拆开，执行顺序始终使用脚本的固定顺序，而非逗号列表的书写顺序。`codex-image` 依赖 `codex-tools` 和 `codex-base`；`simulator-image` 依赖 `source`；`asset-paths` 依赖实际资产；最终 `config` 应在资源位置确定后执行。

已有本地资产但没有本脚本的下载记录时，跳过 `assets`，显式引用目录：

```bash
.venv/bin/python scripts/bootstrap.py \
  --steps asset-paths,cache,config \
  --source-dir /data/RoboDojo \
  --assets-dir /data/Assets \
  --data-dir /data/bole/runtime \
  --model "$BOLE_MODEL" --gpu 0 \
  --apply
```

这里假定镜像已另行准备；该命令只生成路径、缓存目录和配置。复用的资产版本由用户核对，脚本不会把无准备记录的现有目录标为已下载验证。

脚本拒绝覆盖不同来源的 Git checkout、修改过的受跟踪源码、未知来源的同名镜像和不同的已有配置。遇到这些情况应保留原内容，选择新目录或标签，或审阅后使用适用的显式替换选项。

配置和缓存状态记录了绝对路径。移动仓库或数据后，应检查资产链接，并使用新的路径参数重新执行 `asset-paths,cache,config`，通过新配置文件或 `--replace-config` 更新引用。

## 模型与网络

```yaml
harness:
  model: YOUR_MODEL_ID
  reasoning_effort: high
  gateway:
    base_url: https://YOUR_RESPONSES_ENDPOINT/v1
    credential_env: PROVIDER_API_KEY
    proxy: null
    compatibility: false
```

模型服务需要兼容 Bole/Codex 使用的 Responses API。仅能处理 Chat Completions 的 endpoint 不能据此视为可用。模型 ID、推理参数和工具能力由所选服务决定；Bole 接受的推理强度为 `low`、`medium`、`high`、`xhigh`、`max`，服务可能只支持其中一部分。

在初始化时可以通过 `--model`、`--model-base-url`、`--credential-env` 和 `--proxy` 生成这些设置，也可以之后编辑本地资源配置。URL 中不得嵌入用户名或密码，凭据字段填写环境变量名。

网络配置分为三处：

- Python/Git 下载通常使用其标准代理环境变量。
- Docker pull 和构建中的下载需要 Docker daemon、构建网络及相应软件源可达；宿主 shell 的代理不能保证 Docker daemon 自动使用。
- `harness.gateway.proxy` 只配置模型传输；Codex shell 的联网能力由 `harness.shell.network_access` 控制，其 pip 等工具还需要各自网络与包源配置。

Docker daemon 代理配置可参考[官方文档](https://docs.docker.com/engine/daemon/proxy/)。不要把模型代理参数当作所有下载路径的统一代理开关。

## 实验示例

```bash
./bin/bole eval \
  --resources configs/portable.local.yaml \
  --config configs/eval.example.yaml \
  --model "$BOLE_MODEL" --gpu 0

./bin/bole plan \
  --resources configs/portable.local.yaml \
  --config configs/learn.example.yaml \
  --mode learn
```

第一条会执行真实实验；第二条只展开计划。完整命令参数可通过 `./bin/bole eval --help` 和 `./bin/bole learn --help` 查看。
