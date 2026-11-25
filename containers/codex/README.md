# Codex 容器与宿主策略

此目录提供基于官方 `codex-universal` 的薄派生层，以及 Codex 专用 seccomp 和 AppArmor 文件。准备入口为仓库根目录的 `scripts/bootstrap.py`，版本清单为 `configs/resources.lock.json`。

## 容器镜像

官方基础镜像是开发环境。薄派生 Dockerfile 补齐系统 Python、venv、pip、固定版本 Codex CLI 和匹配的 bwrap，并为非 root login shell 使用可访问的工具路径。

Docker 构建上下文由 `codex-tools` 步骤准备，包含经过归档摘要校验的 `codex`、`bwrap` 和 `profile.sh`；它不是仓库根目录。优先通过 bootstrap 的 `codex-tools,codex-base,codex-image` 步骤构建，不要直接把仓库根目录作为该 Dockerfile 的上下文。

首次官方基础镜像 pull 后，实际 digest 记录在本地准备状态中，后续相同状态复用该引用。

## 三个安全组件的位置

| 组件 | 位置与使用方式 |
| --- | --- |
| AppArmor | 在 Docker Linux 宿主机加载命名 profile；容器启动时选择 `apparmor=bole-codex-bwrap` |
| seccomp | JSON 文件在 Docker CLI 主机可读，启动时以 `seccomp=...` 指定，内核限制容器系统调用 |
| bwrap | 在容器内执行，薄派生层放在 Codex 相邻的 `codex-resources/bwrap`；也支持配置宿主只读挂载 |

AppArmor 不会因 pull 镜像而在宿主机自动加载。安装器的预览和执行方式为：

```bash
python3 scripts/install_apparmor.py
sudo python3 scripts/install_apparmor.py --apply
```

从仓库根目录执行。profile 持久文件为 `/etc/apparmor.d/bole-codex-bwrap`。已有不同内容时先审阅，再显式使用 `--replace`，旧文件备份位于 `/var/backups/bole-apparmor/`。

## 策略范围

seccomp 以 Moby seccomp v0.2.4 默认规则为基础，补充当前 Codex/bwrap x86_64 namespace 路径所需的精确 clone flags、user namespace 和挂载相关系统调用。AppArmor 从 Moby AppArmor v0.2.2 派生，为此命名 profile 放行 bwrap 所需的 mount 和 pivot_root。

AppArmor 的这些许可作用于使用该 profile 的整个容器，不能描述成仅授权某一个可执行文件。Bole 继续设置非 root 用户、只读根目录、`cap-drop=ALL`、`no-new-privileges` 和 Codex `workspace-write`。

这些策略只用于 Codex harness，不替换 Docker 的默认策略或仿真镜像策略。当前策略对应 Codex 0.169.0/x86_64；换版本或架构需要核对 namespace 执行路径。新薄派生镜像上的完整配合和宿主重启恢复仍待验收。

策略继承的 Apache 2.0 文本见 [`licenses/Moby-profiles-LICENSE`](../../licenses/Moby-profiles-LICENSE)。Docker 的机制说明见 [AppArmor](https://docs.docker.com/engine/security/apparmor/) 和 [seccomp](https://docs.docker.com/engine/security/seccomp/)。
