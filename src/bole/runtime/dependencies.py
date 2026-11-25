"""Optional host dependencies for existing images; no install or policy changes."""
from __future__ import annotations

import os
import json
import re
from pathlib import Path, PurePosixPath


def codex_bwrap_mount(source, executable):
    """Mount the approved native helper beside the configured Codex executable."""
    if source is None:
        return []
    path = Path(source).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("resources.codex_bwrap must be an executable regular file")
    binary = PurePosixPath(executable)
    if not binary.is_absolute() or ".." in binary.parts:
        raise ValueError("harness.executable must be an absolute container path when codex_bwrap is set")
    destination = binary.parent / "codex-resources/bwrap"
    if "," in str(path) or "," in str(destination):
        raise ValueError("Codex dependency bind mount paths cannot contain commas")
    return ["--mount", f"type=bind,src={path},dst={destination},readonly"]


def codex_security_args(resources):
    """Apply service-specific profiles without adding capabilities."""
    result = []
    if resources.get("codex_seccomp"):
        path = Path(resources["codex_seccomp"]).resolve(strict=True)
        policy = json.loads(path.read_text())
        if not isinstance(policy, dict) or not isinstance(policy.get("syscalls"), list):
            raise ValueError("resources.codex_seccomp must name a seccomp JSON profile")
        result += ["--security-opt", f"seccomp={path}"]
    if resources.get("codex_apparmor"):
        name = resources["codex_apparmor"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name == "unconfined":
            raise ValueError("resources.codex_apparmor requires a named AppArmor profile")
        result += ["--security-opt", f"apparmor={name}"]
    return result


def codex_cli_mount(source, executable):
    """Optional official CLI binary, independently configurable from the image."""
    if source is None:
        return []
    path = Path(source).resolve(strict=True)
    destination = PurePosixPath(executable)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("resources.codex_cli must be an executable regular file")
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("harness.executable must be an absolute container path")
    if "," in str(path) or "," in str(destination):
        raise ValueError("Codex bind mount paths cannot contain commas")
    return ["--mount", f"type=bind,src={path},dst={destination},readonly"]


def asset_override_mounts(directory, container_root):
    """Read-only generated robot configurations; immutable downloaded assets stay unchanged."""
    if not directory:
        return []
    root = Path(directory).resolve(strict=True)
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("asset overrides cannot contain symbolic links")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] != "Robots" or path.name != "curobo.yml":
            raise ValueError(f"unsupported asset override: {relative}")
        destination = PurePosixPath(container_root) / "Assets" / relative.as_posix()
        if "," in str(path) or "," in str(destination):
            raise ValueError("asset bind mount paths cannot contain commas")
        result += ["--mount", f"type=bind,src={path},dst={destination},readonly"]
    if not result:
        raise ValueError("asset overrides directory contains no generated robot configurations")
    return result


def prepare_robot_overrides(directory, assets, staging, container_root):
    """Overlay a robot directory even when the upstream asset lacks curobo.yml.

    Other robot files are container-visible links to a second read-only asset
    mount. This avoids copying large robot assets into every run snapshot.
    """
    if not directory:
        return []
    import shutil
    asset_override_mounts(directory, container_root)  # Validate before writing staging files.
    assets = Path(assets).resolve(strict=True)
    robots = assets / "Robots"
    destination = Path(staging) / "robot-assets"
    destination.mkdir()
    for path in robots.rglob("*"):
        relative = path.relative_to(robots)
        target = destination / relative
        if path.is_symlink() and assets not in path.resolve(strict=True).parents:
            raise ValueError("robot asset links must remain within the asset tree")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(str(PurePosixPath("/opt/bole-assets/Robots") / relative.as_posix()))
    for path in Path(directory).resolve().rglob("curobo.yml"):
        relative = path.relative_to(Path(directory).resolve())
        target = destination.joinpath(*relative.parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        shutil.copy2(path, target)
    return ["--mount", f"type=bind,src={assets},dst=/opt/bole-assets,readonly",
            "--mount", f"type=bind,src={destination},dst={container_root}/Assets/Robots,readonly"]
