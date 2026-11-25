"""Shared CLI/YAML configuration, explicit defaults, path resolution and validation."""

from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = PACKAGE_ROOT.parents[1]
PROJECT_ROOT = (_SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file()
                and (_SOURCE_ROOT / "src/bole").resolve() == PACKAGE_ROOT else Path.cwd())
DEFAULTS = {
    "schema_version": 1,
    "mode": "eval",
    "episodes": None,
    "episodes_selection": "native_prefix",
    "benchmark": {"name": "robodojo", "tasks": "all", "suite": None,
                  "seeds": [0, 1, 2], "instances": None, "options": {}},
    "harness": {"name": "codex", "model": "gpt-6-astra", "reasoning_effort": "high",
                "instruction": str(PACKAGE_ROOT / "prompts/codex.md"),
                "executable": "/opt/bole-tools/codex", "memory": {"enabled": True,
                    "min_rollout_idle_hours": None, "version": None},
                "compact": {"auto_token_limit": None}, "options": {},
                "shell": {"network_access": True},
                "gateway": {"base_url": "https://api.openai.com/v1",
                            "credential_env": "OPENAI_API_KEY",
                            "proxy": None, "compatibility": False}},
    "policy": None,
    "eval": {"session": "isolated", "parallel": 1},
    "learn": {"parallel": 1, "save_image": None},
    "limits": {"timeout_start": "first_observation", "timeout_s": None,
               "control_steps": None, "tool_calls": None, "model_requests": None,
               "startup_timeout_s": 600, "request_timeout_s": 120},
    "termination": {"success_grace_steps": 20},
    "resources": {
        "source": str(PROJECT_ROOT / "vendor/robodojo"),
        "assets": str(PROJECT_ROOT / "assets/robodojo"),
        "source_revision": "f57e1633ffed59761bc8e568cfe7530c4b33381c",
        "asset_revision": "1127bda94f172d757386945af3eb3a6754d1a4e6",
        "simulator_image": "bole/robodojo:f57e1633ffed",
        "simulator_image_id": None,
        "harness_image": "bole/codex:0.169.0",
        "codex_python": "/usr/bin/python3",
        "codex_cli": None,
        "simulator_root": "/workspace/RoboDojo",
        "simulator_python": "python",
        "simulator_entrypoint": None,
        "simulator_kit_args": "--ext-folder /root/.local/share/ov/data/exts/v2 --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        "simulator_env": {},
        "asset_overrides": None,
        "codex_bwrap": None,
        "codex_seccomp": None,
        "codex_apparmor": None,
        "gpus": [],
        "cache_source": None,
        "extensions_source": None,
        "cache_volumes": {"cache": None, "extensions": None},
    },
    "recording": {"video": {"enabled": True, "fps": 5,
                              "cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"]},
                  "safety": {"enabled": True, "roles": None}},
    "learning_assets": {"mode": None, "by_task": {}, "shared": None},
    "output": str(Path.cwd() / "runs"),
}


def merge(base, override, trail=""):
    if not isinstance(override, dict):
        raise ValueError(f"{trail or 'configuration'} must be a mapping")
    result = copy.deepcopy(base)
    extensible = {"learning_assets.by_task", "benchmark.instances", "benchmark.options", "harness.options", "resources.simulator_env"}
    for key, value in override.items():
        path = f"{trail}.{key}".strip(".")
        if key not in base:
            raise ValueError(f"unknown configuration field: {path}")
        if isinstance(base[key], dict) and path not in extensible:
            result[key] = merge(base[key], value, path)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_paths(value, base):
    value = copy.deepcopy(value)
    for keys in [("output",), ("harness", "instruction"), ("resources", "source"),
                 ("resources", "assets"), ("resources", "cache_source"),
                 ("resources", "extensions_source"), ("resources", "codex_bwrap"),
                 ("resources", "codex_seccomp"), ("resources", "codex_cli"),
                 ("resources", "asset_overrides"),
                 ("recording", "safety", "roles"),
                 ("learning_assets", "shared")]:
        current = value
        for key in keys[:-1]:
            current = current.get(key, {})
        leaf = keys[-1]
        if isinstance(current.get(leaf), str):
            path = Path(current[leaf]).expanduser()
            current[leaf] = str((base / path).resolve() if not path.is_absolute() else path)
    assets = value.get("learning_assets", {}).get("by_task", {})
    for key, text in assets.items():
        path = Path(text).expanduser()
        assets[key] = str((base / path).resolve() if not path.is_absolute() else path)
    return value


def positive(value, name, *, optional=False, integer=False, zero=False):
    if value is None and optional:
        return
    valid = type(value) is int if integer else type(value) in (int, float)
    if not valid or not math.isfinite(value) or value < (0 if zero else 1e-300):
        raise ValueError(f"{name} requires a {'nonnegative' if zero else 'positive'} "
                         f"{'integer' if integer else 'number'}")


def validate(c):
    if c["schema_version"] != 1 or c["mode"] not in {"eval", "learn"}:
        raise ValueError("schema_version must be 1 and mode must be eval or learn")
    tasks = c["benchmark"]["tasks"]
    if tasks != "all" and (not isinstance(tasks, list) or not tasks or
                           any(not isinstance(t, str) or not re.fullmatch(r"[\w-]+", t)
                               for t in tasks) or len(set(tasks)) != len(tasks)):
        raise ValueError("benchmark.tasks must be 'all' or a nonempty distinct task list")
    seeds = c["benchmark"]["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("benchmark.seeds must be a nonempty distinct list")
    for seed in seeds:
        positive(seed, "seed", integer=True, zero=True)
    positive(c["episodes"], "episodes", optional=True, integer=True)
    if c["episodes_selection"] not in {"native_prefix", "seed_round_robin"}:
        raise ValueError("unknown episodes_selection")
    for mode in ("eval", "learn"):
        positive(c[mode]["parallel"], mode + ".parallel", integer=True)
    image = c["learn"]["save_image"]
    if image is not None:
        from bole.runtime.image_assets import validate_image_name
        validate_image_name(image.replace("{task}", "task"))
        if "{" in image.replace("{task}", "") or "}" in image.replace("{task}", ""):
            raise ValueError("learn.save_image only supports the {task} placeholder")
    if c["eval"]["session"] != "isolated":
        raise ValueError("eval requires isolated sessions")
    if c["limits"]["timeout_start"] != "first_observation":
        raise ValueError("timeout_start must be first_observation")
    for key, value in c["limits"].items():
        if key == "timeout_start":
            continue
        positive(value, "limits." + key, optional=key not in {
            "startup_timeout_s", "request_timeout_s"}, integer=key in {
                "control_steps", "tool_calls", "model_requests"})
    positive(c["termination"]["success_grace_steps"], "success_grace_steps", integer=True,
             zero=True)
    if not isinstance(c["resources"]["gpus"], list):
        raise ValueError("resources.gpus must be a list")
    for gpu in c["resources"]["gpus"]:
        positive(gpu, "GPU index", integer=True, zero=True)
    if len(set(c["resources"]["gpus"])) != len(c["resources"]["gpus"]):
        raise ValueError("duplicate GPU index")
    for section in (c["harness"]["memory"], c["recording"]["video"], c["recording"]["safety"]):
        if type(section["enabled"]) is not bool:
            raise ValueError("enabled must be a boolean")
    if c["mode"] == "learn" and not c["harness"]["memory"]["enabled"]:
        raise ValueError("Codex learn requires native memory enabled")
    if type(c["harness"]["shell"]["network_access"]) is not bool:
        raise ValueError("harness.shell.network_access must be a boolean")
    idle = c["harness"]["memory"].get("min_rollout_idle_hours")
    if c["harness"]["memory"].get("version") not in {None, "v1", "v2"}:
        raise ValueError("Codex memory.version must be v1 or v2")
    if idle is not None and (type(idle) is not int or not 1 <= idle <= 48):
        raise ValueError("Codex 0.169 memory.min_rollout_idle_hours must be an integer from 1 to 48")
    if c["harness"]["reasoning_effort"] not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("invalid reasoning_effort")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c["harness"]["gateway"]["credential_env"]):
        raise ValueError("credential_env must be an environment variable name, never a key value")
    for name in ("base_url", "proxy"):
        address = c["harness"]["gateway"][name]
        if name == "proxy" and address is None:
            continue
        parsed = urlsplit(address)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"harness.gateway.{name} requires an HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError(f"harness.gateway.{name} cannot contain credentials")
    positive(c["recording"]["video"]["fps"], "video.fps")
    cameras = c["recording"]["video"]["cameras"]
    if (not isinstance(cameras, list) or not cameras or
        any(not isinstance(camera, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", camera)
            for camera in cameras) or len(set(cameras)) != len(cameras)):
        raise ValueError("video.cameras must be a nonempty distinct list")
    if c["learning_assets"]["mode"] not in {None, "by_task", "shared"}:
        raise ValueError("learning_assets.mode must be by_task or shared")
    resources = c["resources"]
    for key in ("codex_python", "simulator_root"):
        text = resources[key]
        if not isinstance(text, str) or not text.startswith("/") or ".." in Path(text).parts:
            raise ValueError(f"resources.{key} requires an absolute container path")
    for key in ("simulator_python", "simulator_image", "harness_image"):
        if not isinstance(resources[key], str) or not resources[key].strip():
            raise ValueError(f"resources.{key} must be a nonempty string")
    expected = resources["simulator_image_id"]
    if expected is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
        raise ValueError("resources.simulator_image_id requires a complete sha256 image ID")
    for key, text in resources["simulator_env"].items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(text, str):
            raise ValueError("resources.simulator_env requires environment names and string values")
    return c


def load(path=None, overrides=None, *, mode="eval", resources_path=None):
    config = copy.deepcopy(DEFAULTS)
    selected_resources = resources_path or os.environ.get("BOLE_RESOURCES")
    resource_file = Path(selected_resources).expanduser().resolve() if selected_resources else PROJECT_ROOT / "resources.yaml"
    if not resources_path and selected_resources and not resource_file.is_file():
        raise FileNotFoundError(f"resource configuration not found: {resource_file}")
    if resource_file.exists():
        content = yaml.safe_load(resource_file.read_text()) or {}
        content = content if "resources" in content else {"resources": content}
        config = merge(config, _resolve_paths(content, resource_file.resolve().parent))
    if path:
        source = Path(path).resolve(strict=True)
        content = yaml.safe_load(source.read_text()) or {}
        config = merge(config, _resolve_paths(content, source.parent))
    if overrides:
        config = merge(config, _resolve_paths(overrides, Path.cwd()))
    config["mode"] = mode
    return validate(config)
