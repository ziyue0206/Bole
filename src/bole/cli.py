"""One CLI for native benchmark plans, isolated eval, and continuous learn."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bole import config as configuration, registry
from bole.engine.runner import new_run_root, preflight
from bole.results.storage import atomic_write, redact
from bole.runtime.snapshots import freeze


def parser():
    root = argparse.ArgumentParser(prog="bole", description="Benchmark-oriented evaluation and continuous harness learning")
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("eval", "learn", "plan", "check"):
        p = sub.add_parser(command)
        p.set_defaults(**{"command": command})
        # Suppression is essential: an omitted CLI option must not overwrite YAML.
        p.add_argument("--config", default=argparse.SUPPRESS)
        p.add_argument("--resources", default=argparse.SUPPRESS,
                       help="resource configuration; overrides BOLE_RESOURCES and project resources.yaml")
        p.add_argument("--simulator-image", default=argparse.SUPPRESS)
        p.add_argument("--bench", default=argparse.SUPPRESS)
        p.add_argument("--harness", default=argparse.SUPPRESS)
        p.add_argument("--policy", default=argparse.SUPPRESS)
        p.add_argument("--task", action="append", default=argparse.SUPPRESS,
                       help="repeat for selected tasks; default is the complete native task catalog")
        p.add_argument("--suite", default=argparse.SUPPRESS)
        p.add_argument("--episodes", type=int, default=argparse.SUPPRESS,
                       help="total episodes per runnable task, taking the native plan prefix")
        p.add_argument("--seed", type=int, action="append", default=argparse.SUPPRESS,
                       help="repeat to select predefined native seed banks; default 0, 1, 2")
        p.add_argument("--success-grace-steps", type=int, default=argparse.SUPPRESS)
        p.add_argument("--parallel", type=int, default=argparse.SUPPRESS)
        p.add_argument("--gpu", type=int, action="append", default=argparse.SUPPRESS)
        p.add_argument("--output", default=argparse.SUPPRESS)
        p.add_argument("--source", default=argparse.SUPPRESS)
        p.add_argument("--assets", default=argparse.SUPPRESS)
        p.add_argument("--harness-image", default=argparse.SUPPRESS,
                       help="local Codex base or learned image; default is resources.harness_image")
        if command != "eval":
            p.add_argument("--save-image", default=argparse.SUPPRESS,
                           help="learn output image tag; use {task} for multiple tasks; default uses run/task IDs")
        p.add_argument("--model", default=argparse.SUPPRESS)
        p.add_argument("--reasoning-effort", default=argparse.SUPPRESS)
        p.add_argument("--instruction", default=argparse.SUPPRESS)
        p.add_argument("--memory", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        p.add_argument("--video", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
        p.add_argument("--timeout", "--timeout-s", dest="timeout_s", type=float, default=argparse.SUPPRESS)
        for limit in ("control-steps", "tool-calls", "model-requests"):
            p.add_argument("--" + limit, type=int, default=argparse.SUPPRESS)
        for limit in ("startup-timeout-s", "request-timeout-s"):
            p.add_argument("--" + limit, type=float, default=argparse.SUPPRESS)
        p.add_argument("--asset-mode", choices=("by_task", "shared"), default=argparse.SUPPRESS)
        p.add_argument("--task-asset", action="append", metavar="TASK=MANIFEST", default=argparse.SUPPRESS)
        p.add_argument("--shared-asset", default=argparse.SUPPRESS)
        if command in {"plan", "check"}:
            p.add_argument("--mode", choices=("eval", "learn"), default="eval")
    plugins = sub.add_parser("plugins", help="list registered benchmark, harness and policy adapters")
    plugins.add_argument("kind", nargs="?", choices=("benchmarks", "harnesses", "policies"))
    memory = sub.add_parser("memory", help="read-only native memory diagnostics")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    inspect = memory_commands.add_parser("inspect", help="inspect files and SQLite counts without starting Codex")
    inspect.add_argument("--home", type=Path, required=True, help="existing private CODEX_HOME for the run")
    inspect.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None,
                         help="effective runtime feature value, only if known; default is unknown")
    return root


def overrides(args):
    """Map only present CLI fields to the shared, validated config schema."""
    values = vars(args)
    mode = values.get("mode", values["command"])
    out = {}
    mapping = {
        "bench": ("benchmark", "name"), "harness": ("harness", "name"),
        "suite": ("benchmark", "suite"), "seed": ("benchmark", "seeds"),
        "episodes": ("episodes",), "success_grace_steps": ("termination", "success_grace_steps"),
        "parallel": (mode, "parallel"), "gpu": ("resources", "gpus"), "output": ("output",),
        "source": ("resources", "source"), "assets": ("resources", "assets"),
        "simulator_image": ("resources", "simulator_image"),
        "harness_image": ("resources", "harness_image"), "save_image": ("learn", "save_image"),
        "model": ("harness", "model"), "reasoning_effort": ("harness", "reasoning_effort"),
        "instruction": ("harness", "instruction"), "memory": ("harness", "memory", "enabled"),
        "video": ("recording", "video", "enabled"),
        "asset_mode": ("learning_assets", "mode"), "shared_asset": ("learning_assets", "shared"),
    }
    for name in ("timeout_s", "control_steps", "tool_calls", "model_requests", "startup_timeout_s", "request_timeout_s"):
        mapping[name] = ("limits", name)
    for argument, keys in mapping.items():
        if argument not in values:
            continue
        target = out
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = values[argument]
    if "simulator_image" in values:
        out["resources"]["simulator_image_id"] = None
    if "task" in values:
        tasks = values["task"]
        if "all" in tasks and tasks != ["all"]:
            raise ValueError("--task all cannot be combined with individual tasks")
        out.setdefault("benchmark", {})["tasks"] = "all" if tasks == ["all"] else tasks
    if "policy" in values:
        out["policy"] = {"name": values["policy"]}
    if "task_asset" in values:
        assets = {}
        for entry in values["task_asset"]:
            task, separator, path = entry.partition("=")
            if not separator or not task or not path or task in assets:
                raise ValueError("--task-asset requires distinct TASK=MANIFEST entries")
            assets[task] = path
        out.setdefault("learning_assets", {})["by_task"] = assets
    return out


def _write(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())


def launch(config, plan, benchmark, harness, policy=None):
    if config["mode"] == "learn" and config["learn"].get("save_image"):
        from bole.runtime.image_assets import require_unused_name
        from bole.harnesses.codex import command
        import asyncio
        for task in plan["tasks"]:
            asyncio.run(require_unused_name(config["learn"]["save_image"].replace("{task}", task["id"]), command))
    report = preflight(config, benchmark)
    if not report["ok"]:
        raise ValueError("; ".join(report["errors"]))
    root = new_run_root(config)
    frozen, manifest = freeze(config, root, plugins=[p for p in (benchmark, harness, policy) if p is not None])
    # Every episode starts from the same inspected image, even if its tag moves.
    image_id = report.get("benchmark", {}).get("resources", {}).get("harness_image_id")
    if image_id:
        frozen["resources"]["harness_image"] = image_id
    # These internal inputs are separate from the evidence files written by the
    # child (which include run identity and redaction).
    _write(root / "execution-config.json", frozen)
    _write(root / "execution-plan.json", plan)
    _write(root / "preflight.json", report)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(manifest["python_paths"])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    print(json.dumps({"run_path": str(root), "code_snapshot": str(root / "snapshot")}), flush=True)
    process = subprocess.Popen([sys.executable, "-m", "bole.engine.runner",
        "--child-config", str(root / "execution-config.json"),
        "--child-plan", str(root / "execution-plan.json"), "--run-dir", str(root)],
        cwd=root / "snapshot" / "bole", env=environment)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # The same terminal sends SIGINT to the child. Give its cancellation
        # path time to close environments and save partial learning assets.
        return process.wait()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "memory":
            from bole.runtime.learning_assets import inspect_memory
            report = inspect_memory(args.home.expanduser(), enabled=args.enabled)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        if args.command == "plugins":
            kinds = (args.kind,) if args.kind else ("benchmarks", "harnesses", "policies")
            print(json.dumps({kind: registry.discover(kind) for kind in kinds}, indent=2))
            return 0
        mode = args.command if args.command in {"eval", "learn"} else args.mode
        config = configuration.load(getattr(args, "config", None), overrides(args), mode=mode,
                                    resources_path=(Path(args.resources).expanduser().resolve(strict=True)
                                                    if hasattr(args, "resources") else None))
        if hasattr(args, "harness_image") and (config["learning_assets"].get("by_task") or config["learning_assets"].get("shared")):
            raise ValueError("select either --harness-image or manifest learning assets, not both")
        benchmark = registry.load("benchmarks", config["benchmark"]["name"])
        if args.command == "check":
            report = preflight(config, benchmark, runtime=False)
            print(json.dumps(redact(report), indent=2, ensure_ascii=False))
            return 0 if report["ok"] else 1
        plan = benchmark.plan(config)
        if args.command == "plan":
            print(json.dumps(redact(plan), indent=2, ensure_ascii=False))
            return 0
        harness = registry.load("harnesses", config["harness"]["name"])
        policy = None
        if config.get("policy"):
            name = config["policy"]["name"] if isinstance(config["policy"], dict) else config["policy"]
            policy = registry.load("policies", name)
        return launch(config, plan, benchmark, harness, policy)
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
        print("bole: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
