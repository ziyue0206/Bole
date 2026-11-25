"""Freeze the code used by a run; large immutable resources remain references.

This is a file snapshot, not an image build or a content-addressed archive. The
child process imports Bole from this copy so editing the checkout cannot change
an already launched run.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import shutil
import subprocess

from bole.results.storage import atomic_write
from bole import registry

NATIVE_PATHS = ("env", "env_cfg", "task", "utils", "scripts", "src",
                "XPolicyLab/client_server", "XPolicyLab/policy/demo_policy/deploy.yml", "XPolicyLab/LICENSE",
                "pyproject.toml", "LICENSE", ".openplatform-assets-revision")
IGNORED = {"__pycache__", ".git", ".venv", ".pytest_cache", "Assets"}
PROVENANCE_FILE = ".bole-source-provenance.json"


def source_provenance(source, declared_revision=None):
    """Read only a checkout's own Git metadata, never an enclosing repository.

    Copied source carries an explicit provenance record. A declared revision is
    retained as a declaration, and is never represented as an observed HEAD.
    """
    source = Path(source).resolve()
    inherited = source / PROVENANCE_FILE
    if inherited.is_file():
        value = json.loads(inherited.read_text())
        if value.get("schema_version") != 1:
            raise ValueError("unsupported frozen source provenance schema")
        return {**value, "read_from_snapshot": str(inherited)}
    result = {"schema_version": 1, "source": str(source), "declared_revision": declared_revision,
              "observed_git_revision": None, "git_changes": None,
              "verification": "not_a_standalone_git_checkout"}
    if not (source / ".git").exists():
        return result
    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], check=True, text=True,
                              capture_output=True, timeout=10).stdout.strip()
    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != source:
            result["verification"] = "git_root_differs_from_source"
            return result
        result.update(observed_git_revision=git("rev-parse", "HEAD"),
                      git_changes=git("status", "--short").splitlines(),
                      verification="observed_source_checkout_head_and_status")
    except (OSError, subprocess.SubprocessError) as exc:
        result["verification"] = "git_metadata_unavailable"
        result["error_type"] = type(exc).__name__
    return result


def _package_source(project_root=None):
    """Locate the running package for source, editable, and installed wheels."""
    if project_root is not None:
        root = Path(project_root).resolve()
        package = root / "src/bole"
        if not package.is_dir():
            raise ValueError("explicit project_root must contain src/bole")
        return package, root
    package = Path(__file__).resolve().parents[1]
    candidate = package.parent.parent
    source_tree = ((candidate / "pyproject.toml").is_file()
                   and (candidate / "src/bole").resolve() == package)
    return package, candidate if source_tree else None


def _copy(source, destination):
    """Dereference source code links so the copy never imports a live checkout."""
    source, destination = Path(source), Path(destination)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False,
                        ignore=shutil.ignore_patterns(*IGNORED, "*.pyc", "*.pyo"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def freeze(config, run_root, *, project_root=None, plugins=()):
    """Return (execution_config, manifest) after making an independent code copy.

    ``plugins`` contains the instantiated adapters selected for this run. Their
    top-level Python packages are copied alongside Bole; installed third-party
    dependencies and container images are explicitly retained as references.
    """
    package_root, source_root = _package_source(project_root)
    run_root = Path(run_root).resolve()
    destination = run_root / "snapshot"
    destination.mkdir(parents=True, exist_ok=False)
    frozen = copy.deepcopy(config)
    code_root = destination / "bole"
    _copy(package_root, code_root / "src/bole")
    if source_root and (source_root / "prompts").exists():
        _copy(source_root / "prompts", code_root / "prompts")
    for filename in ("pyproject.toml", "resources.yaml"):
        if source_root and (source_root / filename).is_file():
            _copy(source_root / filename, code_root / filename)
    prompt = Path(config["harness"]["instruction"]).resolve(strict=True)
    prompt_copy = code_root / "prompts" / "selected-instruction.md"
    _copy(prompt, prompt_copy)
    frozen["harness"]["instruction"] = str(prompt_copy)
    entries = [{"kind": "bole", "original": str(package_root), "snapshot": str(code_root / "src/bole"),
                "source_tree": str(source_root) if source_root else None,
                "import_mode": "source_tree" if source_root else "installed_package"},
               {"kind": "prompt", "original": str(prompt), "snapshot": str(prompt_copy)}]
    python_paths = [str(code_root / "src")]
    seccomp = config.get("resources", {}).get("codex_seccomp")
    if seccomp:
        policy = Path(seccomp).resolve(strict=True)
        policy_copy = destination / "configuration/codex-seccomp.json"
        _copy(policy, policy_copy)
        frozen["resources"]["codex_seccomp"] = str(policy_copy)
        entries.append({"kind": "codex_seccomp", "original": str(policy), "snapshot": str(policy_copy)})
    overrides = config.get("resources", {}).get("asset_overrides")
    if overrides:
        override_copy = destination / "configuration/asset-overrides"
        _copy(Path(overrides).resolve(strict=True), override_copy)
        frozen["resources"]["asset_overrides"] = str(override_copy)
        entries.append({"kind": "asset_overrides", "original": str(overrides), "snapshot": str(override_copy)})
    roles = config.get("recording", {}).get("safety", {}).get("roles")
    if roles:
        roles = Path(roles).resolve(strict=True)
        roles_copy = destination / "configuration" / "safety-roles.yaml"
        _copy(roles, roles_copy)
        frozen["recording"]["safety"]["roles"] = str(roles_copy)
        entries.append({"kind": "safety_roles", "original": str(roles), "snapshot": str(roles_copy)})
    if config["benchmark"]["name"] == "robodojo":
        source = Path(config["resources"]["source"]).resolve(strict=True)
        native_copy = destination / "benchmark"
        native_copy.mkdir()
        copied = []
        for relative in NATIVE_PATHS:
            path = source / relative
            if path.exists():
                _copy(path, native_copy / relative)
                copied.append(relative)
        frozen["resources"]["source"] = str(native_copy)
        provenance = source_provenance(source, config["resources"].get("source_revision"))
        atomic_write(native_copy / PROVENANCE_FILE, (json.dumps(provenance, indent=2) + "\n").encode())
        entries.append({"kind": "benchmark", "original": str(source),
                        "snapshot": str(native_copy), "included_paths": copied,
                        "source_revision": config["resources"].get("source_revision"),
                        "provenance": provenance})
    copied_packages = set()
    modules = [type(plugin).__module__ for plugin in plugins]
    # Entry-point factories can live in a different package from their returned
    # adapter. Freeze both, and freeze discovery so metadata changes cannot
    # redirect a running child to a new factory.
    selected_registry = {}
    for kind, section in (("benchmarks", "benchmark"), ("harnesses", "harness"), ("policies", "policy")):
        selected = config.get(section)
        name = selected.get("name") if isinstance(selected, dict) else selected
        found = registry.discover(kind)
        selected_registry[kind] = {name: found[name]} if name in found else {}
        if name in found:
            modules.append(found[name].split(":", 1)[0])
    atomic_write(code_root / "src/bole/_frozen_plugins.json",
                 (json.dumps(selected_registry, indent=2) + "\n").encode())
    for module_name in modules:
        package_name = module_name.split(".")[0]
        if package_name == "bole" or package_name in copied_packages:
            continue
        package = importlib.import_module(package_name)
        filename = getattr(package, "__file__", None)
        if not filename:
            raise ValueError(f"plugin {module_name} needs a file-backed package for code freezing")
        origin = Path(filename).resolve()
        package_source = origin.parent if hasattr(package, "__path__") else origin
        package_target = destination / "plugins" / (package_name if package_source.is_dir()
                                                     else package_name + ".py")
        _copy(package_source, package_target)
        copied_packages.add(package_name)
        entries.append({"kind": "plugin", "module": module_name,
                        "original": str(package_source), "snapshot": str(package_target)})
    if copied_packages:
        python_paths.insert(0, str(destination / "plugins"))
    manifest = {"schema_version": 1, "strategy": "copied_code_and_prompt_no_build_no_hash",
                "code": entries, "python_paths": python_paths,
                "external_resources": copy.deepcopy(config["resources"]),
                "external_dependencies": "Python environment and existing container images; no download or rebuild",
                "asset_copy": False, "learning_history_resumed": False}
    atomic_write(run_root / "snapshot.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return frozen, manifest


__all__ = ["freeze", "NATIVE_PATHS", "source_provenance", "PROVENANCE_FILE"]
