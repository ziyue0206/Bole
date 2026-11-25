"""Local Docker packaging for saved state; no build, model call or memory gate."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
import re
import tempfile
from uuid import uuid4

SEED_PATH = "/opt/bole-learning"
ASSET_LABEL = "io.bole.learning-asset"


def validate_image_name(name):
    if (not isinstance(name, str) or len(name) > 384 or not name
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", name)
            or name.startswith("sha256:") or "//" in name or name.endswith("/")):
        raise ValueError("save_image must be a local Docker repository/tag, not a path or digest")
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    repository, tag = name.rsplit(":", 1)
    if (repository != repository.lower() or not repository
            or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag)):
        raise ValueError("invalid Docker repository/tag for save_image")
    return name


async def require_unused_name(name, command):
    name = validate_image_name(name)
    found = await command("docker", "image", "ls", "--quiet", "--filter", "reference=" + name)
    if found.strip():
        raise FileExistsError("learning image already exists; choose a new --save-image: " + name)
    return name


def output_image_name(config, run_id, task_id, *, partial=False):
    selected = config.get("learn", {}).get("save_image")
    name = selected.replace("{task}", task_id) if selected else f"bole/codex-learn:{run_id}-{task_id}"
    name = validate_image_name(name)
    # A partial run never takes the name requested for a completed run.
    if partial:
        repository, tag = name.rsplit(":", 1)
        name = repository + ":" + tag[:100] + "-partial-" + uuid4().hex[:8]
    return validate_image_name(name)


async def publish_image(seed, base_image, name, command, *, secrets=()):
    """Add only a saved-state layer to an existing image; never start a process."""
    name = await require_unused_name(name, command)
    helper = "bole-pack-" + uuid4().hex
    try:
        await command("docker", "create", "--pull=never", "--network=none", "--name", helper,
                      "--entrypoint", "/bin/true", base_image)
        await command("docker", "cp", str(seed), helper + ":" + SEED_PATH)
        image_id = (await command("docker", "commit", "--change", "LABEL " + ASSET_LABEL + "=1", helper)).strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise RuntimeError("Docker did not return an immutable saved image ID")
        inspected = await command("docker", "image", "inspect", image_id)
        if any(secret and secret in inspected for secret in secrets):
            raise RuntimeError("credential detected in saved image metadata")
        await require_unused_name(name, command)
        await command("docker", "tag", image_id, name)
        return {"image_id": image_id, "name": name, "import": "local_docker_image",
                "exists": True, "self_contained": True, "seed_path": SEED_PATH,
                "external_reference": True, "archive": None,
                "archive_reason": "saved_in_local_docker_store"}
    finally:
        with contextlib.suppress(OSError, RuntimeError):
            await command("docker", "rm", "-f", "-v", helper)


async def restore_image(image_id, inspected, workspace, home, command):
    """Copy stored bytes to fresh host mounts; inspect structure, not learning."""
    labels = (inspected.get("Config") or {}).get("Labels") or {}
    if ASSET_LABEL not in labels:
        return None  # A normal base image starts with fresh empty mounts.
    if labels[ASSET_LABEL] != "1":
        raise ValueError("unsupported Bole image asset format")
    helper = "bole-restore-" + uuid4().hex
    from bole.runtime.learning_assets import restore_components
    try:
        await command("docker", "create", "--pull=never", "--network=none", "--name", helper,
                      "--entrypoint", "/bin/true", image_id)
        with tempfile.TemporaryDirectory(prefix="bole-image-state-") as temp:
            seed = Path(temp) / "seed"
            await command("docker", "cp", helper + ":" + SEED_PATH, str(seed))
            manifest = json.loads((seed / "manifest.json").read_text())
            if manifest.get("schema_version") != 1:
                raise ValueError("unsupported Bole image state format")
            restore_components(seed / "manifest.json", manifest, workspace, home)
        return {"selection_mode": "explicit_harness_image", "image_id": image_id,
                "source_task_id": manifest.get("source_task_id"),
                "source_run_id": manifest.get("source_run_id"), "status": manifest.get("status"),
                "restore": {"workspace": True, "native_state": True, "resume_history": False}}
    finally:
        with contextlib.suppress(OSError, RuntimeError):
            await command("docker", "rm", "-f", "-v", helper)
