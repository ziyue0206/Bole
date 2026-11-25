"""Executed by the existing image's Python before the native app-server."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import venv


def prepare_workspace(workspace):
    workspace = Path(workspace)
    environment = workspace / ".venv"
    if environment.is_symlink():
        raise RuntimeError("workspace .venv must be a real directory")
    if not environment.exists():
        # Copies keep exported workspaces free of links to image-only binaries.
        venv.EnvBuilder(with_pip=True, symlinks=False).create(environment)
    if not (environment / "pyvenv.cfg").is_file() or not (environment / "bin/python").is_file():
        raise RuntimeError("existing workspace .venv is incomplete; refusing to overwrite it")
    # The native sandbox needs an existing target for this narrow write grant.
    # Create only the directory; existing autonomous skills are left intact.
    (workspace / ".agents/skills").mkdir(parents=True, exist_ok=True)
    os.environ.update(VIRTUAL_ENV=str(environment),
                      PATH=str(environment / "bin") + ":" + os.environ.get("PATH", "/usr/bin:/bin"),
                      PIP_REQUIRE_VIRTUALENV="true", PIP_CACHE_DIR=str(workspace / ".cache/pip"))


if __name__ == "__main__":
    prepare_workspace("/workspace")
    os.execv(sys.argv[1], sys.argv[1:])
