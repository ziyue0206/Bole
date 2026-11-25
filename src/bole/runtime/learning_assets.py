"""Explicit workspace/memory export; Docker commits never include bind mounts.

Image commit primitives preserve workspace and native state. Native
session audit is separate. No reflection notes or replay history are imported.
"""
from __future__ import annotations

import json
import contextlib
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path

from bole.results.storage import atomic_write, redact

MEMORY_DIRS = ("memories", "memories_v2")
_CREDENTIAL = re.compile(r"^(auth\.json|\.env(?:\..*)?|.*\.key)$", re.I)
_PRIVATE_PEM = re.compile(br"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
_SQLITE_METADATA_TABLES = frozenset({"_sqlx_migrations", "schema_migrations", "migrations", "consolidation_progress"})
_SQLITE_PROCESSING_TABLES = frozenset({"jobs"})


def _inspect_memory_database(path):
    """Inspect unknown native schemas without treating schema setup as memory.

    Row counts describe storage only. Native revisions use different tables;
    neither a nonempty database nor any one text column proves consolidation.
    Read-only mode deliberately retains WAL visibility (not immutable=1).
    """
    item = {"path": path.name, "bytes": path.stat().st_size, "state": "uninspected",
            "tables": [], "stage1_output_rows": None, "data_rows": None,
            "processing_rows": None}
    deadline = time.monotonic() + 2
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2) as db:
            db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            names = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            for (name,) in names:
                if name.startswith("sqlite_"):
                    continue
                quoted = '"' + name.replace('"', '""') + '"'
                columns = [row[1] for row in db.execute(f"PRAGMA table_info({quoted})")]
                count = db.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
                entry = {"name": name, "rows": count, "columns": columns,
                         "metadata_only": name in _SQLITE_METADATA_TABLES,
                         "processing_state_only": name in _SQLITE_PROCESSING_TABLES}
                # Return counts, never values, including for an unfamiliar schema.
                texts = {}
                for column in columns:
                    if column not in {"raw_memory", "rollout_summary", "memory", "summary", "content", "text"}:
                        continue
                    field = '"' + column.replace('"', '""') + '"'
                    texts[column] = db.execute(f"SELECT count(*) FROM {quoted} WHERE {field} IS NOT NULL AND length(trim(CAST({field} AS TEXT))) > 0").fetchone()[0]
                if texts:
                    entry["nonempty_text_rows"] = texts
                item["tables"].append(entry)
                if name == "stage1_outputs":
                    item["stage1_output_rows"] = count
            item["data_rows"] = sum(t["rows"] for t in item["tables"]
                                    if not t["metadata_only"] and not t["processing_state_only"])
            item["processing_rows"] = sum(t["rows"] for t in item["tables"] if t["processing_state_only"])
            item["state"] = ("rows_present_unverified" if item["data_rows"] else
                             "processing_state_only" if item["processing_rows"] else "schema_only")
    except sqlite3.Error as exc:
        item.update(state="inspection_error", inspection_error=str(exc))
    return item


def inspect_memory(home, *, enabled=None):
    """Read file metadata and SQLite counts only; never return memory text.

    `enabled` must come from the effective runtime configuration. Files alone
    cannot establish it, nor prove that a new session used generated memory.
    """
    home = Path(home)
    if not home.is_dir():
        raise FileNotFoundError("Codex home is not an existing directory: " + str(home))
    files, databases = [], []
    for name in MEMORY_DIRS:
        directory = home / name
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink():
                files.append({"path": path.relative_to(home).as_posix(), "bytes": path.stat().st_size})
    for path in sorted(home.glob("memories*.sqlite")):
        if path.is_symlink() or not path.is_file():
            continue
        databases.append(_inspect_memory_database(path))
    evidence = [x["path"] for x in files if x["bytes"] and
                (x["path"].endswith("/memory_summary.md") or x["path"].endswith("/MEMORY.md")
                 or "/rollout_summaries/" in x["path"])]
    if evidence:
        state = "generated_files_present_unverified"
    elif any(d["state"] == "rows_present_unverified" for d in databases):
        state = "database_rows_present_unverified"
    elif any(d["state"] == "inspection_error" for d in databases):
        state = "inspection_incomplete"
    elif files:
        state = "files_present_unverified"
    elif any(d["state"] == "processing_state_only" for d in databases):
        state = "processing_state_only"
    elif databases:
        state = "schema_only"
    else:
        state = "empty"
    return {"schema_version": 1, "enabled": enabled, "state": state,
            "files": files, "databases": databases, "generation_evidence_files": evidence,
            "generation_verified": False, "new_session_reuse_verified": False,
            "content_disclosed": False,
            "interpretation": "row_counts_and_files_are_storage_evidence_only; require_native_provenance_and_new_session_verification"}


def _check_file(path, secrets, *, root=None):
    if path.is_symlink():
        link = Path(os.readlink(path))
        if (root is None or link.is_absolute() or
                not path.resolve(strict=True).is_relative_to(Path(root).resolve())):
            raise ValueError(f"learning asset contains an escaping symlink: {path.name}")
    if _CREDENTIAL.fullmatch(path.name):
        raise ValueError(f"credential-like file cannot enter a learning asset: {path.name}")
    patterns = [s.encode() for s in secrets if s]
    pem = path.suffix.lower() == ".pem"
    if (not patterns and not pem) or not path.is_file():
        return
    keep = max([len(p) for p in patterns] + [256 if pem else 1]) - 1
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            data = tail + chunk
            if pem and _PRIVATE_PEM.search(data):
                raise ValueError(f"private key cannot enter a learning asset: {path.name}")
            if any(p in data for p in patterns):
                raise ValueError(f"credential value detected in learning asset: {path.name}")
            tail = data[-keep:] if keep else b""


def archive_tree(source, target, *, secrets=()):
    source, target = Path(source), Path(target)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("asset source must be an existing real directory")
    temporary = target.with_suffix(target.suffix + ".partial")
    files = []
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for path in sorted(source.rglob("*")):
                _check_file(path, secrets, root=source)
                if not path.is_file() and not path.is_dir():
                    raise ValueError(f"unsupported asset file: {path.name}")
                name = path.relative_to(source).as_posix()
                # Preserve validated relative links; avoid implicit tar hard
                # links, whose extraction has a separate attack surface.
                archive.inodes.clear()
                archive.add(path, arcname=name, recursive=False)
                if path.is_file():
                    files.append(name)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return files


def extract_tree(source, target):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        names = set()
        for item in members:
            p = Path(item.name)
            if p.is_absolute() or ".." in p.parts or item.name in names:
                raise ValueError("unsafe or duplicate path in learning asset")
            names.add(item.name)
            destination = (target / p).resolve()
            if not destination.is_relative_to(target) or not (item.isdir() or item.isfile() or item.issym()):
                raise ValueError("learning assets require local files, directories or relative links")
            if item.issym():
                link = Path(item.linkname)
                if link.is_absolute() or not (destination.parent / link).resolve().is_relative_to(target):
                    raise ValueError("symlink escapes learning asset")
        links = {item.name for item in members if item.issym()}
        if any(any(parent.as_posix() in links for parent in Path(item.name).parents) for item in members):
            raise ValueError("archive member must not traverse a symlink")
        for item in members:
            if item.issym():
                continue
            p = target / item.name
            if p.is_symlink() or any(parent.is_symlink() for parent in p.parents if parent != target):
                raise ValueError("destination contains a symlink")
            if item.isdir():
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as src, p.open("wb") as out:
                    shutil.copyfileobj(src, out)
                p.chmod(item.mode & 0o777)
        for item in members:
            if item.issym():
                p = target / item.name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.symlink_to(item.linkname)
        for name in links:
            if not (target / name).resolve(strict=True).is_relative_to(target):
                raise ValueError("symlink target escapes learning asset")


def snapshot_memory(home, destination, *, secrets=()):
    """Call only while the owned harness is frozen. SQLite backup includes WAL.

    state_*.sqlite and sessions are intentionally excluded: the new native
    thread is created through thread/start, with memory data, not prior history.
    """
    home, destination = Path(home), Path(destination)
    if not home.is_dir() or home.is_symlink():
        raise ValueError("native home must be an existing real directory")
    destination.mkdir(parents=True, exist_ok=False)
    for name in MEMORY_DIRS:
        source = home / name
        if source.exists() or source.is_symlink():
            _check_file(source, secrets)
            if not source.is_dir():
                raise ValueError("native memory component must be a directory")
            for p in source.rglob("*"):
                _check_file(p, secrets)
            shutil.copytree(source, destination / name, symlinks=False)
    databases = []
    for source in sorted(home.glob("memories*.sqlite")):
        _check_file(source, secrets)
        target = destination / source.name
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=2) as src:
            with sqlite3.connect(target) as dst:
                deadline = time.monotonic() + 10

                def progress(_status, _remaining, _total):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("memory database snapshot remained busy while the container was frozen")

                src.backup(dst, pages=128, progress=progress, sleep=0.05)
                if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("memory database snapshot failed integrity check")
        _check_file(target, secrets)
        databases.append(source.name)
    files = [p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()]
    generated = [name for name in files if
                 (name.endswith("memory_summary.md") or name.endswith("MEMORY.md")
                  or "/rollout_summaries/" in name) and (destination / name).stat().st_size > 0]
    return {"directories": [name for name in MEMORY_DIRS if (destination / name).exists()],
            "databases": databases, "files": files, "generation_evidence_files": generated,
            "state": "present_unverified" if generated else "not_generated",
            "storage_inspection": inspect_memory(destination),
            "generation_verified": False, "new_session_reuse_verified": False,
            "verification": "new_session_reuse_not_yet_verified",
            "generation_trigger": "native_asynchronous_startup_pipeline; no_forced_summary_or_compaction",
            "excludes": ["sessions", "archived_sessions", "history.jsonl", "auth.json", "state_*.sqlite"],
            "export_supported": True}


def _component_path(manifest_path, component):
    path = Path(component["path"])
    root = Path(manifest_path).resolve().parent
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("component path must remain inside the learning asset")
    result = (root / path).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("invalid learning asset component")
    return result


def restore_components(manifest_path, manifest, workspace, home):
    """Restore saved bytes, without inspecting memory generation or usefulness."""
    parts = manifest["components"]
    for directory in (Path(workspace), Path(home)):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("learning assets require fresh isolated destination directories")
    extract_tree(_component_path(manifest_path, parts["workspace"]), workspace)
    state = parts.get("home", parts.get("memory"))  # Existing manifest assets remain readable.
    Path(home).mkdir(parents=True, exist_ok=True)
    if state:
        extract_tree(_component_path(manifest_path, state), home)
    forbidden = [str(p.relative_to(home)) for p in Path(home).rglob("*") if _excluded_home(p.relative_to(home))]
    if forbidden:
        raise ValueError("saved home contains conversation history or credentials: " + ", ".join(forbidden))


def load_asset(manifest_path, workspace, home, *, verification_only=False):
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("export_status") != "saved":
        raise ValueError("learning asset was not successfully saved")
    if manifest.get("status") not in {"complete", "partial"}:
        raise ValueError("learning asset completeness is not declared")
    image = manifest["components"]["image"]
    if image.get("import") != "local_docker_image" or not re.fullmatch(r"sha256:[0-9a-f]{64}", image.get("image_id", "")):
        raise ValueError("learning asset requires an immutable local Docker image ID")
    restore_components(manifest_path, manifest, workspace, home)
    return manifest


_HISTORY = {"sessions", "archived_sessions", "history.jsonl", "logs", "log", "shell_snapshots",
            "tmp", ".tmp", "thread-writer-locks"}


def _excluded_home(relative):
    return (relative.parts[0] in _HISTORY or any(_CREDENTIAL.fullmatch(p) for p in relative.parts)
            or relative.name.startswith(("logs_", "queue_"))
            or relative.name.endswith(("-wal", "-shm", ".lock", ".sock", ".pid")))


def snapshot_home(home, destination, *, secrets=()):
    """Preserve arbitrary persistent files, excluding native history/auth/runtime files.

    Native indexes are kept; first-user-message text is removed from the thread
    index. SQLite backups include WAL without changing the original database.
    No artifact is interpreted as proof of learning or memory generation.
    """
    home, destination = Path(home), Path(destination)
    if not home.is_dir() or home.is_symlink():
        raise ValueError("native home must be an existing real directory")
    destination.mkdir(parents=True, exist_ok=False)
    excluded = []
    for path in sorted(home.rglob("*")):
        relative = path.relative_to(home)
        if _excluded_home(relative):
            excluded.append(relative.as_posix())
            continue
        _check_file(path, secrets, root=home)
        target = destination / relative
        if path.is_symlink():
            # A valid link must not reintroduce an excluded native history tree.
            resolved = path.resolve(strict=True).relative_to(home.resolve())
            if _excluded_home(resolved):
                excluded.append(relative.as_posix())
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(path))
        elif path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as stream:
                sqlite = stream.read(16) == b"SQLite format 3\x00"
            if sqlite:
                with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as src:
                    with contextlib.closing(sqlite3.connect(target)) as dst:
                        deadline = time.monotonic() + 10
                        def progress(*_):
                            if time.monotonic() > deadline:
                                raise TimeoutError("native state snapshot remained busy")
                        src.backup(dst, pages=128, progress=progress, sleep=.05)
                        if re.fullmatch(r"state_\d+\.sqlite", path.name):
                            tables = {r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                            if "threads" in tables:
                                columns = {r[1] for r in dst.execute('PRAGMA table_info("threads")')}
                                if "first_user_message" in columns:
                                    dst.execute("UPDATE threads SET first_user_message=''")
                        dst.commit()
                        dst.execute("PRAGMA journal_mode=DELETE")
                        dst.execute("VACUUM")  # Do not retain deleted text in free pages.
                _check_file(target, secrets)
            else:
                shutil.copy2(path, target)
    return {"files": [p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()],
            "excluded": excluded, "history_resumed": False, "memory_generation": "not_checked",
            "content_scope": "persistent_native_home_without_raw_history_or_credentials"}


async def export_asset(destination, *, container, workspace, home, metadata, command, secrets=(),
                       base_image=None, image_name=None, state_quiescent=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise FileExistsError("existing learning assets must never be overwritten")
    manifest = {"schema_version": 1, **metadata, "export_status": "pending", "components": {}}
    manifest.setdefault("status", "partial")
    frozen, quiescent = False, state_quiescent
    errors = []
    try:
        if not quiescent:
            try:
                await command("docker", "pause", container)
                frozen = quiescent = True
            except Exception as exc:
                manifest["pause_error"] = str(exc)
                try:
                    state = json.loads(await command("docker", "inspect", container))[0]["State"]
                    quiescent = state.get("Running") is False or state.get("Paused") is True
                except Exception as inspect_error:
                    manifest["container_state_error"] = str(inspect_error)
        manifest["consistency"] = "quiescent_snapshot" if quiescent else "best_effort_unverified"
        if not quiescent:
            manifest["status"] = "partial"
            errors.append({"component": "consistency", "message": "container quiescence could not be established"})
        # Salvage each readable component independently, even after container
        # death; failure of one component must not suppress the others.
        try:
            files = archive_tree(workspace, destination / "workspace.tar.gz", secrets=secrets)
            manifest["components"]["workspace"] = {"path": "workspace.tar.gz", "exists": True, "files": files}
        except Exception as exc:
            errors.append({"component": "workspace", "message": str(exc)})
        try:
            with tempfile.TemporaryDirectory(prefix="bole-state-snapshot-") as temp:
                snapshot = Path(temp) / "state"
                report = snapshot_home(home, snapshot, secrets=secrets)
                archive_tree(snapshot, destination / "home.tar.gz", secrets=secrets)
            manifest["components"]["home"] = {"path": "home.tar.gz", "exists": True, **report}
        except Exception as exc:
            errors.append({"component": "home", "message": str(exc)})
        if quiescent and not errors:
            try:
                from bole.runtime.image_assets import publish_image
                if not base_image or not image_name:
                    raise ValueError("self-contained export requires the selected base image and output image name")
                with tempfile.TemporaryDirectory(prefix="bole-image-seed-", dir=destination) as temp:
                    seed = Path(temp) / "seed"
                    seed.mkdir()
                    for filename in ("workspace.tar.gz", "home.tar.gz"):
                        os.link(destination / filename, seed / filename)
                    atomic_write(seed / "manifest.json", (json.dumps(redact({
                        **manifest, "export_status": "saved", "history_resumed": False,
                        "memory_generation": "not_checked"}, secrets), indent=2) + "\n").encode())
                    manifest["components"]["image"] = await publish_image(
                        seed, base_image, image_name, command, secrets=secrets)
            except Exception as exc:
                errors.append({"component": "image", "message": str(exc)})
        manifest["export_status"] = "failed" if errors else "saved"
        manifest["memory_generation"] = "not_checked"
        manifest["learning_effect"] = "not_evaluated"
        manifest["history_resumed"] = False
        manifest["status_meanings"] = {"export_status": "physical_snapshot_and_image_reference_only",
                                       "status": "learning_chain_completion_only"}
        if errors:
            manifest["errors"] = errors
    except Exception as exc:
        manifest["export_status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if frozen:
            try:
                await command("docker", "unpause", container)
            except Exception as exc:
                manifest["unpause_error"] = str(exc)
        atomic_write(destination / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read native memory metadata only; never start Codex or print memory text.")
    parser.add_argument("home", type=Path, help="Existing private CODEX_HOME for the run")
    parser.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None,
                        help="Effective runtime feature value, if known; files cannot establish it")
    args = parser.parse_args()
    try:
        result = inspect_memory(args.home, enabled=args.enabled)
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
