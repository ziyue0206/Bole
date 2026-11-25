"""Bole evidence writer with atomic files and append-only events.

Preserves atomic document replacement and append-only event recording, without
the old parallel evidence format or mandatory content hashes.
"""
from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

SCHEMA_VERSION = 1
_SECRET_KEYS = re.compile(r"^(authorization|proxy.authorization|api.?key|.*_api_key|access_token|refresh_token|id_token|password|secret|client_secret|cookie|set.cookie)$", re.I)
_BEARER = re.compile(r"(?i)\b(Bearer\s+)[^\s\"'<>]+")
_ASSIGNMENT = re.compile(r"(?i)(\b(?:[A-Z_]*API_KEY|access_token|refresh_token|password|client_secret)\s*[=:]\s*)([\"']?)([^\s,;}\"']+)([\"']?)")


def redact(value, secrets=()):
    """Remove credential fields, bearer headers, and caller-supplied secret values.

    Credential reference names (credential_env / credential_file) remain intact.
    Callers must pass actual secrets used by the run, including proxy passwords.
    """
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if _SECRET_KEYS.fullmatch(str(k)) else redact(v, secrets)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        for secret in sorted((s for s in secrets if isinstance(s, str) and s), key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        value = _BEARER.sub(r"\1[REDACTED]", value)
        return _ASSIGNMENT.sub(r"\1[REDACTED]", value)
    return value


def atomic_write(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_jsonl(path):
    """Read complete records; never claim a damaged tail was recorded fully."""
    path = Path(path)
    if not path.exists():
        return {"records": [], "complete": True, "invalid_line": None}
    records = []
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            if not line.endswith(b"\n"):
                raise ValueError("unterminated record")
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            return {"records": records, "complete": False, "invalid_line": index + 1}
    return {"records": records, "complete": True, "invalid_line": None}


class RecordStore:
    def __init__(self, root: Path, identity: dict, *, secrets=()):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.identity = dict(identity)
        self.secrets = tuple(secrets)
        self._lock = threading.RLock()
        self.clock_id = f"process-{os.getpid()}-{uuid4().hex}"
        target = self.path("trajectory/events.jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            try:
                current = read_jsonl(target)
                if not current["complete"]:
                    raise ValueError("event stream has an incomplete tail; preserve it and use a new recorder directory")
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def path(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative to the record directory")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError("artifact path escapes the record directory")
        return target

    def _record(self, data):
        if not isinstance(data, dict):
            data = {"value": data}
        for key, value in self.identity.items():
            if key in data and data[key] != value:
                raise ValueError(f"record identity mismatch: {key}")
        return redact({**data, **self.identity, "schema_version": SCHEMA_VERSION}, self.secrets)

    def write_json(self, relative, data):
        with self._lock:
            value = self._record(data)
            atomic_write(self.path(relative), (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())
        return str(relative)

    def append_jsonl(self, relative, data):
        with self._lock:
            target = self.path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            value = self._record(data)
            payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
            with target.open("ab") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        return str(relative)

    @staticmethod
    def _last_sequence(stream):
        """Read the tail while holding its inode lock, without scanning the run."""
        end = stream.seek(0, os.SEEK_END)
        if not end:
            return 0
        stream.seek(end - 1)
        if stream.read(1) != b"\n":
            raise ValueError("event stream has an incomplete tail; preserve it and use a new recorder directory")
        position, tail = end - 1, b""
        while position:
            width = min(position, 8192)
            position -= width
            stream.seek(position)
            tail = stream.read(width) + tail
            if b"\n" in tail:
                tail = tail.rsplit(b"\n", 1)[1]
                break
        last = json.loads(tail)
        sequence = last.get("sequence")
        if type(sequence) is not int or sequence < 1:
            raise ValueError("event stream tail has no valid sequence")
        return sequence

    def event(self, kind, **fields):
        if not isinstance(kind, str) or not kind:
            raise ValueError("event kind is required")
        with self._lock:
            event_id = fields.pop("event_id", None) or "event-" + uuid4().hex
            # Runner and simulator have separate processes and RecordStore
            # instances for the same episode. Sequence allocation and append
            # must be one transaction on the shared stream's inode. A cached
            # counter or a per-instance lock cannot provide this ordering.
            with self.path("trajectory/events.jsonl").open("a+b") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    value = self._record({
                        **fields, "kind": kind, "event_id": event_id,
                        "sequence": self._last_sequence(stream) + 1,
                        "source": fields.get("source", "bole"), "control_step": fields.get("control_step"),
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "monotonic_ns": time.monotonic_ns(), "clock_id": self.clock_id,
                    })
                    stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
            return event_id

    def asset(self, relative, data: bytes, *, text=False):
        """Store binary media; textual model/log records must use text=True or JSON methods."""
        if text:
            data = redact(data.decode("utf-8"), self.secrets).encode()
        with self._lock:
            atomic_write(self.path(relative), data)
        return str(relative)
