"""Episode artifact index for migrated Bole evidence and recoverable video.

Paths are relative to the episode directory. File existence is separate from
video decode verification and from whether an interrupted recording is complete.
"""
from __future__ import annotations

import json
from pathlib import Path


def _journal_status(path):
    """Validate the persisted prefix without loading a long telemetry stream."""
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("unterminated record")
                json.loads(line)
            except (ValueError, UnicodeDecodeError):
                return {"records": count, "complete": False, "invalid_line": count + 1}
            count += 1
    return {"records": count, "complete": True, "invalid_line": None}


def episode_video_records(records):
    """Translate VideoRecorder's video-directory references exactly once."""
    translated = []
    for source in records:
        record = dict(source)
        if record.get("path_base") != "episode":
            for key in ("path", "capture_ref", "encoded_frame_map"):
                if record.get(key):
                    path = Path(record[key])
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("video reference escapes its recording directory")
                    record[key] = str(Path("videos") / path)
        record["path_base"] = "episode"
        record["frame_reference_base"] = "videos"
        translated.append(record)
    return translated


def write_artifact_index(store, *, video=None, partial=False, collection_errors=(), video_enabled=None):
    """Inventory actual files, plus explicit missing required files/collections.

    Call again after runner-side usage, result, feedback and audit exports. This
    replaces only the index; it never generates evidence from an empty directory.
    """
    previous = {}
    target = store.root / "artifacts.json"
    if target.is_file():
        try:
            previous = json.loads(target.read_text())
            if not isinstance(previous, dict):
                previous = {}
        except (ValueError, OSError):
            pass
    videos = episode_video_records(previous.get("video", []) if video is None else video)
    partial = bool(partial or previous.get("partial"))
    errors = [*previous.get("collection_errors", []), *collection_errors]
    errors = list({json.dumps(item, sort_keys=True): item for item in errors}.values())
    known_videos = {record["path"]: record for record in videos if record.get("path")}
    records, present = [], set()
    for path in sorted(store.root.rglob("*")):
        relative = path.relative_to(store.root)
        name = str(relative)
        if not path.is_file() or path.is_symlink() or name == "artifacts.json":
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue  # Atomic-write temporaries are never completed artifacts.
        present.add(name)
        kind = ("video" if path.suffix == ".mp4" else "image" if path.suffix == ".png"
                else relative.parts[0] if len(relative.parts) > 1 else path.stem)
        item = {**store.identity, "schema_version": 1, "type": kind, "path": name,
                "path_base": "episode", "status": "saved", "partial": partial,
                "reason": "episode_or_collection_incomplete" if partial else None,
                "size_bytes": path.stat().st_size}
        if path.suffix == ".jsonl":
            try:
                journal = _journal_status(path)
                item.update(records=journal["records"], journal_complete=journal["complete"],
                            invalid_line=journal["invalid_line"])
                if not journal["complete"]:
                    item.update(status="partial", partial=True, reason="incomplete_or_invalid_jsonl_tail")
            except OSError as error:
                item.update(records=None, journal_complete=None, status="collection_failed", partial=True,
                            reason=f"journal_validation_failed: {type(error).__name__}: {error}")
        if path.suffix == ".mp4":
            if name in known_videos:
                item.update(known_videos[name])
            else:
                item.update(status="unknown", partial=True, playable=None,
                            reason="file_exists_but_decode_verification_unavailable")
        records.append(item)
    for video_record in videos:
        if not video_record.get("path"):
            records.append({**store.identity, "schema_version": 1, "type": "video", **video_record})
    for name, kind in (("result.json", "result"), ("usage.json", "usage"),
                       ("safety.json", "safety"), ("trajectory/control-steps.jsonl", "control_steps"),
                       ("native-evaluation/result.json", "native_result")):
        if name not in present:
            records.append({**store.identity, "schema_version": 1, "type": kind, "path": name,
                            "path_base": "episode", "status": "not_collected", "partial": True,
                            "reason": "file_not_recorded"})
    collections = []
    for name in ("trajectory", "telemetry", "observations", "videos", "model-events", "native-evaluation", "feedback", "logs"):
        count = sum(path.startswith(name + "/") for path in present)
        collections.append({**store.identity, "schema_version": 1, "type": "collection", "path": name,
            "path_base": "episode", "file_count": count,
            "status": "saved" if count else "not_collected", "partial": partial if count else True,
            "reason": "episode_or_collection_incomplete" if count and partial else None if count else "no_recorded_files"})
    manifest = {"path_base": "episode", "manifest_ref": "artifacts.json", "artifacts": records,
                "collections": collections, "video": videos, "partial": partial or bool(errors),
                "collection_errors": errors,
                "video_enabled": previous.get("video_enabled") if video_enabled is None else video_enabled}
    for key in ("recording_overhead_s",):
        if key in previous:
            manifest[key] = previous[key]
    store.write_json("artifacts.json", manifest)
    return manifest
