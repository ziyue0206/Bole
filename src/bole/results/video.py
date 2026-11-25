"""Recoverable simulator-time video recording and encoding.

Keeps atomic frame files and CPU finalization. The old head-only default, 3000
slot limit, and automatic thinning are deliberately removed. This module never
steps or renders the simulator: the adapter supplies PNGs or a no-physics render
callback at each due control boundary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

from .storage import RecordStore, atomic_write, read_jsonl

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def png_size(data):
    if not isinstance(data, bytes) or data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24 or data[12:16] != b"IHDR":
        raise ValueError("camera frame must be encoded PNG bytes")
    width, height = struct.unpack(">II", data[16:24])
    if not width or not height:
        raise ValueError("empty PNG frame")
    return [width, height]


class VideoRecorder:
    def __init__(self, root, *, cameras=CAMERAS, fps=5, identity=None):
        self.root = Path(root).resolve()
        self.cameras = tuple(cameras)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in self.cameras):
            raise ValueError("video camera names must be unique nonempty safe identifiers")
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("video fps must be positive and finite")
        if (self.root / "meta.json").exists():
            raise FileExistsError("video capture already exists; use recover() for CPU finalization")
        self.fps = float(fps)
        self.store = RecordStore(self.root, identity or {})
        self.last_slot = -1
        self.last_time = -1.0
        self.rows = []
        self.sizes = {}
        self.journal_complete = True
        self.collection_error = None
        self._recovered = False
        self.store.write_json("meta.json", {"cameras": self.cameras, "fps": self.fps,
            "sampling_clock": "sim_time_s", "automatic_thinning": False,
            "identity": identity or {}, "format": "png", "schema_version": 1})

    @classmethod
    def recover(cls, root):
        """Re-encode a crashed/interrupted capture solely from persisted PNGs."""
        root = Path(root).resolve()
        meta = json.loads((root / "meta.json").read_text())
        self = cls.__new__(cls)
        self.root, self.cameras, self.fps = root, tuple(meta["cameras"]), float(meta["fps"])
        self.store = RecordStore(root, meta.get("identity", {}))
        recovered = read_jsonl(root / "capture.jsonl")
        self.rows = recovered["records"]
        self.journal_complete = recovered["complete"]
        self.collection_error = None
        errors = read_jsonl(root / "collection-errors.jsonl")
        if errors["records"]:
            self.collection_error = errors["records"][-1].get("error", "capture_failed")
        if not errors["complete"]:
            self.collection_error = "capture_error_journal_has_incomplete_tail"
        self.last_slot = max((r["slot"] for r in self.rows), default=-1)
        self.last_time = max((r["sim_time_s"] for r in self.rows), default=-1)
        self.sizes = {r["camera"]: r["size"] for r in self.rows if r.get("status") == "captured"}
        self._recovered = True
        return self

    def mark_collection_error(self, error, *, control_step=None, sim_time_s=None):
        """Preserve an adapter-side clock/capture failure without raising.

        An unavailable clock is None, never a fabricated zero. Since the
        missing time range cannot be reconstructed, stop capture and retain
        existing frames for a partial video. This never changes task grading.
        """
        message = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        self.collection_error = message
        row = {"error": message, "control_step": control_step,
            "sim_time_s": sim_time_s, "last_captured_sim_time_s": self.last_time if self.last_time >= 0 else None,
            "coverage_after_error": "unknown", "capture_stopped": True}
        try:
            self.store.append_jsonl("collection-errors.jsonl", row)
        except Exception as exc:
            self.journal_complete = False
            row["recording_error"] = f"{type(exc).__name__}: {exc}"
        return row

    def wants(self, sim_time_s):
        if type(sim_time_s) not in (int, float) or not math.isfinite(sim_time_s) or sim_time_s < 0:
            raise ValueError("simulation time must be finite and nonnegative")
        return self.collection_error is None and math.floor(sim_time_s * self.fps + 1e-9) > self.last_slot

    def offer(self, control_step, sim_time_s, frames):
        """Record one current frame per camera on a due slot.

        If multiple slots were skipped, they are recorded as missing. A callback
        is invoked only once and must not advance physics. A capture failure is
        separate from action execution and does not stop the environment.
        """
        if self._recovered:
            raise RuntimeError("a recovered recorder is for finalization, not continuation of an unknown physical episode")
        if type(control_step) is not int or control_step < 0:
            raise ValueError("control step must be a nonnegative integer")
        if sim_time_s < self.last_time:
            raise ValueError("simulation time moved backwards")
        if not self.wants(sim_time_s):
            return []
        start = time.monotonic()
        slot = math.floor(sim_time_s * self.fps + 1e-9)
        additions = []
        capture_error = None
        try:
            frames = frames() if callable(frames) else frames
            if not isinstance(frames, dict):
                raise ValueError("frame supplier did not return a camera mapping")
        except Exception as exc:
            frames = {}
            capture_error = f"{type(exc).__name__}: {exc}"
        for index in range(self.last_slot + 1, slot + 1):
            for camera in self.cameras:
                row = {"camera": camera, "slot": index, "scheduled_sim_time_s": index / self.fps,
                    "sim_time_s": float(sim_time_s), "control_step": control_step,
                    "status": "missing", "path": None, "size": None,
                    "reason": "sampling_boundary_not_observed" if index < slot else capture_error or "camera_frame_not_supplied"}
                if index == slot and camera in frames:
                    try:
                        size = png_size(frames[camera])
                        if camera in self.sizes and self.sizes[camera] != size:
                            raise ValueError("camera resolution changed during the episode")
                        path = f"{camera}/frames/{index:09d}.png"
                        self.store.asset(path, frames[camera])
                        self.sizes[camera] = size
                        row.update(status="captured", path=path, size=size, reason=None)
                    except Exception as exc:
                        row["reason"] = f"{type(exc).__name__}: {exc}"
                try:
                    self.store.append_jsonl("capture.jsonl", row)
                except OSError as exc:
                    self.collection_error = f"{type(exc).__name__}: {exc}"
                    self.journal_complete = False
                    row.update(status="missing", reason="capture_journal_write_failed", path=None)
                additions.append(row)
        self.rows.extend(additions)
        self.last_slot, self.last_time = slot, float(sim_time_s)
        try:
            self.store.append_jsonl("capture-timing.jsonl", {"control_step": control_step,
                "sim_time_s": sim_time_s, "capture_and_write_seconds": time.monotonic() - start})
        except OSError as exc:
            self.collection_error = f"{type(exc).__name__}: {exc}"
            self.journal_complete = False
        return additions

    def _encode(self, camera, *, partial, timeout, ffmpeg, ffprobe):
        rows = [r for r in self.rows if r["camera"] == camera]
        captured = [r for r in rows if r["status"] == "captured" and (self.root / r["path"]).is_file()]
        result = {"schema_version": 1, "type": "video", "camera": camera,
            "path": None, "fps": self.fps, "sampling_clock": "sim_time_s",
            "expected_samples": len(rows), "captured_frames": len(captured),
            "encoded_frames": None, "repeated_frames": None, "duration_seconds": None,
            "size": self.sizes.get(camera), "status": "missing", "playable": False,
            "partial": partial or self.collection_error is not None or not self.journal_complete or len(captured) != len(rows),
            "capture_journal_complete": self.journal_complete,
            "collection_error": self.collection_error,
            "capture_ref": "capture.jsonl", "encoded_frame_map": None,
            "missing_samples": len(rows) - len(captured), "reason": None,
            "coverage_control_steps": [captured[0]["control_step"], captured[-1]["control_step"]] if captured else None,
            "coverage_sim_time_s": [captured[0]["sim_time_s"], captured[-1]["sim_time_s"]] if captured else None}
        if not captured:
            result["reason"] = "no_captured_frames"
            return result
        if len(captured) == 1:
            result.update(reason="only_one_frame_not_a_running_episode_video", status="partial", partial=True)
            return result
        if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
            result.update(status="collection_failed", reason="ffmpeg_or_ffprobe_unavailable", partial=True)
            return result
        by_slot = {r["slot"]: r for r in captured}
        first, last = min(by_slot), max((r["slot"] for r in rows), default=max(by_slot))
        mapping, lines = [], ["ffconcat version 1.0"]
        current = None
        for index in range(first, last + 1):
            current = by_slot.get(index, current)
            if current is None:
                continue
            rel = current["path"]
            mapping.append({"schema_version": 1, "camera": camera, "encoded_index": len(mapping),
                "video_time_s": len(mapping) / self.fps, "scheduled_sim_time_s": index / self.fps,
                "captured_sim_time_s": current["sim_time_s"], "control_step": current["control_step"],
                "source_frame": rel, "new_capture": index in by_slot})
            lines.extend([f"file '{rel}'", f"duration {1 / self.fps:.12f}"])
        lines.append(f"file '{mapping[-1]['source_frame']}'")
        concat = self.root / f".{camera}.ffconcat"
        atomic_write(concat, ("\n".join(lines) + "\n").encode())
        map_path = f"{camera}/encoded-frames.jsonl"
        atomic_write(self.root / map_path, b"".join((json.dumps({**r, **self.store.identity}) + "\n").encode() for r in mapping))
        target = self.root / f"{camera}.mp4"
        temporary = self.root / f".{camera}.partial.mp4"
        started = time.monotonic()
        try:
            subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", "-f", "concat", "-safe", "1",
                "-i", str(concat), "-r", str(self.fps), "-frames:v", str(len(mapping)),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(temporary)], check=True, capture_output=True, timeout=timeout)
            probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=width,height,nb_read_frames,duration", "-of", "json", str(temporary)],
                check=True, capture_output=True, timeout=min(timeout, 60))
            stream = json.loads(probe.stdout)["streams"][0]
            count = int(stream["nb_read_frames"])
            if count != len(mapping) or count < 2:
                raise ValueError(f"encoded frame count {count} differs from frame map {len(mapping)}")
            # Decode the whole short or long stream, not only its container header.
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(temporary), "-f", "null", "-"],
                check=True, capture_output=True, timeout=timeout)
            temporary.replace(target)
            result.update(path=target.name, encoded_frames=count, repeated_frames=sum(not r["new_capture"] for r in mapping),
                encoded_frame_map=map_path, duration_seconds=float(stream.get("duration") or count / self.fps),
                encoded_size=[stream["width"], stream["height"]],
                encoding_transform="pad_right_bottom_to_even_size_if_needed", playable=True,
                status="partial" if result["partial"] else "saved")
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
            result.update(status="collection_failed", partial=True, reason=f"{type(exc).__name__}: {exc}")
        result["encode_and_verify_seconds"] = time.monotonic() - started
        return result

    def finalize(self, *, partial=False, timeout=300, ffmpeg="ffmpeg", ffprobe="ffprobe"):
        records = []
        for camera in self.cameras:
            try:
                record = self._encode(camera, partial=partial or self._recovered,
                    timeout=timeout, ffmpeg=ffmpeg, ffprobe=ffprobe)
            except OSError as exc:
                record = {"schema_version": 1, "type": "video", "camera": camera,
                    "status": "collection_failed", "path": None, "partial": True,
                    "playable": False, "reason": f"{type(exc).__name__}: {exc}"}
            records.append(record)
        try:
            self.store.write_json("artifacts.json", {"artifacts": records,
                "automatic_thinning": False, "frames_retained_for_reencoding": True})
        except OSError as exc:
            for record in records:
                record["manifest_write_error"] = f"{type(exc).__name__}: {exc}"
        return records
