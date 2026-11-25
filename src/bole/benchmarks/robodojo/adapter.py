"""RoboDojo adapter with an independent native worker.

The simulator and native evaluator stay in the selected A image/source pair.
The harness gets public observations and execution receipts over Bole tools;
it never receives this socket, simulator assets or native result mounts.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from uuid import uuid4

from bole.results.artifacts import write_artifact_index
from bole.results.storage import read_jsonl
from bole.results.video import VideoRecorder
from bole.runtime.dependencies import asset_override_mounts, prepare_robot_overrides
from .catalog import Catalog

ROOT = "/workspace/RoboDojo"
JOINTS = {"left_arm_joint_state": 6, "left_ee_joint_state": 1,
          "right_arm_joint_state": 6, "right_ee_joint_state": 1}
SOURCE_DIRECTORIES = ("env", "env_cfg", "src", "task", "utils", "XPolicyLab")


def action_schema():
    return {"type": "array", "minItems": 1, "items": {
        "type": "object", "required": list(JOINTS), "additionalProperties": False,
        "properties": {name: {"type": "array", "minItems": width, "maxItems": width,
            "items": {"type": "number", **({"minimum": 0, "maximum": 1} if width == 1 else {})}}
            for name, width in JOINTS.items()}}}


def validate_actions(actions):
    if not isinstance(actions, list) or not actions:
        raise ValueError("ARX X5 actions must be a nonempty list")
    for action in actions:
        if not isinstance(action, dict) or set(action) != set(JOINTS):
            raise ValueError("each action requires both six-joint arms and both grippers")
        for name, width in JOINTS.items():
            values = action[name]
            if (not isinstance(values, list) or len(values) != width or
                any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.4e38
                    for v in values)):
                raise ValueError(f"{name} requires {width} finite float32 values")
            if width == 1 and not 0 <= values[0] <= 1:
                raise ValueError("gripper opening fraction must be in [0,1]")


def command(args, timeout=30):
    return subprocess.run([str(v) for v in args], check=True, capture_output=True,
                          text=True, timeout=timeout).stdout


class RoboDojoAdapter:
    requires_gpu = True

    def plan(self, config):
        from bole.engine.planning import plan
        return plan(config, Catalog(Path(config["resources"]["source"]),
                                    Path(config["resources"]["assets"])))

    def summary(self, plan, results):
        from .scoring import summarize_native
        return summarize_native(plan, results)

    def check(self, config):
        resources = config["resources"]
        issues, evidence = [], {}
        source, assets = Path(resources["source"]), Path(resources["assets"])
        try:
            plan = self.plan(config)
            evidence["planned_episodes"] = len(plan["episodes"])
            evidence["coverage"] = plan["coverage"]
        except (OSError, ValueError) as error:
            issues.append({"component": "catalog", "message": str(error)})
        if not (source / "XPolicyLab/client_server/ws/model_client.py").exists():
            issues.append({"component": "native_dependency", "message":
                "A requires its existing XPolicyLab/client_server package; no clone is attempted"})
        marker = source / ".openplatform-assets-revision"
        evidence["asset_revision"] = marker.read_text().strip() if marker.exists() else None
        evidence["assets"] = {"path": str(assets), "exists": assets.is_dir(), "mount": "read-only"}
        if not assets.is_dir():
            issues.append({"component": "assets", "message": f"asset directory not found: {assets}"})
        if resources.get("asset_overrides"):
            try:
                evidence["asset_overrides"] = asset_override_mounts(
                    resources["asset_overrides"], resources.get("simulator_root", ROOT))
            except (OSError, ValueError) as error:
                issues.append({"component": "asset_overrides", "message": str(error)})
        for key in ("cache_source", "extensions_source"):
            if resources.get(key) and not Path(resources[key]).is_dir():
                issues.append({"component": key, "message": f"cache directory not found: {resources[key]}"})
        from bole.runtime.snapshots import source_provenance
        evidence["source_provenance"] = source_provenance(source, resources.get("source_revision"))
        evidence["source_revision"] = evidence["source_provenance"]["observed_git_revision"]
        evidence["source_changes"] = evidence["source_provenance"]["git_changes"]
        if resources.get("codex_seccomp") or resources.get("codex_apparmor"):
            from bole.runtime.dependencies import codex_security_args
            try:
                evidence["codex_security_profiles"] = {"docker_args": codex_security_args(resources),
                    "runtime_execution": "not_verified_by_static_resource_check"}
            except (OSError, ValueError) as error:
                issues.append({"component": "codex_security_profiles", "message": str(error)})
        if resources.get("codex_cli"):
            from bole.runtime.dependencies import codex_cli_mount
            try:
                evidence["codex_cli"] = codex_cli_mount(resources["codex_cli"], config["harness"]["executable"])
            except (OSError, ValueError) as error:
                issues.append({"component": "codex_cli", "message": str(error)})
        if resources.get("codex_bwrap"):
            from bole.runtime.dependencies import codex_bwrap_mount
            try:
                mount = codex_bwrap_mount(resources["codex_bwrap"], config["harness"]["executable"])
                evidence["codex_bwrap"] = {"path": resources["codex_bwrap"], "exists": True,
                    "mount": mount[-1], "runtime_execution": "not_verified_by_static_resource_check"}
            except (OSError, ValueError) as error:
                issues.append({"component": "codex_bwrap", "message": str(error)})
        try:
            inspected = json.loads(command(["docker", "image", "inspect", resources["simulator_image"]]))[0]
            evidence["simulator_image_id"] = inspected["Id"]
            expected = resources.get("simulator_image_id")
            if expected and inspected["Id"] != expected:
                issues.append({"component": "simulator_image", "message": "image differs from resources.simulator_image_id"})
            evidence["harness_image_id"] = json.loads(command([
                "docker", "image", "inspect", resources["harness_image"]]))[0]["Id"]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            issues.append({"component": "images", "message": str(error)})
        gpu_ids = resources.get("gpus", [])
        if not gpu_ids:
            issues.append({"component": "gpu", "message": "no GPU selected; planning and CPU validation remain available"})
        else:
            for gpu in gpu_ids:
                try:
                    command(["nvidia-smi", "--id", gpu, "--query-gpu=uuid,name", "--format=csv,noheader"])
                except (OSError, subprocess.SubprocessError) as error:
                    issues.append({"component": "gpu", "gpu": gpu, "message": str(error)})
        return {"schema_version": 1, "ready": not issues, "ok": not issues,
                "issues": issues, "resources": evidence,
                "real_simulation_verified": False,
                "capability_validation": "source inspection and CPU contracts; GPU smoke still required"}

    def session(self, context):
        return RoboDojoSession(context)


class RoboDojoSession:
    def __init__(self, context):
        self.context = context
        self.config, self.episode, self.store = context["config"], context["episode"], context["store"]
        self.name = "bole-sim-" + uuid4().hex
        self.temp = self.process = self.connection = self.stream = self.log = None
        self.lock = threading.RLock()
        self.last_result = None
        self.broken = False
        self.opened = False
        self.worker_stopped = False

    def _cache(self, kind, destination):
        resources = self.config["resources"]
        source = resources.get("cache_source" if kind == "cache" else "extensions_source")
        destination.mkdir()
        if source:
            if not Path(source).is_dir():
                raise ValueError(f"missing existing cache source: {source}")
            shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True)
            return
        volume = resources.get("cache_volumes", {}).get(kind)
        if not volume:
            return
        command(["docker", "volume", "inspect", volume])  # Never silently create an empty seed.
        script = (
            "import os,shutil; "
            "size=sum(os.stat(os.path.join(p,n),follow_symlinks=False).st_size "
            "for p,ds,fs in os.walk('/seed') for n in fs); "
            "assert shutil.disk_usage('/copy').free > size+268435456, 'insufficient space for isolated cache copy'; "
            "shutil.copytree('/seed','/copy',dirs_exist_ok=True,symlinks=True); "
            f"[os.chown(os.path.join(p,n),{os.getuid()},{os.getgid()},follow_symlinks=False) "
            "for p,ds,fs in os.walk('/copy') for n in ds+fs]"
        )
        helper = self.name + "-" + kind
        remaining = self._startup_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("startup budget exhausted while preparing isolated caches")
        try:
            command(["docker", "run", "--rm", "--name", helper, "--pull=never", "--network=none", "--read-only",
                     "--mount", f"type=volume,src={volume},dst=/seed,readonly",
                     "--mount", f"type=bind,src={destination},dst=/copy",
                     "--entrypoint", "python", resources["simulator_image"], "-c", script],
                    timeout=remaining)
        finally:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                command(["docker", "rm", "--force", helper], timeout=15)

    def _argv(self, private):
        resources, episode = self.config["resources"], self.episode
        container_root = resources.get("simulator_root", ROOT)
        source, assets = Path(resources["source"]).resolve(), Path(resources["assets"]).resolve()
        code = Path(__file__).resolve().parents[3]
        raw = self.store.root / "native-evaluation/raw"
        raw.mkdir(parents=True, exist_ok=True)
        for kind in ("cache", "extensions"):
            self._cache(kind, private / kind)
        args = ["docker", "run", "--rm", "--pull=never", "--init", "--name", self.name,
                "--network=host", "--shm-size=2g", "--gpus", f"device={self.context['gpu']}",
                "--workdir", container_root,
                "--env", "NVIDIA_DRIVER_CAPABILITIES=all", "--env", "CUDA_VISIBLE_DEVICES=0",
                "--env", "OMNI_KIT_ACCEPT_EULA=YES", "--env", "PYTHONDONTWRITEBYTECODE=1",
                "--env", "EVAL_NUM=1", "--env", "EVAL_ENV_TYPE=sim",
                "--env", f"ROBODOJO_RUN_ID={self.name}",
                "--env", f"PYTHONPATH=/opt/bole:{container_root}:{container_root}/XPolicyLab",
                "--mount", f"type=bind,src={code},dst=/opt/bole,readonly",
                "--mount", f"type=bind,src={private},dst=/run/bole,readonly",
                "--mount", f"type=bind,src={self.store.root},dst=/run/bole-results",
                "--mount", f"type=bind,src={raw},dst={container_root}/eval_result",
                "--mount", f"type=bind,src={assets},dst={container_root}/Assets,readonly",
                "--mount", f"type=bind,src={assets},dst={assets},readonly",
                "--mount", f"type=bind,src={private / 'cache'},dst=/root/.cache",
                "--mount", f"type=bind,src={private / 'extensions'},dst=/root/.local/share/ov"]
        for name in SOURCE_DIRECTORIES:
            path = source / name
            if path.exists():
                args += ["--mount", f"type=bind,src={path},dst={container_root}/{name},readonly"]
        args += prepare_robot_overrides(resources.get("asset_overrides"), assets, private, container_root)
        for key, value in resources.get("simulator_env", {}).items():
            args += ["--env", f"{key}={value}"]
        if resources.get("simulator_entrypoint"):
            args += ["--entrypoint", resources["simulator_entrypoint"]]
        # A's image entrypoint activates its RoboDojo environment. Overriding
        # it with 'python' selects base Conda, which has no simulator packages.
        return args + [resources["simulator_image"], resources.get("simulator_python", "python"), "-u", "-m",
            "bole.benchmarks.robodojo.worker", "--task_name", episode["task_id"],
            "--env_cfg_type", "arx_x5", "--num_envs", "1", "--device_id", "0",
            "--policy_name", "demo_policy", "--port", "1", "--additional_info", "action_type=joint",
            "--seed", str(episode["seed"]), "--enable_cameras", "--headless", "--kit_args",
            resources.get("simulator_kit_args", "--ext-folder /root/.local/share/ov/data/exts/v2 --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera")]

    def open(self):
        if self.opened or self.temp:
            raise RuntimeError("environment session cannot be opened twice")
        if self.context.get("gpu") is None:
            raise ValueError("RoboDojo requires a GPU; no simulator was launched")
        self.temp = tempfile.TemporaryDirectory(prefix="bole-rd-")
        private = Path(self.temp.name)
        safety = dict(self.config["recording"]["safety"])
        if isinstance(safety.get("roles"), str):
            import yaml
            mappings = yaml.safe_load(Path(safety["roles"]).read_text())
            safety["roles"] = mappings.get(self.episode["task_id"])
        worker = {"identity": self.store.identity, "episode": self.episode,
                  "limits": self.config["limits"], "termination": self.config["termination"],
                  "recording": {**self.config["recording"], "safety": safety},
                  "owner": {"uid": os.getuid(), "gid": os.getgid()}}
        (private / "config.json").write_text(json.dumps(worker))
        log_path = self.store.root / "logs/simulator.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("wb")
        deadline = time.monotonic() + self.config["limits"]["startup_timeout_s"]
        self._startup_deadline = deadline
        try:
            with socket.socket(socket.AF_UNIX) as server:
                server.bind(str(private / "worker.sock"))
                server.listen(1)
                server.settimeout(0.2)
                self.process = subprocess.Popen(self._argv(private), stdout=self.log, stderr=self.log,
                                                stdin=subprocess.DEVNULL)
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError(f"simulator exited during startup; see {log_path}")
                    try:
                        self.connection, _ = server.accept()
                        break
                    except TimeoutError:
                        continue
                if self.connection is None:
                    raise TimeoutError("simulator connection startup timeout")
            self.connection.settimeout(max(0.1, deadline - time.monotonic()))
            self.stream = self.connection.makefile("rwb")
            info = self._receive("ready")
            self.connection.settimeout(self.config["limits"]["request_timeout_s"])
            self.opened = True
            return info
        except BaseException:
            self.broken = True
            self._cleanup()
            raise

    def _receive(self, expected):
        line = self.stream.readline()
        if not line:
            raise ConnectionError("simulator worker disconnected")
        response = json.loads(line)
        if response["type"] == "error":
            raise RuntimeError(response["data"])
        if response["type"] != expected:
            raise RuntimeError(f"expected {expected}, received {response['type']}")
        return response["data"]

    def _request(self, operation, data, response):
        with self.lock:
            if not self.opened or self.broken:
                raise RuntimeError("environment is unavailable")
            try:
                self.stream.write((json.dumps({"operation": operation, **data}, allow_nan=False) + "\n").encode())
                self.stream.flush()
                return self._receive(response)
            except BaseException:
                self.broken = True
                raise

    def observe(self):
        return self._request("observe", {}, "observation")

    def delivered(self):
        self._request("delivered", {}, "acknowledged")

    def execute(self, actions, call_id):
        validate_actions(actions)
        return self._request("execute", {"actions": actions, "call_id": call_id}, "receipt")

    def close(self, reason):
        if self.last_result is not None:
            return self.last_result
        errors = []
        try:
            if self.opened and not self.broken:
                self.last_result = self._request("close", {"reason": reason}, "result")
        except Exception as error:
            self.broken = True
            errors.append({"component": "worker_close", "error": f"{type(error).__name__}: {error}"})
        finally:
            try:
                self._cleanup()
            except Exception as error:
                errors.append({"component": "worker_cleanup", "error": f"{type(error).__name__}: {error}"})
        if self.last_result is None:
            self.last_result = self._recover_persisted(reason, errors)
        elif errors:
            self.last_result.setdefault("collection_errors", []).extend(errors)
        return self.last_result

    def _load_record(self, relative, errors):
        """Read identity-matched records only; a neighbouring verdict is not ours."""
        path = self.store.root / relative
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or any(value.get(key) != expected
                    for key, expected in self.store.identity.items()):
                raise ValueError("persisted record identity does not match this episode")
            return value
        except (OSError, ValueError) as error:
            errors.append({"component": relative, "error": f"{type(error).__name__}: {error}"})
            return None

    def _recover_persisted(self, reason, errors):
        """CPU-only recovery after the worker stops; never re-run native scoring."""
        native = self._load_record("native-evaluation/result.json", errors)
        if not native or native.get("available") is not True or type(native.get("success")) is not bool:
            native = {"available": False, "success": None, "score": None,
                      "missing_reason": "worker unavailable and no identity-matched final native verdict was persisted"}
        elif native.get("score") is not None and (type(native["score"]) not in (int, float)
                                                or not math.isfinite(native["score"])):
            native["score"] = None
            native["score_missing_reason"] = "nonfinite_or_invalid_native_score"
        boundary = self._load_record("trajectory/native-boundary.json", errors) or {}
        if type(boundary.get("actual_control_steps")) is not int or boundary.get("actual_control_steps", -1) < 0:
            try:
                journal = read_jsonl(self.store.root / "trajectory/control-steps.jsonl")
                rows = journal["records"]
                if not journal["complete"]:
                    errors.append({"component": "control_step_recovery", "error": "incomplete journal tail; using confirmed prefix only"})
            except (OSError, ValueError) as error:
                rows = []
                errors.append({"component": "control_step_recovery", "error": str(error)})
            confirmed = [row["control_step"] for row in rows if isinstance(row, dict)
                and type(row.get("control_step")) is int and row["control_step"] >= 0
                and all(row.get(key) == value for key, value in self.store.identity.items())]
            boundary = {"actual_control_steps": max(confirmed, default=0),
                        "counter_source": "last_persisted_control_step_or_no_confirmation",
                        "in_flight_execution": "unknown"}
        boundary["counter_coverage"] = "last_persisted_confirmation; later_execution_unknown"
        boundary.setdefault("termination_reason", reason)
        if boundary["termination_reason"] is None:
            boundary["termination_reason"] = reason
        safety = self._load_record("safety.json", errors) or {
            "status": "not_collected", "reason": "worker stopped before safety finalization", "constraints": []}
        videos = None
        meta = self._load_record("videos/meta.json", errors)
        if meta:
            if self.worker_stopped:
                try:
                    recorder = VideoRecorder.recover(self.store.root / "videos")
                    videos = recorder.finalize(partial=True)
                except Exception as error:
                    errors.append({"component": "video_recovery", "error": f"{type(error).__name__}: {error}"})
                    videos = [{"type": "video", "camera": camera, "status": "collection_failed",
                        "path": None, "partial": True, "playable": False, "reason": str(error)}
                        for camera in meta.get("cameras", [])]
            else:
                errors.append({"component": "video_recovery", "error": "worker_stop_not_confirmed; retained frames without concurrent encoding"})
        try:
            artifacts = write_artifact_index(self.store, video=videos, partial=True,
                collection_errors=errors, video_enabled=self.config.get("recording", {}).get("video", {}).get("enabled"))
        except Exception as error:
            errors.append({"component": "artifact_index", "error": f"{type(error).__name__}: {error}"})
            artifacts = {"partial": True, "collection_errors": errors, "status": "collection_failed"}
        return {**boundary, "native_result": native, "safety": safety, "artifacts": artifacts,
                "recovered_from_persisted_records": True, "recovery_errors": errors,
                "infrastructure_error": reason != "cancelled", "artifacts_status": "partial_or_unavailable"}

    def _cleanup(self):
        self.opened = False
        for resource in (self.stream, self.connection):
            if resource:
                with contextlib.suppress(OSError):
                    resource.close()
        self.worker_stopped = self.process is None or self.process.poll() is not None
        if self.process and self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
                self.worker_stopped = True
            except subprocess.TimeoutExpired:
                try:
                    command(["docker", "stop", "--time", "5", self.name], timeout=15)
                    self.worker_stopped = True
                except (OSError, subprocess.SubprocessError):
                    pass
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=5)
        if self.log:
            self.log.close()
        if self.temp:
            # Helpers may create root-owned cache files. Removal is restricted
            # to this session's exact temporary mount, never a shared cache.
            private = Path(self.temp.name)
            try:
                self.temp.cleanup()
            except PermissionError:
                self.store.event("cache_cleanup_pending", path=str(private),
                                 reason="private cache ownership; retained for operator cleanup")
            self.temp = None
