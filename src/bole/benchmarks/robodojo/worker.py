"""Private simulator worker for the selected RoboDojo environment.

Uses RoboDojo A's constructors, control interpolation, task predicates and final
evaluator. Only the episode-stop decision is separated from native success.
"""

from __future__ import annotations

import base64
import copy
import importlib
import io
import json
import math
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch

from bole.engine.termination import EpisodeBoundary
from bole.results import RecordStore
from bole.results.artifacts import episode_video_records, write_artifact_index
from bole.results.video import VideoRecorder
from .adapter import action_schema
from .probes import NativeProbe


def reply(stream, kind, value):
    stream.write((json.dumps({"type": kind, "data": value}, allow_nan=False) + "\n").encode())
    stream.flush()


def wire(value):
    if isinstance(value, dict):
        return {str(k): wire(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [wire(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def measured_state(env):
    manager = env.robot_manager
    state, poses, semantics, validity = {}, {}, {}, {}

    def read(name, getter, width):
        try:
            value = wire(getter())
            if not isinstance(value, list) or len(value) != width or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in value):
                raise ValueError("measurement has missing, nonfinite, or unexpected components")
            validity[name] = {"status": "valid", "reason": None}
            return value
        except Exception as error:
            validity[name] = {"status": "missing", "reason": f"{type(error).__name__}: {error}"}
            return None

    for index, robot in enumerate(manager.robot_list):
        if robot.type != "target":
            continue
        side = robot.arm_name.split("_")[0]
        arm = manager.process_name(robot.arm_name)
        gripper = manager.process_name(robot.gripper_name)
        state[arm] = read(arm, lambda: manager.get_joint(robot, env_idx_list=[0])[0], len(robot.arm_joint_indices))
        finger = read(gripper, lambda: [float(manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0])], 1)
        raw = finger[0] if finger is not None else None
        scale = [None, None]
        state[gripper] = None
        try:
            lo, hi = map(float, robot.gripper_scale)
            if not all(math.isfinite(value) for value in (lo, hi)) or hi <= lo:
                raise ValueError("gripper measurement scale is invalid")
            scale = [lo, hi]
            if raw is not None:
                fraction = (raw - lo) / (hi - lo)
                if robot.gripper_move["sign"] != 1:
                    fraction = 1 - fraction
                state[gripper] = [max(0.0, min(1.0, fraction))]
        except Exception as error:
            validity[gripper] = {"status": "missing", "reason": f"{type(error).__name__}: {error}"}
        pose_name = f"{side}_ee_pose"
        poses[pose_name] = read(pose_name, lambda: manager.get_real_endpose(robot, env_idx_list=[0], is_relative=False)[0], 7)
        articulation = manager.robot_key[index]
        semantics[side] = {"joint_names": [articulation.joint_names[i] for i in robot.arm_joint_indices],
                           "arm_unit": "rad", "gripper_unit": "opening_fraction_0_1",
                           "measured_primary_finger_m": raw, "gripper_scale_m": scale,
                           "end_effector_pose": "world xyz metres, wxyz quaternion"}
    return {"state": state, "end_effector_poses": poses, "measurement_semantics": semantics,
            "measurement_validity": validity}


def public_observation(env):
    import numpy as np
    from PIL import Image
    native = env.get_obs()
    measured = measured_state(env)
    vision = {}
    for name, camera in native["vision"].items():
        pixels = np.asarray(camera["color"])
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
            raise ValueError(f"unsupported native color frame: {name}")
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG", compress_level=1)
        vision[name] = {"color": {"encoding": "png;base64", "dtype": "uint8",
                                  "shape": list(pixels.shape),
                                  "data": base64.b64encode(stream.getvalue()).decode("ascii")}}
    if not vision or not measured["state"]:
        raise RuntimeError("native public cameras or measured robot state are missing")
    return {"state": measured["state"], "end_effector_poses": measured["end_effector_poses"],
            "measurement_validity": measured["measurement_validity"], "vision": vision, "instruction": native["instruction"],
            "control_step": int(env.take_action_cnt[0])}


class WorkerEpisode:
    def __init__(self, env, config, store):
        self.env, self.config, self.store = env, config, store
        limit = config["limits"]["control_steps"]
        limit = min(int(env.step_lim), limit) if limit is not None else int(env.step_lim)
        self.boundary = EpisodeBoundary(grace_steps=config["termination"]["success_grace_steps"],
                                        step_limit=limit, timeout_s=config["limits"]["timeout_s"])
        self.native_is_episode_end = env.is_episode_end
        self.native_step = env.step
        self.native_process_control = env.process_control_info
        self.controller_sequence = None
        self.pending_native = None
        self.in_action = False
        self.last_error = None
        self.probe = None
        self.start_physics_step = None
        self.video = None
        self.recording_overhead_s = 0.0
        self.control_recording_timing = {"measurement_collection_seconds": 0.0,
            "trajectory_write_seconds": 0.0, "measurement_attempts": 0,
            "trajectory_write_attempts": 0, "trajectory_write_failures": 0}
        self.collection_errors = []
        if config["recording"]["safety"]["enabled"]:
            self.probe = NativeProbe(env, store, roles=config["recording"]["safety"].get("roles"))
            self.probe.before_reset()
        env.is_episode_end = self.check_native
        env.process_control_info = self.process_control
        env.step = self.step

    def after_reset(self):
        step = getattr(getattr(self.env.sim, "unwrapped", self.env.sim), "_sim_step_counter", None)
        self.start_physics_step = step if type(step) is int else None
        if self.probe:
            self.probe.start()
        recording = self.config["recording"]["video"]
        if recording["enabled"]:
            self.video = VideoRecorder(self.store.root / "videos", cameras=recording["cameras"],
                                       fps=recording["fps"], identity=self.store.identity)
            self.capture_video()

    def sim_time(self):
        step = getattr(getattr(self.env.sim, "unwrapped", self.env.sim), "_sim_step_counter", None)
        if type(step) is not int or self.start_physics_step is None:
            raise RuntimeError("native simulation step clock unavailable")
        return (step - self.start_physics_step) * float(self.env.obs_manager.dt)

    def capture_video(self):
        if self.video is None:
            return
        started = time.monotonic()
        def frames():
            observation = public_observation(self.env)
            return {name: base64.b64decode(camera["color"]["data"], validate=True)
                    for name, camera in observation["vision"].items()}
        try:
            stamp = self.sim_time()
            if self.video.wants(stamp):
                self.video.offer(int(self.env.take_action_cnt[0]), stamp, frames)
        except Exception as error:
            self.video.mark_collection_error(error, control_step=int(self.env.take_action_cnt[0]))
            self.record_error("video_capture", error)
        self.recording_overhead_s += time.monotonic() - started

    def record_error(self, component, error):
        record = {"component": component, "error": str(error), "control_step": self.boundary.steps}
        self.collection_errors.append(record)
        try:
            self.store.event("collection_error", **record)
        except Exception:
            pass  # Retain this evidence in the final private receipt.

    def checkpoint_boundary(self, *, in_flight=False):
        """Persist only confirmed counters; an in-flight action is never inferred."""
        try:
            self.store.write_json("trajectory/native-boundary.json", {
                **self.boundary.report(), "native_control_counter": int(self.env.take_action_cnt[0]),
                "in_flight_execution": "unknown" if in_flight else "none",
                "counter_source": "worker_confirmed_control_steps"})
        except Exception as error:
            self.record_error("boundary_checkpoint", error)

    def step(self, *args, **kwargs):
        result = self.native_step(*args, **kwargs)
        self.capture_video()  # Render on the physics clock; never add physics steps.
        return result

    def process_control(self, *args, **kwargs):
        result = self.native_process_control(*args, **kwargs)
        self.controller_sequence = wire(copy.deepcopy(result))
        return result

    def check_native(self):
        # These are exactly A's native final-check conditions. Keep the native
        # success/liveness flag unchanged until its own predicates change it.
        final_check = int(self.env.take_action_cnt[0]) >= int(self.env.step_lim) or (
            not self.env.success[0] and not self.env.end_flag[0])
        rewards = self.env.reward_manager.get_reward(final_check=final_check)
        success = bool(rewards[0] > 1 - 1e-3)
        scores = self.env.reward_manager.get_score() if hasattr(self.env, "get_score") else None
        native = {"available": True, "success": success,
                  "score": 1.0 if success else float(scores[0]) / 100 if scores is not None else 0.0,
                  "semantics": "RoboDojo A native predicate/history semantics"}
        self.pending_native = native
        if not self.in_action:
            self.boundary.observe_native(native)
        if not self.env.success[0] and not success:
            self.boundary.end("native_terminal_failure")
        if not self.in_action:
            self.checkpoint_boundary()
        return self.boundary.reason is not None

    def execute(self, request):
        import numpy as np
        actions = request["actions"]
        start = self.boundary.steps
        confirmed = 0
        error = None
        for index, action in enumerate(actions):
            if self.boundary.check():
                break
            if self.boundary.delivered_at is None:
                raise RuntimeError("actions are unavailable before initial observation delivery")
            if self.probe:
                self.probe.control_context(request["call_id"], self.boundary.steps + 1, index)
            self.controller_sequence = None
            before = time.monotonic()
            self.checkpoint_boundary(in_flight=True)
            self.in_action = True
            try:
                self.env.take_action({key: np.asarray(values, dtype=np.float32)
                                      for key, values in action.items()})
                if int(self.env.take_action_cnt[0]) != start + confirmed + 1:
                    raise RuntimeError("native counter failed to confirm one complete control step")
                # A predicate can mark a task failure inside take_action. This
                # still counts the control step which finished successfully.
                terminal = self.boundary.reason
                self.boundary.reason = None
                self.boundary.executed(self.pending_native)
                if terminal:
                    self.boundary.end(terminal)
                confirmed += 1
                self.checkpoint_boundary()
                measurement_started = time.perf_counter()
                try:
                    measured = measured_state(self.env)
                    measurement_status = "valid" if all(v["status"] == "valid" for v in measured["measurement_validity"].values()) else "missing"
                    missing_reason = None if measurement_status == "valid" else "see measured_after_execution.measurement_validity"
                except Exception as exc:
                    measured = {"state": None, "end_effector_poses": None}
                    measurement_status, missing_reason = "collection_failed", str(exc)
                try:
                    stamp, clock_error = self.sim_time(), None
                except Exception as exc:
                    stamp, clock_error = None, str(exc)
                    self.record_error("simulation_clock", exc)
                measurement_seconds = time.perf_counter() - measurement_started
                self.control_recording_timing["measurement_collection_seconds"] += measurement_seconds
                self.control_recording_timing["measurement_attempts"] += 1
                row = {
                    "call_id": request["call_id"], "action_chunk_index": index,
                    "control_step": self.boundary.steps, "wall_time_ns": time.time_ns(),
                    "simulation_time_s": stamp, "simulation_time_missing_reason": clock_error,
                    "requested_action": action,
                    "controller_sequence": self.controller_sequence,
                    "controller_target_semantics": "native process_control_info interpolation; physical arm radians and finger metres",
                    "measured_after_execution": measured, "measurement_status": measurement_status,
                    "measurement_collection_seconds": measurement_seconds,
                    "missing_reason": missing_reason, "execution_wall_s": time.monotonic() - before}
                write_started = time.perf_counter()
                write_error = None
                try:
                    self.store.append_jsonl("trajectory/control-steps.jsonl", row)
                except Exception as exc:
                    write_error = exc
                    self.control_recording_timing["trajectory_write_failures"] += 1
                finally:
                    self.control_recording_timing["trajectory_write_seconds"] += time.perf_counter() - write_started
                    self.control_recording_timing["trajectory_write_attempts"] += 1
                if write_error is not None:
                    self.record_error("control_trajectory", write_error)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.last_error = error
                self.boundary.end("infrastructure_error")
                self.checkpoint_boundary(in_flight=True)
                self.store.event("in_flight_action_error", call_id=request["call_id"],
                    action_chunk_index=index, confirmed_control_steps=self.boundary.steps,
                    native_counter=int(self.env.take_action_cnt[0]), execution_of_current_step="unknown",
                    error=error)
                break
            finally:
                self.in_action = False
        private = {**self.boundary.report(), "error": error, "collection_errors": self.collection_errors,
                   "in_flight_native_counter": int(self.env.take_action_cnt[0]) if error else None}
        public = {"execution_status": "failed" if error else "completed" if confirmed == len(actions) else "partial",
                  "requested_control_steps": len(actions), "confirmed_control_steps": confirmed,
                  "control_step_range": [start, start + confirmed],
                  "unexecuted_steps": None if error else len(actions) - confirmed,
                  "in_flight_execution": "unknown" if error else "none",
                  "episode_closed": self.boundary.reason is not None}
        return {"private": private, "public": public}

    def settle(self, reason):
        if reason == "finish":
            self.boundary.finish()
        else:
            self.boundary.end(reason)
        # Preserve A's native final check, as used by the old independent
        # worker at an explicit truncation. First success never forces this
        # final result back to true.
        limit = self.env.step_lim
        try:
            self.env.step_lim = int(self.env.take_action_cnt[0])
            self.native_is_episode_end()
        finally:
            self.env.step_lim = limit
            self.checkpoint_boundary()

    def interact(self, stream):
        self.check_native()
        measured = measured_state(self.env)
        info = {"task": self.env.task_name, "step_limit": self.boundary.step_limit,
                "private_boundary": self.boundary.report(),
                "instruction": self.env.obs_manager.instruction[0],
                "capabilities": {"action_schema": action_schema(), "control_type": "joint",
                    "action_units": "arm radians; gripper opening fraction [0,1]",
                    "control_frequency_hz": float(self.env.obs_manager.collect_freq),
                    "physics_dt_s": float(self.env.obs_manager.dt),
                    "robot": "arx_x5", "measurement_semantics": measured["measurement_semantics"],
                    "observations": "RGB cameras and measured robot joints/grippers/world end-effector poses",
                    "reset_available_to_harness": False,
                    "success_feedback_during_episode": False}}
        reply(stream, "ready", info)
        while line := stream.readline():
            request = json.loads(line)
            operation = request["operation"]
            if operation == "observe":
                reply(stream, "observation", public_observation(self.env))
            elif operation == "delivered":
                self.boundary.delivered()
                self.checkpoint_boundary()
                reply(stream, "acknowledged", {"ok": True})
            elif operation == "execute":
                reply(stream, "receipt", self.execute(request))
            elif operation == "close":
                self.settle(request["reason"])
                return
            else:
                raise ValueError("unsupported worker operation")
        raise ConnectionError("runner disconnected")

    def artifacts(self, partial):
        try:
            safety = self.probe.finish(partial=partial) if self.probe else {
                "status": "not_collected", "reason": "safety probes disabled", "constraints": {}}
        except Exception as error:
            self.record_error("safety_finalize", error)
            safety = {"status": "collection_failed", "reason": str(error), "constraints": {}}
        finally:
            if self.probe:
                try:
                    self.probe.close()
                except Exception as error:
                    self.record_error("safety_close", error)
        try:
            videos = self.video.finalize(partial=partial) if self.video else []
        except Exception as error:
            self.record_error("video_finalize", error)
            videos = [{"status": "collection_failed", "reason": str(error), "partial": True}]
        artifacts = {"video": episode_video_records(videos), "partial": partial or bool(self.collection_errors),
                     "video_enabled": self.video is not None,
                     "collection_errors": self.collection_errors,
                     "recording_overhead_s": self.recording_overhead_s}
        try:
            self.store.write_json("trajectory/recording-timing.json", {
                **self.control_recording_timing,
                "video_collection_and_write_seconds": self.recording_overhead_s,
                "scope": "worker_control_trajectory_measurement_and_append; video_is_separate",
                "measurement_scope": "post_action_measured_state_and_simulation_clock; includes clock_error_recording_if_needed",
                "write_scope": "control_steps_jsonl_serialization_lock_append_flush_fsync_including_failed_attempts",
                "excluded": ["native_action_execution", "video_collection", "physics_probe_collection",
                             "boundary_checkpoint_writes", "this_timing_document_write"],
                "physics_probe_timing": {"path": "safety.json", "path_base": "episode",
                    "field": "collection_and_write_seconds"},
                "overlapping_with_episode_wall_time": True,
                "partial": partial or bool(self.collection_errors)})
        except Exception as error:
            self.record_error("recording_timing", error)
            artifacts["partial"] = True
        try:
            self.store.write_json("safety.json", safety)
        except Exception as error:
            self.record_error("safety.json", error)
            artifacts["partial"] = True
        try:
            artifacts.update(write_artifact_index(self.store, video=videos,
                partial=artifacts["partial"], collection_errors=self.collection_errors,
                video_enabled=self.video is not None))
            self.store.write_json("artifacts.json", artifacts)
        except Exception as error:
            self.record_error("artifacts.json", error)
            artifacts["partial"] = True
        return {"safety": safety, "artifacts": artifacts}


class NoPolicyCallback:
    def __init__(self, **_):
        pass

    def call(self, func_name):
        if func_name != "reset":
            raise RuntimeError(f"unexpected native policy callback: {func_name}")


def share_owned_records(root, owner):
    if os.getuid() == owner["uid"]:
        return
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            try:
                os.chown(path, owner["uid"], owner["gid"])
            except OSError:
                pass


def run(stream):
    config = json.loads(Path("/run/bole/config.json").read_text())
    store = RecordStore(Path("/run/bole-results"), config["identity"])
    from src.eval_client import main as native
    eval_env = importlib.import_module("src.eval_client.eval_env")
    class Configured(Exception):
        pass
    def capture(*args, **kwargs):
        raise Configured(args, kwargs)
    env = episode = None
    completed = False
    try:
        with patch.object(native, "create_eval_env", capture):
            try:
                native.main()
            except Configured as captured:
                args, kwargs = captured.args
            else:
                raise RuntimeError("native launcher did not configure an environment")
        with patch.object(eval_env, "WsModelClient", NoPolicyCallback):
            env = eval_env.create_eval_env(*args, **kwargs)
        layout = config["episode"]["layout_id"]
        selected = Path(env.seed_manager.seed_info[layout]["scene_layout"])
        expected = Path(config["episode"]["layout_path"])
        if selected.name != expected.name or selected.parent.name != str(config["episode"]["seed"]):
            raise RuntimeError("native seed manager selected a different instance")
        env._stream_vision = lambda env_idx, frame: None
        episode = WorkerEpisode(env, config, store)
        env.reset(seed=[layout])
        episode.after_reset()
        env.eval_one_episode = lambda: episode.interact(stream)
        env.run_eval()
        data = json.loads((Path(env.save_dir) / "_result.json").read_text())
        detail = data["details"].get("0")
        native_result = {"available": detail is not None, "success": None, "score": None,
                         "source": "RoboDojo A run_eval/_result.json",
                         "semantics": "native history/process predicates and native final check",
                         "missing_reason": None if detail is not None else "native evaluator emitted no valid episode detail"}
        if detail is not None:
            if detail["layout_id"] != layout or detail != env.eval_result["details"][0]:
                raise RuntimeError("native result does not correspond to the selected episode")
            native_result.update(detail)
        try:
            store.write_json("native-evaluation/result.json", native_result)
        except Exception as error:
            episode.record_error("native_result_file", error)
        episode.checkpoint_boundary()
        partial = episode.boundary.reason in {"infrastructure_error", "cancelled"}
        artifacts = episode.artifacts(partial=partial)
        result = {"native_result": native_result, **episode.boundary.report(), **artifacts}
        share_owned_records(store.root, config["owner"])
        reply(stream, "result", result)
        completed = True
    finally:
        if episode and not completed:
            try:
                episode.artifacts(partial=True)
            except Exception as error:
                store.event("partial_artifact_recovery_failed", error=str(error))
        if env:
            env.close()
        share_owned_records(store.root, config["owner"])
        native.simulation_app.close()


def main():
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect("/run/bole/worker.sock")
        with connection.makefile("rwb") as stream:
            try:
                run(stream)
            except Exception as error:
                reply(stream, "error", f"{type(error).__name__}: {error}")
                raise


if __name__ == "__main__":
    main()
