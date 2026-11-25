"""CPU contracts around WorkerEpisode; no Isaac, PhysX, Docker, or GPU imports.

The controlled environment follows the inspected A eval_env counter/control/
final-check API. It deliberately returns measured joints unlike command targets
and can overturn a transient success during native final evaluation. These
contracts are not a claim of real simulator acceptance.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bole.results import RecordStore, read_jsonl

DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in ("numpy", "PIL", "yaml"))
if DEPENDENCIES:
    import numpy as np
    from bole.benchmarks.robodojo import worker
    from bole.benchmarks.robodojo.adapter import validate_actions


def configuration(*, grace=3, limit=50, safety=False):
    return {"limits": {"control_steps": limit, "timeout_s": None},
        "termination": {"success_grace_steps": grace},
        "recording": {"safety": {"enabled": safety}, "video": {"enabled": False}}}


def action():
    return {"left_arm_joint_state": [1.] * 6, "left_ee_joint_state": [.95],
            "right_arm_joint_state": [-1.] * 6, "right_ee_joint_state": [.95]}


class ControlledRobotManager:
    def __init__(self):
        self.robot_list = [SimpleNamespace(type="target", arm_name=side + "_arm",
            gripper_name=side + "_ee", gripper_scale=[0., .04],
            gripper_move={"sign": 1 if side == "left" else -1}, arm_joint_indices=list(range(6)))
            for side in ("left", "right")]
        self.robot_key = [SimpleNamespace(joint_names=["j" + str(i) for i in range(6)]) for _ in self.robot_list]

    def process_name(self, name):
        return name + "_joint_state"

    def get_joint(self, robot, env_idx_list):
        assert env_idx_list == [0]
        return {0: np.array([.125] * 6)}

    def get_end_effector_real_val(self, robot, env_idx_list):
        assert env_idx_list == [0]
        return {0: np.array([.01, .01])}

    def get_real_endpose(self, robot, env_idx_list, is_relative):
        assert env_idx_list == [0] and is_relative is False
        return {0: np.array([1., 2., 3., 1., 0., 0., 0.])}


class ControlledRewardManager:
    def __init__(self, env, successes, final_success):
        self.env, self.successes, self.final_success = env, set(successes), final_success
        self.checks = []

    def get_reward(self, final_check):
        self.checks.append(final_check)
        value = self.env.take_action_cnt[0] in self.successes
        if final_check and self.final_success is not None:
            value = self.final_success
        return [1. if value else 0.]

    def get_score(self):
        return [25.]


class ControlledNative:
    """Single-env A API stand-in; one action executes two physical substeps."""
    def __init__(self, successes=(), *, final_success=None, step_lim=50):
        self.task_name, self.step_lim = "controlled_task", step_lim
        self.take_action_cnt, self.success, self.end_flag = [0], [True], [False]
        self.sim = SimpleNamespace(_sim_step_counter=0)
        self.obs_manager = SimpleNamespace(dt=.1, collect_freq=5., instruction=["Move the object."])
        self.robot_manager = ControlledRobotManager()
        self.reward_manager = ControlledRewardManager(self, successes, final_success)
        self.actions, self.observation_reads = [], 0
        self.get_score = lambda: None

    def process_control_info(self, control_info, env_idx):
        assert env_idx == 0
        return [{key: {"position": value.tolist()} for key, value in control_info.items()}]

    def step(self):
        self.sim._sim_step_counter += 1

    def take_action(self, data):
        # The real A validator is a simulator-private bound method. The
        # controlled test validates Bole's public shape before native dispatch.
        validate_actions([{key: value.tolist() for key, value in data.items()}])
        if self.take_action_cnt[0] >= self.step_lim or self.end_flag[0]:
            return
        self.take_action_cnt[0] += 1
        self.actions.append(data)
        self.process_control_info(data, 0)
        self.step()
        self.step()
        self.is_episode_end()

    def is_episode_end(self):
        final_check = self.take_action_cnt[0] >= self.step_lim or (not self.success[0] and not self.end_flag[0])
        reward = self.reward_manager.get_reward(final_check)[0]
        if not self.end_flag[0]:
            if reward > 1 - 1e-3:
                self.success[0], self.end_flag[0] = True, True
            elif self.take_action_cnt[0] >= self.step_lim or not self.success[0]:
                self.success[0], self.end_flag[0] = False, True
        return self.end_flag[0]

    def get_obs(self):
        self.observation_reads += 1
        return {"state": {"left_arm_joint_state": [99.] * 6},
            "vision": {"cam_head": {"color": np.zeros((4, 4, 3), dtype=np.uint8)}},
            "instruction": "Move the object.", "success": True, "reward": 1,
            "native_result": {"success": True}, "object_poses": {"hidden": [1, 2, 3]}}


class ControlledProbe:
    def __init__(self, env, store, roles=None):
        self.calls = []
    def before_reset(self):
        self.calls.append("before_reset")
    def start(self):
        self.calls.append("start")
    def control_context(self, call_id, step, chunk_index):
        self.calls.append((call_id, step, chunk_index))
    def finish(self, partial=False):
        self.calls.append(("finish", partial))
        return {"constraints": [], "complete": not partial}
    def close(self):
        self.calls.append("close")


@unittest.skipUnless(DEPENDENCIES, "CPU worker tests require numpy, Pillow, and PyYAML; no GPU needed")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = RecordStore(Path(self.directory.name), {"run_id": "r", "episode_id": "e"})

    def episode(self, env=None, **config):
        episode = worker.WorkerEpisode(env or ControlledNative(), configuration(**config), self.store)
        episode.after_reset()
        episode.check_native()
        episode.boundary.delivered()
        return episode

    def test_first_success_fixes_grace_despite_later_success(self):
        episode = self.episode(ControlledNative(successes=[2, 4]))
        first = episode.execute({"call_id": "one", "actions": [action()] * 3})
        second = episode.execute({"call_id": "two", "actions": [action()] * 4})
        self.assertEqual(first["public"]["confirmed_control_steps"], 3)
        self.assertEqual(second["public"]["confirmed_control_steps"], 2)
        self.assertEqual(second["public"]["unexecuted_steps"], 2)
        self.assertEqual(episode.boundary.first_success_step, 2)
        self.assertEqual(episode.boundary.steps, 5)
        self.assertEqual(episode.boundary.reason, "success_grace_reached")
        self.assertEqual(len(episode.env.actions), 5)

    def test_native_final_failure_survives_first_success(self):
        env = ControlledNative(successes=[1], final_success=False)
        episode = self.episode(env, grace=1)
        receipt = episode.execute({"call_id": "one", "actions": [action()] * 3})
        self.assertTrue(receipt["private"]["success_observed"])
        episode.settle("success_grace_reached")
        self.assertFalse(env.success[0])
        self.assertTrue(env.end_flag[0])
        self.assertTrue(env.reward_manager.checks[-1])
        self.assertEqual(env.step_lim, 50)
        self.assertEqual(env.take_action_cnt[0], 2)

    def test_action_chunk_stops_exactly_at_control_limit(self):
        episode = self.episode(limit=3)
        receipt = episode.execute({"call_id": "chunk", "actions": [action()] * 10})
        self.assertEqual(receipt["public"]["execution_status"], "partial")
        self.assertEqual(receipt["public"]["control_step_range"], [0, 3])
        self.assertEqual(receipt["public"]["unexecuted_steps"], 7)
        self.assertEqual(episode.env.sim._sim_step_counter, 6)
        self.assertEqual(episode.boundary.reason, "control_step_limit")

    def test_reads_use_measurements_without_advancing_physics_or_leaking_grade(self):
        episode = self.episode()
        before = episode.env.sim._sim_step_counter
        for _ in range(2):
            observation = worker.public_observation(episode.env)
            self.assertEqual(observation["state"]["left_arm_joint_state"], [.125] * 6)
            self.assertAlmostEqual(observation["state"]["left_ee_joint_state"][0], .25)
            self.assertAlmostEqual(observation["state"]["right_ee_joint_state"][0], .75)
            self.assertEqual(set(observation), {"state", "end_effector_poses", "measurement_validity", "vision", "instruction", "control_step"})
        self.assertEqual(episode.env.sim._sim_step_counter, before)
        self.assertEqual(episode.boundary.steps, 0)

    def test_nonfinite_measurement_is_null_with_field_specific_validity(self):
        episode = self.episode()
        manager = episode.env.robot_manager
        with patch.object(manager, "get_end_effector_real_val", return_value={0: np.array([float("nan"), .01])}):
            observation = worker.public_observation(episode.env)
            self.assertIsNone(observation["state"]["left_ee_joint_state"])
            self.assertEqual(observation["measurement_validity"]["left_ee_joint_state"]["status"], "missing")
            self.assertEqual(observation["state"]["left_arm_joint_state"], [.125] * 6)
            json.dumps(observation, allow_nan=False)
            episode.execute({"call_id": "missing-gripper", "actions": [action()]})
        row = read_jsonl(self.store.root / "trajectory/control-steps.jsonl")["records"][0]
        self.assertEqual(row["measurement_status"], "missing")
        self.assertIsNone(row["measured_after_execution"]["measurement_semantics"]["left"]["measured_primary_finger_m"])

    def test_initial_success_ready_is_private_and_persisted_before_grace_zero_close(self):
        episode = worker.WorkerEpisode(ControlledNative(successes=[0]), configuration(grace=0), self.store)
        episode.after_reset()
        class Stream:
            def __init__(self):
                self.input = io.BytesIO(b'{"operation":"delivered"}\n{"operation":"close","reason":"success_grace_reached"}\n')
                self.output = io.BytesIO()
            def write(self, data): return self.output.write(data)
            def flush(self): pass
            def readline(self): return self.input.readline()
        stream = Stream()
        episode.interact(stream)
        ready = json.loads(stream.output.getvalue().splitlines()[0])["data"]
        self.assertEqual(ready["private_boundary"]["first_success_control_step"], 0)
        self.assertEqual(ready["private_boundary"]["termination_reason"], "success_grace_reached")
        self.assertNotIn("private_boundary", worker.public_observation(episode.env))
        self.assertEqual(episode.boundary.steps, 0)
        self.assertGreaterEqual(episode.boundary.report()["elapsed_after_first_observation_s"], 0)
        saved = json.loads((self.store.root / "trajectory/native-boundary.json").read_text())
        self.assertEqual(saved["first_success_control_step"], 0)
        self.assertEqual(saved["actual_control_steps"], 0)

    def test_public_receipt_has_only_execution_fields_and_records_actual_state(self):
        episode = self.episode(ControlledNative(successes=[1]), grace=0)
        receipt = episode.execute({"call_id": "action", "actions": [action()]})
        self.assertEqual(set(receipt["public"]), {"execution_status", "requested_control_steps",
            "confirmed_control_steps", "control_step_range", "unexecuted_steps", "in_flight_execution", "episode_closed"})
        for forbidden in ("success", "score", "reward", "first_success", "termination_reason"):
            self.assertNotIn(forbidden, json.dumps(receipt["public"]))
        rows = read_jsonl(self.store.root / "trajectory/control-steps.jsonl")["records"]
        self.assertEqual(rows[0]["requested_action"]["left_arm_joint_state"], [1.] * 6)
        self.assertEqual(rows[0]["measured_after_execution"]["state"]["left_arm_joint_state"], [.125] * 6)
        self.assertIsNotNone(rows[0]["controller_sequence"])
        saved = json.loads((self.store.root / "trajectory/native-boundary.json").read_text())
        self.assertEqual(saved["actual_control_steps"], 1)
        self.assertEqual(saved["in_flight_execution"], "none")

    def test_probe_context_names_current_step_and_finishes(self):
        with patch.object(worker, "NativeProbe", ControlledProbe):
            episode = self.episode(safety=True)
        episode.execute({"call_id": "chunk", "actions": [action()] * 2})
        episode.artifacts(partial=False)
        self.assertEqual(episode.probe.calls[:4], ["before_reset", "start", ("chunk", 1, 0), ("chunk", 2, 1)])
        self.assertIn(("finish", False), episode.probe.calls)

    def test_trajectory_collection_and_write_time_exclude_video_and_native_action(self):
        episode = self.episode()
        episode.recording_overhead_s = 17.0  # Independently measured video work.
        with patch.object(worker.time, "perf_counter", side_effect=[10., 12., 20., 23.]):
            episode.execute({"call_id": "timed", "actions": [action()]})
        rows = read_jsonl(self.store.root / "trajectory/control-steps.jsonl")["records"]
        self.assertEqual(rows[0]["measurement_collection_seconds"], 2.)
        episode.artifacts(partial=False)
        timing = json.loads((self.store.root / "trajectory/recording-timing.json").read_text())
        self.assertEqual(timing["measurement_collection_seconds"], 2.)
        self.assertEqual(timing["trajectory_write_seconds"], 3.)
        self.assertEqual(timing["video_collection_and_write_seconds"], 17.)
        self.assertEqual(timing["measurement_attempts"], 1)
        self.assertEqual(timing["trajectory_write_attempts"], 1)
        self.assertEqual(timing["trajectory_write_failures"], 0)
        manifest = json.loads((self.store.root / "artifacts.json").read_text())
        self.assertTrue(any(row["path"] == "trajectory/recording-timing.json" for row in manifest["artifacts"]))

    def test_recording_clock_failure_does_not_end_task_or_invent_time(self):
        episode = self.episode()
        with patch.object(episode, "sim_time", side_effect=RuntimeError("clock missing")):
            receipt = episode.execute({"call_id": "chunk", "actions": [action()] * 2})
        self.assertEqual(receipt["public"]["confirmed_control_steps"], 2)
        self.assertFalse(receipt["public"]["episode_closed"])
        self.assertIsNone(episode.boundary.reason)
        rows = read_jsonl(self.store.root / "trajectory/control-steps.jsonl")["records"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["simulation_time_s"] is None for row in rows))

    def test_recording_write_failure_does_not_stop_action_chunk(self):
        episode = self.episode()
        native_append = self.store.append_jsonl
        def append(relative, data):
            if relative == "trajectory/control-steps.jsonl":
                raise OSError("controlled evidence disk error")
            return native_append(relative, data)
        with patch.object(self.store, "append_jsonl", side_effect=append):
            receipt = episode.execute({"call_id": "chunk", "actions": [action()] * 2})
        self.assertEqual(receipt["public"]["confirmed_control_steps"], 2)
        self.assertFalse(receipt["public"]["episode_closed"])
        self.assertIsNone(episode.boundary.reason)

        self.assertEqual(episode.control_recording_timing["measurement_attempts"], 2)
        self.assertEqual(episode.control_recording_timing["trajectory_write_attempts"], 2)
        self.assertEqual(episode.control_recording_timing["trajectory_write_failures"], 2)
        self.assertGreaterEqual(episode.control_recording_timing["trajectory_write_seconds"], 0)

    def test_artifact_failures_keep_native_final_grade_and_close_probe(self):
        env = ControlledNative(successes=[1], final_success=False)
        with patch.object(worker, "NativeProbe", ControlledProbe):
            episode = self.episode(env, grace=0, safety=True)
        episode.execute({"call_id": "one", "actions": [action()]})
        episode.settle("success_grace_reached")
        episode.video = SimpleNamespace(finalize=lambda **_: (_ for _ in ()).throw(OSError("video finalize failed")))
        with patch.object(episode.probe, "finish", side_effect=OSError("probe finalize failed")):
            with patch.object(self.store, "write_json", side_effect=OSError("manifest write failed")):
                artifacts = episode.artifacts(partial=False)
        self.assertFalse(env.success[0])
        self.assertTrue(episode.boundary.report()["success_observed"])
        self.assertIn("close", episode.probe.calls)
        self.assertEqual(artifacts["safety"]["status"], "collection_failed")
        self.assertEqual(artifacts["artifacts"]["video"][0]["status"], "collection_failed")
        self.assertTrue(artifacts["artifacts"]["partial"])


if __name__ == "__main__":
    unittest.main()
