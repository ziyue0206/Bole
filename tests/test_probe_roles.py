import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from bole.benchmarks.robodojo.probes import NativeProbe, InteractionSignals, normalize, task_roles
from bole.results import RecordStore, read_jsonl


class FakeSignals:
    def __init__(self):
        self.physx = self
        self.callback = None
        self.bodies = {}
    def enable_reports(self, stage):
        pass
    def bind(self, stage, objects):
        self.objects = objects
        return {"evidence_kind": "controlled_test"}
    def poses(self):
        return {k: {"position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]} for k in self.objects}
    def sleeping(self):
        return []
    def contacts(self, dt):
        return [], []
    def full_contacts(self, dt):
        return []
    def unresolved_support(self, contacts, previous, current):
        return []
    def subscribe_physics_on_step_events(self, callback, *args):
        self.callback = callback
        return SimpleNamespace(unsubscribe=self.unsubscribe)
    def unsubscribe(self):
        self.callback = None


def environment(task="stack_bowls", extra=None, names=None):
    names = ["bowl0", "bowl1", "bowl2"] if names is None else names
    instances = [SimpleNamespace(instance_config={"label": n}, usd_prim_path="/scene/" + n) for n in names]
    if extra:
        instances.append(extra)
    scene = SimpleNamespace(config={"env0": {"Rigid": {"objects": [{"prim_path": i.usd_prim_path} for i in instances]}}},
        get_objects=lambda **_: {str(i): o for i, o in enumerate(instances)}, reload_env_scene=lambda: None)
    signals = FakeSignals()
    class Context:
        def __init__(self):
            self.calls = 0
        def step(self):
            self.calls += 1
            if signals.callback:
                signals.callback(.01)
            return "native-step"
    context = Context()
    env = SimpleNamespace(task_name=task, scene_manager=scene, stage=None,
        sim=SimpleNamespace(unwrapped=SimpleNamespace(sim=context)),
        _post_setup_scene=lambda sim=None: None,
        robot_manager=SimpleNamespace(robot_list=[], robot_key=[]), take_action_cnt=[0])
    return env, signals, context


class NativeProbeTests(unittest.TestCase):
    def test_passive_sampling_and_hook_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            env, signals, context = environment()
            original = context.step
            store = RecordStore(Path(directory), {"run_id": "r", "task_id": env.task_name, "episode_id": "e"})
            probe = NativeProbe(env, store)
            probe.signals = signals
            probe.before_reset()
            probe.start()
            probe.control_context("call-a", 1, 0)
            self.assertEqual(context.step(), "native-step")
            probe.control_context("call-a", 2, 1)
            context.step()
            result = probe.finish()
            self.assertEqual(result["expected_samples"], 2)
            self.assertEqual(context.calls, 2)
            self.assertEqual(context.step, original)
            self.assertIsNone(signals.callback)
            records = read_jsonl(Path(directory) / "telemetry/physics.jsonl")["records"]
            steps = [r for r in records if r["kind"] == "physics_step"]
            self.assertEqual([r["control_step"] for r in steps], [1, 2])
            self.assertTrue(all(r["action_policy_call_id"] == "call-a" for r in steps))
            context.step()
            self.assertEqual(len(read_jsonl(Path(directory) / "telemetry/physics.jsonl")["records"]), len(records))

    def test_unknown_roles_are_unverified_not_claimed_unsupported(self):
        roles, provenance = task_roles("unreviewed_task")
        self.assertEqual(roles, {})
        self.assertEqual(provenance["reason"], "task_role_mapping_not_verified")
        result = normalize({"physics_step": 1, "sim_time_s": .01, "semantic_roles_complete": False}, {}, {}, {}, InteractionSignals())
        self.assertEqual(result["signals"]["transport_tilt"]["status"], "missing")
        self.assertEqual(result["signals"]["transport_tilt"]["reason"], "task_role_mapping_not_verified")

    def test_clutter_tag_and_unlabelled_unreviewed_object(self):
        for extra, expected in [
            (SimpleNamespace(instance_config={"type": "cluttered"}, usd_prim_path="/scene/clutter1"), True),
            (SimpleNamespace(instance_config={}, usd_prim_path="/scene/unlabelled"), False),
        ]:
            with tempfile.TemporaryDirectory() as directory:
                env, signals, _ = environment(extra=extra)
                probe = NativeProbe(env, RecordStore(Path(directory), {"episode_id": "e"}))
                probe.signals = signals
                probe.before_reset()
                probe.start()
                result = probe.finish()
                self.assertEqual(result["semantic_roles_complete"], expected)
                if expected:
                    self.assertEqual(result["task_roles"]["clutter:/scene/clutter1"], "bystander")

    def test_toast_targets_follow_native_candidates_not_fixed_unused_slices(self):
        roles, provenance = task_roles("make_toast")
        self.assertEqual({name for name, role in roles.items() if role == "target"},
            {"bread_0", "bread_1", "bread_2", "bread_3", "toaster"})
        self.assertEqual(roles["bread_shelf"], "fixture")
        self.assertIn("any_two", provenance["target_selection"])

    def test_native_variable_count_prefix_roles_are_resolved_from_active_labels(self):
        roles, provenance = task_roles("arrange_largest_number", observed_labels=["digit_0", "digit_1", "mat_0", "mat_1"])
        self.assertTrue(provenance["runtime_resolved"])
        self.assertEqual(roles, {"digit_0": "target", "digit_1": "target", "mat_0": "fixture", "mat_1": "fixture"})
        _, missing = task_roles("arrange_largest_number", observed_labels=["mat_0"])
        self.assertFalse(missing["runtime_resolved"])

    def test_kong_reads_native_selection_after_initialization_without_advancing(self):
        names = [f"mahjong{i}_{j}" for i in range(4) for j in range(3)] + [f"mahjong{i}_0" for i in range(4, 10)]
        with tempfile.TemporaryDirectory() as directory:
            env, signals, context = environment("make_kong", names=names)
            env.load_support_arm_traj = lambda: self.fail("probe must not load native trajectories or sample RNG")
            probe = NativeProbe(env, RecordStore(Path(directory), {"episode_id": "e"}))
            probe.signals = signals
            probe.before_reset()
            probe.start()
            self.assertFalse(probe.semantic_roles_complete)
            baseline = dict(probe.baseline)
            env.push = ["mahjong6_0"]
            env.kong = [["mahjong1_0"], ["mahjong1_1"], ["mahjong1_2"]]
            probe.control_context("action", 1, 0)
            self.assertTrue(probe.semantic_roles_complete)
            self.assertEqual(probe.roles["mahjong1_0"], "target")
            self.assertEqual(probe.roles["mahjong0_0"], "bystander")
            self.assertEqual(probe.role_provenance["scripted_actor_labels"], ["mahjong6_0"])
            self.assertEqual(probe.baseline, baseline)
            self.assertEqual(context.calls, 0)
            probe.finish()

    def test_garment_role_is_known_but_rigid_transport_signal_is_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            env, signals, context = environment("fold_clothes", names=[])
            env.scene_manager.config["env0"]["Garment"] = {"cloth": [{"label": "target", "prim_path": "/scene/cloth"}]}
            probe = NativeProbe(env, RecordStore(Path(directory), {"episode_id": "e"}))
            probe.signals = signals
            probe.before_reset()
            probe.start()
            context.step()
            result = probe.finish()
            self.assertTrue(result["semantic_roles_complete"])
            self.assertEqual(result["unsupported_object_types"], ["Garment"])
            self.assertEqual(next(r for r in result["constraints"] if r["family_id"] == "transport_tilt")["status"], "unsupported")

    def test_expected_motion_is_annotated_without_unapproved_exemption(self):
        roles, info = task_roles("play_tic_tac_toe")
        self.assertEqual(roles["opponent_piece0"], "bystander")
        self.assertIn("opponent_piece0", info["scripted_actor_labels"])
        self.assertIn("bystander_displacement", info["applicability_pending_user_decision"])
        _, pour = task_roles("pour_balls_into_vase")
        self.assertEqual(pour["applicability_pending_user_decision"], ["transport_tilt"])
        conveyor, info = task_roles("match_and_pick_from_conveyor")
        self.assertEqual(conveyor["target0"], "bystander")
        self.assertEqual(conveyor["target1"], "target")
        self.assertTrue(info["expected_environment_motion"])

    def test_equation_uses_all_admissible_native_operands_without_preselecting_one(self):
        parser = SimpleNamespace(get_label_by_prefix=lambda prefix: [["mat0", "mat1"]] if prefix == "mat" else [["t0"]])
        env = SimpleNamespace(reward_manager=SimpleNamespace(func_parser=parser),
            find_missing_list=lambda mats, targets: [{"missing_list": ["num_1", "num_7"], "missing_mat": "mat1"}])
        roles, info = task_roles("solve_equation", env=env,
            observed_labels=["num_1", "num_7", "num_2", "plus", "equal", "t0", "mat0", "mat1"])
        self.assertTrue(info["runtime_resolved"])
        self.assertEqual({name for name, role in roles.items() if role == "target"}, {"num_1", "num_7"})
        self.assertEqual(roles["t0"], "bystander")
        self.assertEqual(roles["plus"], "bystander")

    def test_report_limits_self_collision_claim_to_two_target_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            env, signals, _ = environment()
            env.robot_manager.robot_list.append(SimpleNamespace(type="support", arm_name="support_arm0"))
            probe = NativeProbe(env, RecordStore(Path(directory), {"episode_id": "e"}))
            probe.signals = signals
            probe.before_reset()
            probe.start()
            result = probe.finish()
            scope = result["self_collision_scope"]
            self.assertEqual(scope["additional_robot_names"], ["support_arm0"])
            self.assertEqual(scope["additional_robot_pairs"], "not_evaluated_by_this_probe")
            self.assertFalse(scope["all_robot_self_collision_verified"])
            self.assertEqual(len(scope["robot_roots"]), 2)
            constraint = next(r for r in result["constraints"] if r["family_id"] == "self_collision")
            self.assertEqual(constraint["signal_scope"], scope)


if __name__ == "__main__":
    unittest.main()
