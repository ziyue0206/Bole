"""CPU contract tests using synthetic source/layout files, never the simulator."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from bole.benchmarks.robodojo.catalog import Catalog, CatalogError
from bole.engine.planning import PlanningError, plan


# The shape matches A's 12 paired + 30 standalone logical tasks, while names and
# bodies remain fixtures.  A separate test optionally checks the real checkout.
DIMENSIONS = {
    "Generalization": ["make_toast", "hang_mugs"] + [f"paired_{i}" for i in range(10)],
    "Precision": [f"precision_{i}" for i in range(8)],
    "Long-Horizon": [f"long_horizon_{i}" for i in range(8)],
    "Memory": [f"memory_{i}" for i in range(6)],
    "Open": [f"open_{i}" for i in range(8)],
}


def fixture(root, dimensions=None, quota_override=None, seeds=(0, 1, 2)):
    dimensions = dimensions or DIMENSIONS
    source, assets = root / "source", root / "assets"
    native = source / "task/RoboDojo"
    (native / "tasks").mkdir(parents=True)
    (native / "config").mkdir()
    (native / "task_registry.py").write_text("# registry metadata fixture\n")
    summary = source / "scripts/internal/summarize_result.py"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "raise RuntimeError('Planning must not execute source code')\n"
        f"DIMENSIONS = {dimensions!r}\nEXPECTED_SEEDS = {list(seeds)!r}\n"
    )
    paired = set(dimensions.get("Generalization", []))
    tasks = {name for names in dimensions.values() for name in names}
    tasks.update(name + "_random" for name in paired)
    settings = {}
    for task in sorted(tasks):
        logical = task.removesuffix("_random")
        quota = quota_override if quota_override is not None else 25 if logical in paired else 50
        settings[task] = {"eval_nums": quota}
        (native / "tasks" / (task + ".py")).write_text(
            "raise RuntimeError('Do not import native task modules')\n"
            "class Common:\n"
            "    def gen_instruction(self, env_idx):\n"
            "        templates = ['Use the native task instruction.']\n"
            "        return templates\n"
            f"class {task}(Common):\n    pass\n"
        )
        (native / "config" / (task + ".yml")).write_text("{}\n")
        for seed in seeds:
            bank = assets / "Eval_Layout/RoboDojo/arx_x5" / str(seed)
            bank.mkdir(parents=True, exist_ok=True)
            # Extra disk layouts are reserve instances, not native quotas.
            for index in range(quota + 2):
                (bank / f"{task}_{index}.json").write_text("{}\n")
    (native / "config/_task.yml").write_text(
        yaml.safe_dump({"common": {"eval_nums": 50}, "tasks": settings})
    )
    return Catalog(source, assets)


def config(**overrides):
    result = {
        "benchmark": {"name": "robodojo", "tasks": "all", "suite": None, "seeds": [0, 1, 2]},
        "mode": "eval",
        "episodes": None,
        "termination": {"success_grace_steps": 20},
    }
    result.update(overrides)
    return result


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.catalog = fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def selected(self, *names, **overrides):
        value = config(**overrides)
        value["benchmark"]["tasks"] = list(names)
        return value

    def test_full_native_plan_covers_54_runnable_42_logical_and_6300_episodes(self):
        cfg = config()
        before = copy.deepcopy(cfg)
        result = plan(cfg, self.catalog)
        self.assertEqual(cfg, before)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(len(result["tasks"]), 54)
        self.assertEqual(len(result["episodes"]), 6300)
        self.assertEqual(result["coverage"]["selected_logical_tasks"], 42)
        self.assertTrue(result["coverage"]["full_native_coverage"])
        self.assertTrue(result["coverage"]["configured_seed_selection_complete"])
        self.assertTrue(result["coverage"]["native_seed_coverage_complete"])
        self.assertEqual(result["coverage"]["episodes_per_seed"], {"0": 2100, "1": 2100, "2": 2100})
        self.assertEqual(result["coverage"]["runnable_tasks_per_seed"], {"0": 54, "1": 54, "2": 54})
        self.assertEqual(result["coverage"]["planned_task_seed_cells"], 162)
        self.assertEqual(result["coverage"]["missing_native_task_seed_cells"], [])
        self.assertEqual(result["coverage"]["native_reference_episodes"], 6300)
        self.assertFalse(result["protocol"]["official_verified"])
        self.assertFalse(result["protocol"]["is_official_termination"])
        self.assertTrue(all(row["predecessor"] is None for row in result["episodes"]))
        self.assertEqual(len({row["episode_id"] for row in result["episodes"]}), 6300)

    def test_single_runnable_task_does_not_implicitly_add_random_variant(self):
        result = plan(self.selected("make_toast"), self.catalog)
        self.assertEqual([row["id"] for row in result["tasks"]], ["make_toast"])
        self.assertEqual(len(result["episodes"]), 75)
        self.assertEqual([row["seed"] for row in result["episodes"]], [0] * 25 + [1] * 25 + [2] * 25)
        self.assertEqual(result["coverage"]["per_task"]["make_toast"]["episodes_per_seed"],
                         {"0": 25, "1": 25, "2": 25})
        self.assertFalse(result["coverage"]["full_native_coverage"])

    def test_suite_and_variant_metadata(self):
        cfg = config()
        cfg["benchmark"]["suite"] = "Generalization"
        result = plan(cfg, self.catalog)
        self.assertEqual(len(result["tasks"]), 24)
        self.assertEqual(len(result["episodes"]), 1800)
        random = next(row for row in result["tasks"] if row["id"] == "make_toast_random")
        self.assertEqual(random["logical_task_id"], "make_toast")
        self.assertEqual(random["variant"], "random")

    def test_episodes_override_is_per_task_total_not_per_seed(self):
        cfg = self.selected("make_toast", "hang_mugs", episodes=5, episodes_selection="seed_round_robin")
        result = plan(cfg, self.catalog)
        self.assertEqual(len(result["episodes"]), 10)
        for task in ["make_toast", "hang_mugs"]:
            rows = [row for row in result["episodes"] if row["task_id"] == task]
            self.assertEqual([row["seed"] for row in rows], [0, 1, 2, 0, 1])
            self.assertEqual([row["layout_id"] for row in rows], [0, 0, 0, 1, 1])
        self.assertFalse(result["coverage"]["native_instance_coverage_complete"])

    def test_confirmed_native_prefix_default_and_native_pool_limit(self):
        cfg = self.selected("make_toast", episodes=5)
        result = plan(cfg, self.catalog)
        self.assertEqual([row["seed"] for row in result["episodes"]], [0] * 5)
        self.assertEqual(result["protocol"]["episodes_selection"], "native_prefix")
        coverage = result["coverage"]
        self.assertTrue(coverage["configured_seed_selection_complete"])
        self.assertFalse(coverage["native_seed_coverage_complete"])
        self.assertEqual(coverage["configured_seeds"], [0, 1, 2])
        self.assertEqual(coverage["planned_seeds"], [0])
        self.assertEqual(coverage["episodes_per_seed"], {"0": 5, "1": 0, "2": 0})
        self.assertEqual(coverage["runnable_tasks_per_seed"], {"0": 1, "1": 0, "2": 0})
        self.assertEqual(coverage["per_task"]["make_toast"]["planned_seeds"], [0])
        self.assertFalse(coverage["per_task"]["make_toast"]["native_seed_coverage_complete"])
        self.assertEqual(coverage["missing_native_task_seed_cells"], [
            {"task_id": "make_toast", "seed": 1}, {"task_id": "make_toast", "seed": 2}])
        cfg["episodes"] = 76
        with self.assertRaisesRegex(PlanningError, "provide explicit instances"):
            plan(cfg, self.catalog)

    def test_seed_coverage_tracks_cross_seed_prefix_without_claiming_full_instances(self):
        result = plan(self.selected("make_toast", episodes=30), self.catalog)
        coverage = result["coverage"]
        self.assertEqual(coverage["planned_seeds"], [0, 1])
        self.assertEqual(coverage["episodes_per_seed"], {"0": 25, "1": 5, "2": 0})
        self.assertFalse(coverage["native_seed_coverage_complete"])
        result = plan(self.selected("make_toast", episodes=51), self.catalog)
        self.assertTrue(result["coverage"]["native_seed_coverage_complete"])
        self.assertFalse(result["coverage"]["native_instance_coverage_complete"])

    def test_seed_coverage_requires_each_selected_task_not_just_global_seed_union(self):
        cfg = self.selected("make_toast", "hang_mugs")
        cfg["benchmark"]["instances"] = {
            "make_toast": [{"seed": seed, "layout_id": 0} for seed in (0, 1, 2)],
            "hang_mugs": [{"seed": 0, "layout_id": 0}],
        }
        coverage = plan(cfg, self.catalog)["coverage"]
        self.assertEqual(coverage["planned_seeds"], [0, 1, 2])
        self.assertTrue(coverage["per_task"]["make_toast"]["native_seed_coverage_complete"])
        self.assertFalse(coverage["per_task"]["hang_mugs"]["native_seed_coverage_complete"])
        self.assertFalse(coverage["native_seed_coverage_complete"])
        self.assertEqual(coverage["runnable_tasks_per_seed"], {"0": 2, "1": 1, "2": 1})
        self.assertEqual(coverage["planned_task_seed_cells"], 4)
        self.assertEqual(coverage["native_reference_task_seed_cells_selected_tasks"], 6)
        self.assertEqual(coverage["missing_native_task_seed_cells"], [
            {"task_id": "hang_mugs", "seed": 1}, {"task_id": "hang_mugs", "seed": 2}])

    def test_configured_seed_subset_is_separate_from_actual_coverage(self):
        cfg = self.selected("make_toast", episodes=5)
        cfg["benchmark"]["seeds"] = [2, 0]
        coverage = plan(cfg, self.catalog)["coverage"]
        self.assertEqual(coverage["configured_seeds"], [2, 0])
        self.assertEqual(coverage["planned_seeds"], [2])
        self.assertFalse(coverage["configured_seed_selection_complete"])
        self.assertFalse(coverage["native_seed_coverage_complete"])
        self.assertEqual(coverage["per_task"]["make_toast"]["missing_native_seeds"], [0, 1])

    def test_learn_chains_link_only_the_same_task(self):
        cfg = self.selected("make_toast", "hang_mugs", mode="learn", episodes=3,
                            episodes_selection="seed_round_robin")
        result = plan(cfg, self.catalog)
        for task in ["make_toast", "hang_mugs"]:
            rows = [row for row in result["episodes"] if row["task_id"] == task]
            self.assertEqual([row["predecessor"] for row in rows],
                             [None, rows[0]["episode_id"], rows[1]["episode_id"]])
            self.assertTrue(all(row["learning_chain_id"] == task for row in rows))

    def test_explicit_instances_preserve_order_and_intentional_repeats(self):
        cfg = self.selected("make_toast", mode="learn", episodes=3)
        cfg["benchmark"]["instances"] = {"make_toast": [
            {"seed": 2, "layout_id": 4}, {"seed": 0, "layout_id": 0}, {"seed": 2, "layout_id": 4},
        ]}
        result = plan(cfg, self.catalog)
        self.assertEqual([(row["seed"], row["layout_id"]) for row in result["episodes"]],
                         [(2, 4), (0, 0), (2, 4)])
        cfg["episodes"] = 4
        with self.assertRaisesRegex(PlanningError, "need 4, found 3"):
            plan(cfg, self.catalog)

    def test_explicit_instances_are_fully_validated(self):
        cfg = self.selected("make_toast", episodes=1)
        cfg["benchmark"]["instances"] = {"make_toast": [
            {"seed": 0, "layout_id": 0}, {"seed": 0, "layout_id": 9999},
        ]}
        with self.assertRaisesRegex(PlanningError, "unavailable"):
            plan(cfg, self.catalog)
        cfg["benchmark"]["instances"]["make_toast"][1] = {"seed": 42, "layout_id": 0}
        with self.assertRaisesRegex(PlanningError, "not selected"):
            plan(cfg, self.catalog)

    def test_unknown_task_suite_and_invalid_counts_are_errors(self):
        for cfg in [self.selected("missing"), self.selected("make_toast", episodes=True),
                    self.selected("make_toast", episodes=0)]:
            with self.assertRaises(PlanningError):
                plan(cfg, self.catalog)
        cfg = self.selected("make_toast")
        cfg["benchmark"]["suite"] = "Memory"
        with self.assertRaisesRegex(PlanningError, "outside suite"):
            plan(cfg, self.catalog)
        cfg["benchmark"]["suite"] = "not-a-suite"
        with self.assertRaisesRegex(PlanningError, "Unknown RoboDojo suite"):
            plan(cfg, self.catalog)

    def test_asset_matching_is_strict_and_shared_mode_is_traceable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "source_task_id": "make_toast",
                                        "source_run_id": "learn-1", "status": "complete", "export_status": "saved"}))
            cfg = self.selected("make_toast", "hang_mugs", episodes=1, episodes_selection="native_prefix")
            cfg["learning_assets"] = {"mode": "by_task", "by_task": {"make_toast": str(path)}}
            with self.assertRaisesRegex(PlanningError, "hang_mugs"):
                plan(cfg, self.catalog)
            cfg["learning_assets"]["by_task"]["hang_mugs"] = str(path)
            with self.assertRaisesRegex(PlanningError, "differs from target"):
                plan(cfg, self.catalog)
            cfg["learning_assets"] = {"mode": "shared", "shared": str(path)}
            result = plan(cfg, self.catalog)
            assets = [row["input_asset"] for row in result["episodes"]]
            self.assertEqual([row["target_task_id"] for row in assets], ["make_toast", "hang_mugs"])
            self.assertTrue(all(row["source_task_id"] == "make_toast" for row in assets))
            self.assertTrue(all(row["restore"]["resume_history"] is False for row in assets))
            self.assertEqual(assets[0]["manifest_path"], str(path.resolve()))
            path.unlink()
            with self.assertRaisesRegex(PlanningError, "Cannot read learning asset"):
                plan(cfg, self.catalog)

    def test_asset_references_default_to_task_matching_but_shared_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "source_task_id": "make_toast",
                                        "status": "complete", "export_status": "saved"}))
            cfg = self.selected("make_toast", episodes=1)
            cfg["learning_assets"] = {"mode": None, "by_task": {"make_toast": str(path)}}
            asset = plan(cfg, self.catalog)["episodes"][0]["input_asset"]
            self.assertEqual(asset["selection_mode"], "by_task")
            self.assertEqual(asset["source_task_id"], asset["target_task_id"])
            cfg["benchmark"]["tasks"] = ["make_toast", "hang_mugs"]
            with self.assertRaisesRegex(PlanningError, "hang_mugs"):
                plan(cfg, self.catalog)
            cfg["learning_assets"] = {"mode": None, "shared": str(path)}
            with self.assertRaisesRegex(PlanningError, "mode=shared"):
                plan(cfg, self.catalog)
            cfg["benchmark"]["tasks"] = ["make_toast"]
            cfg["learning_assets"] = {"by_task": {"make_toast": str(path)}}
            manifest = json.loads(path.read_text())
            manifest["finalize_status"] = "pending"
            path.write_text(json.dumps(manifest))
            self.assertEqual(plan(cfg, self.catalog)["episodes"][0]["input_asset"]["source_task_id"], "make_toast")


    def test_multiple_learning_tasks_need_distinct_output_names(self):
        cfg = self.selected('make_toast', episodes=1)
        cfg['mode'] = 'learn'
        cfg['benchmark']['tasks'] = ['make_toast','hang_mugs']
        cfg['learn'] = {'save_image':'fixture:same-name'}
        with self.assertRaisesRegex(PlanningError, 'containing {task}'):
            plan(cfg, self.catalog)
        cfg['learn']['save_image'] = 'fixture:{task}'
        self.assertEqual(len(plan(cfg, self.catalog)['tasks']), 2)


class CatalogValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.catalog = fixture(self.root, {"Precision": ["test_task"]}, quota_override=2)

    def tearDown(self):
        self.directory.cleanup()

    def test_numeric_order_and_layout_id_are_distinct_from_filename_suffix(self):
        bank = self.catalog.assets / "Eval_Layout/RoboDojo/arx_x5/0"
        for path in bank.iterdir():
            path.unlink()
        for name in ["test_task_30.json", "test_task_2.json", "test_task_10.json", "test_task_random_1.json"]:
            (bank / name).write_text("{}")
        rows = self.catalog.instances("test_task", 0)
        self.assertEqual([row["layout_id"] for row in rows], [0, 1, 2])
        self.assertEqual([Path(row["path"]).name for row in rows],
                         ["test_task_2.json", "test_task_10.json", "test_task_30.json"])

    def test_missing_bank_and_insufficient_layouts_fail_before_execution(self):
        with self.assertRaisesRegex(CatalogError, "Missing RoboDojo layout bank"):
            self.catalog.instances("test_task", 8)
        bank = self.catalog.assets / "Eval_Layout/RoboDojo/arx_x5/0"
        for path in bank.iterdir():
            if path.name != "test_task_0.json":
                path.unlink()
        with self.assertRaisesRegex(PlanningError, "Insufficient native layouts"):
            plan(config(), self.catalog)

    def test_empty_layout_and_ambiguous_numeric_filename_are_errors(self):
        bank = self.catalog.assets / "Eval_Layout/RoboDojo/arx_x5/0"
        (bank / "test_task_0.json").write_text("")
        with self.assertRaisesRegex(CatalogError, "empty"):
            self.catalog.instances("test_task", 0)
        (bank / "test_task_0.json").write_text("{}")
        (bank / "test_task_00.json").write_text("{}")
        with self.assertRaisesRegex(CatalogError, "Ambiguous"):
            self.catalog.instances("test_task", 0)

    def test_missing_task_implementation_is_not_silently_skipped(self):
        (self.catalog.source / "task/RoboDojo/tasks/test_task.py").unlink()
        with self.assertRaisesRegex(CatalogError, "Missing native task implementations"):
            self.catalog.tasks()

    def test_description_is_native_literal_or_runtime_required(self):
        task = self.catalog.tasks()[0]
        self.assertEqual(task["description"], "Use the native task instruction.")
        path = self.catalog.source / "task/RoboDojo/tasks/test_task.py"
        path.write_text("class test_task:\n    def gen_instruction(self, i):\n        return [self.hidden_scene[i]]\n")
        task = Catalog(self.catalog.source, self.catalog.assets).tasks()[0]
        self.assertIsNone(task["description"])
        self.assertEqual(task["description_status"], "runtime_required")


if __name__ == "__main__":
    unittest.main()
