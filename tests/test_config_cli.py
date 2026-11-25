"""Configuration precedence and frozen-launch tests; no network or simulator."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from bole.cli import overrides, parser, main, launch
from bole.config import DEFAULTS, load
import bole.config as config_module
from bole.runtime.snapshots import freeze, source_provenance


class ConfigCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def test_absent_cli_fields_do_not_replace_yaml(self):
        path = self.root / "config.yaml"
        path.write_text("episodes: 9\nbenchmark:\n  seeds: [2]\nharness:\n  model: yaml-model\nlearn:\n  parallel: 3\n")
        args = parser().parse_args(["learn", "--config", str(path)])
        self.assertEqual(overrides(args), {})
        config = load(path, overrides(args), mode="learn", resources_path=self.root / "absent")
        self.assertEqual(config["episodes"], 9)
        self.assertEqual(config["benchmark"]["seeds"], [2])
        self.assertEqual(config["harness"]["model"], "yaml-model")
        self.assertEqual(config["learn"]["parallel"], 3)

    def test_explicit_cli_fields_win_and_repeated_fields_replace_yaml(self):
        path = self.root / "config.yaml"
        path.write_text("episodes: 9\nbenchmark:\n  seeds: [2]\n  tasks: [old_task]\n")
        args = parser().parse_args(["eval", "--config", str(path), "--episodes", "3", "--task", "one", "--task", "two",
                                   "--seed", "0", "--seed", "1", "--success-grace-steps", "0", "--parallel", "4", "--gpu", "2"])
        config = load(path, overrides(args), resources_path=self.root / "absent")
        self.assertEqual(config["episodes"], 3)
        self.assertEqual(config["benchmark"]["tasks"], ["one", "two"])
        self.assertEqual(config["benchmark"]["seeds"], [0, 1])
        self.assertEqual(config["termination"]["success_grace_steps"], 0)
        self.assertEqual(config["eval"]["parallel"], 4)
        self.assertEqual(config["resources"]["gpus"], [2])

    def test_yaml_paths_are_relative_to_config_and_cli_paths_to_cwd(self):
        path = self.root / "config.yaml"
        path.write_text("output: outputs\nharness:\n  instruction: prompt.txt\nresources:\n  source: native\n")
        config = load(path, {"output": "cli-output"}, resources_path=self.root / "absent")
        self.assertEqual(config["output"], str(Path.cwd() / "cli-output"))
        self.assertEqual(config["harness"]["instruction"], str(self.root / "prompt.txt"))
        self.assertEqual(config["resources"]["source"], str(self.root / "native"))

    def test_native_defaults_and_explicit_false_are_preserved(self):
        args = parser().parse_args(["eval", "--no-video", "--no-memory"])
        config = load(overrides=overrides(args), resources_path=self.root / "absent")
        self.assertEqual(config["benchmark"]["seeds"], [0, 1, 2])
        self.assertIsNone(config["episodes"])
        self.assertEqual(config["episodes_selection"], "native_prefix")
        self.assertFalse(config["recording"]["video"]["enabled"])
        self.assertFalse(config["harness"]["memory"]["enabled"])

    def test_policy_and_learning_asset_mapping_are_explicit(self):
        args = parser().parse_args(["eval", "--policy", "external", "--asset-mode", "by_task",
                                   "--task-asset", "a=one.json", "--task-asset", "b=two.json"])
        values = overrides(args)
        self.assertEqual(values["policy"], {"name": "external"})
        self.assertEqual(values["learning_assets"], {"mode": "by_task", "by_task": {"a": "one.json", "b": "two.json"}})
        with self.assertRaises(ValueError):
            overrides(parser().parse_args(["eval", "--task", "all", "--task", "a"]))

    def test_unknown_yaml_and_duplicate_gpu_fail_before_runtime(self):
        path = self.root / "bad.yaml"
        path.write_text("unknown_budget: 2\n")
        with self.assertRaisesRegex(ValueError, "unknown configuration"):
            load(path, resources_path=self.root / "absent")
        with self.assertRaisesRegex(ValueError, "duplicate GPU"):
            load(overrides={"resources": {"gpus": [0, 0]}}, resources_path=self.root / "absent")

    def test_plan_and_check_do_not_launch_or_open_a_harness(self):
        class Adapter:
            requires_gpu = True
            def check(self, config):
                return {"ok": True}
            def plan(self, config):
                return {"episodes": [], "schema_version": 1}
        with patch("bole.cli.registry.load", return_value=Adapter()) as loader, patch("bole.cli.launch") as launch, patch("builtins.print"):
            self.assertEqual(main(["plan"]), 0)
            self.assertEqual(main(["check"]), 0)
        self.assertEqual([call.args[0] for call in loader.call_args_list], ["benchmarks", "benchmarks"])
        launch.assert_not_called()

    def test_snapshot_copies_code_prompt_and_minimal_native_dependencies_not_assets(self):
        project = self.root / "project"
        package = project / "src/bole"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VERSION = 'frozen'\n")
        (project / "prompts").mkdir()
        prompt = project / "prompts/instruction.md"
        prompt.write_text("frozen prompt")
        source = self.root / "native"
        for relative in ("env", "task", "XPolicyLab/client_server", "XPolicyLab/giant_training_data", "Assets"):
            (source / relative).mkdir(parents=True)
            (source / relative / "content.py").write_text("original")
        config = copy.deepcopy(DEFAULTS)
        config["resources"]["source"] = str(source)
        config["harness"]["instruction"] = str(prompt)
        seccomp = self.root / "seccomp.json"
        seccomp.write_text('{"defaultAction":"SCMP_ACT_ERRNO","syscalls":[]}')
        config["resources"]["codex_seccomp"] = str(seccomp)
        result, manifest = freeze(config, self.root / "run", project_root=project)
        (package / "__init__.py").write_text("VERSION = 'edited'\n")
        prompt.write_text("edited prompt")
        seccomp.write_text('{"changed_after_start":true}')
        self.assertEqual(json.loads(Path(result["resources"]["codex_seccomp"]).read_text())["syscalls"], [])
        self.assertEqual(Path(result["harness"]["instruction"]).read_text(), "frozen prompt")
        self.assertIn("frozen", (Path(manifest["python_paths"][0]) / "bole/__init__.py").read_text())
        native = Path(result["resources"]["source"])
        self.assertTrue((native / "XPolicyLab/client_server/content.py").exists())
        self.assertFalse((native / "XPolicyLab/giant_training_data").exists())
        self.assertFalse((native / "Assets").exists())
        self.assertFalse(manifest["asset_copy"])
        self.assertEqual(config["resources"]["source"], str(source))

    def test_launch_executes_frozen_package_after_original_source_changes(self):
        project = self.root / "project"
        engine = project / "src/bole/engine"
        engine.mkdir(parents=True)
        (engine.parent / "__init__.py").write_text("")
        (engine / "__init__.py").write_text("")
        child = engine / "runner.py"
        child.write_text("import json,sys,pathlib\nargs=dict(zip(sys.argv[1::2],sys.argv[2::2]))\n"
                         "config=json.loads(pathlib.Path(args['--child-config']).read_text())\n"
                         "pathlib.Path(args['--run-dir'],'child-marker.txt').write_text('frozen:' + "
                         "pathlib.Path(config['harness']['instruction']).read_text())\n")
        prompt = project / "prompt.txt"
        prompt.write_text("prompt-v1")
        config = copy.deepcopy(DEFAULTS)
        config["benchmark"]["name"] = "fixture"
        config["harness"]["instruction"] = str(prompt)
        config["output"] = str(self.root / "runs")
        def frozen_copy(config, root, plugins):
            result = freeze(config, root, project_root=project)
            child.write_text("raise RuntimeError('mutable source was executed')\n")
            prompt.write_text("prompt-v2")
            return result
        class Adapter:
            def check(self, config):
                return {"ok": True}
        with patch("bole.cli.freeze", side_effect=frozen_copy), patch("builtins.print"):
            code = launch(config, {"episodes": []}, Adapter(), object())
        self.assertEqual(code, 0)
        outputs = list((self.root / "runs").glob("*/child-marker.txt"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].read_text(), "frozen:prompt-v1")

    def test_installed_package_freezes_without_a_source_tree(self):
        package = self.root / "site-packages/bole"
        (package / "runtime").mkdir(parents=True)
        (package / "__init__.py").write_text("installed_code = True\n")
        (package / "prompts").mkdir()
        prompt = package / "prompts/codex.md"
        prompt.write_text("bundled wheel prompt")
        config = copy.deepcopy(DEFAULTS)
        config["benchmark"]["name"] = "fixture"
        config["harness"]["instruction"] = str(prompt)
        with patch("bole.runtime.snapshots.__file__", str(package / "runtime/snapshots.py")):
            frozen, manifest = freeze(config, self.root / "wheel-run")
        self.assertEqual(manifest["code"][0]["import_mode"], "installed_package")
        self.assertIsNone(manifest["code"][0]["source_tree"])
        self.assertEqual((Path(manifest["python_paths"][0]) / "bole/__init__.py").read_text(), "installed_code = True\n")
        self.assertEqual(Path(frozen["harness"]["instruction"]).read_text(), "bundled wheel prompt")

    def test_frozen_source_does_not_inherit_enclosing_bole_git_revision(self):
        outer = self.root / "bole"
        (outer / ".git").mkdir(parents=True)
        source = outer / "runs/a/snapshot/benchmark"
        source.mkdir(parents=True)
        with patch("bole.runtime.snapshots.subprocess.run") as git:
            provenance = source_provenance(source, "declared-A-revision")
        git.assert_not_called()
        self.assertIsNone(provenance["observed_git_revision"])
        self.assertEqual(provenance["declared_revision"], "declared-A-revision")
        self.assertEqual(provenance["verification"], "not_a_standalone_git_checkout")

    def test_copied_source_provenance_is_preserved_as_observation_of_original(self):
        source = self.root / "frozen-native"
        source.mkdir()
        record = {"schema_version": 1, "source": "/original/RoboDojo", "declared_revision": "revision-A",
                  "observed_git_revision": "revision-A", "git_changes": [],
                  "verification": "observed_source_checkout_head_and_status"}
        (source / ".bole-source-provenance.json").write_text(json.dumps(record))
        with patch("bole.runtime.snapshots.subprocess.run") as git:
            provenance = source_provenance(source, "other-declaration")
        git.assert_not_called()
        self.assertEqual(provenance["source"], "/original/RoboDojo")
        self.assertEqual(provenance["observed_git_revision"], "revision-A")
        self.assertEqual(provenance["read_from_snapshot"], str(source / ".bole-source-provenance.json"))

    def test_installed_config_defaults_use_bundled_prompt_and_working_directory(self):
        package = self.root / "site-packages/bole"
        (package / "prompts").mkdir(parents=True)
        (package / "prompts/codex.md").write_text("bundled prompt")
        installed_config = package / "config.py"
        installed_config.write_text(Path(config_module.__file__).read_text())
        working = self.root / "work"
        working.mkdir()
        (working / "resources.yaml").write_text("source: ./native-source\n")
        with patch("pathlib.Path.cwd", return_value=working):
            module = runpy.run_path(str(installed_config))
            result = module["load"]()
        self.assertEqual(module["PACKAGE_ROOT"], package)
        self.assertEqual(module["DEFAULTS"]["resources"]["source"], str(working / "vendor/robodojo"))
        self.assertTrue(module["DEFAULTS"]["harness"]["shell"]["network_access"])
        self.assertEqual(result["harness"]["instruction"], str(package / "prompts/codex.md"))
        self.assertEqual(result["output"], str(working / "runs"))
        self.assertEqual(result["resources"]["source"], str(working / "native-source"))

    def test_memory_inspect_empty_directory_is_read_only_and_enabled_is_unknown(self):
        output = io.StringIO()
        with patch("bole.cli.configuration.load") as load_config, patch("bole.cli.registry.load") as load_plugin, \
                patch("bole.cli.subprocess.Popen") as process, redirect_stdout(output):
            code = main(["memory", "inspect", "--home", str(self.root)])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["state"], "empty")
        self.assertIsNone(report["enabled"])
        self.assertFalse(report["generation_verified"])
        self.assertFalse(report["new_session_reuse_verified"])
        self.assertFalse(report["content_disclosed"])
        self.assertEqual(list(self.root.iterdir()), [])
        load_config.assert_not_called()
        load_plugin.assert_not_called()
        process.assert_not_called()

    def test_memory_inspect_native_schema_and_processing_rows_do_not_prove_generation(self):
        path = self.root / "memories_v2.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE _sqlx_migrations (version INTEGER)")
            db.execute("INSERT INTO _sqlx_migrations VALUES (1)")
            db.execute("CREATE TABLE consolidation_progress (last_consolidated_at INTEGER)")
            db.execute("INSERT INTO consolidation_progress VALUES (0)")
            db.execute("CREATE TABLE stage1_outputs (thread_id TEXT, raw_memory TEXT, rollout_summary TEXT)")
            db.execute("CREATE TABLE jobs (thread_id TEXT, status TEXT)")
        for expected in ("schema_only", "processing_state_only"):
            before = path.read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["memory", "inspect", "--home", str(self.root), "--enabled"])
            report = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(report["state"], expected)
            self.assertTrue(report["enabled"])
            self.assertEqual(report["databases"][0]["stage1_output_rows"], 0)
            self.assertEqual(report["databases"][0]["data_rows"], 0)
            self.assertFalse(report["generation_verified"])
            self.assertFalse(report["new_session_reuse_verified"])
            self.assertEqual(path.read_bytes(), before)
            if expected == "schema_only":
                with sqlite3.connect(path) as db:
                    db.execute("INSERT INTO jobs VALUES ('private thread identifier', 'pending')")
            else:
                self.assertNotIn("private thread identifier", output.getvalue())


if __name__ == "__main__":
    unittest.main()
