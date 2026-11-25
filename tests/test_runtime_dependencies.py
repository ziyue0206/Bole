import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path

from bole.config import load
from bole.runtime.dependencies import codex_bwrap_mount, codex_security_args
from bole.runtime.codex_bootstrap import prepare_workspace


class RuntimeDependenciesTests(unittest.TestCase):
    def test_config_relative_helper_and_readonly_mount(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            helper = root / "bwrap"
            helper.write_bytes(b"fixture")
            helper.chmod(0o755)
            config_file = root / "run.yaml"
            config_file.write_text("resources:\n  codex_bwrap: bwrap\n")
            config = load(config_file, resources_path=root / "absent")
            self.assertEqual(config["resources"]["codex_bwrap"], str(helper))
            mount = codex_bwrap_mount(config["resources"]["codex_bwrap"], "/opt/coding-bin/codex")
            self.assertEqual(mount, ["--mount", f"type=bind,src={helper},dst=/opt/coding-bin/codex-resources/bwrap,readonly"])
            helper.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "executable regular"):
                codex_bwrap_mount(helper, "/opt/coding-bin/codex")

    def test_missing_dependency_does_not_silently_launch_without_it(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                codex_bwrap_mount(Path(temp) / "missing", "/opt/coding-bin/codex")
        self.assertEqual(codex_bwrap_mount(None, "/opt/coding-bin/codex"), [])

    def test_named_security_profiles_preserve_capability_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            policy = Path(temp) / "seccomp.json"
            policy.write_text('{"defaultAction":"SCMP_ACT_ERRNO","syscalls":[]}')
            result = codex_security_args({"codex_seccomp": str(policy), "codex_apparmor": "bole-codex-bwrap"})
            self.assertEqual(result, ["--security-opt", f"seccomp={policy.resolve()}",
                                      "--security-opt", "apparmor=bole-codex-bwrap"])
            with self.assertRaises(ValueError):
                codex_security_args({"codex_apparmor": "unconfined"})
            policy.unlink()
            with self.assertRaises(FileNotFoundError):
                codex_security_args({"codex_seccomp": str(policy)})

    def test_existing_workspace_environment_is_preserved_and_global_pip_is_guarded(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=False):
            workspace = Path(temp)
            environment = workspace / ".venv"
            (environment / "bin").mkdir(parents=True)
            (environment / "pyvenv.cfg").write_text("fixture")
            (environment / "bin/python").write_text("fixture")
            (environment / "keep.txt").write_text("installed dependency")
            prepare_workspace(workspace)
            self.assertEqual((environment / "keep.txt").read_text(), "installed dependency")
            self.assertEqual(os.environ["VIRTUAL_ENV"], str(environment))
            self.assertEqual(os.environ["PIP_REQUIRE_VIRTUALENV"], "true")
            self.assertEqual(os.environ["PATH"].split(":")[0], str(environment / "bin"))
            skills = workspace / '.agents/skills'
            self.assertTrue(skills.is_dir())
            (skills / 'existing-skill.txt').write_text('keep autonomous content')
            prepare_workspace(workspace)
            self.assertEqual((skills / 'existing-skill.txt').read_text(), 'keep autonomous content')
            (environment / "pyvenv.cfg").unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                prepare_workspace(workspace)
