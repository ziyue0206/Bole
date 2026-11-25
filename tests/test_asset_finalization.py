"""Regression coverage for salvaging, preserving and finalizing native state."""
import asyncio
import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from bole.config import DEFAULTS
from bole.runtime.learning_assets import (archive_tree, extract_tree, export_asset,
                                          load_asset, snapshot_home)
from bole.runtime.snapshots import freeze


class AssetPreservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stopped_container_still_saves_workspace_memory_and_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "work").mkdir()
            (root / "work/learned.py").write_text("learned = True")
            (root / "home/memories").mkdir(parents=True)
            (root / "home/memories/memory_summary.md").write_text("native fixture")
            async def command(*args):
                if args[1] == "pause":
                    raise RuntimeError("container is not running")
                if args[1] == "inspect":
                    return json.dumps([{"State": {"Running": False}}])
                if args[1] == "commit":
                    return "sha256:" + "a" * 64
                if args[1:3] == ("image", "inspect"):
                    return json.dumps([{"Id": "sha256:" + "a" * 64}])
                if args[1] in {"create", "cp", "tag", "rm"} or args[1:3] == ("image", "ls"):
                    return ""
                raise AssertionError(args)
            m = await export_asset(root / "asset", container="stopped", workspace=root / "work",
                home=root / "home", metadata={"status": "partial"}, command=command,
                base_image="base", image_name="fixture:partial")
            self.assertEqual(m["export_status"], "saved")
            self.assertEqual(m["consistency"], "quiescent_snapshot")
            self.assertEqual(m["status"], "partial")
            self.assertEqual(set(m["components"]), {"workspace", "home", "image"})

    async def test_missing_container_still_salvages_readable_components(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "work").mkdir()
            (root / "home").mkdir()
            async def command(*args):
                raise RuntimeError("container gone")
            m = await export_asset(root / "asset", container="gone", workspace=root / "work",
                home=root / "home", metadata={}, command=command)
            self.assertEqual(m["export_status"], "failed")
            self.assertEqual(m["consistency"], "best_effort_unverified")
            self.assertTrue((root / "asset/workspace.tar.gz").is_file())
            self.assertTrue((root / "asset/home.tar.gz").is_file())

    async def test_persistent_state_is_saved_without_raw_history_or_a_memory_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "work").mkdir()
            (root / "home/sessions").mkdir(parents=True)
            (root / "home/sessions/rollout.jsonl").write_text('{"fixture":true}\n')
            (root / "home/auth.json").write_text('{"api_key":"must not export"}')
            with sqlite3.connect(root / "home/state_5.sqlite") as db:
                db.execute("CREATE TABLE threads(id TEXT, updated_at INTEGER)")
                db.execute("INSERT INTO threads VALUES ('source-thread',123)")
            async def command(*args):
                if args[1] == "commit": return "sha256:" + "b" * 64
                if args[1:3] == ("image", "inspect"): return '[{}]'
                return ""
            m = await export_asset(root / "asset", container="owned", workspace=root / "work",
                home=root / "home", metadata={"status": "complete"}, command=command,
                base_image="base", image_name="fixture:saved")
            self.assertEqual(m["export_status"], "saved")
            self.assertNotIn("finalize_status", m)
            manifest = root / "asset/manifest.json"
            load_asset(manifest, root / "eval-work", root / "eval-home")
            restored = root / "eval-home"
            self.assertFalse((restored / "sessions/rollout.jsonl").exists())
            self.assertFalse((restored / "auth.json").exists())
            with sqlite3.connect(restored / "state_5.sqlite") as db:
                self.assertEqual(db.execute("SELECT updated_at FROM threads").fetchone()[0], 123)

    def test_in_tree_relative_links_survive_export_import_without_dereferencing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source/lib").mkdir(parents=True)
            (root / "source/lib/module.py").write_text("value=1")
            (root / "source/module.py").symlink_to("lib/module.py")
            archive_tree(root / "source", root / "tree.tar.gz")
            extract_tree(root / "tree.tar.gz", root / "target")
            self.assertTrue((root / "target/module.py").is_symlink())
            self.assertEqual((root / "target/module.py").read_text(), "value=1")
            (root / "source/escape").symlink_to("../../outside")
            with self.assertRaises((ValueError, FileNotFoundError)):
                archive_tree(root / "source", root / "bad.tar.gz")

    def test_factory_and_roles_are_frozen_alongside_returned_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "audit_impl.py").write_text("class Adapter: pass\n")
            (root / "audit_factory.py").write_text("from audit_impl import Adapter\ndef create(): return Adapter()\n")
            sys.path.insert(0, str(root))
            try:
                import audit_factory
                prompt = root / "prompt.txt"; prompt.write_text("fixture")
                roles = root / "roles.yaml"; roles.write_text("fixture: original")
                config = copy.deepcopy(DEFAULTS)
                config["benchmark"]["name"] = "audit"
                config["harness"]["instruction"] = str(prompt)
                config["recording"]["safety"]["roles"] = str(roles)
                def discover(kind):
                    return {"audit": "audit_factory:create"} if kind == "benchmarks" else {}
                with patch("bole.runtime.snapshots.registry.discover", side_effect=discover):
                    frozen, manifest = freeze(config, root / "run", plugins=[audit_factory.create()])
                (root / "audit_factory.py").write_text("raise RuntimeError('changed')")
                roles.write_text("fixture: changed")
                plugin_root = root / "run/snapshot/plugins"
                self.assertIn("def create", (plugin_root / "audit_factory.py").read_text())
                self.assertTrue((plugin_root / "audit_impl.py").is_file())
                self.assertEqual(Path(frozen["recording"]["safety"]["roles"]).read_text(), "fixture: original")
                self.assertEqual(json.loads((root / "run/snapshot/bole/src/bole/_frozen_plugins.json").read_text())["benchmarks"],
                                 {"audit": "audit_factory:create"})
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("audit_factory", None)
                sys.modules.pop("audit_impl", None)


if __name__ == "__main__":
    unittest.main()
