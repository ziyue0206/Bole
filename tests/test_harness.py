import asyncio
import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from bole.harnesses.codex import CodexHarness
from bole.harnesses.transport import ModelHTTPBridge, StreamingRedactor
from bole.results.storage import RecordStore
from bole.results.usage import UsageLedger
from bole.runtime.learning_assets import archive_tree, extract_tree, export_asset, inspect_memory, load_asset, snapshot_memory


SERVER = r'''
import json, sys
serial = 0
turn = None
def send(x):
    print(json.dumps(x), flush=True)
def complete(status):
    send({"method":"turn/completed","params":{"threadId":"thread-1","turn":{"id":turn,"status":status}}})
for raw in sys.stdin:
    m=json.loads(raw)
    method=m.get("method")
    if method == "initialize": send({"id":m["id"],"result":{}})
    elif method == "thread/start": send({"id":m["id"],"result":{"thread":{"id":"thread-1"}}})
    elif method == "thread/memory/read": send({"id":m["id"],"result":{"summary":None}})
    elif method == "memory/status": send({"id":m["id"],"result":{"v2ConsolidatedThreads":0,"v2Ready":False}})
    elif method == "turn/start":
        serial += 1
        turn = "turn-"+str(serial)
        text=m["params"]["input"][0]["text"]
        send({"id":m["id"],"result":{"turn":{"id":turn,"status":"inProgress"}}})
        send({"method":"item/agentMessage/delta","params":{"threadId":"thread-1","turnId":turn,"delta":"working"}})
        if text in ("finish", "stale"):
            send({"id":"tool-"+str(serial),"method":"item/tool/call","params":{
                "threadId":"bad" if text=="stale" else "thread-1", "turnId":turn,
                "tool":"finish","arguments":{},"callId":"call-"+str(serial)}})
            if text=="stale": complete("completed")
        elif text == "failed": complete("failed")
        elif text != "hold": complete("completed")
    elif method == "turn/interrupt":
        send({"id":m["id"],"result":{}})
        complete("interrupted")
'''


class Tools:
    inputs = []

    def __init__(self):
        self.calls = []

    def definitions(self):
        return [{"name": "finish", "description": "End this episode", "inputSchema": {"type": "object"}}]

    async def call(self, name, args, call_id, parent_id):
        self.calls.append((name, call_id))
        return {"content": [{"type": "text", "text": "Episode ended."}], "isError": False}


class NativeProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tools = Tools()
        self.store = RecordStore(self.root / "records", {"run_id": "run", "task_id": "task", "episode_id": "one"})
        config = {"harness": {"model": "fixture", "reasoning_effort": "high", "memory": {"enabled": True},
                  "gateway": {"base_url": "http://127.0.0.1:9/v1", "credential_env": "BOLE_UNUSED_TEST_KEY"}},
                  "resources": {"harness_image": "fixture"}}
        self.h = CodexHarness(native_command=[sys.executable, "-u", "-c", SERVER])
        await self.h.open({"config": config, "root": self.root / "private", "store": self.store,
                           "run_id": "run", "task_id": "task"}, self.tools)
        self.delivered = 0

    async def asyncTearDown(self):
        await self.h.close()
        self.temp.cleanup()

    async def delivered_callback(self):
        self.delivered += 1

    async def test_normal_final_response_is_not_finish_or_infrastructure_failure(self):
        result = await self.h.run_episode("done", [{"type": "text", "text": "first observation"}], self.delivered_callback, 2)
        self.assertEqual(result["termination"], "final_response_without_finish")
        self.assertFalse(result["finish_called"])
        self.assertTrue(result["session_usable"])
        self.assertEqual(self.delivered, 1)
        events = [json.loads(line) for line in (self.store.root / "model-events/codex-native.jsonl").read_text().splitlines()]
        self.assertTrue(any(e["message"].get("method") == "item/agentMessage/delta" for e in events))
        start = next(e["message"]["params"] for e in events if e["message"].get("method") == "thread/start")
        self.assertNotIn("environments", start, "omit environments so native shell/code execution remains configured")
        methods = [e['message'].get('method') for e in events if e['direction']=='client']
        self.assertNotIn('thread/memory/read', methods)
        self.assertNotIn('memory/status', methods)
        self.assertNotIn('thread/resume', methods)
        native = json.loads((self.store.root/'model-events/native-config.json').read_text())['config']
        self.assertEqual(native['sandbox_mode'], 'workspace-write')
        self.assertEqual(native['sandbox_workspace_write.writable_roots'], ['/workspace/.agents/skills'])

    async def test_finish_and_budget_preserve_same_native_thread(self):
        first = await self.h.run_episode("finish", [], self.delivered_callback, 2)
        self.assertEqual(first["termination"], "finish")
        self.assertTrue(first["finish_called"])
        self.assertEqual(self.tools.calls[0][0], "finish")
        await self.h.feedback("Episode has ended. Native result unavailable in fixture.")
        second_store = RecordStore(self.root / "second", {"run_id": "run", "task_id": "task", "episode_id": "two"})
        await self.h.bind_episode(second_store, Tools())
        second = await self.h.run_episode("hold", [], self.delivered_callback, 0.03)
        self.assertEqual(second["termination"], "time_limit")
        self.assertEqual(first["thread_id"], second["thread_id"])
        self.assertNotEqual(first["turn_id"], second["turn_id"])
        self.assertTrue(second["session_usable"])

    async def test_unrequested_native_failure_stops_chain(self):
        with self.assertRaises(RuntimeError):
            await self.h.run_episode("failed", [], self.delivered_callback, 2)
        self.assertFalse(self.h.session_usable)

    async def test_stale_thread_cannot_execute_finish(self):
        result = await self.h.run_episode("stale", [], self.delivered_callback, 2)
        self.assertFalse(result["finish_called"])
        self.assertEqual(self.tools.calls, [])

    async def test_external_control_limit_stops_without_finish(self):
        task = asyncio.create_task(self.h.run_episode("hold", [], self.delivered_callback, 2))
        while not self.delivered:
            await asyncio.sleep(0)
        await self.h.stop_episode("control_step_limit")
        result = await task
        self.assertEqual(result["termination"], "control_step_limit")
        self.assertFalse(result["finish_called"])

    async def test_finish_called_survives_runner_stop_race(self):
        harness = self.h

        class RacingTools(Tools):
            async def call(self, *args):
                await harness.stop_episode("finish")
                return await super().call(*args)

        self.h.tools = RacingTools()
        result = await self.h.run_episode("finish", [], self.delivered_callback, 2)
        self.assertTrue(result["finish_called"])
        self.assertEqual(result["termination"], "finish")


class AssetTests(unittest.TestCase):
    def test_real_0169_probe_schema_progress_and_jobs_are_not_generated_memory(self):
        # Column names and fresh-home counts observed in the real CPU probe.
        # This regression fixture does not claim to run native generation.
        jobs_columns = "kind,job_key,status,worker_id,ownership_token,started_at,finished_at,lease_until,retry_at,retry_remaining,last_error,input_watermark,last_success_watermark"
        output_columns = "thread_id,source_updated_at,raw_memory,rollout_summary,rollout_slug,generated_at,usage_count,last_usage,selected_for_phase2,selected_for_phase2_source_updated_at"
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            for name in ("memories_1.sqlite", "memories_v2_1.sqlite"):
                with sqlite3.connect(home / name) as db:
                    db.execute("CREATE TABLE _sqlx_migrations (version)")
                    db.executemany("INSERT INTO _sqlx_migrations VALUES (?)", [(1,), (2,)])
                    db.execute("CREATE TABLE consolidation_progress (singleton,max_thread_count)")
                    db.execute("INSERT INTO consolidation_progress VALUES (1,0)")
                    db.execute("CREATE TABLE jobs (" + jobs_columns + ")")
                    db.execute("CREATE TABLE stage1_outputs (" + output_columns + ")")
            report = inspect_memory(home, enabled=True)
            self.assertEqual(report["state"], "schema_only")
            self.assertEqual(len(report["databases"]), 2)
            for item in report["databases"]:
                self.assertEqual(item["data_rows"], 0)
                self.assertEqual(item["processing_rows"], 0)
                self.assertEqual(item["stage1_output_rows"], 0)
                progress = next(t for t in item["tables"] if t["name"] == "consolidation_progress")
                self.assertEqual(progress["rows"], 1)
                self.assertTrue(progress["metadata_only"])
            with sqlite3.connect(home / "memories_v2_1.sqlite") as db:
                db.execute("INSERT INTO jobs (kind,job_key,status) VALUES ('phase1','fixture','running')")
            pending = inspect_memory(home, enabled=True)
            self.assertEqual(pending["state"], "processing_state_only")
            self.assertFalse(pending["generation_verified"])
            with sqlite3.connect(home / "memories_v2_1.sqlite") as db:
                db.execute("INSERT INTO stage1_outputs (thread_id,raw_memory,rollout_summary) VALUES ('fixture-thread','private fixture memory','private fixture summary')")
            extracted = inspect_memory(home, enabled=True)
            self.assertEqual(extracted["state"], "database_rows_present_unverified")
            v2 = next(d for d in extracted["databases"] if d["path"] == "memories_v2_1.sqlite")
            self.assertEqual(v2["stage1_output_rows"], 1)
            self.assertEqual(v2["data_rows"], 1)
            self.assertEqual(v2["processing_rows"], 1)
            self.assertFalse(extracted["generation_verified"])
            self.assertFalse(extracted["new_session_reuse_verified"])
            self.assertNotIn("private fixture", json.dumps(extracted))

    def test_memory_database_inspection_counts_unfamiliar_schema_and_visible_wal(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            # A schema fixture, not evidence of a particular native revision.
            db = sqlite3.connect(home / "memories_v2.sqlite")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE _sqlx_migrations (version INTEGER)")
            db.execute("INSERT INTO _sqlx_migrations VALUES (1)")
            db.execute("CREATE TABLE memory_items (content TEXT)")
            db.commit()
            try:
                empty = inspect_memory(home, enabled=True)
                self.assertEqual(empty["state"], "schema_only")
                self.assertEqual(empty["databases"][0]["data_rows"], 0)
                self.assertIsNone(empty["databases"][0]["stage1_output_rows"])
                db.execute("INSERT INTO memory_items VALUES ('private database fixture body')")
                db.commit()
                found = inspect_memory(home, enabled=True)
                self.assertEqual(found["state"], "database_rows_present_unverified")
                self.assertEqual(found["databases"][0]["data_rows"], 1)
                table = next(t for t in found["databases"][0]["tables"] if t["name"] == "memory_items")
                self.assertEqual(table["nonempty_text_rows"], {"content": 1})
                self.assertFalse(found["generation_verified"])
                self.assertFalse(found["new_session_reuse_verified"])
                self.assertNotIn("private database fixture body", json.dumps(found))
            finally:
                db.close()

    def test_corrupt_memory_database_is_not_reported_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            (home / "memories_v2.sqlite").write_bytes(b"not a SQLite database")
            found = inspect_memory(home)
            self.assertEqual(found["state"], "inspection_incomplete")
            self.assertEqual(found["databases"][0]["state"], "inspection_error")
            self.assertIsNone(found["databases"][0]["data_rows"])

    def test_memory_inspection_distinguishes_empty_and_unverified_without_body(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            empty = inspect_memory(home, enabled=True)
            self.assertTrue(empty["enabled"])
            self.assertEqual(empty["state"], "empty")
            (home / "memories").mkdir()
            (home / "memories/memory_summary.md").write_text("private native fixture body")
            found = inspect_memory(home, enabled=True)
            self.assertEqual(found["state"], "generated_files_present_unverified")
            self.assertFalse(found["generation_verified"])
            self.assertFalse(found["new_session_reuse_verified"])
            self.assertNotIn("private native fixture body", json.dumps(found))

    def test_streaming_redaction_covers_secret_across_chunk_boundaries(self):
        secret = "real-secret-example"
        for cut in range(1, len(secret)):
            scrub = StreamingRedactor((secret,))
            out = scrub.feed(("prefix " + secret[:cut]).encode())
            out += scrub.feed((secret[cut:] + " suffix").encode())
            out += scrub.feed(b"", final=True)
            self.assertEqual(out, "prefix [REDACTED] suffix")

    def test_memory_snapshot_omits_history_and_captures_sqlite_wal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            (home / "memories").mkdir(parents=True)
            (home / "memories/memory_summary.md").write_text("Native generated memory fixture")
            (home / "sessions").mkdir()
            (home / "sessions/rollout.jsonl").write_text("history must not be imported")
            db = sqlite3.connect(home / "memories_1.sqlite")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE stage1_outputs (raw_memory TEXT)")
            db.execute("INSERT INTO stage1_outputs VALUES ('fixture')")
            db.commit()
            try:
                report = snapshot_memory(home, root / "snapshot")
                with sqlite3.connect(root / "snapshot/memories_1.sqlite") as snapshot:
                    self.assertEqual(snapshot.execute("SELECT raw_memory FROM stage1_outputs").fetchone()[0], "fixture")
            finally:
                db.close()
            self.assertEqual(report["state"], "present_unverified")
            self.assertFalse((root / "snapshot/sessions").exists())
            self.assertEqual(report["verification"], "new_session_reuse_not_yet_verified")

    def test_archive_rejects_escape_and_secret(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = root / "bad.tar.gz"
            with tarfile.open(bad, "w:gz") as t:
                info = tarfile.TarInfo("../escaped")
                info.size = 1
                t.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                extract_tree(bad, root / "restore")
            source = root / "source"
            source.mkdir()
            (source / "notes.txt").write_text("secret-value")
            with self.assertRaises(ValueError):
                archive_tree(source, root / "saved.tar.gz", secrets=("secret-value",))

    def test_fresh_import_keeps_mutations_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            asset = root / "asset"
            asset.mkdir()
            source = root / "source"
            source.mkdir()
            (source / "work.txt").write_text("original")
            archive_tree(source, asset / "workspace.tar.gz")
            mem = root / "mem"
            (mem / "memories").mkdir(parents=True)
            (mem / "memories/memory_summary.md").write_text("native fixture")
            archive_tree(mem, asset / "memory.tar.gz")
            manifest = {"schema_version": 1, "export_status": "saved", "status": "complete", "components": {
                "image": {"image_id": "sha256:" + "a" * 64, "import": "local_docker_image"},
                "workspace": {"path": "workspace.tar.gz"}, "memory": {"path": "memory.tar.gz", "state": "present_unverified"}}}
            path = asset / "manifest.json"
            path.write_text(json.dumps(manifest))
            load_asset(path, root / "w1", root / "h1")
            load_asset(path, root / "w2", root / "h2")
            (root / "w1/work.txt").write_text("changed")
            self.assertEqual((root / "w2/work.txt").read_text(), "original")
            self.assertFalse((root / "h1/sessions").exists())
            manifest["components"]["memory"]["state"] = "not_generated"
            path.write_text(json.dumps(manifest))
            load_asset(path, root / "w3", root / "h3")
            manifest["components"]["memory"]["state"] = "present_unverified"
            path.write_text(json.dumps(manifest))
            (mem / "memories/memory_summary.md").unlink()
            archive_tree(mem, asset / "memory.tar.gz")
            load_asset(path, root / "w4", root / "h4")


class ExportFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_complete_chain_without_memory_can_be_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "workspace").mkdir()
            (root / "home").mkdir()
            image_id = "sha256:" + "a" * 64

            async def fake_command(*args):
                if args[1] == "commit":
                    return image_id
                if args[1:3] == ("image", "inspect"):
                    return json.dumps([{"Id": image_id}])
                return ""

            manifest = await export_asset(root / "out", container="owned-test-container", workspace=root / "workspace",
                home=root / "home", metadata={"status": "complete", "source_task_id": "t"}, command=fake_command,
                base_image="base", image_name="fixture:saved")
            self.assertEqual(manifest["export_status"], "saved")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["memory_generation"], "not_checked")
            self.assertEqual(manifest["learning_effect"], "not_evaluated")
            self.assertNotIn("finalize_status", manifest)
            self.assertTrue(manifest["components"]["image"]["self_contained"])
            load_asset(root / "out/manifest.json", root / "eval-work", root / "eval-home")

    async def test_failure_unpauses_and_does_not_claim_saved_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "workspace").mkdir()
            (root / "workspace/auth.json").write_text("{}")
            (root / "home").mkdir()
            calls = []

            async def fake_command(*args):
                calls.append(args)
                return ""

            manifest = await export_asset(root / "out", container="owned-test-container", workspace=root / "workspace",
                home=root / "home", metadata={"status": "partial", "source_task_id": "t"}, command=fake_command)
            self.assertEqual(manifest["export_status"], "failed")
            self.assertEqual(calls, [("docker", "pause", "owned-test-container"), ("docker", "unpause", "owned-test-container")])

    async def test_model_budget_rejects_before_spawning_transport(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"BOLE_TEST_TRANSPORT_KEY": "fixture-key"}):
            store = RecordStore(Path(temp), {"run_id": "r", "task_id": "t", "episode_id": "e"})
            ledger, stopped = UsageLedger(), []

            async def on_limit(reason):
                stopped.append(reason)

            bridge = ModelHTTPBridge({"base_url": "http://127.0.0.1:9/v1", "credential_env": "BOLE_TEST_TRANSPORT_KEY"},
                current_context=lambda: (store, ledger, {"phase": "episode", "episode_id": "e", "model_request_limit": 0}),
                on_limit=on_limit)
            body = json.dumps({"model": "fixture", "input": "hello"}).encode()
            reader = asyncio.StreamReader()
            reader.feed_data(b"POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer bole-local-transport\r\nContent-Length: "
                             + str(len(body)).encode() + b"\r\n\r\n" + body)
            reader.feed_eof()

            class Writer:
                output = b""

                def write(self, data):
                    self.output += data

                async def drain(self):
                    pass

                def close(self):
                    pass

                async def wait_closed(self):
                    pass

            writer = Writer()
            with patch("asyncio.create_subprocess_exec", side_effect=AssertionError("must not forward an over-budget request")):
                await bridge.handle(reader, writer)
            self.assertEqual(stopped, ["model_request_limit"])
            self.assertEqual(ledger.report()["counts"]["started"], 0)
            self.assertNotIn(b"fixture-key", writer.output)


if __name__ == "__main__":
    unittest.main()
