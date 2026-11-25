"""Regression probes use fake subprocess streams; no model/network/GPU calls."""
import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from bole.engine.termination import EpisodeBoundary
from bole.harnesses.codex import CodexHarness
from bole.harnesses.transport import ModelHTTPBridge
from bole.results.storage import RecordStore
from bole.results.usage import UsageLedger
from bole.tools import EpisodeTools


def stream(data=b""):
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


class Sink:
    def __init__(self):
        self.data = bytearray()
    def write(self, data):
        self.data.extend(data)
    async def drain(self):
        pass
    def close(self):
        pass
    async def wait_closed(self):
        pass


class TransportObservationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RecordStore(self.root / "records", {"run_id": "r", "task_id": "t", "episode_id": "e"})

    def events(self):
        return [json.loads(line) for line in (self.store.root / "trajectory/events.jsonl").read_text().splitlines()]

    async def test_bridge_uses_effective_timeout_and_keeps_http_and_response_ids_separate(self):
        process = SimpleNamespace(returncode=0, stdin=Sink(), stderr=stream(), wait=AsyncMock(return_value=0),
            stdout=stream(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nx-request-id: http-id\r\n\r\n"
                          b'{"id":"response-id","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}'))
        ledger = UsageLedger()
        body = b'{ "model" : "fixture", "input" : [] }\n'
        reader = stream(b"POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer bole-local-transport\r\nContent-Length: "
                        + str(len(body)).encode() + b"\r\n\r\n" + body)
        with patch.dict("os.environ", {"BOLE_FAKE_KEY": "fixture-key"}):
            bridge = ModelHTTPBridge({"credential_env": "BOLE_FAKE_KEY", "base_url": "https://fixture.invalid/v1"},
                request_timeout_s=2.5, current_context=lambda: (self.store, ledger, {"phase": "episode", "episode_id": "e", "control_step": 7}))
            with patch("bole.harnesses.transport.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as spawn:
                await bridge.handle(reader, Sink())
        argv = spawn.await_args.args
        self.assertEqual(argv[argv.index("--max-time") + 1], "2.5")
        row = ledger.report()["requests"][0]
        self.assertEqual(row["provider_request_id"], "http-id")
        self.assertEqual(row["provider_response_id"], "response-id")
        response = json.loads(next((self.store.root / "model-events").glob("*-response.json")).read_text())
        self.assertEqual(response["upstream_headers"]["x-request-id"], "http-id")
        event = next(e for e in self.events() if e["kind"] == "model_request")
        metadata = json.loads((self.store.root / event["payload_format_ref"]).read_text())
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["episode_id"], "e")
        self.assertEqual(metadata["format"], "openai_responses_request_json")
        self.assertEqual(metadata["payloads"]["native_input"]["path"], event["native_request_ref"])
        self.assertEqual((self.store.root / event["native_request_ref"]).read_bytes(), body)
        self.assertNotIn("schema_version", json.loads((self.store.root / event["request_ref"]).read_text()))
        self.assertTrue(all(e["control_step"] == 7 for e in self.events() if e["kind"] in {"model_request", "model_response"}))

    async def test_harness_passes_configured_timeout_before_starting_native_process(self):
        harness = CodexHarness()
        context = {"config": {"limits": {"request_timeout_s": 3}, "harness": {"gateway": {}},
                              "resources": {"harness_image": "fixture"}},
                   "store": self.store, "root": self.root / "home"}
        with patch("bole.harnesses.codex.command", new=AsyncMock(return_value='[{"Id":"fixture-image"}]')), \
             patch("bole.harnesses.codex.ModelHTTPBridge", side_effect=RuntimeError("fixture stop")) as bridge, \
             patch("asyncio.create_subprocess_exec") as spawn:
            with self.assertRaisesRegex(RuntimeError, "fixture stop"):
                await harness.open(context, SimpleNamespace(definitions=lambda: []))
        self.assertEqual(bridge.call_args.kwargs["request_timeout_s"], 3)
        spawn.assert_not_called()

    async def test_failed_completion_racing_budget_stop_marks_session_unusable(self):
        harness = CodexHarness()
        harness.store, harness.thread_id, harness.settings = self.store, "thread", {}
        delivered = asyncio.Event()
        harness.request = AsyncMock(return_value={"turn": {"id": "turn"}})
        async def on_delivered():
            delivered.set()
        async def fail_at_boundary():
            await delivered.wait()
            await asyncio.sleep(.001)
            harness.notifications.put_nowait({"method": "turn/completed", "params": {
                "threadId": "thread", "turn": {"id": "turn", "status": "failed"}}})
            await harness.stop_episode("control_step_limit")
        race = asyncio.create_task(fail_at_boundary())
        with self.assertRaisesRegex(RuntimeError, "native turn failed"):
            await harness.run_episode("fixture", [], on_delivered, 1)
        await race
        self.assertFalse(harness.session_usable)
        self.assertEqual(harness.stop_reason, "control_step_limit")

    async def test_settle_does_not_accept_queued_failed_completion(self):
        harness = CodexHarness()
        harness.store, harness.thread_id, harness.turn_id = self.store, "thread", "turn"
        harness.notifications.put_nowait({"method": "turn/completed", "params": {
            "threadId": "thread", "turn": {"id": "turn", "status": "failed"}}})
        with self.assertRaisesRegex(RuntimeError, "native turn failed"):
            await harness._interrupt_and_settle()
        self.assertFalse(harness.session_usable)

    async def prepared_tools(self):
        class Session:
            def observe(self):
                return {"state": {}, "control_step": 7, "measurement_validity": {"joint": "valid"},
                        "vision": {"head": {"color": {"encoding": "png;base64", "data": base64.b64encode(b"fixture png").decode(), "shape": [1, 1, 3]}}}}
            def delivered(self):
                pass
        boundary = EpisodeBoundary()
        boundary.steps = 7
        tools = EpisodeTools(Session(), {"capabilities": {"action_schema": {"type": "array"}}},
                             {"limits": {"tool_calls": None}}, self.store, boundary)
        await tools.first_observation()
        await tools.mark_delivered()
        return tools

    async def test_later_image_is_marked_delivered_only_after_native_send_flush(self):
        tools = await self.prepared_tools()
        gate, entered = asyncio.Event(), asyncio.Event()
        class DelayedSink(Sink):
            async def drain(self):
                entered.set()
                await gate.wait()
        harness = CodexHarness()
        harness.store, harness.tools = self.store, tools
        harness.config = {"limits": {}}
        harness.thread_id, harness.turn_id, harness.accepting_tools = "thread", "turn", True
        harness.process = SimpleNamespace(stdin=DelayedSink())
        request = {"id": "rpc", "method": "item/tool/call", "params": {
            "threadId": "thread", "turnId": "turn", "tool": "observe", "arguments": {}, "callId": "observe-call"}}
        task = asyncio.create_task(harness._server_request(request))
        await entered.wait()
        self.assertFalse(any(e["kind"] == "observation_delivered" for e in self.events()))
        gate.set()
        await task
        delivery = next(e for e in self.events() if e["kind"] == "observation_delivered")
        self.assertEqual(delivery["call_id"], "observe-call")
        self.assertEqual(delivery["captured_control_step"], 7)
        self.assertEqual(delivery["control_step"], 7)
        self.assertEqual(delivery["images"][0]["camera"], "head")
        self.assertTrue(delivery["evidence"]["delivery_event_id"])
        self.assertEqual(harness._transport_context()[2]["observation_ids"], [delivery["observation_id"]])
        self.assertEqual(tools.current_observation["measurement_validity"], {"joint": "valid"})

    async def test_action_references_latest_delivered_image_while_next_result_is_pending(self):
        tools = await self.prepared_tools()
        first = tools.delivered_inputs[0]
        second_result = await tools.call("observe", {}, "capture-second")
        self.assertFalse(second_result["isError"])
        second = tools.inputs[0]
        self.assertNotEqual(first, second)
        tools.session.execute = lambda actions, call: {"public":{}, "private":{"actual_control_steps":8}}
        rejected = await tools.call("execute", {"actions":[{}], "observation_id":second}, "too-early")
        self.assertTrue(rejected["isError"])
        accepted = await tools.call("execute", {"actions":[{}], "observation_id":first}, "delivered-action")
        self.assertFalse(accepted["isError"])
        await tools.mark_result_delivered("capture-second")
        stale = await tools.call("execute", {"actions":[{}], "observation_id":first}, "now-stale")
        self.assertTrue(stale["isError"])

    async def test_failed_native_send_does_not_claim_image_delivery(self):
        tools = await self.prepared_tools()
        class BrokenSink(Sink):
            async def drain(self):
                raise BrokenPipeError("fixture broken pipe")
        harness = CodexHarness()
        harness.store, harness.tools = self.store, tools
        harness.thread_id, harness.turn_id, harness.accepting_tools = "thread", "turn", True
        harness.process = SimpleNamespace(stdin=BrokenSink())
        await harness._server_request({"id": "rpc", "method": "item/tool/call", "params": {
            "threadId": "thread", "turnId": "turn", "tool": "observe", "arguments": {}, "callId": "observe-call"}})
        self.assertFalse(any(e["kind"] == "observation_delivered" for e in self.events()))
        self.assertTrue(any(e["kind"] == "native_message_send_failed" for e in self.events()))
