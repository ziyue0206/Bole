"""CPU lifecycle tests exercise the real scheduler, tools, stores and boundaries."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import yaml

from bole.engine.runner import PreflightError, preflight, run
from bole.results.usage import UsageLedger


def settings(root, mode="eval"):
    prompt = root / "instruction.txt"
    prompt.write_text("Task {task_name}. {task_description}\n{robot_interface}\n{episode_rules}\n{budgets}")
    return {"mode": mode, "benchmark": {"name": "fixture"}, "harness": {
        "name": "fixture", "instruction": str(prompt), "gateway": {"credential_env": "BOLE_TEST_UNUSED_KEY"}},
        "policy": None, "limits": {"timeout_s": None, "control_steps": None, "tool_calls": None,
                                    "model_requests": None, "startup_timeout_s": 1, "request_timeout_s": 1},
        "termination": {"success_grace_steps": 2}, "eval": {"parallel": 2}, "learn": {"parallel": 2},
        "resources": {"gpus": []}, "output": str(root / "runs")}


def fixture_plan(mode="eval", tasks=("a",), count=3):
    episodes = []
    for task in tasks:
        previous = None
        for index in range(count):
            ident = f"{task}-{index}"
            episodes.append({"task_id": task, "episode_id": ident, "seed": 0, "layout_id": index,
                "input_asset": None, "predecessor": previous if mode == "learn" else None,
                "learning_chain_id": task if mode == "learn" else None})
            previous = ident
    return {"schema_version": 1, "episodes": episodes, "tasks": [{"id": t} for t in tasks]}


class FakeBenchmark:
    requires_gpu = False

    def __init__(self, log, behavior=None):
        self.log = log
        self.behavior = behavior or {}
        self.sessions = []
        self.active = 0
        self.max_active = 0

    def check(self, config):
        return {"ok": True, "simulation_started": False}

    def session(self, context):
        session = FakeSession(self, context)
        self.sessions.append(session)
        return session

    def summary(self, plan, results):
        valid = [r for r in results if r.get("native_result", {}).get("available")]
        return {"success_rate": 100 * sum(r["native_result"]["success"] for r in valid) / len(valid)
                if valid and len(valid) == len(plan["episodes"]) else None,
                "aggregation": "fixture_native", "coverage": {"planned": len(plan["episodes"]), "valid": len(valid)}}


class FakeSession:
    def __init__(self, benchmark, context):
        self.benchmark, self.context = benchmark, context
        self.steps = 0
        self.closed = False

    def open(self):
        self.benchmark.active += 1
        self.benchmark.max_active = max(self.benchmark.max_active, self.benchmark.active)
        self.benchmark.log.append(("environment_open", self.context["episode_id"], self.context["gpu"]))
        return {"instruction": "Use measured public state.", "step_limit": 5,
                "capabilities": {"action_schema": {"type": "array", "items": {"type": "object"}}}}

    def observe(self):
        return {"state": {"joint": 1}, "instruction": "Use measured public state.",
                "vision": {}, "control_step": self.steps, "native_hidden": "secret-native-verdict"}

    def delivered(self):
        self.benchmark.log.append(("delivered", self.context["episode_id"]))

    def execute(self, actions, call_id):
        if self.benchmark.behavior.get(self.context["episode_id"]) == "infra":
            raise RuntimeError("fixture worker died")
        self.steps += len(actions)
        return {"public": {"actual_control_steps": self.steps},
                "private": {"actual_control_steps": self.steps, "native_hidden": "secret-native-verdict"}}

    def close(self, reason):
        assert not self.closed
        self.closed = True
        self.benchmark.active -= 1
        ident = self.context["episode_id"]
        self.benchmark.log.append(("environment_close", ident, reason))
        if self.benchmark.behavior.get(ident) == "infra":
            return {"native_result": {"available": False, "reason": "worker_dead"}}
        success = self.context["episode"]["layout_id"] != 0
        return {"native_result": {"available": True, "success": success, "score": int(success)},
                "artifacts": {"fixture": True}}


class FakeHarness:
    def __init__(self, log, behavior=None, delay=0):
        self.log = log
        self.behavior = behavior or {}
        self.delay = delay
        self.ledger = UsageLedger()
        self.stop = asyncio.Event()
        self.seen = []

    async def open(self, context, tools):
        self.context, self.tools, self.store = context, tools, context["store"]
        self.log.append(("harness_open", id(self), context["task_id"], context.get("input_asset")))

    async def bind_episode(self, store, tools):
        self.store, self.tools = store, tools
        self.log.append(("harness_bind", id(self), store.identity["episode_id"]))

    async def run_episode(self, prompt, initial_content, on_delivered, timeout_s):
        self.stop.clear()
        ident = self.store.identity["episode_id"]
        self.seen.append((prompt, initial_content))
        await on_delivered()
        if self.delay:
            await asyncio.sleep(self.delay)
        self.ledger.start(ident, episode_id=ident)
        self.ledger.finish(ident, {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
        behavior = self.behavior.get(ident, "finish")
        if behavior == "hang":
            await self.stop.wait()
            return {"termination": "time_limit", "session_usable": True}
        if behavior == "model_limit":
            return {"termination": "model_request_limit", "session_usable": True}
        if behavior == "final":
            return {"termination": "final_response_without_finish", "session_usable": True}
        if behavior == "infra":
            await self.tools.call("execute", {"actions": [{}], "observation_id": self.tools.inputs[-1]}, ident + "-action")
            return {"termination": "infrastructure_error", "session_usable": False}
        await self.tools.call("finish", {"reason": "done"}, ident + "-finish")
        return {"termination": "finish", "finish_called": True, "session_usable": True}

    async def stop_episode(self, reason):
        self.stop.set()
        self.log.append(("harness_stop", id(self), reason))

    async def feedback(self, text):
        self.log.append(("feedback", id(self), text))

    async def export_asset(self, destination, metadata, partial=False):
        self.log.append(("export", id(self), partial, metadata))
        return {"export_status": "saved", "status": "partial" if partial else "complete"}

    async def close(self):
        self.log.append(("harness_close", id(self)))


class EngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.log, self.harnesses = [], []

    def tearDown(self):
        self.temporary.cleanup()

    def harness_factory(self, behavior=None, delay=0):
        def create():
            harness = FakeHarness(self.log, behavior, delay)
            self.harnesses.append(harness)
            return harness
        return create

    def read_results(self, result):
        return json.loads((Path(result["run_path"]) / "results.json").read_text())["results"]

    async def test_eval_each_episode_has_independent_harness_no_feedback_or_export(self):
        benchmark = FakeBenchmark(self.log)
        result = await run(settings(self.root), fixture_plan(), benchmark=benchmark,
                           harness_factory=self.harness_factory(), root=self.root / "run")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.harnesses), 3)
        self.assertEqual(len(benchmark.sessions), 3)
        self.assertFalse(any(row[0] in {"harness_bind", "feedback", "export"} for row in self.log))
        self.assertEqual(result["success_rate"], 200 / 3)
        self.assertEqual(result["episode_micro"]["scored_subset"]["success_rate"], 2 / 3)
        self.assertTrue(all(s.closed for s in benchmark.sessions))
        for harness in self.harnesses:
            self.assertNotIn("secret-native-verdict", json.dumps(harness.seen))

    async def test_initial_open_failure_attempts_partial_export_before_close(self):
        log = self.log
        class FailedOpen(FakeHarness):
            async def open(self, context, tools):
                context['root'].mkdir(parents=True)
                (context['root'] / 'available-state.txt').write_text('available')
                raise RuntimeError('failed after creating state')
        result = await run(settings(self.root, 'learn'), fixture_plan('learn', count=2),
            benchmark=FakeBenchmark(log), harness_factory=lambda: FailedOpen(log), root=self.root/'failed-open')
        exports = [r for r in log if r[0] == 'export']
        self.assertEqual(len(exports), 1)
        self.assertTrue(exports[0][2])
        self.assertLess(next(i for i,r in enumerate(log) if r[0]=='export'),
                        next(i for i,r in enumerate(log) if r[0]=='harness_close'))
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue((self.root/'failed-open/tasks/a/session/available-state.txt').is_file())

    async def test_resource_index_preserves_observed_baseline_and_marks_unknown_versions(self):
        class ObservedResources(FakeBenchmark):
            def check(self, config):
                return {"ok": True, "resources": {"source_revision": "observed-revision",
                    "source_changes": [" M native.py"], "simulator_image_id": "sha256:sim-observed",
                    "harness_image_id": "sha256:harness-observed", "asset_revision": "asset-observed"}}
        config = settings(self.root)
        config["benchmark"]["name"] = "robodojo"
        config["resources"].update(source="/fixture/frozen-native", source_revision="declared-revision",
            simulator_image="mutable-simulator-tag", harness_image="mutable-harness-tag")
        result = await run(config, fixture_plan(count=1), benchmark=ObservedResources(self.log),
            harness_factory=self.harness_factory(), root=self.root / "resources-run")
        root = Path(result["run_path"])
        record = json.loads((root / "resources.json").read_text())
        observed = record["observed_resources"]
        self.assertEqual(observed["simulator_image_id"], "sha256:sim-observed")
        self.assertEqual(observed["harness_image_id"], "sha256:harness-observed")
        self.assertEqual(observed["source_changes"], [" M native.py"])
        check = json.loads((root / record["preflight_ref"]).read_text())
        self.assertEqual(check["benchmark"]["resources"], observed)
        self.assertEqual(record["native_evaluator"]["declared_source_revision"], "declared-revision")
        self.assertEqual(record["native_evaluator"]["observed_source_revision"], "observed-revision")
        self.assertFalse(record["native_evaluator"]["runtime_execution_verified_by_resource_check"])
        for name in ("harness_executable", "simulator_dependencies"):
            value = record["runtime_versions"][name]
            self.assertEqual(value["status"], "not_collected")
            self.assertIsNone(value.get("version", value.get("versions")))
            self.assertTrue(value["reason"])

    async def test_learn_reuses_one_harness_per_task_with_new_environment_and_continues_failures(self):
        config = settings(self.root, "learn")
        behavior = {"a-1": "final", "a-2": "model_limit"}
        result = await run(config, fixture_plan("learn", tasks=("a", "b")), benchmark=FakeBenchmark(self.log),
                           harness_factory=self.harness_factory(behavior), root=self.root / "run")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.harnesses), 2)
        self.assertEqual(sum(row[0] == "harness_bind" for row in self.log), 4)
        self.assertEqual(sum(row[0] == "feedback" for row in self.log), 6)
        self.assertEqual(sum(row[0] == "environment_close" for row in self.log), 6)
        self.assertTrue(all(not row[2] for row in self.log if row[0] == "export"))
        rows = self.read_results(result)
        self.assertFalse(rows[0]["native_result"]["success"])
        self.assertEqual(rows[1]["termination_reason"], "final_response_without_finish")
        self.assertEqual(rows[2]["termination_reason"], "model_request_limit")
        usage = json.loads((Path(result["run_path"]) / "usage.json").read_text())
        self.assertEqual(usage["known_tokens"]["total_tokens"], 72)  # cumulative reports counted once per chain

    async def test_infrastructure_stops_only_its_learning_chain_and_exports_partial(self):
        behavior = {"a-1": "infra"}
        result = await run(settings(self.root, "learn"), fixture_plan("learn", tasks=("a", "b")),
            benchmark=FakeBenchmark(self.log, behavior), harness_factory=self.harness_factory(behavior), root=self.root / "run")
        rows = self.read_results(result)
        self.assertEqual([r["status"] for r in rows],
                         ["completed", "infrastructure_error", "not_run", "completed", "completed", "completed"])
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["episode_micro"]["coverage"]["native_missing"], 2)
        self.assertEqual(sorted(row[2] for row in self.log if row[0] == "export"), [False, True])

    async def test_wall_budget_starts_on_first_delivery_and_is_normal_chain_boundary(self):
        config = settings(self.root, "learn")
        config["limits"]["timeout_s"] = .03
        result = await run(config, fixture_plan("learn", count=2), benchmark=FakeBenchmark(self.log),
                           harness_factory=self.harness_factory({"a-0": "hang"}), root=self.root / "run")
        rows = self.read_results(result)
        self.assertEqual(rows[0]["termination_reason"], "time_limit")
        self.assertGreaterEqual(rows[0]["elapsed_after_first_observation_s"], .03)
        self.assertEqual(rows[1]["status"], "completed")
        self.assertEqual(len(self.harnesses), 1)

    async def test_one_gpu_is_exclusive_even_when_parallel_is_two(self):
        config = settings(self.root)
        config["resources"]["gpus"] = [3]
        benchmark = FakeBenchmark(self.log)
        benchmark.requires_gpu = True
        await run(config, fixture_plan(), benchmark=benchmark, harness_factory=self.harness_factory(delay=.03), root=self.root / "run")
        self.assertEqual(benchmark.max_active, 1)
        self.assertEqual([s.context["gpu"] for s in benchmark.sessions], [3, 3, 3])

    async def test_no_gpu_rejects_real_run_but_read_only_check_is_allowed(self):
        config = settings(self.root)
        benchmark = FakeBenchmark(self.log)
        benchmark.requires_gpu = True
        self.assertTrue(preflight(config, benchmark, runtime=False)["ok"])
        with self.assertRaises(PreflightError):
            await run(config, fixture_plan(), benchmark=benchmark, harness_factory=self.harness_factory())
        self.assertEqual(self.log, [])

    async def test_same_episode_id_across_tasks_still_creates_isolated_eval_harnesses(self):
        plan = fixture_plan(tasks=("a", "b"), count=1)
        for row in plan["episodes"]:
            row["episode_id"] = "episode-1"
        result = await run(settings(self.root), plan, benchmark=FakeBenchmark(self.log),
                           harness_factory=self.harness_factory(), root=self.root / "run")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.harnesses), 2)

    async def test_input_asset_is_explicit_and_no_history_resume_is_inferred(self):
        plan = fixture_plan(count=1)
        plan["episodes"][0]["input_asset"] = {"manifest_path": "/fixture/manifest.json", "source_task_id": "previous"}
        await run(settings(self.root), plan, benchmark=FakeBenchmark(self.log),
                  harness_factory=self.harness_factory(), root=self.root / "run")
        self.assertEqual(self.harnesses[0].context["input_asset"]["source_task_id"], "previous")
        self.assertNotIn("resume", self.harnesses[0].context)

    async def test_cancellation_saves_partial_chain_and_marks_remaining_not_run(self):
        config = settings(self.root, "learn")
        future = asyncio.create_task(run(config, fixture_plan("learn"), benchmark=FakeBenchmark(self.log),
            harness_factory=self.harness_factory({"a-0": "hang"}), root=self.root / "run"))
        for _ in range(100):
            if any(row[0] == "delivered" for row in self.log):
                break
            await asyncio.sleep(.01)
        future.cancel()
        result = await future
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual([r["status"] for r in self.read_results(result)], ["cancelled", "not_run", "not_run"])
        self.assertEqual([row[2] for row in self.log if row[0] == "export"], [True])

    async def test_missing_verdict_keeps_core_primary_null_even_if_adapter_reports_subset(self):
        class MislabelledSubset(FakeBenchmark):
            def summary(self, plan, results):
                return {"success_rate": 100}
        result = await run(settings(self.root), fixture_plan(count=2),
            benchmark=MislabelledSubset(self.log, {"a-0": "infra"}),
            harness_factory=self.harness_factory(), root=self.root / "run")
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["episode_micro"]["coverage"]["planned"], 2)
        self.assertEqual(result["episode_micro"]["coverage"]["native_valid"], 1)
        self.assertEqual(result["episode_micro"]["scored_subset"]["denominator"], 1)

    async def test_feedback_failure_preserves_valid_native_grade_and_exports_partial(self):
        class FeedbackFailure(FakeHarness):
            async def feedback(self, text):
                raise RuntimeError("feedback transport failed")
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=1),
            benchmark=FakeBenchmark(self.log), harness_factory=lambda: FeedbackFailure(self.log), root=self.root / "run")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["success_rate"], 0)
        self.assertEqual(result["episode_micro"]["coverage"]["native_missing"], 0)
        self.assertTrue(self.read_results(result)[0]["infrastructure_error"])
        export = next(row for row in self.log if row[0] == "export")
        self.assertTrue(export[2])
        self.assertEqual(export[3]["native_valid_episodes"], 1)
        self.assertEqual(export[3]["episodes_completed"], 1)
        feedback = json.loads((self.root / "run/tasks/a/episodes/a-0/feedback/native.json").read_text())
        self.assertEqual(feedback["status"], "delivery_failed")
        self.assertFalse(feedback["delivered"])
        self.assertEqual(feedback["error"]["phase"], "feedback")

    async def test_score_absence_does_not_remove_known_success_or_stop_learning(self):
        class NoScore(FakeBenchmark):
            def session(self, context):
                session = super().session(context)
                close = session.close
                def without_score(reason):
                    result = close(reason)
                    result["native_result"].pop("score")
                    return result
                session.close = without_score
                return session
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=2),
            benchmark=NoScore(self.log), harness_factory=self.harness_factory(), root=self.root / "run")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["success_rate"], 50)
        self.assertIsNone(result["episode_micro"]["main_score"])
        self.assertEqual(result["episode_micro"]["scored_subset"]["score_denominator"], 0)

    async def test_stop_boundary_checks_harness_session_usability_before_learning_continues(self):
        class UnusableAfterStop(FakeHarness):
            async def run_episode(self, prompt, initial_content, on_delivered, timeout_s):
                self.stop.clear()
                await on_delivered()
                await self.tools.call("finish", {"reason": "done"}, "finish")
                await self.stop.wait()
                return {"termination": "finish", "session_usable": False}
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=2),
            benchmark=FakeBenchmark(self.log), harness_factory=lambda: UnusableAfterStop(self.log), root=self.root / "run")
        self.assertEqual([r["status"] for r in self.read_results(result)], ["infrastructure_error", "not_run"])
        self.assertEqual(sum(row[0] == "environment_open" for row in self.log), 1)

    async def test_cancellation_during_close_preserves_started_episode_final_counters_and_partial_asset(self):
        entered, release = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        class DelayedClose(FakeBenchmark):
            def session(self, context):
                session = super().session(context)
                close = session.close
                def delayed(reason):
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(1):
                        raise RuntimeError("test did not release close")
                    return {**close(reason), "actual_control_steps": 3, "first_success_control_step": 2,
                            "first_success_native_result": {"success": True, "score": 1}}
                session.close = delayed
                return session

        benchmark = DelayedClose(self.log)
        task = asyncio.create_task(run(settings(self.root, "learn"), fixture_plan("learn", count=2),
            benchmark=benchmark, harness_factory=self.harness_factory(), root=self.root / "run"))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.sleep(.01)
        self.assertFalse(task.done(), "runner must wait for its owned close operation")
        release.set()
        result = await asyncio.wait_for(task, 2)
        rows = self.read_results(result)
        self.assertEqual([row["status"] for row in rows], ["cancelled", "not_run"])
        self.assertTrue(rows[0]["started"])
        self.assertEqual(rows[0]["actual_control_steps"], 3)
        self.assertEqual(rows[0]["first_success_control_step"], 2)
        self.assertTrue(rows[0]["success_observed"])
        self.assertEqual(rows[0]["executed_post_success_steps"], 1)
        self.assertFalse(rows[0]["native_result"]["success"], "first success must not overwrite final native failure")
        self.assertTrue(benchmark.sessions[0].closed)
        self.assertEqual([row[2] for row in self.log if row[0] == "export"], [True])
        self.assertFalse((self.root / "run/tasks/a/episodes/a-1").exists())
        for relative in ("usage.json", "tasks/a/usage.json", "tasks/a/episodes/a-0/usage.json"):
            report = json.loads((self.root / "run" / relative).read_text())
            self.assertEqual(report["counts"]["actual_control_steps"], 3)
        error = rows[0]["errors"][0]
        self.assertEqual(error["phase"], "environment_close")
        self.assertTrue((self.root / "run/tasks/a/episodes/a-0" / error["log_ref"]).is_file())

    async def test_one_startup_deadline_covers_environment_and_first_observation(self):
        for slow_phase in ("environment_initialization", "initial_observation"):
            with self.subTest(slow_phase=slow_phase):
                class SlowStartup(FakeBenchmark):
                    def session(self, context):
                        session = super().session(context)
                        operation = session.open if slow_phase == "environment_initialization" else session.observe
                        def slow():
                            time.sleep(.04)
                            return operation()
                        if slow_phase == "environment_initialization":
                            session.open = slow
                        else:
                            session.observe = slow
                        return session
                config = settings(self.root)
                config["limits"]["startup_timeout_s"] = .01
                benchmark = SlowStartup(self.log)
                result = await run(config, fixture_plan(count=1), benchmark=benchmark,
                    harness_factory=self.harness_factory(), root=self.root / slow_phase)
                row = self.read_results(result)[0]
                self.assertEqual(row["status"], "infrastructure_error")
                self.assertEqual(row["errors"][0]["type"], "TimeoutError")
                self.assertEqual(row["errors"][0]["phase"], slow_phase)
                self.assertIsNone(row["elapsed_after_first_observation_s"])
                self.assertTrue(benchmark.sessions[0].closed)
                self.assertFalse(any(x[0] == "harness_open" for x in self.log))

    async def test_learn_feedback_reports_first_success_finish_and_final_grade_after_close(self):
        class NativeSuccessThenFailure(FakeBenchmark):
            def session(self, context):
                session = super().session(context)
                close = session.close
                session.close = lambda reason: {**close(reason), "actual_control_steps": 21,
                    "first_success_control_step": 1, "first_success_native_result": {"success": True},
                    "termination_reason": "success_grace_reached"}
                return session
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=1),
            benchmark=NativeSuccessThenFailure(self.log), harness_factory=self.harness_factory({"a-0": "final"}), root=self.root / "run")
        feedback = next(row[2] for row in self.log if row[0] == "feedback")
        value = json.loads(feedback.split("Native feedback: ", 1)[1])
        self.assertTrue(value["success_observed"])
        self.assertEqual(value["executed_post_success_steps"], 20)
        self.assertFalse(value["finish_called"])
        self.assertFalse(value["native_success"])
        self.assertFalse(value["next_episode_planned"])
        self.assertLess(next(i for i, row in enumerate(self.log) if row[0] == "environment_close"),
                        next(i for i, row in enumerate(self.log) if row[0] == "feedback"))
        saved = json.loads((Path(result["run_path"]) / "tasks/a/episodes/a-0/feedback/native.json").read_text())
        self.assertTrue(saved["delivered"])
        self.assertEqual(saved["status"], "acknowledged_by_harness")
        self.assertEqual(saved["text"], feedback)

    async def test_episode_task_run_usage_include_feedback_without_session_snapshot_double_count(self):
        class MeteredFeedback(FakeHarness):
            async def feedback(self, text):
                ident = self.store.identity["episode_id"]
                self.ledger.start("feedback-" + ident, phase="feedback", episode_id=ident)
                self.ledger.finish("feedback-" + ident, {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4})
                self.ledger.snapshot("native-thread", {"totalTokens": 9999})
                await super().feedback(text)
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=2),
            benchmark=FakeBenchmark(self.log), harness_factory=lambda: MeteredFeedback(self.log), root=self.root / "run")
        for relative, total in (("usage.json", 32), ("tasks/a/usage.json", 32),
                                ("tasks/a/episodes/a-0/usage.json", 16), ("tasks/a/episodes/a-1/usage.json", 16)):
            report = json.loads((Path(result["run_path"]) / relative).read_text())
            self.assertEqual(report["known_tokens"]["total_tokens"], total)
            self.assertIn("timing", report)
        first = json.loads((self.root / "run/tasks/a/episodes/a-0/usage.json").read_text())
        self.assertEqual({r["episode_id"] for r in first["requests"]}, {"a-0"})
        self.assertEqual({r["phase"] for r in first["requests"]}, {"episode", "feedback"})
        first_result = self.read_results(result)[0]
        for name in ("initialization_s", "elapsed_after_first_observation_s"):
            self.assertEqual(first["timing"][name], first_result[name])
            self.assertGreaterEqual(first["timing"][name], 0)
        sessions = json.loads((self.root / "run/tasks/a/session-finalizations.json").read_text())["sessions"]
        self.assertEqual(len(sessions), 1, "shared learning cleanup is recorded once, not once per episode")
        self.assertEqual(first["session_key"], sessions[0]["session_key"])
        self.assertEqual(sessions[0]["episode_ids"], ["a-0", "a-1"])
        self.assertGreaterEqual(sessions[0]["asset_export_seconds"], 0)
        self.assertGreaterEqual(sessions[0]["harness_close_seconds"], 0)
        self.assertIsNone(sessions[0]["policy_close_seconds"])
        for relative in ("usage.json", "tasks/a/usage.json"):
            usage = json.loads((self.root / "run" / relative).read_text())
            self.assertEqual(usage["timing"]["session_finalizations"], sessions)

    async def test_failed_export_and_harness_close_keep_actual_finalization_durations(self):
        class FailedCleanup(FakeHarness):
            async def export_asset(self, *args, **kwargs):
                await asyncio.sleep(.01)
                raise RuntimeError("fixture export failure")
            async def close(self):
                await asyncio.sleep(.01)
                raise RuntimeError("fixture close failure")
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=1),
            benchmark=FakeBenchmark(self.log), harness_factory=lambda: FailedCleanup(self.log), root=self.root / "run")
        self.assertEqual(result["status"], "incomplete")
        timing = result["timing"]["session_finalizations"][0]
        self.assertGreaterEqual(timing["asset_export_seconds"], .01)
        self.assertGreaterEqual(timing["harness_close_seconds"], .01)
        self.assertEqual(result["learning_chains"]["a"]["finalization_timing"], timing)
        self.assertEqual({e["phase"] for e in result["lifecycle_errors"]}, {"asset_export", "harness_close"})

    async def test_required_reports_are_written_and_missing_safety_stays_unknown(self):
        result = await run(settings(self.root), fixture_plan(count=1), benchmark=FakeBenchmark(self.log),
                           harness_factory=self.harness_factory(), root=self.root / "run")
        root = Path(result["run_path"])
        for relative in ("config.yaml", "safety_summary.json", "constraint_metrics.csv", "tasks/a/summary.json",
                         "tasks/a/safety_summary.json", "tasks/a/usage.json", "tasks/a/episodes/a-0/usage.json",
                         "tasks/a/episodes/a-0/safety.json", "tasks/a/episodes/a-0/artifacts.json"):
            self.assertTrue((root / relative).is_file(), relative)
        config = yaml.safe_load((root / "config.yaml").read_text())
        self.assertEqual(config["mode"], "eval")
        safety = json.loads((root / "safety_summary.json").read_text())
        self.assertIsNone(safety["Safety"])
        self.assertEqual(safety["planned"], 1)
        self.assertEqual(safety["safety_complete"], 0)
        self.assertEqual(len(json.loads((root / "tasks/a/episodes/a-0/safety.json").read_text())["constraints"]), 8)
        index = json.loads((root / "tasks/a/episodes/a-0/artifacts.json").read_text())
        saved_result = next(item for item in index["artifacts"] if item["path"] == "result.json")
        self.assertEqual(saved_result["size_bytes"], (root / "tasks/a/episodes/a-0/result.json").stat().st_size)

    async def test_pending_memory_asset_uses_learning_directory_and_is_reported_pending(self):
        destinations = []
        class PendingAsset(FakeHarness):
            async def export_asset(self, destination, metadata, partial=False):
                destinations.append(destination)
                return {"export_status": "saved", "finalize_status": "pending",
                        "acceptance_status": "pending_native_memory_finalization"}
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=1),
            benchmark=FakeBenchmark(self.log), harness_factory=lambda: PendingAsset(self.log), root=self.root / "run")
        self.assertEqual(destinations, [(self.root / "run/learning/a").resolve()])
        self.assertEqual(result["pending_memory_assets"][0]["manifest_ref"], "learning/a/manifest.json")
        self.assertEqual(result["pending_memory_assets"][0]["finalize_status"], "pending")

    async def test_recovered_valid_native_grade_does_not_hide_worker_infrastructure_failure(self):
        class Recovered(FakeBenchmark):
            def session(self, context):
                session = super().session(context)
                close = session.close
                session.close = lambda reason: {**close(reason), "infrastructure_error": True,
                                                "recovered_from_persisted_records": True}
                return session
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=2),
            benchmark=Recovered(self.log), harness_factory=self.harness_factory(), root=self.root / "run")
        rows = self.read_results(result)
        self.assertEqual([r["status"] for r in rows], ["infrastructure_error", "not_run"])
        self.assertTrue(rows[0]["native_result"]["available"])
        self.assertEqual(result["episode_micro"]["coverage"]["native_valid"], 1)
        self.assertTrue(next(row[2] for row in self.log if row[0] == "export"))

    async def test_initial_native_boundary_waits_for_first_delivery_and_never_reports_negative_time(self):
        class InitiallySuccessful(FakeBenchmark):
            def session(self, context):
                session = super().session(context)
                open_session = session.open
                session.open = lambda: {**open_session(), "private_boundary": {
                    "actual_control_steps": 0, "first_success_control_step": 0,
                    "first_success_native_result": {"success": True}, "termination_reason": "success_grace_reached"}}
                return session
        config = settings(self.root)
        config["termination"]["success_grace_steps"] = 0
        result = await run(config, fixture_plan(count=1), benchmark=InitiallySuccessful(self.log),
            harness_factory=self.harness_factory({"a-0": "hang"}), root=self.root / "run")
        row = self.read_results(result)[0]
        self.assertEqual(row["termination_reason"], "success_grace_reached")
        self.assertTrue(row["success_observed"])
        self.assertEqual(row["actual_control_steps"], 0)
        self.assertGreaterEqual(row["elapsed_after_first_observation_s"], 0)
        self.assertEqual(sum(event[0] == "delivered" for event in self.log), 1)

    async def test_close_recovery_is_not_cut_off_by_an_individual_request_budget(self):
        class SlowBoundedRecovery(FakeBenchmark):
            def session(self, context):
                session = super().session(context)
                close = session.close
                def recover(reason):
                    time.sleep(.04)  # Distinct bounded cleanup stages can outlast one request.
                    return {**close(reason), "actual_control_steps": 2}
                session.close = recover
                return session
        config = settings(self.root)
        config["limits"]["request_timeout_s"] = .01
        result = await run(config, fixture_plan(count=1), benchmark=SlowBoundedRecovery(self.log),
            harness_factory=self.harness_factory(), root=self.root / "run")
        row = self.read_results(result)[0]
        self.assertEqual(row["status"], "completed")
        self.assertTrue(row["native_result"]["available"])
        self.assertEqual(row["actual_control_steps"], 2)
        self.assertGreaterEqual(row["timing"]["environment_close_seconds"], .04)

    async def test_invalid_close_receipt_does_not_erase_started_episode_or_skip_partial_export(self):
        class InvalidReceipt(FakeBenchmark):
            def session(self, context):
                session = super().session(context)
                close = session.close
                def invalid(reason):
                    close(reason)
                    return "not a native result object"
                session.close = invalid
                return session
        result = await run(settings(self.root, "learn"), fixture_plan("learn", count=2),
            benchmark=InvalidReceipt(self.log), harness_factory=self.harness_factory(), root=self.root / "run")
        rows = self.read_results(result)
        self.assertEqual([r["status"] for r in rows], ["infrastructure_error", "not_run"])
        self.assertTrue(rows[0]["started"])
        self.assertEqual(rows[0]["errors"][0]["phase"], "environment_close")
        self.assertEqual([event[2] for event in self.log if event[0] == "export"], [True])

    async def test_aggregation_error_keeps_episode_evidence_and_phase_log_reference(self):
        class BrokenSummary(FakeBenchmark):
            def summary(self, plan, results):
                raise RuntimeError("fixture aggregation failed")
        result = await run(settings(self.root), fixture_plan(count=1), benchmark=BrokenSummary(self.log),
            harness_factory=self.harness_factory(), root=self.root / "run")
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["success_rate"])
        self.assertEqual(result["success_rate_reason"], "benchmark_aggregation_failed")
        self.assertTrue(self.read_results(result)[0]["native_result"]["available"])
        error = result["lifecycle_errors"][0]
        self.assertEqual(error["phase"], "benchmark_aggregation")
        self.assertTrue((Path(result["run_path"]) / error["log_ref"]).is_file())


if __name__ == "__main__":
    unittest.main()
