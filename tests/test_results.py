import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from bole.results import RecordStore, UsageLedger, read_jsonl, summarize
from bole.probes import SafetyEvaluator, aggregate, load_profile


def _concurrent_writer(directory, barrier, name):
    store = RecordStore(Path(directory), {"run_id": "shared"})
    barrier.wait(timeout=15)
    for index in range(12):
        store.event("concurrent", writer=name, index=index, payload="x" * 9000)


class RecordingTests(unittest.TestCase):
    def test_redaction_identity_and_damaged_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RecordStore(Path(directory), {"run_id": "run-a", "episode_id": "episode-a"}, secrets=["private-test-key"])
            store.event("tool_result", text="value private-test-key", headers={"Authorization": "Bearer token-value"})
            store.write_json("config.json", {"api_key": "secret", "credential_env": "MY_API_KEY"})
            event_path = Path(directory) / "trajectory/events.jsonl"
            text = event_path.read_text()
            self.assertNotIn("private-test-key", text)
            self.assertNotIn("token-value", text)
            self.assertEqual(json.loads((Path(directory) / "config.json").read_text())["credential_env"], "MY_API_KEY")
            with self.assertRaises(ValueError):
                store.write_json("x.json", {"run_id": "wrong"})
            with self.assertRaises(ValueError):
                store.asset("../outside", b"x")
            with event_path.open("ab") as stream:
                stream.write(b'{"interrupted":')
            self.assertFalse(read_jsonl(event_path)["complete"])
            self.assertEqual(len(read_jsonl(event_path)["records"]), 1)
            with self.assertRaises(ValueError):
                RecordStore(Path(directory), {})

    def test_append_after_reopening_uses_new_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            first = RecordStore(Path(directory), {})
            first.event("one")
            second = RecordStore(Path(directory), {})
            second.event("two")
            first.event("three")
            self.assertEqual([e["sequence"] for e in read_jsonl(Path(directory) / "trajectory/events.jsonl")["records"]], [1, 2, 3])

    def test_worker_and_runner_share_one_atomic_event_sequence(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            barrier = context.Barrier(3)
            workers = [context.Process(target=_concurrent_writer, args=(directory, barrier, str(i))) for i in range(2)]
            try:
                for child in workers:
                    child.start()
                _concurrent_writer(directory, barrier, "runner")
                for child in workers:
                    child.join(timeout=20)
                    self.assertEqual(child.exitcode, 0)
                report = read_jsonl(Path(directory) / "trajectory/events.jsonl")
                self.assertTrue(report["complete"])
                rows = report["records"]
                self.assertEqual([r["sequence"] for r in rows], list(range(1, 37)))
                self.assertEqual(len({r["event_id"] for r in rows}), 36)
                self.assertEqual(len({r["clock_id"] for r in rows}), 3)
            finally:
                for child in workers:
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=5)


class UsageTests(unittest.TestCase):
    def test_stream_and_native_duplicate_not_summed(self):
        ledger = UsageLedger()
        usage = {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20,
                 "input_tokens_details": {"cached_tokens": 4},
                 "output_tokens_details": {"reasoning_tokens": 3}}
        ledger.start("r1", source="http")
        ledger.finish("r1", usage, source="http", provider_request_id="provider-1")
        ledger.finish("r1", usage, source="http", provider_request_id="provider-1")
        # This observes the HTTP request's usage; it is not a second attempt.
        ledger.start("native1", source="codex", observation_only=True)
        ledger.finish("native1", usage, source="codex", provider_request_id="provider-1")
        ledger.snapshot("thread", {"total_tokens": 900})
        ledger.start("retry2", phase="memory")
        ledger.finish("retry2", status="failed")
        report = ledger.report()
        self.assertEqual(report["known_tokens"]["total_tokens"], 20)
        self.assertEqual(report["known_tokens"]["cached_input_tokens"], 4)
        self.assertEqual(report["counts"]["started"], 2)
        self.assertEqual(report["counts"]["additional_usage_observations"], 1)
        self.assertFalse(report["complete"])
        self.assertIsNone(report["actual_cost"]["amount"])
        self.assertIsNone(report["estimated_cost"]["amount"])
        self.assertEqual(report["field_coverage"]["input_tokens"]["reported"], 1)

    def test_failed_attempt_with_usage_counts_and_never_fabricates_total(self):
        ledger = UsageLedger()
        ledger.start("failed")
        ledger.finish("failed", {"input_tokens": 10, "output_tokens": 5}, status="failed")
        report = ledger.report()
        self.assertEqual(report["known_tokens"]["input_tokens"], 10)
        self.assertIsNone(report["known_tokens"]["total_tokens"])
        self.assertEqual(report["counts"]["failed"], 1)

    def test_retries_with_same_response_keep_distinct_attempts_and_failed_status(self):
        ledger = UsageLedger()
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        for local, http, status in (("first", "http-1", "failed"), ("retry", "http-2", "completed")):
            ledger.start(local, source="http", episode_id="e", phase="episode")
            ledger.finish(local, usage, source="http", provider_request_id=http,
                          provider_response_id="same-response", status=status)
        report = ledger.report()
        self.assertEqual(report["counts"]["started"], 2)
        self.assertEqual(report["counts"]["failed"], 1)
        self.assertEqual(report["counts"]["completed"], 1)
        self.assertEqual([r["status"] for r in report["requests"]], ["failed", "completed"])
        self.assertEqual(report["known_tokens"]["total_tokens"], 15)
        self.assertEqual(report["counts"]["deduplicated_usage_groups"], 1)
        self.assertEqual(report["usage_groups"][0]["provider_http_request_ids"], ["http-1", "http-2"])

    def test_filtered_report_does_not_allocate_cumulative_session_usage_or_global_tool_counts(self):
        ledger = UsageLedger()
        ledger.start("one", episode_id="one", phase="episode")
        ledger.finish("one", {"total_tokens": 4})
        ledger.start("two", episode_id="two", phase="feedback")
        ledger.finish("two", {"total_tokens": 9})
        ledger.snapshot("native", {"total_tokens": 999})
        ledger.tool_calls = 20
        report = ledger.report(episode_id="one")
        self.assertEqual(report["known_tokens"]["total_tokens"], 4)
        self.assertEqual(report["counts"]["started"], 1)
        self.assertIsNone(report["counts"]["tool_calls"])
        self.assertEqual(report["session_usage_unallocated"][0]["usage"]["total_tokens"], 999)


class SummaryTests(unittest.TestCase):
    def test_missing_verdict_nulls_primary_but_valid_infrastructure_verdict_survives(self):
        plan = [{"task_id": "a", "episode_id": str(i)} for i in range(3)]
        rows = [{**plan[0], "started": True, "termination_reason": "infrastructure_error",
                 "native_result": {"available": True, "value": {"success": True, "score": 1}}},
                {**plan[1], "started": True, "termination_reason": "finish",
                 "native_result": {"available": True, "value": {"success": False, "score": 0}}}]
        report = summarize(rows, plan)
        self.assertIsNone(report["main_success_rate"])
        self.assertEqual(report["scored_subset"]["success_rate"], .5)
        self.assertEqual(report["coverage"]["not_run"], 1)
        self.assertEqual(report["coverage"]["infrastructure_errors"], 1)
        self.assertEqual(report["missing_episodes"][0]["reason"], "not_run")
        with self.assertRaises(ValueError):
            summarize([rows[0], rows[0]], 2)


class SafetyTests(unittest.TestCase):
    def frame(self, step=1, force=0):
        profile = load_profile()
        return {"physics_step": step, "sim_time_s": step * .01, "signals": {
            r.family_id: {"status": "valid", "unit": r.unit, "signal_definition": r.signal_definition,
                          "active": True, "value": force if r.family_id == "arm_furniture_force" else 0}
            for r in profile.rules}}

    def test_unknown_contact_data_cannot_pass_safety(self):
        evaluator = SafetyEvaluator()
        frame = self.frame(force=250)
        frame["signals"]["arm_furniture_force"]["status"] = "missing"
        frame["signals"]["arm_furniture_force"]["reason"] = "contact_buffer_partial"
        evaluator.sample(frame)
        result = evaluator.report("e", complete=True, expected=1)
        force = result["constraints"][0]
        self.assertEqual(force["status"], "missing")
        self.assertTrue(force["violation"])
        result["success"] = True
        summary = aggregate([result], evaluator.profile, planned=1)
        self.assertIsNone(summary["Safety"])
        self.assertIsNone(summary["SBU"])
        self.assertEqual(summary["families"]["arm_furniture_force"]["observed_violations_in_incomplete_data"], 1)

    def test_q_set_and_threshold_boundary(self):
        profile = load_profile()
        safe, unsafe = SafetyEvaluator(), SafetyEvaluator()
        safe.sample(self.frame(force=200))
        unsafe.sample(self.frame(force=300))
        a = safe.report("a", complete=True, expected=1)
        b = unsafe.report("b", complete=True, expected=1)
        a["success"] = False
        b["success"] = True
        summary = aggregate([a, b], profile, planned=3)
        self.assertEqual(summary["safety_complete"], 2)
        self.assertEqual(summary["Safety"], .5)
        self.assertEqual(summary["SBU"], .5)
        self.assertEqual(summary["VSI"], .25)
        self.assertEqual(summary["unscored_planned"], 1)

    def test_gap_and_wrong_units_are_missing_or_unsupported(self):
        evaluator = SafetyEvaluator()
        evaluator.sample(self.frame(1))
        frame = self.frame(3)
        frame["signals"]["joint_torque"]["unit"] = "A"
        evaluator.sample(frame)
        report = evaluator.report("e", complete=True, expected=2)
        self.assertFalse(report["complete"])
        self.assertTrue(report["sample_sequence_gap"])
        self.assertEqual(next(r for r in report["constraints"] if r["family_id"] == "joint_torque")["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
