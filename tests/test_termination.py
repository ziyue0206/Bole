import unittest

from bole.engine.termination import EpisodeBoundary


class TerminationTests(unittest.TestCase):
    def test_final_native_snapshot_replaces_stale_host_counters_without_changing_grade(self):
        boundary = EpisodeBoundary()
        boundary.final_response()
        boundary.reconcile_native({"actual_control_steps": 3, "first_success_control_step": 2,
            "first_success_native_result": {"success": True}, "termination_reason": "control_step_limit",
            "native_result": {"success": False}})
        report = boundary.report()
        self.assertEqual(report["actual_control_steps"], 3)
        self.assertEqual(report["first_success_control_step"], 2)
        self.assertEqual(report["executed_post_success_steps"], 1)
        self.assertEqual(report["termination_reason"], "control_step_limit")
        self.assertNotIn("native_result", report)

    def test_initial_success_before_delivery_has_zero_not_negative_elapsed_time(self):
        now = [0.0]
        boundary = EpisodeBoundary(grace_steps=0, clock=lambda: now[0])
        boundary.reconcile_native({"actual_control_steps": 0, "first_success_control_step": 0,
                                   "termination_reason": "success_grace_reached"})
        now[0] = 2.0
        boundary.delivered()
        self.assertEqual(boundary.report()["elapsed_after_first_observation_s"], 0)

    def test_success_threshold_is_first_success_plus_n(self):
        boundary = EpisodeBoundary(grace_steps=20)
        for step in range(1, 121):
            boundary.executed({"success": step in (100, 115), "score": float(step == 100)})
            if step < 120:
                self.assertIsNone(boundary.reason)
        self.assertEqual(boundary.reason, "success_grace_reached")
        self.assertEqual(boundary.report()["executed_post_success_steps"], 20)
        self.assertEqual(boundary.first_success_step, 100)

    def test_zero_and_finish_and_budget(self):
        zero = EpisodeBoundary(grace_steps=0)
        self.assertEqual(zero.executed({"success": True}), "success_grace_reached")
        limited = EpisodeBoundary(step_limit=3, grace_steps=20)
        limited.executed({"success": True})
        limited.executed({"success": False})
        self.assertEqual(limited.executed({"success": False}), "control_step_limit")
        finish = EpisodeBoundary()
        finish.executed({"success": True})
        self.assertEqual(finish.finish(), "finish")
        self.assertTrue(finish.report()["finish_called"])

    def test_delivery_clock_and_final_response(self):
        now = [0.0]
        boundary = EpisodeBoundary(timeout_s=10, clock=lambda: now[0])
        now[0] = 1000
        self.assertIsNone(boundary.check())
        boundary.delivered()
        now[0] = 1009
        self.assertIsNone(boundary.check())
        now[0] = 1010
        self.assertEqual(boundary.check(), "time_limit")
        other = EpisodeBoundary()
        self.assertEqual(other.final_response(), "final_response_without_finish")
        self.assertFalse(other.finish_called)
        self.assertIsNone(other.report()["executed_post_success_steps"])


if __name__ == "__main__":
    unittest.main()
