import csv
import json
from pathlib import Path
import tempfile
import unittest

from bole.probes.profile import load_profile
from bole.probes.safety import SafetyEvaluator, write_reports


class SafetyReportTests(unittest.TestCase):
    def test_observed_active_partial_sample_and_valid_active_are_separate(self):
        profile = load_profile()
        rule = next(r for r in profile.rules if r.family_id == "peak_contact_force")
        evaluator = SafetyEvaluator(profile)
        for step, status in enumerate(("valid", "missing")):
            evaluator.sample({"physics_step": step, "sim_time_s": step * .1,
                "signals": {rule.family_id: {"status": status, "active": True,
                    "value": 300, "unit": rule.unit, "signal_definition": rule.signal_definition,
                    "reason": "incomplete buffer" if status == "missing" else None}}})
        rows = evaluator.finalize("e", complete=True, expected=2)
        row = next(row for row in rows if row["family_id"] == rule.family_id)
        self.assertEqual((row["active_samples"], row["valid_active_samples"], row["valid_samples"]), (2, 1, 1))
        self.assertEqual(row["status"], "missing")
        self.assertTrue(row["violation"])
        unobserved = next(row for row in rows if row["family_id"] != rule.family_id)
        self.assertEqual(unobserved["active_samples"], 0)
        self.assertEqual(unobserved["active_unknown_samples"], 2)
        self.assertIsNone(unobserved["violation"])

    def test_csv_preserves_heterogeneous_native_signal_scope_and_identity(self):
        profile = load_profile()
        rows = SafetyEvaluator(profile).finalize("e", complete=False, expected=1)
        rows[-1]["signal_scope"] = {"robot_coverage": ["robot0", "robot1"], "all_robots": False}
        rows[-1]["applicability_pending_user_decision"] = ["scripted_actor"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_reports(root, [{"run_id": "r", "task_id": "t", "episode_id": "e",
                "success": True, "constraints": rows, "events": []}], profile, 1, identity={"run_id": "r"})
            with (root / "constraint_metrics.csv").open() as stream:
                written = list(csv.DictReader(stream))
            self.assertEqual(len(written), 8)
            self.assertEqual(json.loads(written[-1]["signal_scope"]), rows[-1]["signal_scope"])
            self.assertEqual(written[-1]["run_id"], "r")
            self.assertIsNone(summary["Safety"])
            self.assertEqual(json.loads((root / "safety_summary.json").read_text())["run_id"], "r")


if __name__ == "__main__":
    unittest.main()
