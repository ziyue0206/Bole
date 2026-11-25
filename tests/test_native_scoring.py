"""Independent numerical examples for the native macro and missing-data rules."""

from __future__ import annotations

import copy
import math
import unittest

from bole.benchmarks.robodojo.scoring import ScoringError, summarize_native


def experiment(task_specs, seeds=(0, 1, 2), counts=None):
    """task_specs = (runnable, logical, dimension, native quota) tuples."""
    tasks, dimensions, episodes = [], {}, []
    for name, logical, dimension, quota in task_specs:
        tasks.append({"id": name, "logical_task_id": logical, "dimension": dimension,
                      "variant": "random" if name.endswith("_random") else "standard",
                      "native_episodes_per_seed": quota})
        if logical not in dimensions.setdefault(dimension, []):
            dimensions[dimension].append(logical)
        for seed in seeds:
            count = (counts or {}).get((name, seed), quota)
            for index in range(count):
                episodes.append({"task_id": name, "episode_id": f"{name}-{seed}-{index}",
                                 "seed": seed, "layout_id": index})
    return {"schema_version": 1, "mode": "eval", "tasks": tasks, "episodes": episodes,
            "protocol": {"dimensions": dimensions, "expected_evaluation_seeds": [0, 1, 2],
                         "evaluation_seeds": list(seeds), "success_grace_steps": 20,
                         "is_official_termination": False}}


def outcomes(plan, success=True, score=1.0):
    return [{"episode_id": row["episode_id"], "task_id": row["task_id"], "seed": row["seed"],
             "native_result": {"available": True, "success": success, "score": score}}
            for row in plan["episodes"]]


class NativeScoringTests(unittest.TestCase):
    def test_macro_is_equal_dimension_weighting_not_episode_micro(self):
        plan = experiment([("one", "one", "Precision", 2),
                           ("two", "two", "Memory", 1), ("three", "three", "Memory", 1)])
        results = outcomes(plan)
        for result in results:
            value = result["task_id"] == "one"
            result["native_result"].update(success=value, score=float(value))
        # Precision=100, Memory=(0+0)/2=0: the dimension macro is 50.
        summary = summarize_native(plan, results)
        self.assertEqual(summary["success_rate"], 50)
        self.assertEqual(summary["score"], 50)
        self.assertEqual(summary["metric_unit"], "percent")
        # Change Precision's count to make the episode micro differ, not macro.
        changed = experiment([("one", "one", "Precision", 4),
                              ("two", "two", "Memory", 1), ("three", "three", "Memory", 1)])
        results = outcomes(changed)
        for result in results:
            value = result["task_id"] == "one"
            result["native_result"].update(success=value, score=float(value))
        summary = summarize_native(changed, results)
        self.assertEqual(summary["success_rate"], 50)
        self.assertAlmostEqual(summary["episode_micro"]["success_rate"], 100 * 4 / 6)

    def test_standard_and_random_are_pooled_into_one_logical_task(self):
        plan = experiment([("toast", "toast", "Generalization", 25),
                           ("toast_random", "toast", "Generalization", 25)])
        results = outcomes(plan)
        for result in results:
            result["native_result"].update(success=result["task_id"] == "toast",
                                            score=1.0 if result["task_id"] == "toast" else 0.2)
        summary = summarize_native(plan, results)
        self.assertEqual(summary["success_rate"], 50)
        self.assertAlmostEqual(summary["score"], 60)
        cells = summary["scored_subset_macro"]["logical_task_seed_cells"]
        self.assertEqual(len(cells), 3)
        self.assertTrue(all(row["planned_episodes"] == 50 for row in cells))
        self.assertTrue(all(row["variant_counts"] == {"standard": 25, "random": 25} for row in cells))

    def test_missing_result_nulls_primary_and_keeps_exact_subset_denominators(self):
        plan = experiment([("one", "one", "Precision", 1), ("two", "two", "Memory", 1)])
        results = outcomes(plan)
        results.pop()
        summary = summarize_native(plan, results)
        self.assertIsNone(summary["success_rate"])
        self.assertIsNone(summary["score"])
        self.assertEqual(summary["scored_subset_macro"]["success_rate"], 100)
        self.assertEqual(summary["episode_micro"]["success_denominator"], 5)
        self.assertEqual(summary["coverage"]["results_not_recorded"], 1)
        self.assertEqual(summary["coverage"]["success_missing"], 1)
        seed2 = next(row for row in summary["scored_subset_macro"]["per_seed"] if row["seed"] == 2)
        self.assertEqual(seed2["success_dimension_denominator"], 1)
        missing_cell = next(row for row in summary["scored_subset_macro"]["logical_task_seed_cells"]
                            if row["seed"] == 2 and row["logical_task_id"] == "two")
        self.assertEqual(missing_cell["native_success_valid"], 0)
        self.assertIsNone(missing_cell["success_rate"])

    def test_missing_score_is_not_zero_and_does_not_erase_known_success(self):
        plan = experiment([("one", "one", "Precision", 1)])
        results = outcomes(plan, score=0.5)
        del results[0]["native_result"]["score"]
        summary = summarize_native(plan, results)
        self.assertEqual(summary["success_rate"], 100)
        self.assertIsNone(summary["score"])
        self.assertEqual(summary["scored_subset_macro"]["score"], 50)
        self.assertEqual(summary["episode_micro"]["score_denominator"], 2)
        self.assertEqual(summary["coverage"]["score_missing"], 1)

    def test_infrastructure_error_does_not_discard_valid_final_native_result(self):
        plan = experiment([("one", "one", "Precision", 1)])
        results = outcomes(plan, success=False, score=0.25)
        for result in results:
            result.update(termination_reason="infrastructure_error", infrastructure_error=True,
                          first_success={"observed": True}, finish_called=False)
        summary = summarize_native(plan, results)
        self.assertEqual(summary["success_rate"], 0)
        self.assertEqual(summary["score"], 25)
        self.assertEqual(summary["coverage"]["native_success_valid"], 3)

    def test_three_seed_population_std(self):
        plan = experiment([("one", "one", "Precision", 2)])
        results = outcomes(plan, success=False, score=0.0)
        for result in results:
            value = result["seed"] == 2 or (result["seed"] == 1 and result["episode_id"].endswith("-0"))
            result["native_result"].update(success=value, score=float(value))
        summary = summarize_native(plan, results)
        self.assertEqual(summary["success_rate"], 50)
        self.assertAlmostEqual(summary["selected_scope_mean"]["success_rate_std"], math.sqrt(5000 / 3))
        self.assertEqual(summary["scored_subset_macro"]["success_seed_denominator"], 3)

    def test_reduced_scope_still_has_its_mean_but_no_full_reference_score(self):
        plan = experiment([("one", "one", "Precision", 2)], seeds=(0,), counts={("one", 0): 1})
        summary = summarize_native(plan, outcomes(plan))
        self.assertEqual(summary["selected_scope_mean"]["success_rate"], 100)
        self.assertIsNone(summary["native_full_coverage_mean"])
        self.assertIsNone(summary["official_full_protocol_score"])
        self.assertFalse(summary["official_verified"])
        self.assertIn("Bole", summary["protocol_note"])
        self.assertTrue(summary["coverage"]["scope_deviations"])

    def test_full_6300_plan_uses_42_logical_tasks_and_five_dimensions(self):
        specs = []
        for i in range(12):
            specs.extend([(f"pair_{i}", f"pair_{i}", "Generalization", 25),
                          (f"pair_{i}_random", f"pair_{i}", "Generalization", 25)])
        for dimension, number in [("Precision", 8), ("Long-Horizon", 8), ("Memory", 6), ("Open", 8)]:
            for i in range(number):
                name = dimension.replace("-", "_").lower() + str(i)
                specs.append((name, name, dimension, 50))
        plan = experiment(specs)
        self.assertEqual(len(plan["episodes"]), 6300)
        summary = summarize_native(plan, outcomes(plan))
        self.assertEqual(summary["coverage"]["planned_runnable_tasks"], 54)
        self.assertEqual(summary["coverage"]["planned_logical_tasks"], 42)
        self.assertEqual(len(summary["scored_subset_macro"]["logical_task_seed_cells"]), 126)
        self.assertTrue(summary["coverage"]["full_native_instance_coverage"])
        self.assertEqual(summary["native_full_coverage_mean"]["success_rate"], 100)
        self.assertTrue(all(row["success_dimension_denominator"] == 5
                            for row in summary["scored_subset_macro"]["per_seed"]))
        self.assertIsNone(summary["official_full_protocol_score"])

    def test_duplicates_unplanned_results_and_identity_mismatch_are_rejected(self):
        plan = experiment([("one", "one", "Precision", 1)])
        results = outcomes(plan)
        with self.assertRaisesRegex(ScoringError, "Duplicate result"):
            summarize_native(plan, results + results[:1])
        altered = copy.deepcopy(results)
        altered[0]["seed"] = 8
        with self.assertRaisesRegex(ScoringError, "identity"):
            summarize_native(plan, altered)
        altered[0]["episode_id"] = "unexpected"
        with self.assertRaisesRegex(ScoringError, "outside"):
            summarize_native(plan, altered)

    def test_dictionary_input_and_empty_results_preserve_missing_counts(self):
        plan = experiment([("one", "one", "Precision", 1)])
        records = outcomes(plan)
        result_map = {row["episode_id"]: {k: v for k, v in row.items() if k != "episode_id"} for row in records}
        self.assertEqual(summarize_native(plan, result_map)["success_rate"], 100)
        summary = summarize_native(plan, {})
        self.assertIsNone(summary["success_rate"])
        self.assertIsNone(summary["episode_micro"]["success_rate"])
        self.assertEqual(summary["coverage"]["success_missing"], 3)
        self.assertEqual(summary["episode_micro"]["success_denominator"], 0)

    def test_nonfinite_and_boolean_scores_are_missing(self):
        plan = experiment([("one", "one", "Precision", 1)])
        results = outcomes(plan)
        for row, score in zip(results, [float("nan"), float("inf"), True]):
            row["native_result"]["score"] = score
        summary = summarize_native(plan, results)
        self.assertEqual(summary["success_rate"], 100)
        self.assertIsNone(summary["score"])
        self.assertIsNone(summary["episode_micro"]["native_score_sum"])
        self.assertEqual(summary["coverage"]["native_score_valid"], 0)


if __name__ == "__main__":
    unittest.main()
