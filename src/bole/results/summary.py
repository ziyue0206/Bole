"""Episode coverage and scored-subset summaries, with no invented verdicts."""
from __future__ import annotations

import math


def native_result(row):
    value = row.get("native_result", row.get("native_evaluation", row.get("native")))
    if not isinstance(value, dict) or value.get("available") is False:
        return None
    value = value.get("value", value)
    if not isinstance(value, dict) or type(value.get("success")) is not bool:
        return None
    return value


def summarize(results, planned):
    """Episode-equal summary, suitable for task or explicitly labelled micro SR.

    The RoboDojo official dimension/task/seed macro aggregation is a separate
    benchmark adapter responsibility. This function never labels micro SR official.
    `planned` is an episode list or an integer count. Repeated result IDs fail.
    """
    results = list(results)
    plan_rows = list(planned) if not isinstance(planned, int) else None
    n = planned if plan_rows is None else len(plan_rows)
    if n < 0 or len(results) > n:
        raise ValueError("results exceed the declared plan")
    keys = [(r.get("task_id"), r["episode_id"]) for r in results]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate episode result")
    if plan_rows is not None:
        expected = {(r.get("task_id"), r["episode_id"]) for r in plan_rows}
        if len(expected) != n or not set(keys).issubset(expected):
            raise ValueError("result IDs must match unique planned episodes")
    scored = [(r, native_result(r)) for r in results if native_result(r) is not None]
    success_n = sum(value["success"] for _, value in scored)
    values = [value["score"] for _, value in scored if type(value.get("score")) in (int, float) and math.isfinite(value["score"])]
    observed = {key: r for key, r in zip(keys, results)}
    missing = []
    if plan_rows is not None:
        for planned_row in plan_rows:
            row = observed.get((planned_row.get("task_id"), planned_row["episode_id"]))
            if row is None or native_result(row) is None:
                missing.append({"task_id": planned_row.get("task_id"), "episode_id": planned_row["episode_id"],
                    "reason": "not_run" if row is None else row.get("termination_reason", "native_result_unavailable")})
    started = sum(r.get("started", r.get("status") != "not_run") is True for r in results)
    return {"schema_version": 1, "aggregation": "episode_equal_micro_not_official_macro",
        "main_success_rate": success_n / n if n and len(scored) == n else None,
        "main_success_rate_reason": None if n and len(scored) == n else "planned_native_results_missing_or_empty_plan",
        "main_score": sum(values) / n if n and len(values) == n else None,
        "scored_subset": {"success_rate": success_n / len(scored) if scored else None,
            "successes": success_n, "denominator": len(scored),
            "score": sum(values) / len(values) if values else None, "score_denominator": len(values)},
        "coverage": {"planned": n, "started": started,
            "completed": sum(r.get("completed", r.get("status") == "completed") is True for r in results),
            "native_valid": len(scored), "native_missing": n - len(scored), "not_run": n - started,
            "infrastructure_errors": sum(r.get("termination_reason") == "infrastructure_error" or r.get("infrastructure_error") is True for r in results),
            "cancelled": sum(r.get("termination_reason") == "cancelled" or r.get("cancelled") is True for r in results),
            "fraction_native_valid": len(scored) / n if n else None,
            "counts_overlap": True}, "missing_episodes": missing}
