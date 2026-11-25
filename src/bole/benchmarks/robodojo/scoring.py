"""Native RoboDojo aggregation with explicit missing-data and scope reporting.

The hierarchy follows A's scripts/internal/summarize_result.py: pool each
logical task/seed's selected episodes (including standard/random variants),
average tasks within a dimension, dimensions within a seed, then seeds.  Raw
episode micro averages remain a separately named diagnostic.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping


class ScoringError(ValueError):
    """Results do not belong unambiguously to the supplied immutable plan."""


def _mean(values):
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _std(values):
    available = [value for value in values if value is not None]
    return statistics.pstdev(available) if available else None


def _records(results) -> list[dict]:
    if isinstance(results, Mapping):
        if "episode_id" in results:
            return [dict(results)]
        for field in ("results", "episodes"):
            if field in results:
                return _records(results[field])
        output = []
        for episode_id, value in results.items():
            if not isinstance(value, Mapping):
                raise ScoringError("Result mappings must contain episode records")
            row = dict(value)
            if "episode_id" in row and row["episode_id"] != episode_id:
                raise ScoringError("Result dictionary key differs from episode_id")
            row.setdefault("episode_id", episode_id)
            output.append(row)
        return output
    if not isinstance(results, Iterable) or isinstance(results, (str, bytes)):
        raise ScoringError("Results must be episode records or an episode-ID mapping")
    output = []
    for value in results:
        if not isinstance(value, Mapping):
            raise ScoringError("Every result must be a mapping")
        output.append(dict(value))
    return output


def _native_metrics(result):
    if result is None:
        return None, None, "result_not_recorded", "result_not_recorded"
    native = result.get("native_result")
    if not isinstance(native, Mapping) or native.get("available") is not True:
        return None, None, "native_result_unavailable", "native_result_unavailable"
    success, score = native.get("success"), native.get("score")
    if type(success) is bool:
        success_value, success_reason = float(success), None
    else:
        success_value, success_reason = None, "native_success_missing_or_invalid"
    if type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1:
        score_value, score_reason = float(score), None
    else:
        score_value, score_reason = None, "native_score_missing_or_invalid"
    return success_value, score_value, success_reason, score_reason


def _native_scope(plan, tasks, episodes):
    """Verify native *coverage*, separately from termination/official status."""
    protocol = plan.get("protocol") or {}
    dimensions = protocol.get("dimensions") or {}
    seeds = protocol.get("expected_evaluation_seeds") or []
    if not dimensions or not seeds:
        return False, ["native_reference_metadata_missing"]
    expected_tasks = {task for names in dimensions.values() for task in names}
    expected_tasks |= {name + "_random" for name in dimensions.get("Generalization", [])}
    reasons = []
    if set(tasks) != expected_tasks:
        reasons.append("task_selection_differs_from_native_reference")
    actual = Counter()
    for episode in episodes:
        layout = episode.get("layout_id")
        if type(layout) is not int or layout < 0:
            reasons.append("native_instance_metadata_missing")
            break
        actual[(episode["task_id"], episode["seed"], layout)] += 1
    expected = Counter()
    for task_id, task in tasks.items():
        quota = task.get("native_episodes_per_seed")
        if type(quota) is not int or quota < 1:
            reasons.append("native_episode_quota_missing")
            break
        for seed in seeds:
            for layout in range(quota):
                expected[(task_id, seed, layout)] += 1
    if actual != expected:
        reasons.append("seed_or_instance_selection_differs_from_native_reference")
    return not reasons, sorted(set(reasons))


def summarize_native(plan: dict, results) -> dict:
    """Return percentage-valued native metrics; missing observations stay null.

    A valid final success verdict and a valid normalized native score have
    separate coverage.  A known verdict remains usable when the service omitted
    its score; no absent score is replaced with zero.  Infrastructure status
    does not discard an independently valid native result.
    """
    task_rows = plan.get("tasks") or []
    tasks = {task["id"]: task for task in task_rows}
    if len(tasks) != len(task_rows):
        raise ScoringError("Plan contains duplicate task IDs")
    episodes = plan.get("episodes") or []
    planned = {}
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in planned:
            raise ScoringError("Plan episode IDs must be nonempty and unique")
        if episode.get("task_id") not in tasks:
            raise ScoringError(f"Plan episode references an unknown task: {episode_id}")
        if type(episode.get("seed")) is not int:
            raise ScoringError(f"Plan episode has no integer evaluation seed: {episode_id}")
        planned[episode_id] = episode
    recorded = {}
    for result in _records(results):
        episode_id = result.get("episode_id")
        if episode_id not in planned:
            raise ScoringError(f"Result is outside the selected plan: {episode_id!r}")
        if episode_id in recorded:
            raise ScoringError(f"Duplicate result for episode: {episode_id}")
        episode = planned[episode_id]
        if result.get("task_id") != episode["task_id"] or result.get("seed") != episode["seed"]:
            raise ScoringError(f"Result identity disagrees with the plan: {episode_id}")
        recorded[episode_id] = result

    cells = {}
    all_successes, all_scores = [], []
    missing_success, missing_score = [], []
    for episode in episodes:
        task = tasks[episode["task_id"]]
        logical, dimension = task.get("logical_task_id"), task.get("dimension")
        if not isinstance(logical, str) or not isinstance(dimension, str):
            raise ScoringError(f"Task has no native logical-task/dimension metadata: {task['id']}")
        key = (episode["seed"], dimension, logical)
        cell = cells.setdefault(key, {
            "seed": episode["seed"], "dimension": dimension, "logical_task_id": logical,
            "planned_episodes": 0, "success_values": [], "score_values": [],
            "runnable_tasks": set(), "variant_counts": Counter(),
        })
        cell["planned_episodes"] += 1
        cell["runnable_tasks"].add(episode["task_id"])
        cell["variant_counts"][task.get("variant", "standard")] += 1
        success, score, success_reason, score_reason = _native_metrics(recorded.get(episode["episode_id"]))
        identity = {"episode_id": episode["episode_id"], "task_id": episode["task_id"], "seed": episode["seed"]}
        if success is None:
            missing_success.append({**identity, "reason": success_reason})
        else:
            cell["success_values"].append(success)
            all_successes.append(success)
        if score is None:
            missing_score.append({**identity, "reason": score_reason})
        else:
            cell["score_values"].append(score)
            all_scores.append(score)

    logical_cells = []
    dimension_cells = defaultdict(list)
    for key in sorted(cells):
        cell = cells[key]
        successes, scores = cell.pop("success_values"), cell.pop("score_values")
        success_mean, score_mean = _mean(successes), _mean(scores)
        cell.update({
            "runnable_tasks": sorted(cell["runnable_tasks"]),
            "variant_counts": dict(sorted(cell["variant_counts"].items())),
            "native_success_valid": len(successes),
            "native_score_valid": len(scores),
            "successes": int(sum(successes)),
            "native_score_sum": sum(scores) if scores else None,
            "success_rate": 100 * success_mean if success_mean is not None else None,
            "score": 100 * score_mean if score_mean is not None else None,
        })
        logical_cells.append(cell)
        dimension_cells[(cell["seed"], cell["dimension"])].append(cell)

    per_dimension_seed, by_seed = [], defaultdict(list)
    for (seed, dimension), selected_cells in sorted(dimension_cells.items()):
        row = {
            "seed": seed, "dimension": dimension,
            "planned_logical_tasks": len(selected_cells),
            "success_task_denominator": sum(cell["success_rate"] is not None for cell in selected_cells),
            "score_task_denominator": sum(cell["score"] is not None for cell in selected_cells),
            "success_rate": _mean([cell["success_rate"] for cell in selected_cells]),
            "score": _mean([cell["score"] for cell in selected_cells]),
        }
        per_dimension_seed.append(row)
        by_seed[seed].append(row)
    per_seed = []
    for seed, dimensions in sorted(by_seed.items()):
        per_seed.append({
            "seed": seed,
            "planned_dimensions": len(dimensions),
            "success_dimension_denominator": sum(row["success_rate"] is not None for row in dimensions),
            "score_dimension_denominator": sum(row["score"] is not None for row in dimensions),
            "success_rate": _mean([row["success_rate"] for row in dimensions]),
            "score": _mean([row["score"] for row in dimensions]),
        })
    seed_successes, seed_scores = [row["success_rate"] for row in per_seed], [row["score"] for row in per_seed]
    macro = {
        "success_rate": _mean(seed_successes),
        "score": _mean(seed_scores),
        "success_rate_std": _std(seed_successes),
        "score_std": _std(seed_scores),
        "success_seed_denominator": sum(value is not None for value in seed_successes),
        "score_seed_denominator": sum(value is not None for value in seed_scores),
        "included_success_episodes": len(all_successes),
        "included_score_episodes": len(all_scores),
        "complete_success_coverage": bool(episodes) and not missing_success,
        "complete_score_coverage": bool(episodes) and not missing_score,
        "logical_task_seed_cells": logical_cells,
        "dimension_seed_cells": per_dimension_seed,
        "per_seed": per_seed,
        "weighting": "pool episodes per logical-task/seed; equal tasks per dimension; equal dimensions per seed; equal seeds",
        "subset_note": "Missing metrics are excluded only from this explicitly labelled known subset; primary metrics remain null when incomplete.",
    }
    selected = {
        "success_rate": macro["success_rate"] if macro["complete_success_coverage"] else None,
        "score": macro["score"] if macro["complete_score_coverage"] else None,
        "success_rate_std": macro["success_rate_std"] if macro["complete_success_coverage"] else None,
        "score_std": macro["score_std"] if macro["complete_score_coverage"] else None,
        "success_rate_status": "complete" if macro["complete_success_coverage"] else "incomplete_or_empty",
        "score_status": "complete" if macro["complete_score_coverage"] else "incomplete_or_empty",
    }
    full_native, scope_reasons = _native_scope(plan, tasks, episodes)
    if not episodes:
        full_native = False
        scope_reasons.append("no_planned_episodes")
    protocol = plan.get("protocol") or {}
    native_mean = selected.copy() if full_native else None
    success_mean, score_mean = _mean(all_successes), _mean(all_scores)
    return {
        "schema_version": 1,
        "metric_unit": "percent",
        "native_input_score_range": [0, 1],
        "success_rate": selected["success_rate"],
        "score": selected["score"],
        "selected_scope_mean": selected,
        "native_full_coverage_mean": native_mean,
        "official_full_protocol_score": None,
        "official_verified": False,
        "protocol_note": (
            "Bole reports native evaluator outcomes with its configured termination policy "
            f"(success_grace_steps={protocol.get('success_grace_steps', 'unknown')}). "
            "Matching native task/seed/layout coverage does not establish the upstream "
            "termination protocol or official verification; this is not an official verified score."
        ),
        "official_full_protocol_score_unavailable_reason": "bole_termination_and_no_official_verification",
        "scored_subset_macro": macro,
        "episode_micro": {
            "success_rate": 100 * success_mean if success_mean is not None else None,
            "score": 100 * score_mean if score_mean is not None else None,
            "successes": int(sum(all_successes)),
            "success_denominator": len(all_successes),
            "native_score_sum": sum(all_scores) if all_scores else None,
            "score_denominator": len(all_scores),
            "scope": "known_episode_subset",
        },
        "coverage": {
            "planned": len(episodes),
            "results_recorded": len(recorded),
            "results_not_recorded": len(episodes) - len(recorded),
            "native_success_valid": len(all_successes),
            "native_score_valid": len(all_scores),
            "success_missing": len(missing_success),
            "score_missing": len(missing_score),
            "success_coverage": len(all_successes) / len(episodes) if episodes else None,
            "score_coverage": len(all_scores) / len(episodes) if episodes else None,
            "planned_runnable_tasks": len({row["task_id"] for row in episodes}),
            "planned_logical_tasks": len({row["logical_task_id"] for row in logical_cells}),
            "planned_seeds": sorted(by_seed),
            "configured_seeds": protocol.get("evaluation_seeds"),
            "full_native_instance_coverage": full_native,
            "scope_deviations": scope_reasons,
            "missing_success": missing_success,
            "missing_score": missing_score,
        },
    }
