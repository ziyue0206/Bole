"""Expand an immutable eval/learn plan before launching any runtime."""

from __future__ import annotations

import copy
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from bole.benchmarks.robodojo.catalog import Catalog


class PlanningError(ValueError):
    """The requested experiment cannot be expanded without an implicit choice."""


def _input_assets(config: dict, tasks: list[dict]) -> dict[str, dict | None]:
    settings = config.get("learning_assets") or {}
    mode = settings.get("mode")
    if mode not in (None, "by_task", "shared"):
        raise PlanningError("learning_assets.mode must be null, by_task, or shared")
    if mode is None:
        if settings.get("shared"):
            raise PlanningError("Cross-task shared learning assets require mode=shared")
        if settings.get("by_task"):
            mode = "by_task"
        else:
            return {task["id"]: None for task in tasks}
    assets = {}
    for task in tasks:
        target = task["id"]
        by_task = settings.get("by_task") or {}
        if not isinstance(by_task, dict):
            raise PlanningError("learning_assets.by_task must be a task-to-manifest mapping")
        reference = by_task.get(target) if mode == "by_task" else settings.get("shared")
        if not reference:
            raise PlanningError(f"No {mode} learning asset selected for task {target}")
        if not isinstance(reference, (str, Path)):
            raise PlanningError(f"Learning asset for {target} must name a manifest file")
        path = Path(reference).expanduser().resolve()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PlanningError(f"Cannot read learning asset manifest for {target}: {path}") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise PlanningError(f"Unsupported learning asset manifest: {path}")
        source = manifest.get("source_task_id", manifest.get("task_id"))
        if not isinstance(source, str) or not source:
            raise PlanningError(f"Learning asset has no source task: {path}")
        if mode == "by_task" and source != target:
            raise PlanningError(f"Task-matched asset source {source!r} differs from target {target!r}")
        assets[target] = {
            "selection_mode": mode,
            "source_task_id": source,
            "target_task_id": target,
            "manifest_path": str(path),
            "external_reference": True,
            "exists": True,
            "import": "bole_learning_asset_manifest",
            "source_run_id": manifest.get("source_run_id", manifest.get("run_id")),
            "completion_status": manifest.get("status"),
            "export_status": manifest.get("export_status"),
            "restore": {"workspace": True, "memory": True, "resume_history": False},
        }
    return assets


def plan(config: dict, catalog=None) -> dict:
    """Expand Bole's shared plan.  This function creates no directories/processes.

    ``episodes`` is a total per selected runnable task, not per seed.  Its
    confirmed default is the prefix of the native seed-major plan.  Explicit
    instance sequences preserve user-specified order and intentional repeats.
    """
    benchmark = config.get("benchmark") or {}
    if benchmark.get("name", "robodojo") != "robodojo" and catalog is None:
        raise PlanningError("A benchmark catalog must be supplied for external benchmarks")
    if catalog is None:
        resources = config.get("resources") or {}
        try:
            catalog = Catalog(Path(resources["source"]), Path(resources["assets"]))
        except KeyError as error:
            raise PlanningError("Planning RoboDojo requires resources.source and resources.assets") from error
    all_tasks = catalog.tasks()
    task_index = {task["id"]: task for task in all_tasks}
    if len(task_index) != len(all_tasks):
        raise PlanningError("Benchmark catalog contains duplicate task IDs")
    suite = benchmark.get("suite")
    if suite in (None, "all"):
        suite_tasks = all_tasks
    elif suite in catalog.dimensions:
        suite_tasks = [task for task in all_tasks if task["dimension"] == suite]
    else:
        raise PlanningError(f"Unknown RoboDojo suite: {suite!r}")
    selection = benchmark.get("tasks", "all")
    if selection == "all":
        tasks = suite_tasks
    elif isinstance(selection, list) and selection and all(isinstance(t, str) for t in selection):
        if len(set(selection)) != len(selection):
            raise PlanningError("Selected tasks must be distinct")
        unknown = set(selection) - set(task_index)
        if unknown:
            raise PlanningError(f"Unknown tasks: {', '.join(sorted(unknown))}")
        excluded = set(selection) - {task["id"] for task in suite_tasks}
        if excluded:
            raise PlanningError(f"Selected tasks are outside suite {suite!r}: {', '.join(sorted(excluded))}")
        tasks = [task_index[name] for name in selection]
    else:
        raise PlanningError("benchmark.tasks must be 'all' or a nonempty list of task IDs")
    if not tasks:
        raise PlanningError("No tasks selected")
    image_name = config.get("learn", {}).get("save_image")
    if config.get("mode") == "learn" and image_name and len(tasks) > 1 and "{task}" not in image_name:
        raise PlanningError("Multiple learning tasks require --save-image containing {task}, or automatic image names")
    seeds = benchmark.get("seeds", list(catalog.expected_seeds))
    if (
        not isinstance(seeds, list) or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise PlanningError("benchmark.seeds must list distinct nonnegative integers")
    count = config.get("episodes")
    if count is not None and (type(count) is not int or count < 1):
        raise PlanningError("episodes must be a positive integer or null")
    mode = config.get("mode", "eval")
    if mode not in ("eval", "learn"):
        raise PlanningError("mode must be eval or learn")
    explicit = benchmark.get("instances") or {}
    if not isinstance(explicit, dict):
        raise PlanningError("benchmark.instances must map task IDs to instance sequences")
    unknown = set(explicit) - set(task_index)
    if unknown:
        raise PlanningError(f"Instances configured for unknown tasks: {', '.join(sorted(unknown))}")
    selection_mode = config.get("episodes_selection")
    if selection_mode is None:
        selection_mode = "native_prefix"
    if selection_mode not in ("native_prefix", "seed_round_robin"):
        raise PlanningError("episodes_selection must be native_prefix or seed_round_robin")
    input_assets = _input_assets(config, tasks)
    instance_cache = {}

    def instances(task_id, seed):
        key = task_id, seed
        if key not in instance_cache:
            instance_cache[key] = catalog.instances(task_id, seed)
        return instance_cache[key]

    episodes = []
    per_task = {}
    expected_seeds = list(catalog.expected_seeds)
    configured_seed_selection_complete = set(seeds) == set(expected_seeds)
    for task in tasks:
        task_id = task["id"]
        quota = task["native_episodes_per_seed"]
        if task_id in explicit:
            rows = explicit[task_id]
            if not isinstance(rows, list) or not rows:
                raise PlanningError(f"Explicit instances must be a nonempty list: {task_id}")
            if count is not None and len(rows) < count:
                raise PlanningError(f"Explicit instances for {task_id}: need {count}, found {len(rows)}")
            # Validate the whole configured sequence, including rows after a
            # caller's explicit count override, to expose invalid identifiers.
            resolved = []
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"seed", "layout_id"}:
                    raise PlanningError(f"Each {task_id} instance needs exactly seed and layout_id")
                seed, layout = row["seed"], row["layout_id"]
                if type(seed) is not int or seed not in seeds:
                    raise PlanningError(f"Instance seed {seed!r} is not selected for {task_id}")
                if type(layout) is not int or layout < 0:
                    raise PlanningError(f"Invalid layout_id for {task_id}: {layout!r}")
                available = instances(task_id, seed)
                if layout >= len(available):
                    raise PlanningError(f"Layout {layout} is unavailable for {task_id}, seed {seed}")
                resolved.append(available[layout])
            schedule = resolved if count is None else resolved[:count]
            arrangement = "explicit"
        else:
            pools = []
            for seed in seeds:
                available = instances(task_id, seed)
                if len(available) < quota:
                    raise PlanningError(
                        f"Insufficient native layouts for {task_id}, seed {seed}: "
                        f"need {quota}, found {len(available)}"
                    )
                pools.append(available[:quota])
            if selection_mode == "seed_round_robin" and count is not None:
                native = [row for group in zip_longest(*pools) for row in group if row is not None]
            else:
                native = [row for pool in pools for row in pool]
            if count is not None and count > len(native):
                raise PlanningError(
                    f"Requested {count} episodes for {task_id}, native sequence has {len(native)}; "
                    "provide explicit instances to select reserve layouts or intentional repeats"
                )
            schedule = native if count is None else native[:count]
            arrangement = "native" if count is None else selection_mode
        actual = Counter((row["seed"], row["layout_id"]) for row in schedule)
        expected = Counter((seed, layout) for seed in seeds for layout in range(quota))
        per_seed = {str(seed): sum(row["seed"] == seed for row in schedule) for seed in seeds}
        planned_seeds = [seed for seed in seeds if per_seed[str(seed)] > 0]
        per_task[task_id] = {
            "planned_episodes": len(schedule),
            "native_expected_episodes_selected_seeds": quota * len(seeds),
            "episodes_per_seed": per_seed,
            "configured_seeds": list(seeds),
            "planned_seeds": planned_seeds,
            "configured_seed_selection_complete": configured_seed_selection_complete,
            "native_seed_coverage_complete": set(planned_seeds) == set(expected_seeds),
            "missing_native_seeds": [seed for seed in expected_seeds if seed not in planned_seeds],
            "arrangement": arrangement,
            "native_instance_coverage": actual == expected,
        }
        previous = None
        for number, instance in enumerate(schedule, start=1):
            episode_id = f"{task_id}-{number:06d}"
            episodes.append({
                "task_id": task_id,
                "logical_task_id": task["logical_task_id"],
                "variant": task["variant"],
                "episode_id": episode_id,
                "episode_number": number,
                "seed": instance["seed"],
                "layout_id": instance["layout_id"],
                "layout_path": instance["path"],
                "learning_chain_id": task_id if mode == "learn" else None,
                "predecessor": previous if mode == "learn" else None,
                "input_asset": copy.deepcopy(input_assets[task_id]),
                "limits": copy.deepcopy(config.get("limits") or {}),
            })
            previous = episode_id
    full_tasks = set(task_index) == {task["id"] for task in tasks}
    # Selecting a seed in configuration does not mean the expanded schedule
    # contains it: a native prefix can end before later seed banks are reached.
    full_seeds = all(row["native_seed_coverage_complete"] for row in per_task.values())
    episodes_per_seed = {str(seed): sum(row["episodes_per_seed"][str(seed)] for row in per_task.values())
                         for seed in seeds}
    runnable_tasks_per_seed = {str(seed): sum(seed in row["planned_seeds"] for row in per_task.values())
                              for seed in seeds}
    missing_task_seeds = [{"task_id": task_id, "seed": seed}
                          for task_id, row in per_task.items() for seed in row["missing_native_seeds"]]
    full_instances = all(row["native_instance_coverage"] for row in per_task.values())
    grace = (config.get("termination") or {}).get("success_grace_steps", 20)
    if type(grace) is not int or grace < 0:
        raise PlanningError("termination.success_grace_steps must be a nonnegative integer")
    return {
        "schema_version": 1,
        "mode": mode,
        "benchmark": benchmark.get("name", "robodojo"),
        "harness_image": config.get("resources", {}).get("harness_image"),
        "tasks": copy.deepcopy(tasks),
        "episodes": episodes,
        "coverage": {
            "catalog_runnable_tasks": len(all_tasks),
            "catalog_logical_tasks": len({task["logical_task_id"] for task in all_tasks}),
            "selected_runnable_tasks": len(tasks),
            "selected_logical_tasks": len({task["logical_task_id"] for task in tasks}),
            "planned_episodes": len(episodes),
            "native_reference_episodes": sum(
                task["native_episodes_per_seed"] * len(catalog.expected_seeds) for task in all_tasks
            ),
            "native_task_coverage_complete": full_tasks,
            "configured_seeds": list(seeds),
            "planned_seeds": [seed for seed in seeds if episodes_per_seed[str(seed)] > 0],
            "configured_seed_selection_complete": configured_seed_selection_complete,
            "episodes_per_seed": episodes_per_seed,
            "runnable_tasks_per_seed": runnable_tasks_per_seed,
            "native_seed_coverage_complete": full_seeds,
            "native_seed_coverage_scope": "every_selected_runnable_task",
            "planned_task_seed_cells": sum(len(row["planned_seeds"]) for row in per_task.values()),
            "native_reference_task_seed_cells_selected_tasks": len(tasks) * len(expected_seeds),
            "missing_native_task_seed_cells": missing_task_seeds,
            "native_instance_coverage_complete": full_instances,
            "native_instance_coverage_scope": "selected_tasks_and_configured_seeds",
            "full_native_coverage": full_tasks and full_seeds and full_instances,
            "per_task": per_task,
            "runtime_validation": "not_performed",
        },
        "protocol": {
            "name": "robodojo-a-public-layouts",
            "expected_evaluation_seeds": list(catalog.expected_seeds),
            "evaluation_seeds": list(seeds),
            "dimensions": copy.deepcopy(catalog.dimensions),
            "layout_order": "numeric_filename_suffix; layout_id_is_sorted_index",
            "episodes_override_total_per_runnable_task": count,
            "episodes_selection": selection_mode,
            "success_grace_steps": grace,
            "termination": "bole_first_native_success_plus_control_steps",
            "official_verified": False,
            "is_official_termination": False,
            "missing_native_result_policy": "primary_null_with_scored_subset_and_coverage",
            "model_history_resume": False,
            "learn_chain_scope": "one_continuous_chain_per_runnable_task" if mode == "learn" else None,
        },
    }
