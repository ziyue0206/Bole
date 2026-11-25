"""Shared eval/learn orchestration. Benchmarks own physics and native verdicts."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import platform
import time
from uuid import uuid4

import yaml

from bole import __version__, registry
from bole.engine.prompts import render
from bole.engine.termination import EpisodeBoundary
from bole.results.storage import RecordStore, redact
from bole.results.summary import summarize as summarize_micro, native_result
from bole.results.usage import UsageLedger
from bole.tools import EpisodeTools


class PreflightError(RuntimeError):
    pass


def new_run_root(config):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    root = Path(config["output"]) / (stamp + "-" + uuid4().hex[:12])
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    return root.resolve()


def preflight(config, benchmark, *, runtime=True):
    """Read-only checks. Plan/check remain available without a GPU."""
    check = benchmark.check(config)
    errors = []
    if isinstance(check, dict) and check.get("ok") is False:
        errors.append("benchmark resource checks failed")
    if runtime and getattr(benchmark, "requires_gpu", False) and not config["resources"]["gpus"]:
        errors.append("This benchmark requires a GPU; select an available GPU with --gpu. plan/check require no GPU.")
    return {"ok": not errors, "runtime_requested": runtime, "benchmark": check,
            "gpu_required": bool(getattr(benchmark, "requires_gpu", False)),
            "selected_gpus": list(config["resources"]["gpus"]), "errors": errors}


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def _error(exc, phase="unknown", log_ref=None):
    return {"type": type(exc).__name__, "message": str(exc), "phase": phase, "log_ref": log_ref}


def _record_error(store, exc, phase):
    error = {**_error(exc, phase, "logs/runner.jsonl"), "error_id": "error-" + uuid4().hex,
             "occurred_at": datetime.now(timezone.utc).isoformat()}
    store.append_jsonl(error["log_ref"], error)
    return error


async def _settle_cleanup(awaitable, on_cancel):
    """Finish a specific owned cleanup even if its caller is cancelled again."""
    task = asyncio.ensure_future(awaitable)
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            on_cancel()
            if task.done():
                return task.result()


def _secrets(config):
    key = os.environ.get(config["harness"].get("gateway", {}).get("credential_env", ""))
    return (key,) if key else ()


def _safe_component(value):
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("task and episode identifiers must be safe path components")
    return value


def _identity(episode):
    return {key: episode.get(key) for key in ("task_id", "episode_id", "seed", "layout_id",
                                              "logical_task_id", "predecessor", "learning_chain_id")}


async def _drive(harness, tools, prompt, content, config, startup_deadline=None):
    """Race native turn completion against explicit episode budgets.

    Episode time starts in the delivered callback, never during simulator or
    harness initialization. The separate startup deadline covers acknowledgement.
    """
    task = asyncio.create_task(harness.run_episode(prompt, content, tools.mark_delivered, None))
    deadline = startup_deadline or (asyncio.get_running_loop().time() + config["limits"]["startup_timeout_s"])
    try:
        while not task.done():
            if not tools.delivery.is_set() and asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("native harness did not acknowledge initial observation before startup timeout")
            reason = tools.boundary.check() if tools.delivery.is_set() else None
            if tools.delivery.is_set() and (reason or tools.finished.is_set()):
                await harness.stop_episode(reason or "infrastructure_error")
                try:
                    await asyncio.wait_for(asyncio.shield(task), 20)
                    break
                except TimeoutError as exc:
                    raise RuntimeError("native harness did not settle after episode closure") from exc
            await asyncio.wait((task,), timeout=.02)
        outcome = task.result() or {}
        if tools.boundary.reason is None:
            reason = outcome.get("termination", outcome.get("termination_reason"))
            if reason == "finish" or outcome.get("finish_called"):
                tools.boundary.finish()
            elif reason and reason != "final_response_without_finish":
                tools.boundary.end(reason)
            else:
                tools.boundary.final_response()
        if outcome.get("session_usable") is False:
            raise RuntimeError("native harness session is not reusable")
        return outcome
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _episode(config, episode, benchmark, harness, *, opened, run_id, run_root,
                   harness_root, gpu, policy):
    episode_root = run_root / "tasks" / episode["task_id"] / "episodes" / episode["episode_id"]
    store = RecordStore(episode_root, {"run_id": run_id, **_identity(episode)}, secrets=_secrets(config))
    context = {"config": config, "run_id": run_id, **_identity(episode), "episode": episode,
               "root": harness_root, "episode_root": episode_root, "store": store,
               "gpu": gpu, "policy": policy, "input_asset": episode.get("input_asset")}
    context["layout_path"] = episode.get("layout_path")
    boundary = EpisodeBoundary(grace_steps=config["termination"]["success_grace_steps"],
                               timeout_s=config["limits"]["timeout_s"],
                               step_limit=config["limits"]["control_steps"])
    session = tools = None
    closed, outcome, errors, timing = {}, {}, [], {}
    started = cancelled = False
    phase = "environment_initialization"
    startup_tasks = []
    began = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    deadline = asyncio.get_running_loop().time() + config["limits"]["startup_timeout_s"]

    def mark_cancelled():
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            boundary.end("cancelled")
            errors.append(_record_error(store, asyncio.CancelledError("run cancelled"), phase))

    async def startup(awaitable, name):
        nonlocal phase
        phase = name
        task = asyncio.ensure_future(awaitable)
        startup_tasks.append(task)
        mark = time.monotonic()
        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                task.cancel()
                raise TimeoutError("startup deadline exceeded before " + name)
            try:
                return await asyncio.wait_for(asyncio.shield(task), remaining)
            except TimeoutError as exc:
                raise TimeoutError("startup deadline exceeded during " + name) from exc
        finally:
            timing[name + "_seconds"] = time.monotonic() - mark

    store.write_json("episode.json", {"mode": config["mode"], "plan": episode, "gpu": gpu,
                                      "input_asset": episode.get("input_asset")})
    store.event("episode_initializing", gpu=gpu)
    try:
        session = benchmark.session(context)
        started = True
        info = await startup(asyncio.to_thread(session.open), "environment_initialization")
        if boundary.step_limit is None:
            boundary.step_limit = info.get("step_limit")
        if info.get("private_boundary"):
            boundary.reconcile_native(info["private_boundary"])
        store.write_json("environment.json", info)
        tools = EpisodeTools(session, info, config, store, boundary, **({"policy": policy} if policy is not None else {}))
        content = await startup(tools.first_observation(), "initial_observation")
        prompt = render(config, episode, info, tools.definitions())
        store.asset("prompt.txt", prompt.encode(), text=True)
        if opened:
            await startup(_maybe_await(harness.bind_episode(store, tools)), "harness_bind")
        else:
            await startup(harness.open(context, tools), "harness_initialization")
            opened = True
        phase = "episode_execution"
        mark = time.monotonic()
        try:
            outcome = await _drive(harness, tools, prompt, content, config, deadline)
        finally:
            timing["harness_execution_seconds"] = time.monotonic() - mark
        if tools.error:
            raise RuntimeError(str(tools.error))
    except asyncio.CancelledError:
        mark_cancelled()
    except Exception as exc:
        boundary.end("infrastructure_error")
        errors.append(_record_error(store, exc, phase))
    finally:
        if tools:
            tools.stop()

        async def close_environment():
            nonlocal closed, phase
            # A timed-out synchronous plugin call can still be running. Settle
            # it before close where possible; never reuse this failed chain.
            phase = "startup_settle"
            settle_began = time.monotonic()
            for task in startup_tasks:
                if task.done():
                    if not task.cancelled():
                        task.exception()  # Retrieve any late exception after a timeout.
                    continue
                try:
                    await asyncio.wait_for(asyncio.shield(task), config["limits"]["request_timeout_s"])
                except (Exception, asyncio.CancelledError) as exc:
                    errors.append(_record_error(store, exc, phase))
                    task.cancel()
            timing["startup_settle_seconds"] = time.monotonic() - settle_began
            phase = "environment_close"
            mark = time.monotonic()
            if session is not None:
                try:
                    # The adapter owns its bounded socket/stop/video recovery
                    # stages. A second request-level timeout here would abandon
                    # a live thread and discard late authoritative receipts.
                    receipt = await asyncio.to_thread(session.close,
                        "cancelled" if cancelled else boundary.reason or "infrastructure_error")
                    if receipt is not None and not isinstance(receipt, dict):
                        raise TypeError("environment close must return a result object or null")
                    closed = receipt or {}
                except (Exception, asyncio.CancelledError) as exc:
                    errors.append(_record_error(store, exc, phase))
            timing["environment_close_seconds"] = time.monotonic() - mark

        await _settle_cleanup(close_environment(), mark_cancelled)
        try:
            boundary.reconcile_native(closed)
        except (TypeError, ValueError) as exc:
            errors.append(_record_error(store, exc, "final_native_boundary"))
        # The close result is authoritative, including counters whose action
        # receipt was interrupted. In-run success never replaces the final grade.
        final_native = closed.get("native_result", {"available": False, "reason": "native_close_result_missing"})
        if native_result({"native_result": final_native}) is None and not cancelled:
            errors.append(_record_error(store, RuntimeError("no valid native final success verdict"), "native_evaluation"))
        if closed.get("infrastructure_error"):
            errors.append(_record_error(store, RuntimeError("environment reported an infrastructure failure; see native close records"), "environment_close"))
        infrastructure = (bool(errors) or bool(closed.get("infrastructure_error"))) and not cancelled
        reason = "cancelled" if cancelled else "infrastructure_error" if infrastructure else boundary.reason
        timing.update(wall_seconds=time.monotonic() - began, started_at=started_at,
                      ended_at=datetime.now(timezone.utc).isoformat())
        result = {**_identity(episode), **closed, **boundary.report(), "native_result": final_native,
                  "mode": config["mode"], "benchmark": config["benchmark"]["name"],
                  "episode_termination_reason": boundary.reason, "termination_reason": reason,
                  "status": "cancelled" if cancelled else "infrastructure_error" if infrastructure else "completed",
                  "started": started, "completed": started and not infrastructure and not cancelled,
                  "infrastructure_error": infrastructure, "cancelled": cancelled,
                  "errors": errors, "harness": outcome, "input_asset": episode.get("input_asset"),
                  "requested_control_steps": tools.requested_steps if tools else 0,
                  "tool_calls": tools.tool_calls if tools else 0, "timing": timing,
                  "references": {"episode": "episode.json", "environment": "environment.json" if (episode_root / "environment.json").is_file() else None,
                      "prompt": "prompt.txt" if (episode_root / "prompt.txt").is_file() else None,
                      "usage": "usage.json", "artifacts": "artifacts.json", "safety": "safety.json",
                      "events": "trajectory/events.jsonl"}}
        store.write_json("result.json", result)
        store.event("episode_closed", result=result)
    return result, opened, store


def _merge_usage(ledger, report, session_key):
    for row in [*report.get("requests", []), *report.get("usage_observations", [])]:
        request_id = session_key + ":" + str(row["request_id"])
        source = row.get("source", "model")
        ignored = {"source", "request_id", "raw_usage", "provider_request_id", "status",
                   "provider_response_id", "actual_cost", "estimated_cost", "aliases", "tokens", "missing_fields"}
        meta = {k: v for k, v in row.items() if k not in ignored}
        ledger.start(request_id, source=source, **meta)
        ledger.finish(request_id, row.get("raw_usage"), source=source,
                      status=row.get("status", "started"), provider_request_id=row.get("provider_request_id"),
                      provider_response_id=row.get("provider_response_id"),
                      actual_cost=row.get("actual_cost"), estimated_cost=row.get("estimated_cost"), **meta)
    for snapshot in report.get("session_usage_unallocated", []):
        values = {k: v for k, v in snapshot.items() if k not in {"session_id", "source", "usage"}}
        ledger.snapshot(session_key + ":" + str(snapshot["session_id"]), snapshot.get("usage"),
                        source=snapshot.get("source", "codex"), **values)
    ledger.issues.extend(report.get("issues", []))


def _episode_usage(report, result):
    """Allocate only identified requests; never turn a session total into deltas."""
    ledger = UsageLedger()
    selected = {"requests": [r for r in report.get("requests", [])
                              if r.get("episode_id") == result["episode_id"]],
                "usage_observations": [r for r in report.get("usage_observations", [])
                                       if r.get("episode_id") == result["episode_id"]],
                "issues": report.get("issues", []), "session_usage_unallocated": []}
    _merge_usage(ledger, selected, result["episode_id"])
    ledger.tool_calls = result.get("tool_calls", 0)
    ledger.control_steps = result.get("actual_control_steps", 0)
    return {**ledger.report(), "timing": {**result.get("timing", {}),
                "initialization_s": result.get("initialization_s"),
                "elapsed_after_first_observation_s": result.get("elapsed_after_first_observation_s"),
                "boundary_timing_source": "result.json authoritative episode boundary"},
            "scope": "identified_episode_requests_including_boundary_feedback",
            "session_finalization_ref": "../../session-finalizations.json",
            "session_key": result.get("harness_session_key"),
            "session_usage_unallocated_ref": "../../usage.json"}


def _safety_record(result, store, profile):
    from bole.probes.safety import SafetyEvaluator
    fallback = SafetyEvaluator(profile).report(result["episode_id"], complete=False, expected=0)
    for constraint in fallback["constraints"]:
        constraint.update(reason="safety_not_collected", collection_status="not_collected")
    value = result.get("safety")
    path = store.root / "safety.json"
    if not isinstance(value, dict) and path.is_file():
        value = json.loads(path.read_text())
    if not isinstance(value, dict):
        value = fallback
    else:
        known = {r["family_id"]: r for r in value.get("constraints", [])}
        value = {**value, "constraints": [known.get(r["family_id"], r) for r in fallback["constraints"]]}
    final = native_result(result)
    value = {**value, **store.identity, "success": final["success"] if final is not None else None}
    store.write_json("safety.json", value)
    return value


async def run(config, plan=None, *, benchmark=None, harness_factory=None, root=None,
              check=True, policy_factory=None):
    """Execute an already frozen config; injected adapters support CPU tests."""
    run_began = time.monotonic()
    run_started_at = datetime.now(timezone.utc).isoformat()
    benchmark = benchmark or registry.load("benchmarks", config["benchmark"]["name"])
    harness_factory = harness_factory or (lambda: registry.load("harnesses", config["harness"]["name"]))
    report = None
    if check:
        report = preflight(config, benchmark)
        if not report["ok"]:
            raise PreflightError("; ".join(report["errors"]))
    plan = plan if plan is not None else benchmark.plan(config)
    episodes = list(plan["episodes"])
    identities = [(e["task_id"], e["episode_id"]) for e in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError("episode identifiers must be unique within each task")
    for task_id, episode_id in identities:
        _safe_component(task_id)
        _safe_component(episode_id)
    root = Path(root).resolve() if root else new_run_root(config)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (root / "results.json").exists() or (root / "trajectory/events.jsonl").exists():
        raise FileExistsError("run evidence already exists; choose a new run directory")
    run_id = root.name
    store = RecordStore(root, {"run_id": run_id}, secrets=_secrets(config))
    store.asset("config.yaml", yaml.safe_dump(redact({**config, "schema_version": 1, "run_id": run_id},
        _secrets(config)), allow_unicode=True, sort_keys=False).encode(), text=True)
    store.write_json("plan.json", plan)
    preflight_ref, check_scope, check_error = None, "not_collected", None
    if report is not None:
        preflight_ref = store.write_json("runtime-preflight.json", report)
        check_scope = "runner_preflight_before_episode_initialization"
    elif (root / "preflight.json").is_file():
        preflight_ref, check_scope = "preflight.json", "parent_preflight_before_code_freeze"
        try:
            report = json.loads((root / preflight_ref).read_text())
            if not isinstance(report, dict):
                raise ValueError("preflight report is not an object")
        except (OSError, ValueError) as exc:
            report, check_error = None, str(exc)
    observed = (report or {}).get("benchmark", {}).get("resources", {})
    evaluator = {"status": "not_reported", "reason": "benchmark_did_not_declare_evaluator_provenance"}
    if config["benchmark"]["name"] == "robodojo":
        source = config["resources"].get("source")
        evaluator = {"status": "declared_source_entry_points", "source_root": source,
            "declared_source_revision": config["resources"].get("source_revision"),
            "observed_source_revision": observed.get("source_revision"),
            "observed_revision_status": "observed" if observed.get("source_revision") else "not_reported",
            "task_implementations": "task/RoboDojo/tasks", "task_configurations": "task/RoboDojo/config",
            "entry_points": ["src/eval_client/eval_env.py:create_eval_env.EvalEnv.run_eval",
                             "reward_manager.get_reward(final_check=True)"],
            "native_output": "_result.json", "path_base": "source_root",
            "episode_evidence": "tasks/<task_id>/episodes/<episode_id>/native-evaluation/result.json",
            "runtime_execution_verified_by_resource_check": False}
    store.write_json("resources.json", {"resources": config["resources"],
        "snapshot_manifest": "snapshot.json" if (root / "snapshot.json").exists() else None,
        "execution_code_frozen": (root / "snapshot.json").exists(),
        "preflight_ref": preflight_ref, "preflight_scope": check_scope,
        "preflight_read_error": check_error, "observed_resources": observed,
        "observed_resources_status": "collected" if observed else "not_reported",
        "observed_resources_reason": None if observed else "preflight_resource_evidence_unavailable",
        "image_evidence_scope": "preflight_selected_base_images; actual harness image IDs are in episode harness_started events",
        "native_evaluator": evaluator,
        "runtime_versions": {
            "bole": {"version": __version__, "status": "observed", "scope": "runner_import"},
            "runner_python": {"version": platform.python_version(), "status": "observed"},
            "runner_pyyaml": {"version": yaml.__version__, "status": "observed"},
            "harness_executable": {"path": config["harness"].get("executable"), "version": None,
                "status": "not_collected", "reason": "native_executable_version_not_probed_for_this_run"},
            "simulator_dependencies": {"versions": None, "status": "not_collected",
                "reason": "installed_simulator_dependency_versions_not_probed_for_this_run",
                "environment": config["resources"].get("simulator_image")}},
        "task_instance_plan_ref": "plan.json"})
    store.event("run_started", mode=config["mode"], planned_episodes=len(episodes))
    units = defaultdict(list)
    for episode in episodes:
        key = episode["task_id"] if config["mode"] == "learn" else episode["task_id"] + ":" + episode["episode_id"]
        units[key].append(episode)
    if config["mode"] == "learn":
        for rows in units.values():
            previous = None
            for row in rows:
                if row.get("predecessor") != previous:
                    raise ValueError("learning episode predecessor does not match its task chain")
                previous = row["episode_id"]
    results, chains, lifecycle_errors, session_finalizations = {}, {}, [], {}
    usage = UsageLedger()
    task_usage = defaultdict(UsageLedger)
    semaphore = asyncio.Semaphore(config[config["mode"]]["parallel"])
    gpu_queue = asyncio.Queue()
    needs_gpu = bool(getattr(benchmark, "requires_gpu", False))
    for gpu in config["resources"]["gpus"]:
        gpu_queue.put_nowait(gpu)

    async def unit(key, rows):
        harness = policy = None
        opened, interrupted, gpu = False, False, None
        last_store = None
        chain = {"task_id": rows[0]["task_id"], "episode_ids": [], "status": "running",
                 "input_asset": rows[0].get("input_asset"), "asset": None}
        finalization = {"session_key": key, "task_id": rows[0]["task_id"],
            "episode_ids": chain["episode_ids"],
            "asset_export_seconds": None, "harness_close_seconds": None, "policy_close_seconds": None,
            "scope": "one_harness_session; shared_learning_chain_cleanup_not_allocated_to_each_episode"}

        def mark_unit_cancelled():
            nonlocal interrupted
            interrupted = True
            chain["status"] = "cancelled"

        async with semaphore:
            if needs_gpu:
                gpu = await gpu_queue.get()
            try:
                harness = harness_factory()
                if config.get("policy"):
                    settings = config["policy"] if isinstance(config["policy"], dict) else {"name": config["policy"]}
                    policy = policy_factory() if policy_factory else registry.load("policies", settings["name"])
                    await _maybe_await(policy.load(settings, config["resources"]))
                harness_root = (root / "tasks" / rows[0]["task_id"] / "session" if config["mode"] == "learn"
                                else root / "tasks" / rows[0]["task_id"] / "episodes" / rows[0]["episode_id"] / "session")
                for row in rows:
                    result, opened, last_store = await _episode(config, row, benchmark, harness,
                        opened=opened, run_id=run_id, run_root=root, harness_root=harness_root, gpu=gpu, policy=policy)
                    result["harness_session_key"] = key
                    results[(row["task_id"], row["episode_id"])] = result
                    chain["episode_ids"].append(row["episode_id"])
                    if result["infrastructure_error"] or result["cancelled"]:
                        interrupted = True
                        chain["status"] = result["status"]
                        break
                    if config["mode"] == "learn":
                        final = native_result(result)
                        feedback = {"episode_id": row["episode_id"],
                                    "native_result": final if final is not None else {"available": False, "reason": "native_result_unavailable"},
                                    "native_success": final["success"] if final is not None else None,
                                    "native_score": final.get("score") if final is not None else None,
                                    "termination_reason": result["termination_reason"],
                                    "finish_called": result["finish_called"],
                                    "success_observed": result["success_observed"],
                                    "first_success_control_step": result["first_success_control_step"],
                                    "executed_post_success_steps": result["executed_post_success_steps"],
                                    "next_episode_planned": row is not rows[-1]}
                        text = "Previous episode closed. Native feedback: " + json.dumps(feedback)
                        delivery = {"text": text, "delivered": False, "status": "prepared",
                                    "prepared_at": datetime.now(timezone.utc).isoformat(),
                                    "delivery_boundary": "harness_feedback_completed; native_request_evidence_is_separate"}
                        last_store.write_json("feedback/native.json", delivery)
                        last_store.event("native_feedback_prepared", feedback_ref="feedback/native.json")
                        mark = time.monotonic()
                        try:
                            delivery.update(status="submitted_to_harness", submitted_at=datetime.now(timezone.utc).isoformat())
                            last_store.write_json("feedback/native.json", delivery)
                            await harness.feedback(text)
                            delivery.update(status="acknowledged_by_harness", delivered=True,
                                            acknowledged_at=datetime.now(timezone.utc).isoformat())
                            last_store.event("native_feedback_delivered", feedback_ref="feedback/native.json")
                        except asyncio.CancelledError:
                            mark_unit_cancelled()
                            result["post_episode_error"] = _record_error(last_store, asyncio.CancelledError("cancelled during feedback"), "feedback")
                            result.update(cancelled=True, status="cancelled", termination_reason="cancelled")
                            result["errors"].append(result["post_episode_error"])
                            delivery.update(status="delivery_cancelled", error=result["post_episode_error"])
                            break
                        except Exception as exc:
                            interrupted = True
                            result["infrastructure_error"] = True
                            result["post_episode_error"] = _record_error(last_store, exc, "feedback")
                            result.update(status="infrastructure_error", termination_reason="infrastructure_error")
                            result["errors"].append(result["post_episode_error"])
                            delivery.update(status="delivery_failed", error=result["post_episode_error"])
                            chain["status"] = "feedback_infrastructure_error"
                            break
                        finally:
                            last_store.write_json("feedback/native.json", delivery)
                            result["timing"]["feedback_seconds"] = time.monotonic() - mark
                            result["references"]["feedback"] = "feedback/native.json"
                            last_store.write_json("result.json", result)
                if not interrupted:
                    chain["status"] = "completed"
            except asyncio.CancelledError:
                mark_unit_cancelled()
            except Exception as exc:
                interrupted = True
                chain["status"] = "infrastructure_error"
                chain["error"] = _record_error(store, exc, "unit_initialization")
                lifecycle_errors.append({"unit": key, **chain["error"]})
            finally:
                async def finalize_unit():
                    try:
                        if harness is not None:
                            if config["mode"] == "learn":
                                destination = root / "learning" / rows[0]["task_id"]
                                export_began = time.monotonic()
                                try:
                                    episode_results = [results[(row["task_id"], row["episode_id"])] for row in rows
                                                       if (row["task_id"], row["episode_id"]) in results]
                                    chain["asset"] = await harness.export_asset(destination,
                                        {"episodes_completed": sum(row.get("completed", False) for row in episode_results),
                                         "episodes_started": sum(row.get("started", False) for row in episode_results),
                                         "native_valid_episodes": sum(native_result(row) is not None for row in episode_results),
                                         "planned_episodes": len(rows), "chain_status": chain["status"]}, partial=interrupted)
                                    chain["asset_ref"] = str((destination / "manifest.json").relative_to(root))
                                    if chain["asset"].get("export_status") != "saved":
                                        raise RuntimeError("learning asset export did not save physical components")
                                except (Exception, asyncio.CancelledError) as exc:
                                    chain["asset"] = chain["asset"] or {"export_status": "failed"}
                                    chain["episode_chain_status"] = chain["status"]
                                    chain["status"] = "asset_export_failed"
                                    error = _record_error(store, exc, "asset_export")
                                    chain["asset"]["error"] = error
                                    lifecycle_errors.append({"unit": key, **error})
                                finally:
                                    finalization["asset_export_seconds"] = time.monotonic() - export_began
                            close_began = time.monotonic()
                            try:
                                await harness.close()
                            except (Exception, asyncio.CancelledError) as exc:
                                lifecycle_errors.append({"unit": key, **_record_error(store, exc, "harness_close")})
                            finally:
                                finalization["harness_close_seconds"] = time.monotonic() - close_began
                            report = harness.ledger.report() if hasattr(harness, "ledger") else UsageLedger().report()
                            _merge_usage(usage, report, key)
                            _merge_usage(task_usage[rows[0]["task_id"]], report, key)
                            for row in rows:
                                result = results.get((row["task_id"], row["episode_id"]))
                                if result is None:
                                    continue
                                episode_store = RecordStore(root / "tasks" / row["task_id"] / "episodes" / row["episode_id"],
                                    {"run_id": run_id, **_identity(row)}, secrets=_secrets(config))
                                episode_store.write_json("usage.json", _episode_usage(report, result))
                    finally:
                        if policy is not None:
                            close_began = time.monotonic()
                            try:
                                await _maybe_await(policy.close())
                            except (Exception, asyncio.CancelledError) as exc:
                                lifecycle_errors.append({"unit": key, **_record_error(store, exc, "policy_close")})
                            finally:
                                finalization["policy_close_seconds"] = time.monotonic() - close_began
                        session_finalizations[key] = finalization
                        if needs_gpu:
                            gpu_queue.put_nowait(gpu)
                        if config["mode"] == "learn":
                            chain["finalization_timing"] = finalization
                            chains[key] = chain
                            RecordStore(root / "tasks" / rows[0]["task_id"], {"run_id": run_id, "task_id": rows[0]["task_id"]},
                                        secrets=_secrets(config)).write_json("chain.json", chain)

                await _settle_cleanup(finalize_unit(), mark_unit_cancelled)

    workers = [asyncio.create_task(unit(key, rows)) for key, rows in units.items()]
    cancelled = False
    try:
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        cancelled = True
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    except Exception as exc:
        lifecycle_errors.append(_record_error(store, exc, "scheduler"))
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    ordered = []
    for episode in episodes:
        ident = (episode["task_id"], episode["episode_id"])
        row = results.get(ident)
        if row is None:
            row = {**_identity(episode), "mode": config["mode"], "benchmark": config["benchmark"]["name"],
                   "started": False, "completed": False, "status": "not_run",
                   "termination_reason": "not_run", "not_run_reason": "run_cancelled" if cancelled else "chain_or_initialization_stopped",
                   "input_asset": episode.get("input_asset"),
                   "native_result": {"available": False, "reason": "not_run"}}
            # Unstarted episodes live in plan/results/summary only. Their absence
            # must not be disguised by fabricating episode output directories.
        ordered.append(row)
    usage.tool_calls = sum(row.get("tool_calls", 0) for row in ordered)
    usage.control_steps = sum(row.get("actual_control_steps", 0) for row in ordered)

    from bole.probes.profile import load_profile
    from bole.probes.safety import write_reports
    from bole.results.artifacts import write_artifact_index
    profile = load_profile()
    safety_rows = []
    for row in ordered:
        if row["status"] == "not_run":
            continue
        episode_store = RecordStore(root / "tasks" / row["task_id"] / "episodes" / row["episode_id"],
            {"run_id": run_id, **_identity(row)}, secrets=_secrets(config))
        try:
            safety_rows.append(_safety_record(row, episode_store, profile))
            if not (episode_store.root / "usage.json").is_file():
                episode_store.write_json("usage.json", _episode_usage({}, row))
            row["references"].update(config="../../../../config.yaml", resources="../../../../resources.json",
                                      plan="../../../../plan.json")
            for field, relative in (("native_evaluation", "native-evaluation/result.json"),
                                    ("control_steps", "trajectory/control-steps.jsonl"),
                                    ("observations", "observations"), ("videos", "videos"),
                                    ("model_events", "model-events")):
                path = episode_store.root / relative
                present = path.is_file() or (path.is_dir() and any(p.is_file() for p in path.rglob("*")))
                row["references"][field] = relative if present else None
            row["artifacts"] = {"manifest_ref": "artifacts.json", "path_base": "episode"}
            episode_store.write_json("result.json", row)
            write_artifact_index(episode_store, partial=not row.get("completed", False),
                video_enabled=config.get("recording", {}).get("video", {}).get("enabled"))
        except Exception as exc:
            error = _record_error(episode_store, exc, "episode_reports")
            row.setdefault("collection_errors", []).append(error)
            lifecycle_errors.append({"task_id": row["task_id"], "episode_id": row["episode_id"], **error,
                "log_ref": str((episode_store.root / error["log_ref"]).relative_to(root)), "path_base": "run"})
            episode_store.write_json("result.json", row)
    try:
        write_reports(root, safety_rows, profile, len(episodes), identity={"run_id": run_id})
    except Exception as exc:
        lifecycle_errors.append(_record_error(store, exc, "safety_aggregation"))
    for task_id in dict.fromkeys(row["task_id"] for row in episodes):
        planned = [row for row in episodes if row["task_id"] == task_id]
        task_results = [row for row in ordered if row["task_id"] == task_id]
        task_store = RecordStore(root / "tasks" / task_id, {"run_id": run_id, "task_id": task_id}, secrets=_secrets(config))
        task_store.write_json("summary.json", {**summarize_micro(task_results, planned), "mode": config["mode"],
            "benchmark": config["benchmark"]["name"], "episode_ids": [r["episode_id"] for r in planned]})
        task_finalizations = [row for row in session_finalizations.values() if row["task_id"] == task_id]
        task_store.write_json("session-finalizations.json", {"sessions": task_finalizations,
            "aggregation": "one_record_per_session; do_not_sum_again_per_episode"})
        ledger = task_usage[task_id]
        ledger.tool_calls = sum(r.get("tool_calls", 0) for r in task_results)
        ledger.control_steps = sum(r.get("actual_control_steps", 0) for r in task_results)
        task_store.write_json("usage.json", {**ledger.report(), "timing": {
            "sum_episode_wall_seconds": sum(r.get("timing", {}).get("wall_seconds", 0) for r in task_results),
            "sum_feedback_seconds": sum(r.get("timing", {}).get("feedback_seconds", 0) for r in task_results),
            "session_finalizations": task_finalizations,
            "aggregation": "episode_duration_sums_not_parallel_wall_time"}})
        try:
            write_reports(task_store.root, [r for r in safety_rows if r["task_id"] == task_id], profile,
                          len(planned), identity={"run_id": run_id, "task_id": task_id})
        except Exception as exc:
            error = _record_error(task_store, exc, "task_safety_aggregation")
            lifecycle_errors.append({"task_id": task_id, **error,
                "log_ref": str((task_store.root / error["log_ref"]).relative_to(root)), "path_base": "run"})
    try:
        benchmark_metrics = benchmark.summary(plan, ordered) if hasattr(benchmark, "summary") else {}
    except Exception as exc:
        benchmark_metrics = {"success_rate": None, "aggregation_error": _record_error(store, exc, "benchmark_aggregation")}
        lifecycle_errors.append(benchmark_metrics["aggregation_error"])
    micro = summarize_micro(ordered, episodes)
    fully_scored = bool(episodes) and micro["coverage"]["native_missing"] == 0
    cancelled = cancelled or any(row.get("cancelled") for row in ordered) or any(
        chain.get("status") == "cancelled" for chain in chains.values())
    pending_memory = [{"task_id": task_id, "manifest_ref": chain.get("asset_ref"),
                       "finalize_status": chain["asset"].get("finalize_status"),
                       "acceptance_status": chain["asset"].get("acceptance_status")}
                      for task_id, chain in chains.items() if isinstance(chain.get("asset"), dict)
                      and chain["asset"].get("finalize_status") == "pending"]
    learning_images = [{"task_id": task_id, "status": chain["asset"].get("status"),
                        **chain["asset"]["components"]["image"]}
                       for task_id, chain in chains.items() if isinstance(chain.get("asset"), dict)
                       and chain["asset"].get("export_status") == "saved"
                       and "image" in chain["asset"].get("components", {})]
    timing = {"wall_seconds": time.monotonic() - run_began, "started_at": run_started_at,
              "ended_at": datetime.now(timezone.utc).isoformat(),
              "session_finalizations": list(session_finalizations.values()),
              "scope": "runner_process_excludes_parent_code_snapshot_preparation"}
    summary = {"schema_version": 1, "benchmark": config["benchmark"]["name"], "mode": config["mode"],
               "success_rate": benchmark_metrics.get("success_rate") if fully_scored else None,
               "success_rate_reason": ("planned_native_results_missing_or_empty_plan" if not fully_scored else
                   "benchmark_aggregation_failed" if "aggregation_error" in benchmark_metrics else
                   "benchmark_metric_not_reported" if benchmark_metrics.get("success_rate") is None else None),
               "benchmark_metrics": benchmark_metrics, "episode_micro": micro,
               "learning_chains": chains, "learning_images": learning_images, "pending_memory_assets": pending_memory,
               "lifecycle_errors": lifecycle_errors, "timing": timing,
               "cancelled": cancelled, "run_path": str(root),
               "references": {"config": "config.yaml", "plan": "plan.json", "resources": "resources.json",
                              "results": "results.json", "usage": "usage.json", "safety": "safety_summary.json",
                              "constraints": "constraint_metrics.csv"}}
    summary["status"] = "cancelled" if cancelled else "incomplete" if lifecycle_errors or any(
        r.get("infrastructure_error") or r.get("status") == "not_run" for r in ordered) else "completed"
    store.write_json("results.json", {"results": ordered})
    store.write_json("usage.json", {**usage.report(), "timing": timing})
    store.write_json("summary.json", summary)
    store.event("run_closed", status=summary["status"])

    return summary


def child_main(argv=None):
    parser = argparse.ArgumentParser(description="Internal frozen Bole runner")
    parser.add_argument("--child-config", required=True)
    parser.add_argument("--child-plan", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.child_config).read_text())
    plan = json.loads(Path(args.child_plan).read_text())
    summary = asyncio.run(run(config, plan, root=args.run_dir))
    print(json.dumps({"run_path": summary["run_path"], "status": summary["status"],
                      "success_rate": summary["success_rate"], "learning_images": summary["learning_images"]}, ensure_ascii=False))
    return 0 if summary["status"] == "completed" else 130 if summary["status"] == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(child_main())
