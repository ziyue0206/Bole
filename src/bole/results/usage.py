"""Request accounting with response usage and request-attempt tracking.

Request snapshots are authoritative; native session cumulative snapshots are
retained separately and never added to request totals.
"""
from __future__ import annotations

import json
import math
import time
from copy import deepcopy

FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens", "total_tokens")


def response_usage(body):
    if isinstance(body, bytes):
        body = body.decode(errors="replace")
    try:
        values = [json.loads(body)] if isinstance(body, str) else [body]
    except (ValueError, TypeError):
        values = []
        for line in str(body).splitlines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                try:
                    values.append(json.loads(line[5:]))
                except ValueError:
                    continue
    usage, provider_id = None, None
    for value in values:
        if not isinstance(value, dict):
            continue
        for item in (value, value.get("message"), value.get("response")):
            if isinstance(item, dict):
                provider_id = item.get("id") or provider_id
                if isinstance(item.get("usage"), dict):
                    usage = {**(usage or {}), **item["usage"]}
    return {"raw_usage": usage, "provider_response_id": provider_id}


def normalize_usage(raw):
    raw = raw or {}
    inp = raw.get("input_tokens", raw.get("inputTokens", raw.get("prompt_tokens")))
    out = raw.get("output_tokens", raw.get("outputTokens", raw.get("completion_tokens")))
    in_details = raw.get("input_tokens_details", raw.get("prompt_tokens_details")) or {}
    out_details = raw.get("output_tokens_details", raw.get("completion_tokens_details")) or {}
    result = dict(zip(FIELDS, (inp, out,
        in_details.get("cached_tokens", raw.get("cached_input_tokens", raw.get("cachedInputTokens"))),
        out_details.get("reasoning_tokens", raw.get("reasoning_output_tokens", raw.get("reasoningOutputTokens"))),
        raw.get("total_tokens", raw.get("totalTokens")))))
    return {k: v if type(v) is int and v >= 0 else None for k, v in result.items()}


class UsageLedger:
    def __init__(self):
        self.requests = {}
        self.session_snapshots = {}
        self.issues = []
        self.tool_calls = 0
        self.control_steps = 0

    def start(self, request_id, *, source="model", phase="episode", **meta):
        if not request_id:
            raise ValueError("a request ID is required")
        key = (source, str(request_id))
        row = self.requests.setdefault(key, {"source": source, "request_id": str(request_id),
            "phase": phase, "status": "started", "raw_usage": None, "provider_request_id": None,
            "provider_response_id": None, "observation_only": False,
            "started_monotonic": time.monotonic(), "actual_cost": None, "estimated_cost": None})
        row.update(meta)
        return row

    def finish(self, request_id, usage=None, *, source="model", status="completed",
               provider_request_id=None, provider_response_id=None,
               actual_cost=None, estimated_cost=None, **meta):
        row = self.requests.get((source, str(request_id)))
        if row is None:
            row = self.start(request_id, source=source)
            row["start_observed"] = False
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("usage must be an object or null")
            row["raw_usage"] = {**(row["raw_usage"] or {}), **usage}
        row.update(status=status, **meta)
        if provider_request_id:
            row["provider_request_id"] = str(provider_request_id)
        if provider_response_id:
            row["provider_response_id"] = str(provider_response_id)
        for name, cost in (("actual_cost", actual_cost), ("estimated_cost", estimated_cost)):
            if cost is not None:
                if not isinstance(cost, dict) or not cost.get("currency") or not cost.get("source"):
                    raise ValueError("cost needs an amount, currency and explicit source")
                amount = cost.get("amount")
                if type(amount) not in (float, int) or not math.isfinite(amount) or amount < 0:
                    raise ValueError("cost must be finite and nonnegative")
                row[name] = dict(cost)
        row["duration_seconds"] = meta.get("duration_seconds", time.monotonic() - row["started_monotonic"])
        return row

    def snapshot(self, session_id, usage, *, source="codex", **meta):
        self.session_snapshots[(source, str(session_id))] = {
            "session_id": str(session_id), "source": source, "level": "session_cumulative",
            "usage": deepcopy(usage), "allocation": "unallocated_not_added_to_request_totals", **meta}

    def report(self, *, episode_id=None, task_id=None, phases=None):
        """Keep every attempt; deduplicate only the separately reported usage.

        Request/response IDs have distinct namespaces. Repeated snapshots of a
        response must not erase failed attempts or their timing/attribution.
        Session cumulative snapshots are never added to request totals.
        """
        selected_phases = {phases} if isinstance(phases, str) else set(phases) if phases is not None else None
        records = []
        for original in self.requests.values():
            if episode_id is not None and original.get("episode_id") != episode_id:
                continue
            if task_id is not None and original.get("task_id") != task_id:
                continue
            if selected_phases is not None and original.get("phase") not in selected_phases:
                continue
            row = deepcopy(original)
            row["tokens"] = normalize_usage(row["raw_usage"])
            row["missing_fields"] = [k for k, v in row["tokens"].items() if v is None]
            row.pop("started_monotonic", None)
            records.append(row)
        attempts = [row for row in records if not row.get("observation_only")]
        observations = [row for row in records if row.get("observation_only")]
        groups = {}
        conflicts = []
        for original in records:
            row = deepcopy(original)
            key = (row["source"], row["request_id"])
            response_id, request_id = row.get("provider_response_id"), row.get("provider_request_id")
            group_key = (("response", row.get("provider_scope", "default"), response_id) if response_id else
                         ("http_request", row.get("provider_scope", "default"), request_id) if request_id else
                         ("attempt", *key))
            if group_key not in groups:
                row["aliases"] = [list(key)]
                row["attempt_statuses"] = [row["status"]]
                row["provider_http_request_ids"] = [request_id] if request_id else []
                groups[group_key] = row
            else:
                existing = groups[group_key]
                existing["aliases"].append(list(key))
                existing["attempt_statuses"].append(row["status"])
                if request_id and request_id not in existing["provider_http_request_ids"]:
                    existing["provider_http_request_ids"].append(request_id)
                merged = existing["raw_usage"] or {}
                for field, value in (row["raw_usage"] or {}).items():
                    if field in merged and merged[field] != value:
                        conflicts.append({"provider_request_id": request_id,
                                          "provider_response_id": response_id, "field": field})
                        merged[field] = None
                    else:
                        merged[field] = value
                existing["raw_usage"] = merged or None
                if row["status"] == "completed":
                    existing["status"] = "completed"
                for field in ("actual_cost", "estimated_cost"):
                    if existing[field] is None:
                        existing[field] = row[field]
        rows = list(groups.values())
        for row in rows:
            row["tokens"] = normalize_usage(row["raw_usage"])
            row["missing_fields"] = [k for k, v in row["tokens"].items() if v is None]
            row.pop("started_monotonic", None)
        totals, field_coverage = {}, {}
        for field in FIELDS:
            values = [r["tokens"][field] for r in rows if r["tokens"][field] is not None]
            totals[field] = sum(values) if values else None
            field_coverage[field] = {"reported": len(values), "expected": len(rows), "complete": bool(rows) and len(values) == len(rows)}
        costs = {}
        for field in ("actual_cost", "estimated_cost"):
            values = [r[field] for r in rows if r[field] is not None]
            currencies = {v["currency"] for v in values}
            valid = bool(rows) and len(values) == len(rows) and len(currencies) == 1
            costs[field] = {"amount": sum(v["amount"] for v in values) if valid else None,
                "currency": next(iter(currencies)) if len(currencies) == 1 else None,
                "known_amount": sum(v["amount"] for v in values) if len(currencies) == 1 else None,
                "covered_requests": len(values), "complete": valid,
                "reason": None if valid else "bill_or_price_not_reported_for_all_requests_or_mixed_currency"}
        filtered = episode_id is not None or task_id is not None or selected_phases is not None
        return {"schema_version": 1, "requests": attempts, "usage_observations": observations, "usage_groups": rows,
            "scope": {"episode_id": episode_id, "task_id": task_id,
                      "phases": sorted(selected_phases) if selected_phases is not None else None},
            "counts": {"started": len(attempts), "completed": sum(r["status"] == "completed" for r in attempts),
                "failed": sum(r["status"] == "failed" for r in attempts),
                "usage_reported": sum(r["raw_usage"] is not None for r in attempts),
                "deduplicated_usage_groups": len(rows),
                "additional_usage_observations": len(observations),
                "tool_calls": None if filtered else self.tool_calls,
                "actual_control_steps": None if filtered else self.control_steps},
            "known_tokens": totals, "field_coverage": field_coverage,
            **costs, "session_usage_unallocated": list(self.session_snapshots.values()),
            "complete": bool(attempts) and all(r["raw_usage"] is not None and r["status"] != "started" for r in attempts) and not conflicts,
            "completeness_scope": "attempt_usage_observation; field_coverage_and_cost_coverage_reported_separately",
            "aggregation": "attempt_counts_preserved; tokens_and_costs_from_deduplicated_usage_groups; nested_tokens_not_added_to_total",
            "issues": self.issues + conflicts}
