"""A clock and control-step boundary shared by eval and learn.

Native success is private. It only locks a threshold; it does not overwrite the
native final grade, cancel the harness's decisions or add fabricated actions.
"""

from __future__ import annotations

import time


class EpisodeBoundary:
    def __init__(self, *, grace_steps=20, step_limit=None, timeout_s=None, clock=time.monotonic):
        if type(grace_steps) is not int or grace_steps < 0:
            raise ValueError("grace_steps must be a nonnegative integer")
        self.clock = clock
        self.grace_steps, self.step_limit, self.timeout_s = grace_steps, step_limit, timeout_s
        self.created_at = clock()
        self.delivered_at = None
        self.steps = 0
        self.first_success_step = None
        self.first_success_result = None
        self.reason = None
        self.finish_called = False
        self.ended_at = None

    def delivered(self):
        if self.delivered_at is not None:
            raise ValueError("first observation was already delivered")
        self.delivered_at = self.clock()

    def end(self, reason):
        if self.reason is None:
            self.reason = reason
            self.ended_at = self.clock()
        return self.reason

    def check(self):
        if self.reason is not None:
            return self.reason
        if self.delivered_at is not None and self.timeout_s is not None:
            if self.clock() - self.delivered_at >= self.timeout_s:
                return self.end("time_limit")
        if self.step_limit is not None and self.steps >= self.step_limit:
            return self.end("control_step_limit")
        if self.first_success_step is not None:
            if self.steps >= self.first_success_step + self.grace_steps:
                return self.end("success_grace_reached")
        return None

    def observe_native(self, native):
        if native and native.get("success") is True and self.first_success_step is None:
            self.first_success_step = self.steps
            self.first_success_result = dict(native)
        return self.check()

    def executed(self, native=None):
        if self.reason is not None:
            raise RuntimeError("cannot confirm a new step after closure")
        self.steps += 1
        return self.observe_native(native)

    def finish(self):
        self.finish_called = True
        return self.end("finish")

    def final_response(self):
        return self.end("final_response_without_finish")

    def reconcile_native(self, snapshot):
        """Adopt authoritative worker counters, including a late close receipt.

        An interrupted tool response may never have reached the host. The final
        worker snapshot is therefore authoritative over the host's last receipt.
        Native grades remain separate; this method never changes their success.
        """
        if not isinstance(snapshot, dict):
            raise ValueError("native boundary snapshot must be an object")
        if "actual_control_steps" in snapshot:
            steps = snapshot["actual_control_steps"]
            if type(steps) is not int or steps < 0:
                raise ValueError("native actual_control_steps must be a nonnegative integer")
            self.steps = steps
        if "first_success_control_step" in snapshot:
            first = snapshot["first_success_control_step"]
            if first is not None and (type(first) is not int or not 0 <= first <= self.steps):
                raise ValueError("native first success step is outside executed control steps")
            self.first_success_step = first
        if "first_success_native_result" in snapshot:
            self.first_success_result = snapshot["first_success_native_result"]
        if snapshot.get("termination_reason"):
            if self.ended_at is None:
                self.ended_at = self.clock()
            self.reason = snapshot["termination_reason"]

    def report(self):
        return {
            "termination_reason": self.reason,
            "finish_called": self.finish_called,
            "final_response_without_finish": self.reason == "final_response_without_finish",
            "actual_control_steps": self.steps,
            "success_observed": self.first_success_step is not None,
            "first_success_control_step": self.first_success_step,
            "first_success_native_result": self.first_success_result,
            "success_grace_steps": self.grace_steps,
            "executed_post_success_steps": None if self.first_success_step is None else
                max(0, self.steps - self.first_success_step),
            "elapsed_after_first_observation_s": None if self.delivered_at is None else
                max(0, (self.ended_at or self.clock()) - self.delivered_at),
            "initialization_s": None if self.delivered_at is None else
                self.delivered_at - self.created_at,
        }
