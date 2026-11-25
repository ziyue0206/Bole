"""Dynamic environment tools; the harness owns the decision loop."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from uuid import uuid4


class EpisodeTools:
    def __init__(self, session, info, config, store, boundary, policy=None):
        self.session, self.info, self.config, self.store = session, info, config, store
        self.boundary = boundary
        self.inputs = []
        self.active = True
        self.finished = asyncio.Event()
        self.delivery = asyncio.Event()
        self.lock = asyncio.Lock()
        self.calls = {}
        self.error = None
        self.requested_steps = 0
        self.tool_calls = 0
        self.first_observation_id = None
        self.last_private = {}
        self.policy = policy
        self.current_observation = None
        self.observations = {}
        self.result_observations = {}
        self.delivered_inputs = []

    def definitions(self):
        empty = {"type": "object", "properties": {}, "additionalProperties": False}
        definitions = [
            {"name": "observe", "description": "Read the current allowed images and measured robot state; does not advance physics.", "inputSchema": empty},
            {"name": "status", "description": "Read remaining configured budgets; does not advance physics.", "inputSchema": empty},
            {"name": "execute", "description": "Execute an action sequence through the declared native robot interface. The reply reports actual completed control steps.",
             "inputSchema": {"type": "object", "properties": {
                 "actions": self.info["capabilities"]["action_schema"],
                 "observation_id": {"type": "string"}},
                 "required": ["actions", "observation_id"], "additionalProperties": False}},
            {"name": "finish", "description": "End or abandon this episode immediately. This is not a task-success claim.",
             "inputSchema": {"type": "object", "properties": {"reason": {"type": "string"}},
                             "required": ["reason"], "additionalProperties": False}},
        ]
        if self.policy is not None:
            definitions.append({"name": "infer_policy", "description":
                "Infer a policy candidate from the latest public observation. This does not execute actions; use execute for an accepted candidate.",
                "inputSchema": empty})
        return definitions

    async def observe(self):
        native = await asyncio.to_thread(self.session.observe)
        observation_id = "observation-" + uuid4().hex
        step = native.get("control_step", self.boundary.steps)
        public = {key: native[key] for key in (
            "state", "end_effector_poses", "measurement_validity", "instruction", "control_step") if key in native}
        public["observation_id"] = observation_id
        self.current_observation = {**public, "vision": native.get("vision", {})}
        images, content = [], [{"type": "text", "text": json.dumps(public, ensure_ascii=False)}]
        for order, (camera, view) in enumerate(native.get("vision", {}).items()):
            color = view["color"]
            if color["encoding"] != "png;base64":
                raise ValueError("unsupported public image encoding")
            payload = base64.b64decode(color["data"], validate=True)
            relative = f"observations/{observation_id}-{camera}.png"
            self.store.asset(relative, payload)
            images.append({"camera": camera, "order": order, "path": relative,
                           "shape": color.get("shape"), "format": "png",
                           "control_step": step, "transformation": None})
            content.extend([{"type": "text", "text": f"Camera: {camera}"},
                            {"type": "image", "mimeType": "image/png", "data": color["data"]}])
        capture_event = self.store.event("observation_captured", observation_id=observation_id,
                         control_step=step, images=images, public_state=public)
        self.observations[observation_id] = {"captured_control_step": step, "images": images,
                                            "capture_event_id": capture_event}
        self.inputs = [observation_id]
        return content

    async def first_observation(self):
        content = await self.observe()
        self.first_observation_id = self.inputs[0]
        return content

    async def mark_delivered(self, **evidence):
        if self.delivery.is_set():
            return
        self.boundary.delivered()
        self.delivered_inputs = [self.first_observation_id]
        self.store.event("first_observation_delivered", observation_id=self.first_observation_id,
                         control_step=self.boundary.steps, evidence=evidence,
                         **self.observations.get(self.first_observation_id, {}),
                         boundary="accepted_by_native_harness; model input separately recorded")
        await asyncio.to_thread(self.session.delivered)
        self.delivery.set()

    async def mark_result_delivered(self, call_id, **evidence):
        """Record completed native transport delivery, never model consumption."""
        observations = self.result_observations.get(call_id, [])
        event_id = self.store.event("tool_result_delivered", call_id=call_id,
            observation_ids=observations, control_step=self.boundary.steps, evidence=evidence,
            boundary="stdin_drain_completed; not_model_consumption_ack")
        for observation_id in observations:
            self.store.event("observation_delivered", observation_id=observation_id,
                call_id=call_id, tool_delivery_event_id=event_id, control_step=self.boundary.steps,
                **self.observations[observation_id], evidence=evidence,
                boundary="stdin_drain_completed; not_model_consumption_ack")
        if observations:
            self.delivered_inputs = list(observations)

    def stop(self):
        self.active = False
        self.finished.set()

    async def end(self, reason):
        self.boundary.end(reason)
        self.stop()

    def status(self):
        limit = self.boundary.step_limit
        timeout = self.boundary.timeout_s
        elapsed = 0 if self.boundary.delivered_at is None else time.monotonic() - self.boundary.delivered_at
        return {"control_steps_executed": self.boundary.steps,
                "control_steps_remaining": None if limit is None else max(0, limit - self.boundary.steps),
                "seconds_remaining": None if timeout is None else max(0, timeout - elapsed),
                "episode_open": self.active and self.boundary.reason is None}

    async def call(self, name, arguments, call_id, parent_id=None):
        async with self.lock:
            if call_id in self.calls:
                return self.calls[call_id]
            self.tool_calls += 1
            self.store.event("tool_requested", call_id=call_id, parent_id=parent_id,
                             tool=name, arguments=arguments, control_step=self.boundary.steps)
            try:
                if not self.active or self.boundary.check():
                    raise ValueError("episode is closed")
                if not self.delivery.is_set():
                    raise ValueError("initial observation delivery has not completed")
                maximum = self.config["limits"]["tool_calls"]
                if maximum is not None and self.tool_calls > maximum:
                    await self.end("tool_call_limit")
                    raise ValueError("tool call budget reached")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                if name in {"observe", "status"}:
                    if arguments:
                        raise ValueError("this tool takes no arguments")
                    content = await self.observe() if name == "observe" else [
                        {"type": "text", "text": json.dumps(self.status())}]
                    if name == "observe":
                        self.result_observations[call_id] = list(self.inputs)
                elif name == "infer_policy" and self.policy is not None:
                    if arguments or self.current_observation is None:
                        raise ValueError("infer_policy requires an existing observation and no arguments")
                    candidate = await asyncio.to_thread(self.policy.infer, self.current_observation)
                    self.store.event("policy_inference", call_id=call_id,
                                     observation_ids=list(self.inputs), output=candidate)
                    content = [{"type": "text", "text": json.dumps({"policy_output": candidate,
                                "observation_id": self.inputs[-1], "executed": False})}]
                elif name == "finish":
                    if set(arguments) != {"reason"} or not isinstance(arguments["reason"], str):
                        raise ValueError("finish requires a text reason")
                    self.boundary.finish()
                    self.stop()
                    content = [{"type": "text", "text": '{"episode_closed":true}'}]
                elif name == "execute":
                    if set(arguments) != {"actions", "observation_id"}:
                        raise ValueError("execute requires actions and observation_id")
                    if arguments["observation_id"] not in self.delivered_inputs:
                        raise ValueError("action must reference the most recent delivered observation")
                    actions = arguments["actions"]
                    if not isinstance(actions, list) or not actions:
                        raise ValueError("actions must be a nonempty array")
                    self.requested_steps += len(actions)
                    response = await asyncio.to_thread(self.session.execute, actions, call_id)
                    private = response.get("private", {})
                    self.last_private = private
                    actual = private.get("actual_control_steps")
                    if type(actual) is not int or actual < self.boundary.steps:
                        raise RuntimeError("environment returned an invalid actual step counter")
                    self.boundary.steps = actual
                    if private.get("first_success_control_step") is not None:
                        self.boundary.first_success_step = private["first_success_control_step"]
                        self.boundary.first_success_result = private.get("first_success_native_result")
                    if private.get("termination_reason"):
                        self.boundary.end(private["termination_reason"])
                    self.boundary.check()
                    if self.boundary.reason:
                        self.stop()
                    self.store.event("action_receipt", call_id=call_id,
                                     requested_control_steps=len(actions), private=private,
                                     public_receipt=response["public"], control_step=actual)
                    content = [{"type": "text", "text": json.dumps(response["public"])}]
                else:
                    raise ValueError(f"unknown tool: {name}")
                result = {"content": content, "isError": False}
            except (TypeError, ValueError) as error:
                result = {"content": [{"type": "text", "text": str(error)}], "isError": True}
            except Exception as error:
                self.error = error
                await self.end("infrastructure_error")
                self.store.event("environment_error", call_id=call_id, error=str(error))
                result = {"content": [{"type": "text", "text": "Environment operation failed; episode is closed."}], "isError": True}
            self.calls[call_id] = result
            self.store.event("tool_result_prepared", call_id=call_id, tool=name,
                             observation_ids=list(self.inputs), is_error=result["isError"],
                             control_step=self.boundary.steps)
            return result
