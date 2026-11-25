"""Native Codex app-server/dynamicTools lifecycle.

One instance is one container and native thread. The runner rebinds its public
environment tools between learn episodes; eval constructs a new instance.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from bole.results.usage import UsageLedger
from bole.runtime.learning_assets import export_asset, load_asset
from bole.runtime.image_assets import output_image_name, require_unused_name, restore_image
from bole.runtime.dependencies import codex_bwrap_mount, codex_security_args, codex_cli_mount
from .transport import ModelHTTPBridge


async def command(*args):
    process = await asyncio.create_subprocess_exec(*map(str, args), stdout=asyncio.subprocess.PIPE,
                                                  stderr=asyncio.subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode:
        raise RuntimeError(f"{args[0]} exited {process.returncode}: {err.decode(errors='replace')[-2000:]}")
    return out.decode()


class CodexHarness:
    name = "codex"

    def __init__(self, *, native_command=None):
        self.native_command = native_command  # Controlled protocol tests only.
        self.process = None
        self.pending = {}
        self.notifications = asyncio.Queue()
        self.tasks = set()
        self.write_lock = asyncio.Lock()
        self.serial = 0
        self.thread_id = self.turn_id = None
        self.reader_task = self.stderr_task = None
        self.bridge = None
        self.container = None
        self.ledger = UsageLedger()
        self.phase = "initialization"
        self.accepting_tools = False
        self.finish_called = False
        self.stopped = asyncio.Event()
        self.stop_reason = None
        self.active = False
        self.session_usable = True

    async def open(self, context, tools):
        self.context, self.tools = context, tools
        self.config = context["config"]
        self.settings = self.config["harness"]
        self.store = context["store"]
        self.root = Path(context["root"]).resolve()
        self.workspace, self.home = self.root / "workspace", self.root / "codex-home"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.workspace.mkdir(mode=0o700)
        self.home.mkdir(mode=0o700)
        image = self.config["resources"]["harness_image"]
        if context.get("input_asset"):
            asset_input = context["input_asset"]
            manifest_path = asset_input.get("manifest_path") if isinstance(asset_input, dict) else asset_input
            self.input_asset = load_asset(manifest_path, self.workspace, self.home)
            image = self.input_asset["components"]["image"]["image_id"]
        else:
            self.input_asset = None
        if not self.native_command:
            inspected = json.loads(await command("docker", "image", "inspect", image))[0]
            self.image_id = inspected["Id"]
            if not self.input_asset:
                self.input_asset = await restore_image(self.image_id, inspected, self.workspace, self.home, command)
            if self.config.get("mode") == "learn":
                await require_unused_name(output_image_name(self.config, context["run_id"], context["task_id"]), command)
            self.bridge = ModelHTTPBridge(self.settings["gateway"], current_context=self._transport_context,
                                          on_limit=self.stop_episode,
                                          request_timeout_s=self.config.get("limits", {}).get("request_timeout_s", 120))
            endpoint = await self.bridge.start()
        else:
            self.image_id = "controlled_protocol_fixture"
            endpoint = "http://127.0.0.1:9/v1"
        config = {
            "model_provider": "bole", "model": self.settings["model"],
            "sandbox_mode": "workspace-write",
            "sandbox_workspace_write.writable_roots": ["/workspace/.agents/skills"],
            "model_providers.bole.name": "Bole recorded model transport",
            "model_providers.bole.base_url": endpoint,
            "model_providers.bole.env_key": "OPENAI_API_KEY",
            "model_providers.bole.wire_api": "responses",
            "features.enable_request_compression": False,
            "features.api_key_model_discovery": False,
            "features.memories": self.settings.get("memory", {}).get("enabled", True),
            "features.apps": False, "web_search": "disabled",
            "check_for_update_on_startup": False,
            "sandbox_workspace_write.network_access": self.settings.get("shell", {}).get("network_access", False),
        }
        memory = self.settings.get("memory", {})
        for key in ("version", "dual_write", "generate_memories", "use_memories", "dedicated_tools",
                    "disable_on_external_context", "extract_model", "consolidation_model",
                    "min_rollout_idle_hours", "max_rollouts_per_startup"):
            if key in memory and memory[key] is not None:
                config[f"memories.{key}"] = memory[key]
        compact = self.settings.get("compact", {}).get("auto_token_limit")
        if compact is not None:
            config["model_auto_compact_token_limit"] = compact
        native = [self.settings.get("executable", "/opt/bole-tools/codex"),
                  "app-server", "--listen", "stdio://"]
        for key, value in config.items():
            native += ["-c", key + "=" + json.dumps(value)]
        self.container = "bole-harness-" + uuid4().hex
        bootstrap = Path(__file__).resolve().parents[1] / "runtime/codex_bootstrap.py"
        docker = ["docker", "run", "--pull=never", "--init", "--interactive", "--read-only",
                  "--name", self.container, "--network=host", "--cap-drop=ALL",
                  "--security-opt=no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
                  *codex_security_args(self.config["resources"]),
                  "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m", "--workdir", "/workspace",
                  "--mount", f"type=bind,src={self.workspace},dst=/workspace",
                  "--mount", f"type=bind,src={self.home},dst=/home/bole/.codex",
                  "--mount", f"type=bind,src={bootstrap},dst=/opt/bole-runtime/bootstrap.py,readonly",
                  "--env", "HOME=/home/bole", "--env", "CODEX_HOME=/home/bole/.codex",
                  "--env", "OPENAI_API_KEY=bole-local-transport",
                  *codex_bwrap_mount(self.config["resources"].get("codex_bwrap"), native[0]),
                  *codex_cli_mount(self.config["resources"].get("codex_cli"), native[0]),
                  "--entrypoint", self.config["resources"].get("codex_python", "/usr/bin/python3"),
                  self.image_id, "-I", "-B", "/opt/bole-runtime/bootstrap.py", *native]
        self.process = await asyncio.create_subprocess_exec(
            *(self.native_command or docker), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=64 * 1024 * 1024)
        self.reader_task = asyncio.create_task(self._read())
        self.stderr_task = asyncio.create_task(self._stderr())
        await self.request("initialize", {"clientInfo": {"name": "bole", "version": "0.1.0"},
                                          "capabilities": {"experimentalApi": True}})
        await self.send({"method": "initialized", "params": {}})
        response = await self.request("thread/start", {
            "model": self.settings["model"], "modelProvider": "bole", "cwd": "/workspace",
            "approvalPolicy": "never", "sandbox": "workspace-write", "ephemeral": False,
            "dynamicTools": [{"type": "function", "deferLoading": False, **d} for d in tools.definitions()],
        })
        self.thread_id = response["thread"]["id"]
        self.sandbox_policy = response.get("sandbox")
        self.store.event("harness_started", container=self.container, thread_id=self.thread_id,
                         image_id=self.image_id, history_resumed=False,
                         input_asset=self.input_asset,
                         memory_enabled=config["features.memories"], memory_reuse_verified=False)
        self.store.write_json("model-events/native-config.json", {"config": config,
                              "workspace": "/workspace", "codex_home": "/home/bole/.codex",
                              "sandbox_policy": self.sandbox_policy,
                              "python_environment": "/workspace/.venv"})
        return self

    def _transport_context(self):
        return self.store, self.ledger, {"phase": self.phase, "native_thread_id": self.thread_id,
            "native_turn_id": self.turn_id, "episode_id": self.store.identity.get("episode_id"),
            "task_id": self.store.identity.get("task_id"),
            "control_step": self._control_step(),
            "observation_ids": list(getattr(self.tools, "delivered_inputs", getattr(self.tools, "inputs", []))),
            "model_request_limit": self.config.get("limits", {}).get("model_requests"),
            "attribution": "active_harness_context; native_background_work_may_be_unallocated"}

    async def bind_episode(self, store, tools):
        if self.active:
            raise RuntimeError("cannot rebind an active native turn")
        self.store, self.tools = store, tools

    def _control_step(self):
        return getattr(getattr(getattr(self, "tools", None), "boundary", None), "steps", None)

    async def send(self, message):
        store = self.store
        message_id = "native-message-" + uuid4().hex
        store.append_jsonl("model-events/codex-native.jsonl", {"direction": "client", "message": message,
            "native_message_id": message_id, "delivery_status": "prepared", "control_step": self._control_step()})
        try:
            async with self.write_lock:
                self.process.stdin.write((json.dumps(message, allow_nan=False) + "\n").encode())
                await self.process.stdin.drain()
        except BaseException as exc:
            store.event("native_message_send_failed", native_message_id=message_id,
                        control_step=self._control_step(), error={"type": type(exc).__name__, "message": str(exc)})
            raise
        event_id = store.event("native_message_sent", native_message_id=message_id,
            rpc_id=message.get("id"), method=message.get("method"), native_thread_id=self.thread_id,
            native_turn_id=self.turn_id, control_step=self._control_step(),
            boundary="stdin_drain_completed; not_model_consumption_ack")
        return {"native_message_id": message_id, "delivery_event_id": event_id}

    async def request(self, method, params, *, timeout=60):
        self.serial += 1
        ident = self.serial
        future = asyncio.get_running_loop().create_future()
        self.pending[ident] = future
        try:
            await self.send({"id": ident, "method": method, "params": params})
            return await asyncio.wait_for(future, timeout)
        finally:
            self.pending.pop(ident, None)

    async def _stderr(self):
        while line := await self.process.stderr.readline():
            self.store.append_jsonl("model-events/codex-stderr.jsonl", {"text": line.decode(errors="replace")})

    async def _read(self):
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                self.store.append_jsonl("model-events/codex-native.jsonl", {"direction": "server", "message": message,
                    "control_step": self._control_step()})
                if "id" in message and "method" not in message:
                    future = self.pending.get(message["id"])
                    if future and not future.done():
                        if "error" in message:
                            future.set_exception(RuntimeError(json.dumps(message["error"])))
                        else:
                            future.set_result(message.get("result"))
                elif "id" in message:
                    task = asyncio.create_task(self._server_request(message))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
                else:
                    await self.notifications.put(message)
            raise EOFError("Codex app-server exited")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.session_usable = False
            for future in tuple(self.pending.values()):
                if not future.done():
                    future.set_exception(exc)
            await self.notifications.put(exc)

    async def _server_request(self, message):
        try:
            tools = self.tools
            if message["method"] != "item/tool/call":
                await self.send({"id": message["id"], "error": {"code": -32601,
                    "message": "interactive approvals and undeclared external operations are unavailable"}})
                return
            p = message["params"]
            accepted = (p.get("threadId") == self.thread_id and self.accepting_tools
                        and (self.turn_id is None or p.get("turnId") == self.turn_id))
            if not accepted:
                result = {"content": [{"type": "text", "text": "No active episode accepts robot tools."}], "isError": True}
            else:
                result = await tools.call(p["tool"], p["arguments"], p["callId"], self.turn_id)
                self.ledger.tool_calls += 1
            finished = accepted and p["tool"].split(".")[-1] == "finish" and not result["isError"]
            if finished:
                self.finish_called = True
            content = [{"type": "inputText", "text": c["text"]} if c["type"] == "text" else
                       {"type": "inputImage", "imageUrl": "data:" + c["mimeType"] + ";base64," + c["data"]}
                       for c in result["content"]]
            receipt = await self.send({"id": message["id"], "result": {"contentItems": content, "success": not result["isError"]}})
            if accepted and hasattr(tools, "mark_result_delivered"):
                await tools.mark_result_delivered(p["callId"], **receipt,
                    native_thread_id=self.thread_id, native_turn_id=p.get("turnId"))
            if finished:
                await self.stop_episode("finish")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.store.event("harness_tool_error", error=str(exc))
            with contextlib.suppress(Exception):
                await self.send({"id": message["id"], "error": {"code": -32603, "message": str(exc)}})

    async def stop_episode(self, reason):
        if not self.stopped.is_set():
            self.stop_reason = reason
            self.accepting_tools = False
            self.stopped.set()

    def _completion(self, event, *, interrupted=False):
        if not isinstance(event, dict) or event.get("method") != "turn/completed":
            return False
        params = event.get("params", {})
        turn = params.get("turn", {})
        if params.get("threadId") != self.thread_id or turn.get("id") != self.turn_id:
            return False
        allowed = {"completed", "interrupted"} if interrupted else {"completed"}
        if turn.get("status") not in allowed:
            self.session_usable = False
            self.store.event("native_turn_failed", native_thread_id=self.thread_id,
                native_turn_id=self.turn_id, control_step=self._control_step(),
                native_status=turn.get("status"), error=turn.get("error"), boundary_reason=self.stop_reason)
            raise RuntimeError("native turn failed: " + str(turn.get("status")))
        return True

    async def _interrupt_and_settle(self):
        if not self.thread_id or not self.turn_id:
            return
        # A native completion can race with a budget/finish notification.
        while not self.notifications.empty():
            event = self.notifications.get_nowait()
            if isinstance(event, Exception):
                raise event
            if self._completion(event, interrupted=True):
                return
        try:
            await self.request("turn/interrupt", {"threadId": self.thread_id, "turnId": self.turn_id}, timeout=5)
        except Exception:
            # A matching completion is authoritative even if interrupt raced
            # with it. Otherwise the bounded wait below fails the session.
            pass
        async with asyncio.timeout(10):
            while True:
                event = await self.notifications.get()
                if isinstance(event, Exception):
                    raise event
                if self._completion(event, interrupted=True):
                    return

    async def _turn(self, prompt, content, *, on_delivered=None, timeout_s=600, phase="episode"):
        if self.active or not self.session_usable:
            raise RuntimeError("native session is unavailable for a new turn")
        self.active = True
        self.phase = phase
        self.turn_id = None
        self.finish_called = False
        self.stop_reason = None
        self.stopped.clear()
        self.accepting_tools = phase == "episode"
        inputs = [{"type": "text", "text": prompt}]
        inputs += [{"type": "text", "text": c["text"]} if c["type"] == "text" else
                   {"type": "image", "url": "data:" + c["mimeType"] + ";base64," + c["data"]} for c in content]
        notification = stopper = None
        try:
            response = await self.request("turn/start", {"threadId": self.thread_id, "input": inputs,
                                                          "effort": self.settings.get("reasoning_effort", "high")})
            self.turn_id = response["turn"]["id"]
            self.store.event("native_input_acknowledged", thread_id=self.thread_id,
                             turn_id=self.turn_id, phase=phase, content_items=len(inputs), control_step=self._control_step())
            if on_delivered:
                await on_delivered()
            async with asyncio.timeout(timeout_s):
                stopper = asyncio.create_task(self.stopped.wait())
                while True:
                    notification = asyncio.create_task(self.notifications.get())
                    done, _ = await asyncio.wait((notification, stopper), return_when=asyncio.FIRST_COMPLETED)
                    if stopper in done:
                        if notification.done():
                            event = notification.result()
                            if self._completion(event, interrupted=True):
                                break
                            await self.notifications.put(event)
                        else:
                            notification.cancel()
                            await asyncio.gather(notification, return_exceptions=True)
                        await self._interrupt_and_settle()
                        break
                    event = notification.result()
                    if isinstance(event, Exception):
                        raise event
                    p = event.get("params", {})
                    if p.get("threadId") != self.thread_id:
                        continue
                    if event.get("method") == "thread/tokenUsage/updated":
                        self.ledger.snapshot(self.thread_id, p.get("tokenUsage"), phase=phase)
                    if self._completion(event):
                        self.stop_reason = "final_response_without_finish" if phase == "episode" else "feedback_delivered"
                        break
        except TimeoutError:
            if self.turn_id is None:
                self.session_usable = False
                raise RuntimeError("native input was not acknowledged before startup timeout")
            self.stop_reason = "time_limit"
            self.accepting_tools = False
            # asyncio.wait does not cancel its child get() task. It must not
            # steal the completion needed to settle an interrupted turn.
            if notification:
                if notification.done() and not notification.cancelled():
                    self.notifications.put_nowait(notification.result())
                else:
                    notification.cancel()
                    await asyncio.gather(notification, return_exceptions=True)
            try:
                await self._interrupt_and_settle()
            except Exception:
                self.session_usable = False
                raise
        except asyncio.CancelledError:
            self.stop_reason = "cancelled"
            self.accepting_tools = False
            if notification:
                if notification.done() and not notification.cancelled():
                    self.notifications.put_nowait(notification.result())
                else:
                    notification.cancel()
                    await asyncio.gather(notification, return_exceptions=True)
            try:
                await asyncio.shield(self._interrupt_and_settle())
            except Exception:
                self.session_usable = False
            raise
        except Exception:
            self.session_usable = False
            raise
        finally:
            for task in (notification, stopper):
                if task and not task.done():
                    task.cancel()
            if notification or stopper:
                await asyncio.gather(*(t for t in (notification, stopper) if t), return_exceptions=True)
            self.accepting_tools = False
            self.active = False
            self.store.write_json("model-events/usage-chain.json", self.ledger.report())
        return {"termination": self.stop_reason, "finish_called": self.finish_called,
                "thread_id": self.thread_id, "turn_id": self.turn_id,
                "session_usable": self.session_usable}

    async def run_episode(self, prompt, initial_content, on_delivered, timeout_s):
        return await self._turn(prompt, initial_content, on_delivered=on_delivered, timeout_s=timeout_s)

    async def feedback(self, text):
        self.store.event("feedback_sent", thread_id=self.thread_id, text=text, control_step=self._control_step())
        result = await self._turn(text, (), phase="feedback")
        if result["termination"] != "feedback_delivered":
            raise RuntimeError("native feedback was not completed; learning chain cannot continue")
        self.store.event("feedback_delivered", thread_id=self.thread_id, turn_id=self.turn_id, control_step=self._control_step())
        return result

    async def _audit_sessions(self):
        for folder in ("sessions", "archived_sessions"):
            for path in (self.home / folder).rglob("*"):
                if path.is_file() and not path.is_symlink():
                    relative = path.relative_to(self.home).as_posix()
                    self.store.asset("model-events/native-audit/" + relative, path.read_bytes(), text=True)

    async def export_asset(self, destination, metadata, partial=False):
        if self.active:
            raise RuntimeError("cannot export during an active native turn")
        if not hasattr(self, "workspace") or not hasattr(self, "home"):
            raise RuntimeError("harness initialization created no available persistent state")
        flush = None
        if self.config.get("mode") == "learn":
            flush = await self._finish_native()
            partial = partial or not flush["native_flush_confirmed"]
        await self._audit_sessions()
        gateway = self.settings["gateway"]
        key = os.environ.get(gateway["credential_env"])
        return await export_asset(destination, container=self.container, workspace=self.workspace,
            home=self.home, metadata={**metadata, "source_task_id": self.context["task_id"],
                "source_run_id": self.context["run_id"], "parent_asset": self.context.get("input_asset"),
                "parent_image_id": getattr(self, "image_id", None),
                "status": "partial" if partial else "complete", "native_thread_id": self.thread_id,
                "native_shutdown": flush,
                "native_history_resumed_on_import": False}, command=command, secrets=(key,),
            base_image=getattr(self, "image_id", self.config["resources"]["harness_image"]),
            image_name=output_image_name(self.config, self.context["run_id"], self.context["task_id"], partial=partial),
            state_quiescent=self.process is None and self.container is None)

    async def _finish_native(self):
        """Request normal native exit, retain its container and mounted state."""
        if not self.process:
            return {"native_flush_confirmed": False, "status": "process_not_started"}
        if self.process.returncode is not None:
            return {"native_flush_confirmed": False, "status": "already_exited_without_observed_flush",
                    "returncode": self.process.returncode}
        self.process.stdin.close()
        try:
            returncode = await asyncio.wait_for(self.process.wait(), 10)
            result = {"native_flush_confirmed": returncode == 0,
                      "status": "normal_stdin_eof_exit" if returncode == 0 else "nonzero_exit",
                      "returncode": returncode}
        except TimeoutError:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await asyncio.wait_for(self.process.wait(), 5)
            result = {"native_flush_confirmed": False, "status": "forced_exit_flush_unknown",
                      "returncode": self.process.returncode}
        if self.bridge:
            await self.bridge.close()
        self.session_usable = False
        self.store.event("native_process_shutdown", control_step=self._control_step(), **result)
        return result

    async def close(self):
        try:
            if self.process and self.process.returncode is None:
                if self.active:
                    with contextlib.suppress(Exception):
                        await self._interrupt_and_settle()
                await self._finish_native()
            for task in (*self.tasks, self.reader_task, self.stderr_task):
                if task:
                    task.cancel()
            await asyncio.gather(*(t for t in (*self.tasks, self.reader_task, self.stderr_task) if t), return_exceptions=True)
            if hasattr(self, "home"):
                await self._audit_sessions()
            if self.container and not self.native_command:
                await command("docker", "rm", "-f", self.container)
        finally:
            if self.bridge:
                await self.bridge.close()


__all__ = ["CodexHarness"]
