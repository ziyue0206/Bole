"""One host-owned HTTP capture boundary.

The native process receives a non-secret placeholder; credentials are attached
only here. Requests and incremental SSE responses remain outside its mounts.
"""
from __future__ import annotations

import asyncio
import codecs
import contextlib
import gzip
import json
import os
import math
import tempfile
import time
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlsplit

from bole.results.usage import response_usage
from .compat import normalize_request, stream_complete


class StreamingRedactor:
    def __init__(self, secrets):
        self.secrets = tuple(x for x in secrets if x)
        self.keep = max(map(len, self.secrets), default=1) - 1
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(self, chunk, *, final=False):
        self.buffer += self.decoder.decode(chunk, final=final)
        cut = len(self.buffer) if final else max(0, len(self.buffer) - self.keep)
        for secret in self.secrets:
            start = self.buffer.find(secret)
            while start >= 0:
                if start < cut < start + len(secret):
                    cut = start
                start = self.buffer.find(secret, start + 1)
        result, self.buffer = self.buffer[:cut], self.buffer[cut:]
        for secret in self.secrets:
            result = result.replace(secret, "[REDACTED]")
        return result


class ModelHTTPBridge:
    def __init__(self, settings, *, current_context, on_limit=None, request_timeout_s=None):
        self.settings = settings
        self.current_context = current_context
        self.on_limit = on_limit
        self.server = None
        self.tasks = set()
        self.request_count = 0
        self.episode_counts = {}
        self.request_timeout_s = request_timeout_s if request_timeout_s is not None else settings.get("request_timeout_s", 600)
        if type(self.request_timeout_s) not in (int, float) or not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("transport request timeout must be a positive finite number")
        self.key = os.environ.get(settings["credential_env"])
        if not self.key:
            raise ValueError(f"missing credential environment variable: {settings['credential_env']}")

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0, limit=65536)
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/v1"

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        process = stderr_task = None
        store, ledger, context = self.current_context()
        request_id = "request-" + uuid4().hex
        started = time.monotonic()
        status, error, content_type = None, None, "application/json"
        response = []
        headers_sent = False
        temporary = None
        request_started = False
        public_headers = {}
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15)
            lines = head.decode("latin1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            if method != "POST" or path not in ("/v1/responses", "/v1/responses/compact"):
                raise ValueError("only the configured Responses and native compact endpoints are allowed")
            headers = {k.lower(): v.strip() for k, v in
                       (line.split(":", 1) for line in lines[1:] if ":" in line)}
            if headers.get("authorization") != "Bearer bole-local-transport":
                raise ValueError("invalid local transport authorization")
            length = int(headers.get("content-length", "0"))
            if not 0 < length <= 64 * 1024 * 1024 or "transfer-encoding" in headers:
                raise ValueError("invalid request size or unsupported chunked request")
            raw = await asyncio.wait_for(reader.readexactly(length), 30)
            if headers.get("content-encoding") == "gzip":
                raw = gzip.decompress(raw)
            elif headers.get("content-encoding") not in (None, "identity"):
                raise ValueError("unsupported request encoding")
            original = json.loads(raw)
            body, changes = (normalize_request(original, "responses")
                             if self.settings.get("compatibility") else (original, []))
            limit = self.settings.get("max_requests")
            episode_key = context.get("episode_id")
            # Explicit zero is used by read-only maintenance verification: it
            # forbids model forwarding even outside an ordinary episode phase.
            configured_limit = context.get("model_request_limit")
            episode_limit = configured_limit if context.get("phase") == "episode" or configured_limit == 0 else None
            if ((limit is not None and self.request_count >= limit) or
                (episode_limit is not None and self.episode_counts.get(episode_key, 0) >= episode_limit)):
                store.event("model_request_rejected", reason="model_request_limit", **context)
                if self.on_limit:
                    await self.on_limit("model_request_limit")
                raise RuntimeError("model request budget exhausted")
            self.request_count += 1
            if context.get("phase") == "episode":
                self.episode_counts[episode_key] = self.episode_counts.get(episode_key, 0) + 1
            phase = "compact" if path.endswith("/compact") else context.get("phase", "episode")
            endpoint_parts = urlsplit(self.settings["base_url"])
            provider_scope = endpoint_parts.scheme + "://" + (endpoint_parts.hostname or "") + (
                ":" + str(endpoint_parts.port) if endpoint_parts.port else "") + endpoint_parts.path
            ledger.start(request_id, model=body.get("model"), provider_scope=provider_scope,
                         transport_timeout_s=self.request_timeout_s, **{**context, "phase": phase})
            request_started = True
            prefix = f"model-events/{request_id}"
            sent_data = json.dumps(body, allow_nan=False).encode()
            native_ref = store.asset(prefix + "-native-input.json", raw, text=True)
            sent_ref = store.asset(prefix + "-sent-input.json", sent_data, text=True)
            format_ref = store.write_json(prefix + "-payload-format.json", {
                "request_id": request_id, "endpoint_path": path, "path_base": "episode",
                "format": "openai_responses_request_json", "media_type": "application/json",
                "schema_owner": "upstream API; Bole schema and identity live in this sidecar",
                "payloads": {"native_input": {"path": native_ref,
                    "representation": "received JSON bytes after optional HTTP gzip decoding"},
                    "sent_input": {"path": sent_ref,
                    "representation": "JSON serialization forwarded to upstream before credential attachment"}},
                "redaction": "configured secrets redacted from saved payloads; no Bole envelope injected",
                "transformations": changes, **context})
            store.event("model_request", request_id=request_id, native_request_ref=native_ref,
                        request_ref=sent_ref, transformations=changes, model=body.get("model"),
                        payload_format_ref=format_ref,
                        transport_timeout_s=self.request_timeout_s, **context)
            with tempfile.NamedTemporaryFile(prefix="bole-request-", delete=False) as f:
                temporary = Path(f.name)
                f.write(sent_data)
            base = self.settings["base_url"].rstrip("/")
            endpoint = base + path.removeprefix("/v1")
            config = ["url = " + json.dumps(endpoint),
                      "header = " + json.dumps("Authorization: Bearer " + self.key),
                      'header = "Content-Type: application/json"']
            for name in ("accept", "openai-beta", "x-client-request-id", "idempotency-key"):
                if name in headers:
                    config.append("header = " + json.dumps(name + ": " + headers[name]))
            if self.settings.get("proxy"):
                config.append("proxy = " + json.dumps(self.settings["proxy"]))
            process = await asyncio.create_subprocess_exec(
                "curl", "--silent", "--show-error", "--no-buffer", "--include",
                "--suppress-connect-headers", "--connect-timeout", "15", "--max-time",
                str(self.request_timeout_s), "--config", "-",
                "--data-binary", "@" + str(temporary), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                limit=128 * 1024)
            stderr_task = asyncio.create_task(process.stderr.read())
            process.stdin.write(("\n".join(config) + "\n").encode())
            await process.stdin.drain()
            process.stdin.close()
            response_head = await process.stdout.readuntil(b"\r\n\r\n")
            while response_head.startswith(b"HTTP/1.1 100"):
                response_head = await process.stdout.readuntil(b"\r\n\r\n")
            response_lines = response_head.decode("latin1").split("\r\n")
            status = int(response_lines[0].split()[1])
            public_headers = {k.lower(): v.strip() for k, v in
                              (x.split(":", 1) for x in response_lines[1:] if ":" in x)
                              if k.lower() in {"content-type", "x-request-id", "request-id"}}
            content_type = public_headers.get("content-type", "application/json")
            writer.write(f"HTTP/1.1 {status} Upstream\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n".encode())
            await writer.drain()
            headers_sent = True
            scrub = StreamingRedactor((self.key,))

            async def save_chunk(text):
                if not text:
                    return
                response.append(text)
                store.append_jsonl(prefix + "-response-stream.jsonl",
                                   {"request_id": request_id, "chunk_index": len(response) - 1,
                                    "text": text, **context})
                writer.write(text.encode())
                await writer.drain()

            while chunk := await process.stdout.read(65536):
                await save_chunk(scrub.feed(chunk))
            await save_chunk(scrub.feed(b"", final=True))
            await process.wait()
            if process.returncode:
                raise RuntimeError(f"model transport exited with status {process.returncode}")
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc).replace(self.key, "[REDACTED]")}
            if not headers_sent:
                with contextlib.suppress(Exception):
                    writer.write(b'HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n{"error":{"message":"Bole model transport failed"}}')
                    await writer.drain()
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            if process and process.returncode is None:
                process.kill()
                await process.wait()
            stderr = (await stderr_task).decode(errors="replace") if stderr_task else ""
            if temporary:
                temporary.unlink(missing_ok=True)
            body_text = "".join(response)
            complete = error is None and status is not None and 200 <= status < 300
            if "text/event-stream" in content_type:
                complete = complete and stream_complete(body_text.encode())
                for line in body_text.splitlines():
                    if line.startswith("data:"):
                        try:
                            event = json.loads(line[5:])
                        except ValueError:
                            continue
                        if isinstance(event, dict) and event.get("type") in {
                            "error", "response.failed", "response.incomplete"
                        }:
                            complete = False
            if request_started:
                parsed = response_usage(body_text)
                http_request_id = public_headers.get("x-request-id") or public_headers.get("request-id")
                response_id = parsed["provider_response_id"]
                ledger.finish(request_id, parsed["raw_usage"], provider_request_id=http_request_id,
                              provider_response_id=response_id,
                              status="completed" if complete else "failed", duration_seconds=time.monotonic() - started)
                store.write_json(f"model-events/{request_id}-response.json",
                                 {"request_id": request_id, "http_status": status,
                                  "provider_request_id": http_request_id, "provider_response_id": response_id,
                                  "upstream_headers": public_headers,
                                  "body": body_text, "complete": complete, "error": error,
                                  "stderr": stderr.replace(self.key, "[REDACTED]"), **context})
                store.event("model_response", request_id=request_id, complete=complete,
                            http_status=status, provider_request_id=http_request_id,
                            provider_response_id=response_id, error=error, **context)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            self.tasks.discard(task)

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        pending = tuple(self.tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
