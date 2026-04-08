"""Claude CLI subprocess implementations (5 types — no ImageGen, no Tools).

All classes run ``claude --print --output-format json`` as a subprocess and
parse the JSON result.  ImageGenLLM and ToolsLLM are not implemented via CLI.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import time
import uuid

from .._retry import retry_text_completion
from ..responses import TextResponse
from ..types import (
    GeneralLLM,
    ImageInspectorLLM,
    ReasoningLLM,
    TextGenLLM,
)

logger = logging.getLogger(__name__)

_CLI_CMD = "claude"


def _schema_instruction(schema: dict) -> str:
    """Return a system-prompt fragment instructing the model to emit schema-valid JSON."""
    return (
        "You must respond with valid JSON that strictly matches the following schema. "
        "Output only the JSON object — no markdown fences, no explanation.\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )


def _inject_schema(messages: list[dict], schema: dict) -> list[dict]:
    """Prepend or merge the schema instruction into the system message."""
    instruction = _schema_instruction(schema)
    msgs = list(messages)
    for i, msg in enumerate(msgs):
        if msg.get("role") == "system":
            existing = msg.get("content", "") or ""
            msgs[i] = {**msg, "content": f"{instruction}\n\n{existing}"}
            return msgs
    return [{"role": "system", "content": instruction}] + msgs


def _run_claude(
    prompt: str,
    *,
    system: str | None = None,
    timeout: int = 120,
    effort: str | None = None,
) -> tuple[str, float]:
    cmd = [_CLI_CMD, "--print", "--output-format", "json"]
    if effort:
        cmd += ["--effort", effort]
    if system:
        cmd += ["--system-prompt", system]
    cmd.append(prompt)

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    duration_ms = (time.monotonic() - t0) * 1000

    if proc.returncode != 0:
        if proc.stderr:
            logger.error("claude stderr: %s", proc.stderr[:2000])
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)

    parsed = json.loads(proc.stdout)
    return parsed.get("result", ""), duration_ms


def _run_claude_stream_json(
    messages: list[dict], *, timeout: int = 120, effort: str | None = None
) -> tuple[str, float]:
    cmd = [
        _CLI_CMD,
        "--output-format",
        "stream-json",
        "--input-format",
        "stream-json",
    ]
    if effort:
        cmd += ["--effort", effort]

    system_content: str | None = None
    user_messages: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "") or ""
        else:
            user_messages.append(msg)

    if system_content and user_messages:
        first = user_messages[0]
        content = first.get("content", "")
        if isinstance(content, str):
            merged = {**first, "content": f"{system_content}\n\n{content}"}
        else:
            merged = {
                **first,
                "content": [{"type": "text", "text": system_content}] + list(content),
            }
        user_messages[0] = merged

    session_id = str(uuid.uuid4())
    lines = [
        json.dumps(
            {"type": "user", "message": msg, "parent_tool_use_id": None, "session_id": session_id}
        )
        for msg in user_messages
    ]
    stdin_data = "\n".join(lines)

    t0 = time.monotonic()
    proc = subprocess.run(cmd, input=stdin_data, capture_output=True, text=True, timeout=timeout)
    duration_ms = (time.monotonic() - t0) * 1000

    if proc.returncode != 0:
        if proc.stderr:
            logger.error("claude stderr: %s", proc.stderr[:2000])
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)

    result = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event.get("result", "")
            break
    return result, duration_ms


class CLIGeneralLLM(GeneralLLM):
    def __init__(
        self,
        model: str = "claude",
        timeout: int = 60,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> TextResponse:
        effective_schema = response_schema if response_schema is not None else self.response_schema
        msgs = _inject_schema(messages, effective_schema) if effective_schema else messages
        system = next((m["content"] for m in msgs if m.get("role") == "system"), None)
        user_msgs = [m for m in msgs if m.get("role") != "system"]
        prompt = "\n\n".join(
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
            for m in user_msgs
        )
        content, duration_ms = _run_claude(prompt, system=system, timeout=self.timeout)
        return TextResponse(content=content, model=self.model, duration_ms=duration_ms, attempts=1)


class CLITextGenLLM(TextGenLLM):
    def __init__(
        self,
        model: str = "claude",
        timeout: int = 120,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def complete(
        self,
        messages: list[dict],
        *,
        max_retries: int = 3,
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> TextResponse:
        effective_schema = response_schema if response_schema is not None else self.response_schema
        base_msgs = _inject_schema(messages, effective_schema) if effective_schema else messages

        def call_fn(msgs: list[dict]) -> tuple[str, str]:
            system = next((m["content"] for m in msgs if m.get("role") == "system"), None)
            user_msgs = [m for m in msgs if m.get("role") != "system"]
            prompt = "\n\n".join(
                m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                for m in user_msgs
            )
            content, _ = _run_claude(prompt, system=system, timeout=self.timeout)
            return content, self.model

        return retry_text_completion(call_fn, base_msgs, max_retries, self.model)


class CLIReasoningLLM(ReasoningLLM):
    """Uses ``--effort high``; thinking_budget is accepted but ignored."""

    def __init__(
        self,
        model: str = "claude",
        timeout: int = 300,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def complete(
        self,
        messages: list[dict],
        *,
        thinking_budget: int | None = None,
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> TextResponse:
        effective_schema = response_schema if response_schema is not None else self.response_schema
        msgs = _inject_schema(messages, effective_schema) if effective_schema else messages
        system = next((m["content"] for m in msgs if m.get("role") == "system"), None)
        user_msgs = [m for m in msgs if m.get("role") != "system"]
        prompt = "\n\n".join(
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
            for m in user_msgs
        )
        content, duration_ms = _run_claude(
            prompt, system=system, timeout=self.timeout, effort="high"
        )
        return TextResponse(content=content, model=self.model, duration_ms=duration_ms, attempts=1)


class CLIImageInspectorLLM(ImageInspectorLLM):
    def __init__(
        self,
        model: str = "claude",
        timeout: int = 300,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

    def inspect(
        self,
        image: bytes,
        system: str,
        prompt: str,
        *,
        max_retries: int = 3,
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> TextResponse:
        b64 = base64.b64encode(image).decode("ascii")
        messages: list[dict] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        effective_schema = response_schema if response_schema is not None else self.response_schema
        base_msgs = _inject_schema(messages, effective_schema) if effective_schema else messages

        def call_fn(msgs: list[dict]) -> tuple[str, str]:
            content, _ = _run_claude_stream_json(msgs, timeout=self.timeout)
            return content, self.model

        return retry_text_completion(call_fn, base_msgs, max_retries, self.model)
