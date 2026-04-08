"""LiteLLM implementations of all LLM interface types."""

from __future__ import annotations

import base64
import json as _json
import time
from typing import Callable

import litellm

from .._litellm_workaround import reset_litellm_client
from .._retry import retry_image_generation, retry_text_completion
from ..responses import ImageResponse, TextResponse, ToolCall, ToolCallResponse
from ..types import (
    GeneralLLM,
    ImageGenLLM,
    ImageInspectorLLM,
    ReasoningLLM,
    TextGenLLM,
    ToolsLLM,
)


def _apply_common(
    kwargs: dict,
    temperature: float | None,
    max_tokens: int | None,
    response_schema: dict | None,
    options: dict | None = None,
) -> None:
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_schema is not None:
        kwargs["response_format"] = response_schema
    if options:
        kwargs["extra_body"] = {"options": options}


class LiteLLMGeneralLLM(GeneralLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 60,
        api_base: str | None = None,
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
        self.api_base = api_base
        reset_litellm_client()

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> TextResponse:
        t0 = time.monotonic()
        kwargs: dict = {"model": self.model, "messages": messages, "timeout": self.timeout}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        effective_schema = response_schema if response_schema is not None else self.response_schema
        effective_temp = temperature if temperature is not None else self.temperature
        _apply_common(kwargs, effective_temp, self.max_tokens, effective_schema, options)
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        return TextResponse(
            content=content,
            model=self.model,
            duration_ms=(time.monotonic() - t0) * 1000,
            attempts=1,
        )


class LiteLLMTextGenLLM(TextGenLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 120,
        api_base: str | None = None,
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
        self.api_base = api_base
        reset_litellm_client()

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
        effective_temp = temperature if temperature is not None else self.temperature

        def call_fn(msgs: list[dict]) -> tuple[str, str]:
            kwargs: dict = {"model": self.model, "messages": msgs, "timeout": self.timeout}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            _apply_common(kwargs, effective_temp, self.max_tokens, effective_schema, options)
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content or "", self.model

        return retry_text_completion(
            call_fn, messages, max_retries, self.model, on_transfer_error=reset_litellm_client
        )


class LiteLLMReasoningLLM(ReasoningLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 300,
        api_base: str | None = None,
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
        self.api_base = api_base
        reset_litellm_client()

    def complete(
        self,
        messages: list[dict],
        *,
        thinking_budget: int | None = None,
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> TextResponse:
        t0 = time.monotonic()
        kwargs: dict = {"model": self.model, "messages": messages, "timeout": self.timeout}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        effective_schema = response_schema if response_schema is not None else self.response_schema
        effective_temp = temperature if temperature is not None else self.temperature
        _apply_common(kwargs, effective_temp, self.max_tokens, effective_schema, options)
        if thinking_budget is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        return TextResponse(
            content=content,
            model=self.model,
            duration_ms=(time.monotonic() - t0) * 1000,
            attempts=1,
        )


class LiteLLMImageGenLLM(ImageGenLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 300,
        api_base: str | None = None,
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
        self.api_base = api_base
        reset_litellm_client()

    def generate(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        validator: Callable[[bytes], bool] | None = None,
        reference_images: list[bytes] | None = None,
        weight: float | None = None,
        width: int = 256,
        height: int = 256,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        options: dict | None = None,
    ) -> ImageResponse:
        def call_fn() -> tuple[bytes, str]:
            kwargs: dict = {"model": self.model, "prompt": prompt, "timeout": self.timeout}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = litellm.image_generation(**kwargs)
            b64 = response.data[0].b64_json if response.data else None
            if not b64:
                return b"", self.model
            return base64.b64decode(b64), self.model

        return retry_image_generation(call_fn, max_retries, self.model, validator=validator)


class LiteLLMImageInspectorLLM(ImageInspectorLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 90,
        api_base: str | None = None,
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
        self.api_base = api_base
        reset_litellm_client()

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
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        effective_schema = response_schema if response_schema is not None else self.response_schema
        effective_temp = temperature if temperature is not None else self.temperature

        def call_fn(msgs: list[dict]) -> tuple[str, str]:
            kwargs: dict = {"model": self.model, "messages": msgs, "timeout": self.timeout}
            if self.api_base:
                kwargs["api_base"] = self.api_base
            _apply_common(kwargs, effective_temp, self.max_tokens, effective_schema, options)
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content or "", self.model

        return retry_text_completion(
            call_fn, messages, max_retries, self.model, on_transfer_error=reset_litellm_client
        )


class LiteLLMToolsLLM(ToolsLLM):
    """LiteLLM implementation of ToolsLLM (function/tool calling)."""

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        api_base: str | None = None,
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
        self.api_base = api_base
        reset_litellm_client()

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        max_retries: int = 3,
        options: dict | None = None,
    ) -> ToolCallResponse:
        t0 = time.monotonic()
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        _apply_common(kwargs, self.temperature, self.max_tokens, None, options)
        response = litellm.completion(**kwargs)
        duration_ms = (time.monotonic() - t0) * 1000

        msg = response.choices[0].message
        content = msg.content or None
        raw_tool_calls = getattr(msg, "tool_calls", None) or []

        tool_calls = []
        for tc in raw_tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id or f"call_{tc.function.name}",
                    name=tc.function.name,
                    arguments=args or {},
                )
            )

        return ToolCallResponse(
            content=content,
            tool_calls=tool_calls,
            model=self.model,
            duration_ms=duration_ms,
            attempts=1,
        )
