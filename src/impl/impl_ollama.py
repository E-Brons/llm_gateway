"""Ollama REST API implementations of all LLM interface types."""

from __future__ import annotations

import base64
import time
from typing import Callable

import requests

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


def _bare_model(model: str) -> str:
    """Strip ``ollama/`` prefix for the Ollama REST API."""
    return model.removeprefix("ollama/")


def _build_options(
    temperature: float | None,
    max_tokens: int | None,
    extra: dict | None = None,
) -> dict:
    opts: dict = {}
    if temperature is not None:
        opts["temperature"] = temperature
    if max_tokens is not None:
        opts["num_predict"] = max_tokens
    if extra:
        opts.update(extra)
    return opts


def _ollama_chat(
    ollama_url: str,
    model: str,
    messages: list[dict],
    timeout: int,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_schema: dict | None = None,
) -> tuple[str, str]:
    payload: dict = {
        "model": _bare_model(model),
        "messages": messages,
        "stream": False,
    }
    options = _build_options(temperature, max_tokens)
    if options:
        payload["options"] = options
    if response_schema is not None:
        payload["format"] = response_schema

    resp = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "") or ""
    used_model = data.get("model", model)
    return content, used_model


def _ollama_generate(
    ollama_url: str,
    model: str,
    prompt: str,
    timeout: int,
    system: str | None = None,
    images: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_schema: dict | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> tuple[str, str]:
    payload: dict = {
        "model": _bare_model(model),
        "prompt": prompt,
        "stream": False,
    }
    if system is not None:
        payload["system"] = system
    if images:
        payload["images"] = images

    gen_options = _build_options(temperature, max_tokens)
    if width:
        gen_options["width"] = width
    if height:
        gen_options["height"] = height
    if seed is not None:
        gen_options["seed"] = seed
    if gen_options:
        payload["options"] = gen_options

    if response_schema is not None:
        payload["format"] = response_schema

    resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "") or "", data.get("model", model)


class OllamaGeneralLLM(GeneralLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 60,
        ollama_url: str = "http://localhost:11434",
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
        self.ollama_url = ollama_url

    def complete(self, messages: list[dict]) -> TextResponse:
        t0 = time.monotonic()
        content, used_model = _ollama_chat(
            self.ollama_url,
            self.model,
            messages,
            self.timeout,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_schema=self.response_schema,
        )
        return TextResponse(
            content=content,
            model=used_model,
            duration_ms=(time.monotonic() - t0) * 1000,
            attempts=1,
        )


class OllamaTextGenLLM(TextGenLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 120,
        ollama_url: str = "http://localhost:11434",
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
        self.ollama_url = ollama_url

    def complete(self, messages: list[dict], *, max_retries: int = 3) -> TextResponse:
        def call_fn(msgs: list[dict]) -> tuple[str, str]:
            return _ollama_chat(
                self.ollama_url,
                self.model,
                msgs,
                self.timeout,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_schema=self.response_schema,
            )

        return retry_text_completion(call_fn, messages, max_retries, self.model)


class OllamaReasoningLLM(ReasoningLLM):
    """Ollama does not support thinking budget natively; the parameter is accepted but ignored."""

    def __init__(
        self,
        model: str,
        timeout: int = 300,
        ollama_url: str = "http://localhost:11434",
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
        self.ollama_url = ollama_url

    def complete(self, messages: list[dict], *, thinking_budget: int | None = None) -> TextResponse:
        t0 = time.monotonic()
        content, used_model = _ollama_chat(
            self.ollama_url,
            self.model,
            messages,
            self.timeout,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_schema=self.response_schema,
        )
        return TextResponse(
            content=content,
            model=used_model,
            duration_ms=(time.monotonic() - t0) * 1000,
            attempts=1,
        )


class OllamaImageGenLLM(ImageGenLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 300,
        ollama_url: str = "http://localhost:11434",
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
        self.ollama_url = ollama_url

    def generate(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        validator: Callable[[bytes], bool] | None = None,
        reference_images: list[bytes] | None = None,
        width: int = 128,
        height: int = 128,
        seed: int | None = None,
    ) -> ImageResponse:
        images_b64 = (
            [base64.b64encode(img).decode("ascii") for img in reference_images]
            if reference_images
            else []
        )

        def call_fn() -> tuple[bytes, str]:
            payload: dict = {
                "model": _bare_model(self.model),
                "prompt": prompt,
                "stream": False,
            }
            if images_b64:
                payload["images"] = images_b64
            options = _build_options(self.temperature, self.max_tokens)
            if width:
                options["width"] = width
            if height:
                options["height"] = height
            if seed is not None:
                options["seed"] = seed
            if options:
                payload["options"] = options
            if self.response_schema is not None:
                payload["format"] = self.response_schema

            resp = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()

            img_b64: str | None = None
            if "images" in data and data["images"]:
                img_b64 = data["images"][0]
            elif "image" in data:
                img_b64 = data["image"]
            if not img_b64:
                return b"", data.get("model", self.model)
            return base64.b64decode(img_b64), data.get("model", self.model)

        return retry_image_generation(call_fn, max_retries, self.model, validator=validator)


class OllamaImageInspectorLLM(ImageInspectorLLM):
    def __init__(
        self,
        model: str,
        timeout: int = 90,
        ollama_url: str = "http://localhost:11434",
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
        self.ollama_url = ollama_url

    def inspect(
        self, image: bytes, system: str, prompt: str, *, max_retries: int = 3
    ) -> TextResponse:
        image_b64 = base64.b64encode(image).decode("ascii")

        def call_fn(msgs: list[dict]) -> tuple[str, str]:
            return _ollama_generate(
                self.ollama_url,
                self.model,
                prompt,
                self.timeout,
                system=system,
                images=[image_b64],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_schema=self.response_schema,
            )

        messages: list[dict] = [{"role": "user", "content": prompt}]
        return retry_text_completion(call_fn, messages, max_retries, self.model)


class OllamaToolsLLM(ToolsLLM):
    """Ollama implementation of ToolsLLM (function/tool calling)."""

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        ollama_url: str = "http://localhost:11434",
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
        self.ollama_url = ollama_url

    def complete(
        self, messages: list[dict], tools: list[dict], *, max_retries: int = 3
    ) -> ToolCallResponse:
        t0 = time.monotonic()
        payload: dict = {
            "model": _bare_model(self.model),
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        options = _build_options(self.temperature, self.max_tokens)
        if options:
            payload["options"] = options

        resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        duration_ms = (time.monotonic() - t0) * 1000

        message = data.get("message", {})
        content = message.get("content") or None
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=tc.get("function", {}).get("name", ""),
                arguments=tc.get("function", {}).get("arguments") or {},
            )
            for i, tc in enumerate(raw_tool_calls)
        ]

        return ToolCallResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", self.model),
            duration_ms=duration_ms,
            attempts=1,
        )
