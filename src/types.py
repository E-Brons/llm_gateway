"""Abstract base classes for the six LLM interface types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class GeneralLLM(ABC):
    """General-purpose text completion — no retry, no structured output."""

    def __init__(
        self,
        model: str,
        timeout: int = 60,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_schema = response_schema

    @abstractmethod
    def complete(self, messages: list[dict]) -> "TextResponse":  # noqa: F821
        """Send *messages* and return a TextResponse."""


class TextGenLLM(ABC):
    """Text generation with validation retry for structured outputs."""

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_schema = response_schema

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        max_retries: int = 3,
    ) -> "TextResponse":  # noqa: F821
        """Send *messages* and return a TextResponse, retrying on empty/invalid output."""


class ReasoningLLM(ABC):
    """Extended-thinking / reasoning-mode text generation."""

    def __init__(
        self,
        model: str,
        timeout: int = 300,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_schema = response_schema

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        thinking_budget: int | None = None,
    ) -> "TextResponse":  # noqa: F821
        """Send *messages* with optional thinking budget and return a TextResponse."""


class ImageGenLLM(ABC):
    """Image generation from a text prompt."""

    def __init__(
        self,
        model: str,
        timeout: int = 300,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_schema = response_schema

    @abstractmethod
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
    ) -> "ImageResponse":  # noqa: F821
        """Generate an image from *prompt* and return an ImageResponse."""


class ImageInspectorLLM(ABC):
    """Vision-language model: inspect / describe an image."""

    def __init__(
        self,
        model: str,
        timeout: int = 90,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_schema = response_schema

    @abstractmethod
    def inspect(
        self,
        image: bytes,
        system: str,
        prompt: str,
        *,
        max_retries: int = 3,
    ) -> "TextResponse":  # noqa: F821
        """Describe or analyse *image* given *system* and *prompt*."""


class ToolsLLM(ABC):
    """Tool-use / function-calling LLM."""

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_schema = response_schema

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        max_retries: int = 3,
    ) -> "ToolCallResponse":  # noqa: F821
        """Send *messages* with *tools* definitions and return a ToolCallResponse."""


# Avoid circular import — import response types only for type annotations
from .responses import ImageResponse, TextResponse, ToolCallResponse  # noqa: E402, F401
