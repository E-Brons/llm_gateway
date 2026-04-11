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
    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> "TextResponse":  # noqa: F821
        """Send *messages* and return a TextResponse.

        Parameters
        ----------
        temperature:
            Per-call temperature override.  Overrides the instance-level
            ``temperature`` set at construction time.
        response_schema:
            Per-call JSON schema for structured output.  Overrides the
            instance-level ``response_schema`` set at construction time.
        """


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
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> "TextResponse":  # noqa: F821
        """Send *messages* and return a TextResponse, retrying on empty/invalid output.

        Parameters
        ----------
        temperature:
            Per-call temperature override.  Overrides the instance-level
            ``temperature`` set at construction time.
        response_schema:
            Per-call JSON schema for structured output.  Overrides the
            instance-level ``response_schema`` set at construction time.
        """


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
        temperature: float | None = None,
        response_schema: dict | None = None,
        options: dict | None = None,
    ) -> "TextResponse":  # noqa: F821
        """Send *messages* with optional thinking budget and return a TextResponse.

        Parameters
        ----------
        temperature:
            Per-call temperature override.  Overrides the instance-level
            ``temperature`` set at construction time.
        response_schema:
            Per-call JSON schema for structured output.  Overrides the
            instance-level ``response_schema`` set at construction time.
        """


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
        ip_adapter_scale: float | None = None,
        width: int = 256,
        height: int = 256,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        negative_prompt: str | None = None,
        cfg_scale: float | None = None,
        lora: str | None = None,
        lora_weight: float = 1.0,
        options: dict | None = None,
    ) -> "ImageResponse":  # noqa: F821
        """Generate an image from *prompt* and return an ImageResponse.

        Parameters
        ----------
        ip_adapter_scale:
            Conditioning strength for reference-image-guided generation
            (e.g. IP-Adapter).  Ignored by backends that do not support it.
        negative_prompt:
            Text describing what to avoid in the generated image.
        cfg_scale:
            Classifier-free guidance scale.  Higher values follow the prompt
            more closely.  Ignored by backends that do not support it.
        lora:
            HuggingFace repo ID or local path to a LoRA weight file.
            Ignored by backends that do not support it.
        lora_weight:
            LoRA conditioning scale (0–1).  Only used when *lora* is set.
        """


class ImageInspectorLLM(ABC):
    """Vision-language model: inspect / describe an image."""

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
    ) -> "TextResponse":  # noqa: F821
        """Describe or analyse *image* given *system* and *prompt*.

        Parameters
        ----------
        temperature:
            Per-call temperature override.  Overrides the instance-level
            ``temperature`` set at construction time.
        response_schema:
            Per-call JSON schema for structured output.  Overrides the
            instance-level ``response_schema`` set at construction time.
        """


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
        options: dict | None = None,
    ) -> "ToolCallResponse":  # noqa: F821
        """Send *messages* with *tools* definitions and return a ToolCallResponse."""


class IPAdapterLLM(ABC):
    """Image generation conditioned on a reference image via IP-Adapter."""

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
        reference_image: bytes,
        *,
        max_retries: int = 3,
        validator: Callable[[bytes], bool] | None = None,
        weight: float = 0.5,
        width: int = 256,
        height: int = 256,
        seed: int | None = None,
        num_inference_steps: int | None = None,
    ) -> "ImageResponse":  # noqa: F821
        """Generate an image conditioned on *reference_image* and return an ImageResponse."""


class IPAdapterFaceIDLLM(ABC):
    """Image generation conditioned on a face image via IP-Adapter FaceID."""

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
        face_image: bytes,
        *,
        max_retries: int = 3,
        validator: Callable[[bytes], bool] | None = None,
        weight: float = 0.5,
        width: int = 256,
        height: int = 256,
        seed: int | None = None,
        num_inference_steps: int | None = None,
    ) -> "ImageResponse":  # noqa: F821
        """Generate an image conditioned on *face_image* and return an ImageResponse."""


# Avoid circular import — import response types only for type annotations
from .responses import ImageResponse, TextResponse, ToolCallResponse  # noqa: E402, F401
