"""Response types for all LLM calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TextResponse(BaseModel, frozen=True):
    """Result from a text-completion LLM call."""

    content: str
    model: str
    duration_ms: float
    attempts: int
    last_error: str | None = None


class ImageResponse(BaseModel, frozen=True):
    """Result from an image-generation LLM call."""

    image: bytes
    model: str
    duration_ms: float
    attempts: int
    last_error: str | None = None


class ToolCall(BaseModel, frozen=True):
    """A single tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallResponse(BaseModel, frozen=True):
    """Result from a tool-use LLM call."""

    content: str | None  # optional text alongside tool calls
    tool_calls: list[ToolCall]
    model: str
    duration_ms: float
    attempts: int
    last_error: str | None = None
