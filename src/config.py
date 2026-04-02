"""Configuration models and YAML loader for the llm_gateway module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class LLMTypeConfig(BaseModel):
    """Per-task-type LLM configuration."""

    implementation: str  # "ollama" | "litellm" | "cli"
    model: str
    api_base: str | None = None
    ollama_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int | None = None
    thinking_budget: int | None = None
    timeout: int | None = None
    response_schema: dict | None = None


class LLMConfig(BaseModel):
    """Top-level configuration mapping each task type to an LLMTypeConfig."""

    general: LLMTypeConfig
    text_gen: LLMTypeConfig
    reasoning: LLMTypeConfig
    image_gen: LLMTypeConfig
    image_inspector: LLMTypeConfig
    tools: LLMTypeConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_llm_config(
    path: str | Path,
    override_path: str | Path | None = None,
) -> LLMConfig:
    """Load a YAML config file, optionally deep-merging a local override.

    Parameters
    ----------
    path:
        Base config file (e.g. ``llm_route.yml``). Must exist.
    override_path:
        Optional local override (e.g. ``local/llm_route.yml``).
        Loaded only if the file exists — missing file is silently ignored.
        Overrides are merged at the key level, so a local file only needs
        to specify the fields it wants to change.
    """
    data: Any = yaml.safe_load(Path(path).read_text())

    if override_path is not None:
        override_file = Path(override_path)
        if override_file.exists():
            override_data: Any = yaml.safe_load(override_file.read_text())
            if override_data:
                data = _deep_merge(data, override_data)

    return LLMConfig.model_validate(data)
