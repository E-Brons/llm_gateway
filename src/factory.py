"""Factory for constructing LLM interface instances from configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import LLMConfig, LLMTypeConfig, load_llm_config
from .impl.impl_cli import (
    CLIGeneralLLM,
    CLIImageInspectorLLM,
    CLIReasoningLLM,
    CLITextGenLLM,
)
from .impl.impl_diffusion_server import (
    DiffusionServerIPAdapterFaceIDLLM,
    DiffusionServerIPAdapterLLM,
)
from .impl.impl_litellm import (
    LiteLLMGeneralLLM,
    LiteLLMImageGenLLM,
    LiteLLMImageInspectorLLM,
    LiteLLMReasoningLLM,
    LiteLLMTextGenLLM,
    LiteLLMToolsLLM,
)
from .impl.impl_ollama import (
    OllamaGeneralLLM,
    OllamaImageGenLLM,
    OllamaImageInspectorLLM,
    OllamaReasoningLLM,
    OllamaTextGenLLM,
    OllamaToolsLLM,
)
from .types import (
    GeneralLLM,
    ImageGenLLM,
    ImageInspectorLLM,
    IPAdapterFaceIDLLM,
    IPAdapterLLM,
    ReasoningLLM,
    TextGenLLM,
    ToolsLLM,
)

_REGISTRY: dict[tuple[str, str], type] = {
    ("general", "ollama"): OllamaGeneralLLM,
    ("general", "litellm"): LiteLLMGeneralLLM,
    ("general", "cli"): CLIGeneralLLM,
    ("text_gen", "ollama"): OllamaTextGenLLM,
    ("text_gen", "litellm"): LiteLLMTextGenLLM,
    ("text_gen", "cli"): CLITextGenLLM,
    ("reasoning", "ollama"): OllamaReasoningLLM,
    ("reasoning", "litellm"): LiteLLMReasoningLLM,
    ("reasoning", "cli"): CLIReasoningLLM,
    ("image_gen", "ollama"): OllamaImageGenLLM,
    ("image_gen", "litellm"): LiteLLMImageGenLLM,
    # CLI does not support image generation
    ("image_inspector", "ollama"): OllamaImageInspectorLLM,
    ("image_inspector", "litellm"): LiteLLMImageInspectorLLM,
    ("image_inspector", "cli"): CLIImageInspectorLLM,
    ("tools", "ollama"): OllamaToolsLLM,
    ("tools", "litellm"): LiteLLMToolsLLM,
    # CLI does not support tool use
    ("ipadapter", "diffusion_server"): DiffusionServerIPAdapterLLM,
    ("ipadapter_faceid", "diffusion_server"): DiffusionServerIPAdapterFaceIDLLM,
}


def _build(type_name: str, cfg: LLMTypeConfig) -> Any:
    key = (type_name, cfg.implementation)
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"No implementation registered for type={type_name!r}, "
            f"impl={cfg.implementation!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )

    kwargs: dict[str, Any] = {"model": cfg.model}
    if cfg.timeout is not None:
        kwargs["timeout"] = cfg.timeout
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.max_tokens is not None:
        kwargs["max_tokens"] = cfg.max_tokens
    if cfg.response_schema is not None:
        kwargs["response_schema"] = cfg.response_schema
    if cfg.implementation == "ollama" and cfg.ollama_url:
        kwargs["ollama_url"] = cfg.ollama_url
    if cfg.implementation in ("litellm", "diffusion_server") and cfg.api_base:
        kwargs["api_base"] = cfg.api_base

    return cls(**kwargs)


class LLMFactory:
    """Construct typed LLM instances from a validated LLMConfig."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def general(self) -> GeneralLLM:
        return _build("general", self._config.general)

    def text_gen(self) -> TextGenLLM:
        return _build("text_gen", self._config.text_gen)

    def reasoning(self) -> ReasoningLLM:
        return _build("reasoning", self._config.reasoning)

    def image_gen(self) -> ImageGenLLM:
        return _build("image_gen", self._config.image_gen)

    def image_inspector(self) -> ImageInspectorLLM:
        return _build("image_inspector", self._config.image_inspector)

    def tools(self) -> ToolsLLM:
        return _build("tools", self._config.tools)

    def ipadapter(self) -> IPAdapterLLM:
        if self._config.ipadapter is None:
            raise ValueError(
                "ipadapter is not configured. Add an 'ipadapter' section to llm_route.yml."
            )
        return _build("ipadapter", self._config.ipadapter)

    def ipadapter_faceid(self) -> IPAdapterFaceIDLLM:
        if self._config.ipadapter_faceid is None:
            raise ValueError(
                "ipadapter_faceid is not configured. "
                "Add an 'ipadapter_faceid' section to llm_route.yml."
            )
        return _build("ipadapter_faceid", self._config.ipadapter_faceid)


def create_factory(config_path: str | Path) -> LLMFactory:
    """Load a YAML config file and return a ready-to-use LLMFactory."""
    return LLMFactory(load_llm_config(config_path))
