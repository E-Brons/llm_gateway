"""Tests for LLMConfig, load_llm_config, and LLMFactory."""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

_SAMPLE_YAML = textwrap.dedent("""\
    general:
      implementation: ollama
      model: ollama/phi3
      ollama_url: http://localhost:11434

    text_gen:
      implementation: litellm
      model: gpt-4o
      api_base: null

    reasoning:
      implementation: litellm
      model: claude-opus-4-6

    image_gen:
      implementation: ollama
      model: ollama/flux
      ollama_url: http://localhost:11434

    image_inspector:
      implementation: ollama
      model: ollama/llava
      ollama_url: http://localhost:11434

    tools:
      implementation: litellm
      model: gpt-4o
""")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_llm_config_from_yaml(tmp_path: Path):
    from src.config import load_llm_config

    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(_SAMPLE_YAML)
    cfg = load_llm_config(cfg_file)

    assert cfg.general.implementation == "ollama"
    assert cfg.general.model == "ollama/phi3"
    assert cfg.text_gen.model == "gpt-4o"
    assert cfg.reasoning.model == "claude-opus-4-6"
    assert cfg.image_gen.model == "ollama/flux"
    assert cfg.tools.implementation == "litellm"


def test_load_llm_config_missing_field_raises(tmp_path: Path):
    from pydantic import ValidationError

    from src.config import load_llm_config

    bad_yaml = textwrap.dedent("""\
        general:
          implementation: ollama
          # missing model
    """)
    cfg_file = tmp_path / "bad.yml"
    cfg_file.write_text(bad_yaml)
    with pytest.raises((ValidationError, Exception)):
        load_llm_config(cfg_file)


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------


def test_factory_general_returns_ollama(tmp_path: Path):
    from src.config import load_llm_config
    from src.factory import LLMFactory
    from src.impl.impl_ollama import OllamaGeneralLLM

    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(_SAMPLE_YAML)
    cfg = load_llm_config(cfg_file)

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        factory = LLMFactory(cfg)
        obj = factory.general()

    assert isinstance(obj, OllamaGeneralLLM)


def test_factory_text_gen_returns_litellm(tmp_path: Path):
    from src.config import load_llm_config
    from src.factory import LLMFactory
    from src.impl.impl_litellm import LiteLLMTextGenLLM

    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(_SAMPLE_YAML)
    cfg = load_llm_config(cfg_file)

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        factory = LLMFactory(cfg)
        obj = factory.text_gen()

    assert isinstance(obj, LiteLLMTextGenLLM)


def test_factory_tools_returns_litellm(tmp_path: Path):
    from src.config import load_llm_config
    from src.factory import LLMFactory
    from src.impl.impl_litellm import LiteLLMToolsLLM

    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(_SAMPLE_YAML)
    cfg = load_llm_config(cfg_file)

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        factory = LLMFactory(cfg)
        obj = factory.tools()

    assert isinstance(obj, LiteLLMToolsLLM)


def test_factory_tools_returns_ollama(tmp_path: Path):
    from src.config import load_llm_config
    from src.factory import LLMFactory
    from src.impl.impl_ollama import OllamaToolsLLM

    yaml = textwrap.dedent("""\
        general:
          implementation: ollama
          model: phi3
          ollama_url: http://localhost:11434
        text_gen:
          implementation: ollama
          model: phi3
          ollama_url: http://localhost:11434
        reasoning:
          implementation: ollama
          model: phi3
          ollama_url: http://localhost:11434
        image_gen:
          implementation: ollama
          model: flux
          ollama_url: http://localhost:11434
        image_inspector:
          implementation: ollama
          model: llava
          ollama_url: http://localhost:11434
        tools:
          implementation: ollama
          model: phi3
          ollama_url: http://localhost:11434
    """)
    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(yaml)
    cfg = load_llm_config(cfg_file)
    factory = LLMFactory(cfg)
    obj = factory.tools()

    assert isinstance(obj, OllamaToolsLLM)


def test_factory_invalid_impl_raises(tmp_path: Path):
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory

    bad_cfg = LLMConfig(
        general=LLMTypeConfig(implementation="bogus", model="x"),
        text_gen=LLMTypeConfig(implementation="ollama", model="x"),
        reasoning=LLMTypeConfig(implementation="ollama", model="x"),
        image_gen=LLMTypeConfig(implementation="ollama", model="x"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="x"),
        tools=LLMTypeConfig(implementation="ollama", model="x"),
    )
    factory = LLMFactory(bad_cfg)
    with pytest.raises(ValueError, match="No implementation registered"):
        factory.general()


def test_factory_tools_cli_raises(tmp_path: Path):
    """CLI does not support tool use — should raise ValueError."""
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="x"),
        text_gen=LLMTypeConfig(implementation="ollama", model="x"),
        reasoning=LLMTypeConfig(implementation="ollama", model="x"),
        image_gen=LLMTypeConfig(implementation="ollama", model="x"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="x"),
        tools=LLMTypeConfig(implementation="cli", model="claude"),
    )
    factory = LLMFactory(cfg)
    with pytest.raises(ValueError, match="No implementation registered"):
        factory.tools()


def test_create_factory_convenience(tmp_path: Path):
    from src.factory import create_factory

    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(_SAMPLE_YAML)

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        factory = create_factory(cfg_file)

    assert factory is not None
