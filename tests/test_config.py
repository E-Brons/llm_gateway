"""Tests for config loading and deep merge."""

import textwrap

_BASE_YAML = textwrap.dedent("""\
    general:
      implementation: ollama
      model: ollama/phi3
      ollama_url: http://localhost:11434
      temperature: 0.8
    text_gen:
      implementation: ollama
      model: ollama/phi3
      ollama_url: http://localhost:11434
    reasoning:
      implementation: ollama
      model: ollama/phi3
      ollama_url: http://localhost:11434
    image_gen:
      implementation: ollama
      model: ollama/flux
      ollama_url: http://localhost:11434
    image_inspector:
      implementation: ollama
      model: ollama/llava
      ollama_url: http://localhost:11434
    tools:
      implementation: ollama
      model: ollama/phi3
      ollama_url: http://localhost:11434
""")


# ── _deep_merge ───────────────────────────────────────────────────────────────


def test_deep_merge_flat():
    from src.config import _deep_merge

    assert _deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested_partial():
    """Override only some keys of a nested dict — others are preserved."""
    from src.config import _deep_merge

    base = {
        "general": {"model": "phi3", "temperature": 0.8, "ollama_url": "http://localhost:11434"}
    }
    override = {"general": {"model": "llama3"}}
    result = _deep_merge(base, override)
    assert result["general"]["model"] == "llama3"
    assert result["general"]["temperature"] == 0.8
    assert result["general"]["ollama_url"] == "http://localhost:11434"


def test_deep_merge_adds_new_top_level_key():
    from src.config import _deep_merge

    result = _deep_merge({"a": 1}, {"b": 2})
    assert result == {"a": 1, "b": 2}


def test_deep_merge_non_dict_value_replaces():
    from src.config import _deep_merge

    result = _deep_merge({"a": {"x": 1}}, {"a": 99})
    assert result["a"] == 99


def test_deep_merge_does_not_mutate_base():
    from src.config import _deep_merge

    base = {"a": {"x": 1}}
    _deep_merge(base, {"a": {"x": 2}})
    assert base["a"]["x"] == 1


# ── load_llm_config with override ────────────────────────────────────────────


def test_load_with_override_merges_task(tmp_path):
    from src.config import load_llm_config

    base = tmp_path / "llm_route.yml"
    base.write_text(_BASE_YAML)

    override = tmp_path / "local.yml"
    override.write_text(
        "general:\n  model: ollama/llama3\ntools:\n  implementation: litellm\n  model: gpt-4o\n"
    )

    cfg = load_llm_config(base, override_path=override)

    assert cfg.general.model == "ollama/llama3"
    assert cfg.general.implementation == "ollama"  # preserved from base
    assert cfg.general.temperature == 0.8  # preserved from base
    assert cfg.tools.implementation == "litellm"
    assert cfg.tools.model == "gpt-4o"


def test_load_with_override_missing_is_silent(tmp_path):
    from src.config import load_llm_config

    base = tmp_path / "llm_route.yml"
    base.write_text(_BASE_YAML)

    cfg = load_llm_config(base, override_path=tmp_path / "nonexistent.yml")
    assert cfg.general.model == "ollama/phi3"


def test_load_with_no_override_arg(tmp_path):
    from src.config import load_llm_config

    base = tmp_path / "llm_route.yml"
    base.write_text(_BASE_YAML)

    cfg = load_llm_config(base)
    assert cfg.general.model == "ollama/phi3"
