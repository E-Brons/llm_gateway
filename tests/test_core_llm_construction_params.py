"""Tests verifying construction-time parameters are forwarded to API calls."""

import textwrap
from unittest.mock import MagicMock, patch

_OLLAMA_URL = "http://localhost:11434"


def _mock_chat_response(content: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"message": {"content": content}, "model": "m"}
    return resp


def _mock_litellm_completion(content: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    return resp


# ---------------------------------------------------------------------------
# Ollama — temperature and max_tokens forwarded as options
# ---------------------------------------------------------------------------


def test_ollama_general_temperature_in_options():
    from src.impl.impl_ollama import OllamaGeneralLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response()
    ) as mock_post:
        llm = OllamaGeneralLLM(model="phi3", ollama_url=_OLLAMA_URL, temperature=0.3)
        llm.complete([{"role": "user", "content": "x"}])

    assert mock_post.call_args[1]["json"]["options"]["temperature"] == 0.3


def test_ollama_general_max_tokens_as_num_predict():
    from src.impl.impl_ollama import OllamaGeneralLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response()
    ) as mock_post:
        llm = OllamaGeneralLLM(model="phi3", ollama_url=_OLLAMA_URL, max_tokens=512)
        llm.complete([{"role": "user", "content": "x"}])

    assert mock_post.call_args[1]["json"]["options"]["num_predict"] == 512


def test_ollama_general_response_schema_as_format():
    from src.impl.impl_ollama import OllamaGeneralLLM

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response()
    ) as mock_post:
        llm = OllamaGeneralLLM(model="phi3", ollama_url=_OLLAMA_URL, response_schema=schema)
        llm.complete([{"role": "user", "content": "x"}])

    assert mock_post.call_args[1]["json"]["format"] == schema


def test_ollama_no_options_when_not_set():
    from src.impl.impl_ollama import OllamaGeneralLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response()
    ) as mock_post:
        llm = OllamaGeneralLLM(model="phi3", ollama_url=_OLLAMA_URL)
        llm.complete([{"role": "user", "content": "x"}])

    payload = mock_post.call_args[1]["json"]
    assert "options" not in payload
    assert "format" not in payload


# ---------------------------------------------------------------------------
# LiteLLM — temperature, max_tokens, response_schema forwarded
# ---------------------------------------------------------------------------


def test_litellm_general_temperature_passed():
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_litellm_completion()
        ) as mock_c:
            llm = LiteLLMGeneralLLM(model="gpt-4o", temperature=0.7)
            llm.complete([{"role": "user", "content": "x"}])

    assert mock_c.call_args[1]["temperature"] == 0.7


def test_litellm_general_max_tokens_passed():
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_litellm_completion()
        ) as mock_c:
            llm = LiteLLMGeneralLLM(model="gpt-4o", max_tokens=1024)
            llm.complete([{"role": "user", "content": "x"}])

    assert mock_c.call_args[1]["max_tokens"] == 1024


def test_litellm_general_response_schema_passed():
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    schema = {"type": "json_object"}

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_litellm_completion()
        ) as mock_c:
            llm = LiteLLMGeneralLLM(model="gpt-4o", response_schema=schema)
            llm.complete([{"role": "user", "content": "x"}])

    assert mock_c.call_args[1]["response_format"] == schema


def test_litellm_no_extras_when_not_set():
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_litellm_completion()
        ) as mock_c:
            llm = LiteLLMGeneralLLM(model="gpt-4o")
            llm.complete([{"role": "user", "content": "x"}])

    call_kwargs = mock_c.call_args[1]
    assert "temperature" not in call_kwargs
    assert "max_tokens" not in call_kwargs
    assert "response_format" not in call_kwargs


# ---------------------------------------------------------------------------
# CLI — params stored but not forwarded (accepted without error)
# ---------------------------------------------------------------------------


def test_cli_accepts_construction_params():
    from src.impl.impl_cli import CLITextGenLLM

    schema = {"type": "object"}
    llm = CLITextGenLLM(model="claude", temperature=0.5, max_tokens=256, response_schema=schema)
    assert llm.temperature == 0.5
    assert llm.max_tokens == 256
    assert llm.response_schema == schema


# ---------------------------------------------------------------------------
# Per-call response_schema overrides construction-time schema
# ---------------------------------------------------------------------------


def test_ollama_per_call_schema_overrides_config_schema():
    """A schema passed to complete() is used instead of the one set at construction time."""
    from src.impl.impl_ollama import OllamaGeneralLLM

    config_schema = {"type": "object", "properties": {"old": {"type": "string"}}}
    call_schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response('{"name":"x"}')
    ) as mock_post:
        llm = OllamaGeneralLLM(model="phi3", ollama_url=_OLLAMA_URL, response_schema=config_schema)
        llm.complete([{"role": "user", "content": "x"}], response_schema=call_schema)

    assert mock_post.call_args[1]["json"]["format"] == call_schema


def test_ollama_per_call_schema_used_when_no_config_schema():
    """A schema passed to complete() is applied even without a config-level schema."""
    from src.impl.impl_ollama import OllamaGeneralLLM

    call_schema = {"type": "object", "properties": {"value": {"type": "integer"}}}

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response('{"value":1}')
    ) as mock_post:
        llm = OllamaGeneralLLM(model="phi3", ollama_url=_OLLAMA_URL)
        llm.complete([{"role": "user", "content": "x"}], response_schema=call_schema)

    assert mock_post.call_args[1]["json"]["format"] == call_schema


def test_litellm_per_call_schema_overrides_config_schema():
    """A schema passed to complete() supersedes the construction-time schema for LiteLLM."""
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    config_schema = {"type": "json_object"}
    call_schema = {"type": "json_schema", "json_schema": {"name": "result", "schema": {}}}

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion",
            return_value=_mock_litellm_completion('{"ok":true}'),
        ) as mock_c:
            llm = LiteLLMGeneralLLM(model="gpt-4o", response_schema=config_schema)
            llm.complete([{"role": "user", "content": "x"}], response_schema=call_schema)

    assert mock_c.call_args[1]["response_format"] == call_schema


def test_cli_per_call_schema_injected_as_system_message():
    """For the CLI backend the schema is injected as a system-prompt instruction."""
    from unittest.mock import patch

    from src.impl.impl_cli import CLITextGenLLM

    call_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    captured: dict = {}

    def fake_stream_json(msgs, *, timeout=120, effort=None):
        captured["msgs"] = msgs
        return '{"answer":"42"}', 100.0

    with patch("src.impl.impl_cli._run_claude_stream_json", side_effect=fake_stream_json):
        llm = CLITextGenLLM(model="claude")
        llm.complete([{"role": "user", "content": "What is 6*7?"}], response_schema=call_schema)

    # The first message should be a system message containing schema instructions
    first = captured["msgs"][0]
    assert first["role"] == "system"
    assert "answer" in first["content"]  # schema property appears in the instruction


# ---------------------------------------------------------------------------
# Factory — passes temperature/max_tokens from config
# ---------------------------------------------------------------------------


def test_factory_passes_temperature_from_config(tmp_path):
    from src.config import load_llm_config
    from src.factory import LLMFactory

    yaml = textwrap.dedent("""\
        general:
          implementation: ollama
          model: phi3
          ollama_url: http://localhost:11434
          temperature: 0.25
          max_tokens: 512

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
    obj = factory.general()

    assert obj.temperature == 0.25
    assert obj.max_tokens == 512
