"""Additional tests to improve coverage across all modules.

Covers gaps identified in:
  src/_retry.py          – _is_client_error, _is_timeout, client-error propagation
  src/factory.py         – timeout/response_schema kwargs, reasoning/image_gen/image_inspector,
                           ipadapter not-configured raises
  src/impl/impl_cli.py   – _inject_schema no-system path, CalledProcessError paths,
                           list-content system merge
  src/impl/impl_ipadapter.py – error response fallback, FaceID path
  src/impl/impl_litellm.py   – api_base forwarding, invalid tool-call JSON args
  src/impl/impl_ollama.py    – _ollama_generate response_schema, image fallback key
  src/types.py           – IPAdapterLLM / IPAdapterFaceIDLLM concrete constructors
  src/server.py          – root, /api/tags, /v1/models, endpoint routes,
                           _load_settings, _minimal_png, diffusion availability
"""

from __future__ import annotations

import base64
import os
import textwrap
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = textwrap.dedent("""\
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
      model: ollama/flux
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

_OLLAMA_URL = "http://localhost:11434"


def _text(content: str = "hello"):
    from src.responses import TextResponse

    return TextResponse(content=content, model="phi3", duration_ms=100.0, attempts=1)


def _image():
    from src.responses import ImageResponse

    return ImageResponse(image=b"\x89PNG", model="flux", duration_ms=200.0, attempts=1)


def _mock_chat_response(content: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"message": {"content": content}, "model": "m"}
    return resp


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import app

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(_MINIMAL_CONFIG)

    mock_factory = MagicMock()
    env = {
        "LLM_GATEWAY_ROUTE": str(cfg),
        "LLM_GATEWAY_ROUTE_LOCAL": str(tmp_path / "no_override.yml"),
        "LLM_GATEWAY_HOST": "127.0.0.1",
        "LLM_GATEWAY_PORT": "4096",
    }

    with patch.dict(os.environ, env):
        with patch("src.server.LLMFactory", return_value=mock_factory):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app) as c:
                    yield c, mock_factory


# ===========================================================================
# _retry.py — _is_client_error
# ===========================================================================


def test_is_client_error_requests_4xx():
    import requests

    from src._retry import _is_client_error

    mock_resp = MagicMock()
    mock_resp.status_code = 422
    exc = requests.exceptions.HTTPError(response=mock_resp)
    assert _is_client_error(exc) is True


def test_is_client_error_requests_5xx_returns_false():
    import requests

    from src._retry import _is_client_error

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    exc = requests.exceptions.HTTPError(response=mock_resp)
    assert _is_client_error(exc) is False


def test_is_client_error_httpx_4xx():
    import httpx

    from src._retry import _is_client_error

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    exc = httpx.HTTPStatusError("bad request", request=MagicMock(), response=mock_resp)
    assert _is_client_error(exc) is True


def test_is_client_error_httpx_5xx_returns_false():
    import httpx

    from src._retry import _is_client_error

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    exc = httpx.HTTPStatusError("service unavailable", request=MagicMock(), response=mock_resp)
    assert _is_client_error(exc) is False


def test_is_client_error_other_exception_returns_false():
    from src._retry import _is_client_error

    assert _is_client_error(RuntimeError("something else")) is False


# ===========================================================================
# _retry.py — _is_timeout (httpx + message fallback)
# ===========================================================================


def test_is_timeout_httpx_timeout_exception():
    import httpx

    from src._retry import _is_timeout

    exc = httpx.TimeoutException("timed out", request=MagicMock())
    assert _is_timeout(exc) is True


def test_is_timeout_httpx_read_timeout():
    import httpx

    from src._retry import _is_timeout

    exc = httpx.ReadTimeout("read timeout", request=MagicMock())
    assert _is_timeout(exc) is True


def test_is_timeout_message_timed_out():
    from src._retry import _is_timeout

    assert _is_timeout(RuntimeError("Request timed out after 30s")) is True


def test_is_timeout_message_timeout():
    from src._retry import _is_timeout

    assert _is_timeout(RuntimeError("Connection timeout")) is True


def test_is_timeout_message_read_timeout():
    from src._retry import _is_timeout

    assert _is_timeout(RuntimeError("read timeout")) is True


def test_is_timeout_unrelated_error_returns_false():
    from src._retry import _is_timeout

    assert _is_timeout(RuntimeError("some other error")) is False


# ===========================================================================
# _retry.py — client error propagation in retry loops
# ===========================================================================


def test_retry_text_client_error_raises_immediately():
    import requests

    from src._retry import retry_text_completion

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    exc = requests.exceptions.HTTPError(response=mock_resp)

    calls = [0]

    def call_fn(msgs):
        calls[0] += 1
        raise exc

    with pytest.raises(requests.exceptions.HTTPError):
        retry_text_completion(call_fn, [{"role": "user", "content": "x"}], 3, "m")

    assert calls[0] == 1  # must not retry on 4xx


def test_retry_image_client_error_raises_immediately():
    import requests

    from src._retry import retry_image_generation

    mock_resp = MagicMock()
    mock_resp.status_code = 422
    exc = requests.exceptions.HTTPError(response=mock_resp)

    calls = [0]

    def call_fn():
        calls[0] += 1
        raise exc

    with pytest.raises(requests.exceptions.HTTPError):
        retry_image_generation(call_fn, 3, "m")

    assert calls[0] == 1


def test_retry_image_httpx_timeout_raises_immediately():
    import httpx

    from src._retry import retry_image_generation

    calls = [0]

    def call_fn():
        calls[0] += 1
        raise httpx.ReadTimeout("read timeout", request=MagicMock())

    with pytest.raises(httpx.ReadTimeout):
        retry_image_generation(call_fn, 3, "m")

    assert calls[0] == 1


# ===========================================================================
# factory.py — timeout / response_schema kwargs passed by _build
# ===========================================================================


def test_factory_build_passes_timeout():
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="phi3", timeout=999),
        text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
        reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
        image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
        tools=LLMTypeConfig(implementation="ollama", model="phi3"),
    )
    factory = LLMFactory(cfg)
    obj = factory.general()
    assert obj.timeout == 999


def test_factory_build_passes_response_schema():
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory

    schema = {"type": "object", "properties": {"result": {"type": "string"}}}
    cfg = LLMConfig(
        general=LLMTypeConfig(
            implementation="ollama", model="phi3", response_schema=schema
        ),
        text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
        reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
        image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
        tools=LLMTypeConfig(implementation="ollama", model="phi3"),
    )
    factory = LLMFactory(cfg)
    obj = factory.general()
    assert obj.response_schema == schema


def test_factory_reasoning_returns_ollama():
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory
    from src.impl.impl_ollama import OllamaReasoningLLM

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="phi3"),
        text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
        reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
        image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
        tools=LLMTypeConfig(implementation="ollama", model="phi3"),
    )
    factory = LLMFactory(cfg)
    assert isinstance(factory.reasoning(), OllamaReasoningLLM)


def test_factory_image_gen_returns_ollama():
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory
    from src.impl.impl_ollama import OllamaImageGenLLM

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="phi3"),
        text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
        reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
        image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
        tools=LLMTypeConfig(implementation="ollama", model="phi3"),
    )
    factory = LLMFactory(cfg)
    assert isinstance(factory.image_gen(), OllamaImageGenLLM)


def test_factory_image_inspector_returns_ollama():
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="phi3"),
        text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
        reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
        image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
        tools=LLMTypeConfig(implementation="ollama", model="phi3"),
    )
    factory = LLMFactory(cfg)
    assert isinstance(factory.image_inspector(), OllamaImageInspectorLLM)


def test_factory_ipadapter_not_configured_raises():
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="phi3"),
        text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
        reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
        image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
        tools=LLMTypeConfig(implementation="ollama", model="phi3"),
    )
    factory = LLMFactory(cfg)
    with pytest.raises(ValueError, match="ipadapter is not configured"):
        factory.ipadapter()


def test_factory_ipadapter_faceid_not_configured_raises():
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="phi3"),
        text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
        reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
        image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
        tools=LLMTypeConfig(implementation="ollama", model="phi3"),
    )
    factory = LLMFactory(cfg)
    with pytest.raises(ValueError, match="ipadapter_faceid is not configured"):
        factory.ipadapter_faceid()


def test_factory_build_passes_litellm_api_base():
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        cfg = LLMConfig(
            general=LLMTypeConfig(
                implementation="litellm",
                model="gpt-4o",
                api_base="http://custom:8080",
            ),
            text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
            reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
            image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
            image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
            tools=LLMTypeConfig(implementation="ollama", model="phi3"),
        )
        factory = LLMFactory(cfg)
        obj = factory.general()
    assert obj.api_base == "http://custom:8080"


# ===========================================================================
# impl_cli.py — _inject_schema: no existing system message → new one prepended
# ===========================================================================


def test_inject_schema_creates_system_message_when_none_present():
    from src.impl.impl_cli import _inject_schema

    msgs = [{"role": "user", "content": "hello"}]
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = _inject_schema(msgs, schema)

    assert result[0]["role"] == "system"
    assert "answer" in result[0]["content"]
    assert result[1] == msgs[0]


def test_inject_schema_merges_into_existing_system_message():
    from src.impl.impl_cli import _inject_schema

    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "hello"},
    ]
    schema = {"type": "object"}
    result = _inject_schema(msgs, schema)

    assert result[0]["role"] == "system"
    assert "Be helpful." in result[0]["content"]
    assert len(result) == 2


def test_run_claude_nonzero_exit_raises():
    import subprocess

    from src.impl.impl_cli import _run_claude

    proc = MagicMock()
    proc.returncode = 1
    proc.stdout = ""
    proc.stderr = "auth error"

    with patch("src.impl.impl_cli.subprocess.run", return_value=proc):
        with pytest.raises(subprocess.CalledProcessError):
            _run_claude("hello")


def test_run_claude_stream_json_nonzero_exit_raises():
    import subprocess

    from src.impl.impl_cli import _run_claude_stream_json

    proc = MagicMock()
    proc.returncode = 2
    proc.stdout = ""
    proc.stderr = "fatal error"

    with patch("src.impl.impl_cli.subprocess.run", return_value=proc):
        with pytest.raises(subprocess.CalledProcessError):
            _run_claude_stream_json([{"role": "user", "content": "hi"}])


def test_run_claude_stream_json_list_content_merged_with_system():
    """When first user message has list content, system text is prepended as text block."""
    from src.impl.impl_cli import _run_claude_stream_json

    result_line = '{"type": "result", "result": "ok"}'
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = result_line
    proc.stderr = ""

    messages = [
        {"role": "system", "content": "Be precise."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "describe"}],
        },
    ]

    with patch("src.impl.impl_cli.subprocess.run", return_value=proc) as mock_run:
        result, _ = _run_claude_stream_json(messages)

    # The stdin passed to the subprocess must contain merged content
    stdin_data = mock_run.call_args[1]["input"]
    assert "Be precise." in stdin_data
    assert result == "ok"


# ===========================================================================
# impl_ipadapter.py — error response fallback + FaceID paths
# ===========================================================================


def test_ipadapter_error_response_json_parse_fallback():
    """When resp.ok is False and resp.json() raises, detail falls back to resp.text.

    A 4xx client error is used so it propagates immediately without retrying.
    """
    import requests

    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    bad_resp = MagicMock()
    bad_resp.ok = False
    bad_resp.status_code = 422
    bad_resp.reason = "Unprocessable Entity"
    bad_resp.json.side_effect = ValueError("not json")
    bad_resp.text = "raw error text"

    with patch("src.impl.impl_ipadapter.requests.post", return_value=bad_resp):
        llm = DiffusionServerIPAdapterLLM(model="ip-adapter", api_base="http://localhost:7860")
        with pytest.raises(requests.exceptions.HTTPError, match="raw error text"):
            llm.generate("a cat", reference_images=[b"ref"], max_retries=1)


def test_ipadapter_faceid_happy_path():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    raw = b"\x89PNG\r\n"
    b64 = base64.b64encode(raw).decode()

    resp = MagicMock()
    resp.ok = True
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"image": b64, "model": "ip-adapter-faceid"}

    with patch("src.impl.impl_ipadapter.requests.post", return_value=resp):
        llm = DiffusionServerIPAdapterFaceIDLLM(
            model="ip-adapter-faceid", api_base="http://localhost:7860"
        )
        result = llm.generate("a portrait", reference_images=[b"face"], max_retries=1)

    assert result.image == raw
    assert result.attempts == 1


def test_ipadapter_faceid_error_response_json_parse_fallback():
    """FaceID: when resp.ok is False and resp.json() raises, fallback to resp.text."""
    import requests

    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    bad_resp = MagicMock()
    bad_resp.ok = False
    bad_resp.status_code = 422
    bad_resp.reason = "Unprocessable"
    bad_resp.json.side_effect = ValueError("not json")
    bad_resp.text = "invalid face image"

    with patch("src.impl.impl_ipadapter.requests.post", return_value=bad_resp):
        llm = DiffusionServerIPAdapterFaceIDLLM(
            model="ip-adapter-faceid", api_base="http://localhost:7860"
        )
        with pytest.raises(requests.exceptions.HTTPError, match="invalid face image"):
            llm.generate("a portrait", reference_images=[b"face"], max_retries=1)


# ===========================================================================
# impl_litellm.py — api_base forwarding for each class
# ===========================================================================


def test_litellm_text_gen_api_base_passed():
    from src.impl.impl_litellm import LiteLLMTextGenLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion",
            return_value=_make_litellm_response("ok"),
        ) as mock_c:
            llm = LiteLLMTextGenLLM(model="ollama/phi3", api_base="http://localhost:11434")
            llm.complete([{"role": "user", "content": "x"}])

    assert mock_c.call_args[1]["api_base"] == "http://localhost:11434"


def test_litellm_reasoning_api_base_passed():
    from src.impl.impl_litellm import LiteLLMReasoningLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion",
            return_value=_make_litellm_response("reasoning"),
        ) as mock_c:
            llm = LiteLLMReasoningLLM(model="claude-3-opus", api_base="http://custom:9000")
            llm.complete([{"role": "user", "content": "think"}])

    assert mock_c.call_args[1]["api_base"] == "http://custom:9000"


def test_litellm_image_gen_api_base_passed():
    from src.impl.impl_litellm import LiteLLMImageGenLLM

    raw = b"\x89PNG"
    b64 = base64.b64encode(raw).decode()
    mock_img_resp = MagicMock()
    mock_img_resp.data = [MagicMock()]
    mock_img_resp.data[0].b64_json = b64

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.image_generation", return_value=mock_img_resp
        ) as mock_c:
            llm = LiteLLMImageGenLLM(model="dall-e-3", api_base="http://img-server")
            llm.generate("a cat", max_retries=1)

    assert mock_c.call_args[1]["api_base"] == "http://img-server"


def test_litellm_image_inspector_api_base_passed():
    from src.impl.impl_litellm import LiteLLMImageInspectorLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion",
            return_value=_make_litellm_response("I see a cat"),
        ) as mock_c:
            llm = LiteLLMImageInspectorLLM(model="gpt-4o", api_base="http://vision-api")
            llm.inspect(b"imgdata", "analyst", "describe")

    assert mock_c.call_args[1]["api_base"] == "http://vision-api"


def test_litellm_tools_api_base_passed():
    import json

    from src.impl.impl_litellm import LiteLLMToolsLLM

    mock_tc = MagicMock()
    mock_tc.id = "c1"
    mock_tc.function.name = "fn"
    mock_tc.function.arguments = json.dumps({"x": 1})

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = None
    mock_resp.choices[0].message.tool_calls = [mock_tc]

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=mock_resp
        ) as mock_c:
            llm = LiteLLMToolsLLM(model="gpt-4o", api_base="http://tools-api")
            llm.complete([{"role": "user", "content": "x"}], [])

    assert mock_c.call_args[1]["api_base"] == "http://tools-api"


def test_litellm_tools_invalid_json_args_become_empty_dict():
    """When tool_calls arguments contain invalid JSON, arguments fall back to {}."""
    from src.impl.impl_litellm import LiteLLMToolsLLM

    mock_tc = MagicMock()
    mock_tc.id = "c1"
    mock_tc.function.name = "broken"
    mock_tc.function.arguments = "NOT_JSON{{{"

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = None
    mock_resp.choices[0].message.tool_calls = [mock_tc]

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch("src.impl.impl_litellm.litellm.completion", return_value=mock_resp):
            llm = LiteLLMToolsLLM(model="gpt-4o")
            result = llm.complete([{"role": "user", "content": "x"}], [])

    assert result.tool_calls[0].arguments == {}


def _make_litellm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    return resp


# ===========================================================================
# impl_ollama.py — _ollama_generate response_schema, image fallback key
# ===========================================================================


def test_ollama_generate_response_schema_sets_format():
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    schema = {"type": "object", "properties": {"label": {"type": "string"}}}

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": "a label", "model": "llava"}

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=mock_resp
    ) as mock_post:
        llm = OllamaImageInspectorLLM(model="llava", ollama_url=_OLLAMA_URL)
        llm.inspect(b"imgdata", "sys", "describe", response_schema=schema)

    payload = mock_post.call_args[1]["json"]
    assert payload["format"] == schema


def test_ollama_image_gen_fallback_to_image_key():
    """If 'images' is absent, fall back to the 'image' key in the response."""
    import base64

    from src.impl.impl_ollama import OllamaImageGenLLM

    raw = b"\x89PNG\r\n"
    b64 = base64.b64encode(raw).decode()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    # Return 'image' (singular) instead of 'images' (plural list)
    mock_resp.json.return_value = {"image": b64, "model": "flux"}

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp):
        llm = OllamaImageGenLLM(model="flux", ollama_url=_OLLAMA_URL)
        result = llm.generate("a cat", max_retries=1)

    assert result.image == raw


# ===========================================================================
# server.py — uncovered routes and helpers
# ===========================================================================


def test_root_route(client):
    c, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "llm_gateway"
    assert "docs" in data


def test_ollama_tags_route(client):
    c, _ = client
    resp = c.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    # All configured models should appear
    model_names = [m["name"] for m in data["models"]]
    assert len(model_names) > 0


def test_ollama_tags_deduplicates_models(client):
    """Models shared across tasks appear only once."""
    c, _ = client
    resp = c.get("/api/tags")
    names = [m["name"] for m in resp.json()["models"]]
    # phi3 appears in general/text_gen/reasoning/tools — should be listed once
    assert names.count("phi3") == 1


def test_openai_models_route(client):
    c, _ = client
    resp = c.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    # Each entry has required fields
    entry = data["data"][0]
    assert "id" in entry
    assert entry["object"] == "model"


def test_general_endpoint(client):
    c, factory = client
    factory.general.return_value.complete.return_value = _text("hello world")
    resp = c.post("/general", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json()["content"] == "hello world"


def test_text_gen_endpoint(client):
    c, factory = client
    factory.text_gen.return_value.complete.return_value = _text("generated text")
    resp = c.post("/text_gen", json={"messages": [{"role": "user", "content": "gen"}]})
    assert resp.status_code == 200
    assert resp.json()["content"] == "generated text"


def test_reasoning_endpoint(client):
    c, factory = client
    factory.reasoning.return_value.complete.return_value = _text("reasoned answer")
    resp = c.post(
        "/reasoning",
        json={"messages": [{"role": "user", "content": "think"}], "thinking_budget": 1024},
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "reasoned answer"


def test_image_gen_endpoint(client):
    c, factory = client
    factory.image_gen.return_value.generate.return_value = _image()
    resp = c.post("/image_gen", json={"prompt": "a red square"})
    assert resp.status_code == 200
    assert base64.b64decode(resp.json()["image_b64"]) == b"\x89PNG"


def test_image_gen_endpoint_with_reference_images(client):
    c, factory = client
    factory.image_gen.return_value.generate.return_value = _image()
    ref_b64 = base64.b64encode(b"ref_data").decode()
    resp = c.post(
        "/image_gen",
        json={"prompt": "a cat", "reference_images_b64": [ref_b64], "width": 512, "height": 512},
    )
    assert resp.status_code == 200
    factory.image_gen.return_value.generate.assert_called_once()
    call_kwargs = factory.image_gen.return_value.generate.call_args[1]
    assert call_kwargs["reference_images"] == [b"ref_data"]
    assert call_kwargs["width"] == 512


def test_image_inspector_endpoint(client):
    c, factory = client
    factory.image_inspector.return_value.inspect.return_value = _text("I see a cat")
    img_b64 = base64.b64encode(b"\x89PNG").decode()
    resp = c.post(
        "/image_inspector",
        json={"image_b64": img_b64, "system": "You are an analyst.", "prompt": "describe"},
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "I see a cat"


def test_models_endpoint_with_ollama_online(client):
    c, _ = client
    online_data = {"models": [{"name": "phi3"}, {"name": "flux"}]}
    with patch(
        "src.server._requests.get",
        return_value=MagicMock(ok=True, json=MagicMock(return_value=online_data)),
    ):
        resp = c.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "phi3" in data["ollama_available"] or "flux" in data["ollama_available"]


def test_models_endpoint_availability_flag_set(client):
    """Models returned by Ollama are flagged available=True."""
    c, _ = client
    with patch(
        "src.server._requests.get",
        return_value=MagicMock(ok=True, json=MagicMock(return_value={"models": [{"name": "phi3"}]})),
    ):
        resp = c.get("/models")
    data = resp.json()
    phi3_entry = next(
        (e for e in data["configured"] if e["model"] == "phi3"), None
    )
    assert phi3_entry is not None
    assert phi3_entry.get("available") is True


# ---------------------------------------------------------------------------
# server.py — _load_settings helper
# ---------------------------------------------------------------------------


def test_load_settings_returns_empty_when_no_files(tmp_path):
    from src.server import _load_settings

    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = _load_settings()
    finally:
        os.chdir(orig)
    assert result == {}


def test_load_settings_reads_base_json(tmp_path):
    import json

    from src.server import _load_settings

    settings = {"key": "value", "nested": {"a": 1}}
    (tmp_path / "settings.json").write_text(json.dumps(settings))

    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = _load_settings()
    finally:
        os.chdir(orig)
    assert result == settings


def test_load_settings_merges_override(tmp_path):
    import json

    from src.server import _load_settings

    base = {"key": "base", "extra": "preserved"}
    override = {"key": "overridden"}

    (tmp_path / "settings.json").write_text(json.dumps(base))
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "settings.json").write_text(json.dumps(override))

    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = _load_settings()
    finally:
        os.chdir(orig)
    assert result["key"] == "overridden"
    assert result["extra"] == "preserved"


# ---------------------------------------------------------------------------
# server.py — _minimal_png helper
# ---------------------------------------------------------------------------


def test_minimal_png_is_valid_png():
    import struct
    import zlib

    from src.server import _minimal_png

    data = _minimal_png()
    # PNG magic bytes
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR chunk — 1×1 pixel
    width, height = struct.unpack(">II", data[16:24])
    assert width == 1
    assert height == 1


# ---------------------------------------------------------------------------
# server.py — diffusion server availability in /models
# ---------------------------------------------------------------------------


def test_models_diffusion_server_available(tmp_path):
    """When a diffusion server is configured and reachable, models are listed."""
    from fastapi.testclient import TestClient

    from src.server import app

    config_with_ipadapter = textwrap.dedent("""\
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
        ipadapter:
          implementation: diffusion_server
          model: ip-adapter_sd15
          api_base: http://localhost:7860
    """)

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(config_with_ipadapter)

    mock_factory = MagicMock()
    env = {
        "LLM_GATEWAY_ROUTE": str(cfg),
        "LLM_GATEWAY_ROUTE_LOCAL": str(tmp_path / "no_override.yml"),
        "LLM_GATEWAY_HOST": "127.0.0.1",
        "LLM_GATEWAY_PORT": "4096",
    }

    diffusion_models_data = {"models": [{"name": "ip-adapter_sd15"}]}

    def mock_get(url, **kwargs):
        if "7860/models" in url:
            return MagicMock(ok=True, json=MagicMock(return_value=diffusion_models_data))
        # Ollama offline
        raise Exception("offline")

    with patch.dict(os.environ, env):
        with patch("src.server.LLMFactory", return_value=mock_factory):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app) as c:
                    with patch("src.server._requests.get", side_effect=mock_get):
                        resp = c.get("/models")

    assert resp.status_code == 200
    data = resp.json()
    assert "ip-adapter_sd15" in data["diffusion_available"]


def test_models_diffusion_server_availability_flag(tmp_path):
    """Diffusion models that match a configured model are flagged available."""
    from fastapi.testclient import TestClient

    from src.server import app

    config_with_ipadapter = textwrap.dedent("""\
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
        ipadapter:
          implementation: diffusion_server
          model: ip-adapter_sd15
          api_base: http://localhost:7860
    """)

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(config_with_ipadapter)

    mock_factory = MagicMock()
    env = {
        "LLM_GATEWAY_ROUTE": str(cfg),
        "LLM_GATEWAY_ROUTE_LOCAL": str(tmp_path / "no_override.yml"),
        "LLM_GATEWAY_HOST": "127.0.0.1",
        "LLM_GATEWAY_PORT": "4096",
    }

    diffusion_models_data = {"models": [{"name": "ip-adapter_sd15"}]}

    def mock_get(url, **kwargs):
        if "7860/models" in url:
            return MagicMock(ok=True, json=MagicMock(return_value=diffusion_models_data))
        raise Exception("offline")

    with patch.dict(os.environ, env):
        with patch("src.server.LLMFactory", return_value=mock_factory):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app) as c:
                    with patch("src.server._requests.get", side_effect=mock_get):
                        resp = c.get("/models")

    data = resp.json()
    ipadapter_entry = next(
        (e for e in data["configured"] if e["implementation"] == "diffusion_server"),
        None,
    )
    assert ipadapter_entry is not None
    assert ipadapter_entry.get("available") is True


# ===========================================================================
# impl_ollama.py — _ollama_generate with width/height/seed, reference_images
# ===========================================================================


def test_ollama_generate_sets_width_height_seed_in_options():
    """_ollama_generate passes width, height, and seed into options payload."""
    from src.impl.impl_ollama import _ollama_generate

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": "desc", "model": "llava"}

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp) as mock_post:
        _ollama_generate(
            _OLLAMA_URL,
            "llava",
            "describe",
            timeout=30,
            width=128,
            height=64,
            seed=77,
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["options"]["width"] == 128
    assert payload["options"]["height"] == 64
    assert payload["options"]["seed"] == 77


def test_ollama_image_gen_sends_reference_images():
    """OllamaImageGenLLM encodes reference_images as base64 in the payload."""
    import base64

    from src.impl.impl_ollama import OllamaImageGenLLM

    ref_bytes = b"reference image data"
    expected_b64 = base64.b64encode(ref_bytes).decode("ascii")
    img_b64 = base64.b64encode(b"result").decode()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"images": [img_b64], "model": "flux"}

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp) as mock_post:
        llm = OllamaImageGenLLM(model="flux", ollama_url=_OLLAMA_URL)
        llm.generate("a cat", max_retries=1, reference_images=[ref_bytes])

    payload = mock_post.call_args[1]["json"]
    assert payload["images"] == [expected_b64]


# ===========================================================================
# types.py — IPAdapterLLM / IPAdapterFaceIDLLM abstract base constructors
# ===========================================================================


def test_ipadapter_llm_construction():
    """IPAdapterLLM base __init__ is exercised via a concrete subclass."""
    from src.responses import ImageResponse
    from src.types import IPAdapterLLM

    class _ConcreteIPAdapter(IPAdapterLLM):
        def generate(self, prompt, reference_image, **kwargs) -> ImageResponse:  # type: ignore[override]
            return ImageResponse(image=b"x", model=self.model, duration_ms=1.0, attempts=1)

    llm = _ConcreteIPAdapter(
        model="ip-adapter",
        timeout=120,
        temperature=0.3,
        max_tokens=512,
        response_schema={"type": "object"},
    )
    assert llm.model == "ip-adapter"
    assert llm.timeout == 120
    assert llm.temperature == 0.3
    assert llm.max_tokens == 512
    assert llm.response_schema == {"type": "object"}


def test_ipadapter_faceid_llm_construction():
    """IPAdapterFaceIDLLM base __init__ is exercised via a concrete subclass."""
    from src.responses import ImageResponse
    from src.types import IPAdapterFaceIDLLM

    class _ConcreteIPAdapterFaceID(IPAdapterFaceIDLLM):
        def generate(self, prompt, face_image, **kwargs) -> ImageResponse:  # type: ignore[override]
            return ImageResponse(image=b"y", model=self.model, duration_ms=1.0, attempts=1)

    llm = _ConcreteIPAdapterFaceID(
        model="ip-adapter-faceid",
        timeout=200,
        temperature=0.5,
        max_tokens=256,
    )
    assert llm.model == "ip-adapter-faceid"
    assert llm.timeout == 200


# ===========================================================================
# impl_cli.py — remaining branches
# ===========================================================================


def test_run_claude_stream_json_string_content_merged_with_system():
    """When first user message content is a string, system is prepended as text."""
    from src.impl.impl_cli import _run_claude_stream_json

    result_line = '{"type": "result", "result": "merged"}'
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = result_line
    proc.stderr = ""

    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "describe the image"},
    ]

    with patch("src.impl.impl_cli.subprocess.run", return_value=proc) as mock_run:
        result, _ = _run_claude_stream_json(messages)

    stdin_data = mock_run.call_args[1]["input"]
    # The merged content should contain both the system text and user text
    assert "Be precise." in stdin_data
    assert "describe the image" in stdin_data
    assert result == "merged"


def test_run_claude_stream_json_skips_empty_lines_and_invalid_json():
    """Empty lines and non-JSON lines in stdout are silently skipped."""
    from src.impl.impl_cli import _run_claude_stream_json

    # Output with empty lines and invalid JSON, then a valid result
    output = "\n\nnot-json-at-all\n{}\n" + '{"type": "result", "result": "final"}'
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = output
    proc.stderr = ""

    with patch("src.impl.impl_cli.subprocess.run", return_value=proc):
        result, _ = _run_claude_stream_json([{"role": "user", "content": "hi"}])

    assert result == "final"


# ===========================================================================
# server.py — _f() 503 when factory not initialized, _log_startup with faceid
# ===========================================================================


def test_factory_not_initialized_returns_503(tmp_path):
    """Calling an endpoint before the factory is set returns 503."""
    import src.server as server_mod
    from fastapi.testclient import TestClient

    from src.server import app

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(_MINIMAL_CONFIG)

    env = {
        "LLM_GATEWAY_ROUTE": str(cfg),
        "LLM_GATEWAY_ROUTE_LOCAL": str(tmp_path / "no_override.yml"),
        "LLM_GATEWAY_HOST": "127.0.0.1",
        "LLM_GATEWAY_PORT": "4096",
    }

    with patch.dict(os.environ, env):
        with patch("src.server.LLMFactory", return_value=MagicMock()):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app, raise_server_exceptions=False) as c:
                    # Temporarily null out the factory to simulate pre-init
                    original = server_mod._factory
                    server_mod._factory = None
                    try:
                        resp = c.post(
                            "/general", json={"messages": [{"role": "user", "content": "x"}]}
                        )
                    finally:
                        server_mod._factory = original

    assert resp.status_code == 503


def test_log_startup_with_ipadapter_faceid(tmp_path):
    """_log_startup runs without error when ipadapter_faceid is configured."""
    from src.config import LLMConfig, LLMTypeConfig
    from src.server import _log_startup

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="phi3"),
        text_gen=LLMTypeConfig(implementation="ollama", model="phi3"),
        reasoning=LLMTypeConfig(implementation="ollama", model="phi3"),
        image_gen=LLMTypeConfig(implementation="ollama", model="flux"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="llava"),
        tools=LLMTypeConfig(implementation="ollama", model="phi3"),
        ipadapter=LLMTypeConfig(
            implementation="diffusion_server",
            model="ip-adapter_sd15",
            api_base="http://localhost:7860",
        ),
        ipadapter_faceid=LLMTypeConfig(
            implementation="diffusion_server",
            model="ip-adapter-faceid",
            api_base="http://localhost:7860",
        ),
    )
    # Should not raise; output goes to logging
    _log_startup(cfg, {}, "llm_route.yml", "local.yml", False, "127.0.0.1", "4096")


def test_ollama_tags_returns_empty_when_config_none():
    """GET /api/tags returns empty models list when _config is None."""
    import src.server as server_mod
    from fastapi.testclient import TestClient

    from src.server import app

    original = server_mod._config
    server_mod._config = None
    try:
        # Use a raw test client that bypasses lifespan
        from starlette.testclient import TestClient as StarletteClient

        c = StarletteClient(app, raise_server_exceptions=False)
        # Calling without proper lifespan, so directly invoke the endpoint
        resp = c.get("/api/tags")
    finally:
        server_mod._config = original

    assert resp.status_code == 200
    assert resp.json() == {"models": []}


def test_openai_models_returns_empty_when_config_none():
    """GET /v1/models returns empty data when _config is None."""
    import src.server as server_mod
    from starlette.testclient import TestClient as StarletteClient

    from src.server import app

    original = server_mod._config
    server_mod._config = None
    try:
        c = StarletteClient(app, raise_server_exceptions=False)
        resp = c.get("/v1/models")
    finally:
        server_mod._config = original

    assert resp.status_code == 200
    data = resp.json()
    assert data == {"object": "list", "data": []}


# ===========================================================================
# server.py — remaining gaps: /models 503, diffusion faceid api_base, exception path
# ===========================================================================


def test_models_returns_503_when_config_none():
    """GET /models raises 503 when _config is None."""
    import src.server as server_mod
    from starlette.testclient import TestClient as StarletteClient

    from src.server import app

    original = server_mod._config
    server_mod._config = None
    try:
        c = StarletteClient(app, raise_server_exceptions=False)
        resp = c.get("/models")
    finally:
        server_mod._config = original

    assert resp.status_code == 503


def test_models_diffusion_ipadapter_faceid_api_base_queried(tmp_path):
    """When both ipadapter and ipadapter_faceid have api_base, both are queried."""
    from fastapi.testclient import TestClient

    from src.server import app

    config_both = textwrap.dedent("""\
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
        ipadapter:
          implementation: diffusion_server
          model: ip-adapter_sd15
          api_base: http://localhost:7860
        ipadapter_faceid:
          implementation: diffusion_server
          model: ip-adapter-faceid
          api_base: http://localhost:7861
    """)

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(config_both)

    mock_factory = MagicMock()
    env = {
        "LLM_GATEWAY_ROUTE": str(cfg),
        "LLM_GATEWAY_ROUTE_LOCAL": str(tmp_path / "no_override.yml"),
        "LLM_GATEWAY_HOST": "127.0.0.1",
        "LLM_GATEWAY_PORT": "4096",
    }

    def mock_get(url, **kwargs):
        if "7860/models" in url or "7861/models" in url:
            return MagicMock(ok=True, json=MagicMock(return_value={"models": [{"name": "ip-model"}]}))
        raise Exception("offline")

    with patch.dict(os.environ, env):
        with patch("src.server.LLMFactory", return_value=mock_factory):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app) as c:
                    with patch("src.server._requests.get", side_effect=mock_get):
                        resp = c.get("/models")

    assert resp.status_code == 200
    assert "ip-model" in resp.json()["diffusion_available"]


def test_models_diffusion_server_exception_is_silenced(tmp_path):
    """If the diffusion server query raises, the exception is silently swallowed."""
    from fastapi.testclient import TestClient

    from src.server import app

    config_with_ipadapter = textwrap.dedent("""\
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
        ipadapter:
          implementation: diffusion_server
          model: ip-adapter_sd15
          api_base: http://localhost:7860
    """)

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(config_with_ipadapter)

    mock_factory = MagicMock()
    env = {
        "LLM_GATEWAY_ROUTE": str(cfg),
        "LLM_GATEWAY_ROUTE_LOCAL": str(tmp_path / "no_override.yml"),
        "LLM_GATEWAY_HOST": "127.0.0.1",
        "LLM_GATEWAY_PORT": "4096",
    }

    with patch.dict(os.environ, env):
        with patch("src.server.LLMFactory", return_value=mock_factory):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app) as c:
                    # Both Ollama and diffusion server requests fail
                    with patch(
                        "src.server._requests.get",
                        side_effect=Exception("all offline"),
                    ):
                        resp = c.get("/models")

    assert resp.status_code == 200
    data = resp.json()
    assert data["diffusion_available"] == []
