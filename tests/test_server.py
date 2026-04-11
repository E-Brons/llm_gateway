"""Tests for the FastAPI gateway server."""

import base64
import os
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from src.responses import ImageResponse, TextResponse, ToolCall, ToolCallResponse

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


def _text(content: str = "hello") -> TextResponse:
    return TextResponse(content=content, model="phi3", duration_ms=100.0, attempts=1)


def _image() -> ImageResponse:
    return ImageResponse(image=b"\x89PNG", model="flux", duration_ms=200.0, attempts=1)


def _tool_resp() -> ToolCallResponse:
    tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    return ToolCallResponse(
        content=None, tool_calls=[tc], model="phi3", duration_ms=100.0, attempts=1
    )


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


# ── health ────────────────────────────────────────────────────────────────────


def test_health(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── models ────────────────────────────────────────────────────────────────────


def test_models_ollama_offline(client):
    c, _ = client
    with patch("src.server._requests.get", side_effect=Exception("offline")):
        resp = c.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured" in data
    assert data["ollama_available"] == []


def test_models_ollama_online(client):
    c, _ = client
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {"models": [{"name": "phi3:latest"}, {"name": "llava:latest"}]}
    with patch("src.server._requests.get", return_value=mock_resp):
        resp = c.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "phi3:latest" in data["ollama_available"]
    ollama_entries = [e for e in data["configured"] if e["implementation"] == "ollama"]
    for entry in ollama_entries:
        assert "available" in entry


# ── text endpoints ────────────────────────────────────────────────────────────


def test_general(client):
    c, factory = client
    factory.general.return_value.complete.return_value = _text("general response")
    resp = c.post("/general", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json()["content"] == "general response"


def test_text_gen(client):
    c, factory = client
    factory.text_gen.return_value.complete.return_value = _text("text gen response")
    resp = c.post("/text_gen", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json()["content"] == "text gen response"


def test_reasoning(client):
    c, factory = client
    factory.reasoning.return_value.complete.return_value = _text("reasoning response")
    resp = c.post("/reasoning", json={"messages": [{"role": "user", "content": "think"}]})
    assert resp.status_code == 200
    assert resp.json()["content"] == "reasoning response"


def test_reasoning_with_thinking_budget(client):
    c, factory = client
    factory.reasoning.return_value.complete.return_value = _text("deep thought")
    resp = c.post(
        "/reasoning",
        json={"messages": [{"role": "user", "content": "think"}], "thinking_budget": 2048},
    )
    assert resp.status_code == 200
    factory.reasoning.return_value.complete.assert_called_once_with(
        [{"role": "user", "content": "think"}],
        thinking_budget=2048,
        temperature=None,
        response_schema=None,
    )


# ── structured output ─────────────────────────────────────────────────────────


def test_general_forwards_response_schema(client):
    c, factory = client
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    factory.general.return_value.complete.return_value = _text('{"answer":"yes"}')
    resp = c.post(
        "/general",
        json={"messages": [{"role": "user", "content": "hi"}], "response_schema": schema},
    )
    assert resp.status_code == 200
    factory.general.return_value.complete.assert_called_once_with(
        [{"role": "user", "content": "hi"}], temperature=None, response_schema=schema
    )


def test_text_gen_forwards_response_schema(client):
    c, factory = client
    schema = {"type": "object", "properties": {"result": {"type": "number"}}}
    factory.text_gen.return_value.complete.return_value = _text('{"result":42}')
    resp = c.post(
        "/text_gen",
        json={"messages": [{"role": "user", "content": "compute"}], "response_schema": schema},
    )
    assert resp.status_code == 200
    factory.text_gen.return_value.complete.assert_called_once_with(
        [{"role": "user", "content": "compute"}],
        max_retries=3,
        temperature=None,
        response_schema=schema,
    )


def test_reasoning_forwards_response_schema(client):
    c, factory = client
    schema = {"type": "object", "properties": {"chain": {"type": "string"}}}
    factory.reasoning.return_value.complete.return_value = _text('{"chain":"step1"}')
    resp = c.post(
        "/reasoning",
        json={"messages": [{"role": "user", "content": "reason"}], "response_schema": schema},
    )
    assert resp.status_code == 200
    factory.reasoning.return_value.complete.assert_called_once_with(
        [{"role": "user", "content": "reason"}],
        thinking_budget=None,
        temperature=None,
        response_schema=schema,
    )


def test_image_inspector_forwards_response_schema(client):
    c, factory = client
    schema = {"type": "object", "properties": {"color": {"type": "string"}}}
    factory.image_inspector.return_value.inspect.return_value = _text('{"color":"red"}')
    img_b64 = base64.b64encode(b"fake image").decode()
    resp = c.post(
        "/image_inspector",
        json={
            "image_b64": img_b64,
            "system": "analyst",
            "prompt": "color?",
            "response_schema": schema,
        },
    )
    assert resp.status_code == 200
    factory.image_inspector.return_value.inspect.assert_called_once_with(
        b"fake image", "analyst", "color?", max_retries=3, temperature=None, response_schema=schema
    )


def test_response_schema_optional_defaults_to_none(client):
    """Omitting response_schema from the request body defaults to None."""
    c, factory = client
    factory.general.return_value.complete.return_value = _text("plain")
    resp = c.post("/general", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    factory.general.return_value.complete.assert_called_once_with(
        [{"role": "user", "content": "hi"}], temperature=None, response_schema=None
    )


# ── image endpoints ───────────────────────────────────────────────────────────


def test_image_gen(client):
    c, factory = client
    factory.image_gen.return_value.generate.return_value = _image()
    resp = c.post("/image_gen", json={"prompt": "a cat"})
    assert resp.status_code == 200
    assert base64.b64decode(resp.json()["image_b64"]) == b"\x89PNG"
    # reference_images should be None when not provided
    factory.image_gen.return_value.generate.assert_called_once_with(
        "a cat",
        reference_images=None,
        width=256,
        height=256,
        seed=None,
        num_inference_steps=3,
        max_retries=3,
    )


def test_image_gen_with_reference_images(client):
    c, factory = client
    factory.image_gen.return_value.generate.return_value = _image()
    ref_b64 = base64.b64encode(b"ref_png").decode()
    resp = c.post(
        "/image_gen",
        json={"prompt": "a cat", "reference_images_b64": [ref_b64], "width": 64, "height": 64},
    )
    assert resp.status_code == 200
    factory.image_gen.return_value.generate.assert_called_once_with(
        "a cat",
        reference_images=[b"ref_png"],
        width=64,
        height=64,
        seed=None,
        num_inference_steps=3,
        max_retries=3,
    )


def test_image_inspector(client):
    c, factory = client
    factory.image_inspector.return_value.inspect.return_value = _text("a dog")
    img_b64 = base64.b64encode(b"fake image").decode()
    resp = c.post(
        "/image_inspector",
        json={"image_b64": img_b64, "system": "You are an analyst.", "prompt": "What is this?"},
    )
    assert resp.status_code == 200
    assert resp.json()["content"] == "a dog"


# ── tools endpoint ────────────────────────────────────────────────────────────


def test_tools(client):
    c, factory = client
    factory.tools.return_value.complete.return_value = _tool_resp()
    resp = c.post(
        "/tools",
        json={
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["name"] == "get_weather"


# ── ipadapter endpoints ───────────────────────────────────────────────────────


def test_ipadapter(client):
    c, factory = client
    factory.ipadapter.return_value.generate.return_value = _image()
    ref_b64 = base64.b64encode(b"ref_png").decode()
    resp = c.post("/ipadapter", json={"prompt": "a cat", "reference_image_b64": ref_b64})
    assert resp.status_code == 200
    assert base64.b64decode(resp.json()["image_b64"]) == b"\x89PNG"
    factory.ipadapter.return_value.generate.assert_called_once_with(
        "a cat",
        b"ref_png",
        ip_adapter_scale=0.5,
        width=256,
        height=256,
        seed=None,
        num_inference_steps=3,
        max_retries=3,
    )


def test_ipadapter_with_params(client):
    c, factory = client
    factory.ipadapter.return_value.generate.return_value = _image()
    ref_b64 = base64.b64encode(b"ref_png").decode()
    resp = c.post(
        "/ipadapter",
        json={
            "prompt": "a cat",
            "reference_image_b64": ref_b64,
            "ip_adapter_scale": 0.8,
            "width": 512,
            "height": 512,
            "seed": 42,
            "optimize": "quality",
        },
    )
    assert resp.status_code == 200
    factory.ipadapter.return_value.generate.assert_called_once_with(
        "a cat",
        b"ref_png",
        ip_adapter_scale=0.8,
        width=512,
        height=512,
        seed=42,
        num_inference_steps=4,
        max_retries=3,
    )


def test_ipadapter_faceid(client):
    c, factory = client
    factory.ipadapter_faceid.return_value.generate.return_value = _image()
    face_b64 = base64.b64encode(b"face_png").decode()
    resp = c.post("/ipadapter_faceid", json={"prompt": "a portrait", "face_image_b64": face_b64})
    assert resp.status_code == 200
    assert base64.b64decode(resp.json()["image_b64"]) == b"\x89PNG"
    factory.ipadapter_faceid.return_value.generate.assert_called_once_with(
        "a portrait",
        b"face_png",
        ip_adapter_scale=0.5,
        width=256,
        height=256,
        seed=None,
        num_inference_steps=3,
        max_retries=3,
    )


def test_ipadapter_faceid_with_params(client):
    c, factory = client
    factory.ipadapter_faceid.return_value.generate.return_value = _image()
    face_b64 = base64.b64encode(b"face_png").decode()
    resp = c.post(
        "/ipadapter_faceid",
        json={
            "prompt": "a portrait",
            "face_image_b64": face_b64,
            "ip_adapter_scale": 0.9,
            "width": 64,
            "height": 64,
            "seed": 1,
            "optimize": "fast",
        },
    )
    assert resp.status_code == 200
    factory.ipadapter_faceid.return_value.generate.assert_called_once_with(
        "a portrait",
        b"face_png",
        ip_adapter_scale=0.9,
        width=64,
        height=64,
        seed=1,
        num_inference_steps=2,
        max_retries=3,
    )


# ── error paths ───────────────────────────────────────────────────────────────


def test_timeout_returns_504(client):
    import requests.exceptions
    from fastapi.testclient import TestClient

    from src.server import app

    c, factory = client
    factory.general.return_value.complete.side_effect = requests.exceptions.ReadTimeout(
        "read timed out"
    )
    # raise_server_exceptions=False lets the exception handler return the response
    with TestClient(app, raise_server_exceptions=False) as tc:
        tc.app_state = c.app_state  # reuse existing factory state
        resp = tc.post("/general", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 504
    assert "timeout" in resp.json()["detail"].lower()


def test_unhandled_error_returns_502(client):
    from fastapi.testclient import TestClient

    from src.server import app

    c, factory = client
    factory.general.return_value.complete.side_effect = RuntimeError("something broke")
    with TestClient(app, raise_server_exceptions=False) as tc:
        tc.app_state = c.app_state
        resp = tc.post("/general", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 502
    assert "something broke" in resp.json()["detail"]


def test_config_not_found(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import app

    with patch.dict(
        os.environ,
        {
            "LLM_GATEWAY_ROUTE": str(tmp_path / "nonexistent.yml"),
            "LLM_GATEWAY_ROUTE_LOCAL": "",
            "LLM_GATEWAY_HOST": "127.0.0.1",
            "LLM_GATEWAY_PORT": "4096",
        },
    ):
        with pytest.raises(Exception, match="Base config not found"):
            with TestClient(app):
                pass


def test_local_override_loaded(tmp_path):
    """local/llm_route.yml override is merged into base config at startup."""
    from fastapi.testclient import TestClient

    from src.server import app

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(_MINIMAL_CONFIG)

    override = tmp_path / "local_llm_route.yml"
    override.write_text("general:\n  model: ollama/llama3\n")

    mock_factory = MagicMock()
    with patch.dict(
        os.environ,
        {
            "LLM_GATEWAY_ROUTE": str(cfg),
            "LLM_GATEWAY_ROUTE_LOCAL": str(override),
            "LLM_GATEWAY_HOST": "127.0.0.1",
            "LLM_GATEWAY_PORT": "4096",
        },
    ):
        with patch("src.server.LLMFactory", return_value=mock_factory):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app) as c:
                    resp = c.get("/health")
                    assert resp.status_code == 200


# ── discovery routes ──────────────────────────────────────────────────────────


def test_root(client):
    c, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "llm_gateway"
    assert "docs" in data


def test_ollama_tags(client):
    c, _ = client
    resp = c.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


def test_ollama_tags_deduplicates_models(client):
    """Models shared across tasks appear only once in the /api/tags list."""
    c, _ = client
    resp = c.get("/api/tags")
    names = [m["name"] for m in resp.json()["models"]]
    # phi3 appears in general/text_gen/reasoning/tools — should be listed once
    assert names.count("phi3") == 1


def test_openai_models(client):
    c, _ = client
    resp = c.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    entry = data["data"][0]
    assert "id" in entry
    assert entry["object"] == "model"


def test_ollama_tags_returns_empty_when_config_none():
    """GET /api/tags returns an empty models list when _config is None."""
    from starlette.testclient import TestClient as StarletteClient

    import src.server as server_mod
    from src.server import app

    original = server_mod._config
    server_mod._config = None
    try:
        c = StarletteClient(app, raise_server_exceptions=False)
        resp = c.get("/api/tags")
    finally:
        server_mod._config = original

    assert resp.status_code == 200
    assert resp.json() == {"models": []}


def test_openai_models_returns_empty_when_config_none():
    """GET /v1/models returns empty data when _config is None."""
    from starlette.testclient import TestClient as StarletteClient

    import src.server as server_mod
    from src.server import app

    original = server_mod._config
    server_mod._config = None
    try:
        c = StarletteClient(app, raise_server_exceptions=False)
        resp = c.get("/v1/models")
    finally:
        server_mod._config = original

    assert resp.status_code == 200
    assert resp.json() == {"object": "list", "data": []}


def test_models_returns_503_when_config_none():
    """GET /models returns 503 when _config is None."""
    from starlette.testclient import TestClient as StarletteClient

    import src.server as server_mod
    from src.server import app

    original = server_mod._config
    server_mod._config = None
    try:
        c = StarletteClient(app, raise_server_exceptions=False)
        resp = c.get("/models")
    finally:
        server_mod._config = original

    assert resp.status_code == 503


# ── _f() guard ───────────────────────────────────────────────────────────────


def test_factory_not_initialized_returns_503(tmp_path):
    """Calling an endpoint before the factory is initialised returns 503."""
    from fastapi.testclient import TestClient

    import src.server as server_mod
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
                    original = server_mod._factory
                    server_mod._factory = None
                    try:
                        resp = c.post(
                            "/general",
                            json={"messages": [{"role": "user", "content": "x"}]},
                        )
                    finally:
                        server_mod._factory = original

    assert resp.status_code == 503


# ── _load_settings ────────────────────────────────────────────────────────────


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
    import json as _json

    from src.server import _load_settings

    settings = {"key": "value", "nested": {"a": 1}}
    (tmp_path / "settings.json").write_text(_json.dumps(settings))

    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = _load_settings()
    finally:
        os.chdir(orig)
    assert result == settings


def test_load_settings_merges_override(tmp_path):
    import json as _json

    from src.server import _load_settings

    base = {"key": "base", "extra": "preserved"}
    override = {"key": "overridden"}

    (tmp_path / "settings.json").write_text(_json.dumps(base))
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "settings.json").write_text(_json.dumps(override))

    orig = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = _load_settings()
    finally:
        os.chdir(orig)
    assert result["key"] == "overridden"
    assert result["extra"] == "preserved"


# ── _minimal_png ──────────────────────────────────────────────────────────────


def test_minimal_png_is_valid_png():
    import struct

    from src.server import _minimal_png

    data = _minimal_png()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert width == 1
    assert height == 1


# ── _log_startup with ipadapter_faceid ───────────────────────────────────────


def test_log_startup_with_ipadapter_faceid():
    """_log_startup runs without error when both ipadapter and ipadapter_faceid are configured."""
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
    _log_startup(cfg, {}, "llm_route.yml", "local.yml", False, "127.0.0.1", "4096")


# ── diffusion server availability in /models ──────────────────────────────────


_IPADAPTER_CONFIG = textwrap.dedent("""\
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

_IPADAPTER_BOTH_CONFIG = textwrap.dedent("""\
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


@pytest.fixture
def ipadapter_client(tmp_path):
    from fastapi.testclient import TestClient

    from src.server import app

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(_IPADAPTER_CONFIG)

    env = {
        "LLM_GATEWAY_ROUTE": str(cfg),
        "LLM_GATEWAY_ROUTE_LOCAL": str(tmp_path / "no_override.yml"),
        "LLM_GATEWAY_HOST": "127.0.0.1",
        "LLM_GATEWAY_PORT": "4096",
    }

    with patch.dict(os.environ, env):
        with patch("src.server.LLMFactory", return_value=MagicMock()):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app) as c:
                    yield c


def test_models_diffusion_server_listed(ipadapter_client):
    """When a diffusion server is reachable its models appear in diffusion_available."""

    def mock_get(url, **kwargs):
        if "7860/models" in url:
            return MagicMock(
                ok=True, json=MagicMock(return_value={"models": [{"name": "ip-adapter_sd15"}]})
            )
        raise Exception("offline")

    with patch("src.server._requests.get", side_effect=mock_get):
        resp = ipadapter_client.get("/models")

    assert resp.status_code == 200
    assert "ip-adapter_sd15" in resp.json()["diffusion_available"]


def test_models_diffusion_server_availability_flag(ipadapter_client):
    """A configured diffusion model is flagged available=True when the server reports it."""

    def mock_get(url, **kwargs):
        if "7860/models" in url:
            return MagicMock(
                ok=True, json=MagicMock(return_value={"models": [{"name": "ip-adapter_sd15"}]})
            )
        raise Exception("offline")

    with patch("src.server._requests.get", side_effect=mock_get):
        resp = ipadapter_client.get("/models")

    data = resp.json()
    entry = next(
        (e for e in data["configured"] if e["implementation"] == "diffusion_server"),
        None,
    )
    assert entry is not None
    assert entry.get("available") is True


def test_models_diffusion_server_exception_is_silenced(ipadapter_client):
    """If the diffusion server query raises, the exception is silently swallowed."""
    with patch("src.server._requests.get", side_effect=Exception("all offline")):
        resp = ipadapter_client.get("/models")

    assert resp.status_code == 200
    assert resp.json()["diffusion_available"] == []


def test_models_diffusion_both_adapters_queried(tmp_path):
    """When both ipadapter and ipadapter_faceid have different api_base URLs, both are queried."""
    from fastapi.testclient import TestClient

    from src.server import app

    cfg = tmp_path / "llm_route.yml"
    cfg.write_text(_IPADAPTER_BOTH_CONFIG)

    env = {
        "LLM_GATEWAY_ROUTE": str(cfg),
        "LLM_GATEWAY_ROUTE_LOCAL": str(tmp_path / "no_override.yml"),
        "LLM_GATEWAY_HOST": "127.0.0.1",
        "LLM_GATEWAY_PORT": "4096",
    }

    def mock_get(url, **kwargs):
        if "7860/models" in url or "7861/models" in url:
            return MagicMock(
                ok=True, json=MagicMock(return_value={"models": [{"name": "ip-model"}]})
            )
        raise Exception("offline")

    with patch.dict(os.environ, env):
        with patch("src.server.LLMFactory", return_value=MagicMock()):
            with patch("src.server._run_sanity_checks"):
                with TestClient(app) as c:
                    with patch("src.server._requests.get", side_effect=mock_get):
                        resp = c.get("/models")

    assert resp.status_code == 200
    assert "ip-model" in resp.json()["diffusion_available"]
