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
        [{"role": "user", "content": "think"}], thinking_budget=2048
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
        weight=0.5,
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
            "weight": 0.8,
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
        weight=0.8,
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
        weight=0.5,
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
            "weight": 0.9,
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
        weight=0.9,
        width=64,
        height=64,
        seed=1,
        num_inference_steps=2,
        max_retries=3,
    )


# ── error paths ───────────────────────────────────────────────────────────────


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
