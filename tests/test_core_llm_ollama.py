"""Tests for Ollama implementations of all LLM interface types."""
import base64
import json
from unittest.mock import MagicMock, patch

import pytest

_OLLAMA_URL = "http://localhost:11434"


def _mock_chat_response(content: str, model: str = "test-model",
                        tool_calls: list | None = None) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    data: dict = {"message": {"content": content, "role": "assistant"}, "model": model}
    if tool_calls is not None:
        data["message"]["tool_calls"] = tool_calls
    resp.json.return_value = data
    return resp


def _mock_generate_response(response: str = "", model: str = "test-model",
                             images: list | None = None) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    data: dict = {"response": response, "model": model}
    if images:
        data["images"] = images
    resp.json.return_value = data
    return resp


# ---------------------------------------------------------------------------
# OllamaGeneralLLM
# ---------------------------------------------------------------------------

def test_ollama_general_happy_path():
    from src.impl.impl_ollama import OllamaGeneralLLM

    with patch("src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("hi")) as mock_post:
        llm = OllamaGeneralLLM(model="ollama/phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete([{"role": "user", "content": "hello"}])

    assert result.content == "hi"
    assert result.attempts == 1
    assert mock_post.call_args[1]["json"]["model"] == "phi3"


def test_ollama_general_strips_prefix():
    from src.impl.impl_ollama import OllamaGeneralLLM

    with patch("src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("ok")) as mock_post:
        llm = OllamaGeneralLLM(model="ollama/llama3", ollama_url=_OLLAMA_URL)
        llm.complete([{"role": "user", "content": "q"}])

    assert mock_post.call_args[1]["json"]["model"] == "llama3"


# ---------------------------------------------------------------------------
# OllamaTextGenLLM
# ---------------------------------------------------------------------------

def test_ollama_text_gen_happy_path():
    from src.impl.impl_ollama import OllamaTextGenLLM

    with patch("src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("result")):
        llm = OllamaTextGenLLM(model="phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete([{"role": "user", "content": "x"}])

    assert result.content == "result"
    assert result.attempts == 1


def test_ollama_text_gen_empty_retries():
    from src.impl.impl_ollama import OllamaTextGenLLM

    responses = [_mock_chat_response(""), _mock_chat_response(""), _mock_chat_response("good")]
    with patch("src.impl.impl_ollama.requests.post", side_effect=responses):
        llm = OllamaTextGenLLM(model="phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete([{"role": "user", "content": "x"}], max_retries=3)

    assert result.content == "good"
    assert result.attempts == 3


def test_ollama_text_gen_exhaustion_raises():
    from src.impl.impl_ollama import OllamaTextGenLLM

    with patch("src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("")):
        llm = OllamaTextGenLLM(model="phi3", ollama_url=_OLLAMA_URL)
        with pytest.raises(ValueError):
            llm.complete([{"role": "user", "content": "x"}], max_retries=2)


# ---------------------------------------------------------------------------
# OllamaReasoningLLM
# ---------------------------------------------------------------------------

def test_ollama_reasoning_happy_path():
    from src.impl.impl_ollama import OllamaReasoningLLM

    with patch("src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("thought")):
        llm = OllamaReasoningLLM(model="phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete([{"role": "user", "content": "think"}])

    assert result.content == "thought"


def test_ollama_reasoning_thinking_budget_ignored():
    from src.impl.impl_ollama import OllamaReasoningLLM

    with patch("src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("ok")):
        llm = OllamaReasoningLLM(model="phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete([{"role": "user", "content": "x"}], thinking_budget=1024)

    assert result.content == "ok"


# ---------------------------------------------------------------------------
# OllamaImageGenLLM
# ---------------------------------------------------------------------------

def _image_gen_response(img_b64: str, model: str = "flux") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"images": [img_b64], "model": model}
    return resp


def test_ollama_image_gen_happy_path():
    from src.impl.impl_ollama import OllamaImageGenLLM

    raw = b"\x89PNG\r\n"
    b64 = base64.b64encode(raw).decode()

    with patch("src.impl.impl_ollama.requests.post", return_value=_image_gen_response(b64)):
        llm = OllamaImageGenLLM(model="flux", ollama_url=_OLLAMA_URL)
        result = llm.generate("a cat", max_retries=1)

    assert result.image == raw
    assert result.attempts == 1


def test_ollama_image_gen_no_image_retries_then_raises():
    from src.impl.impl_ollama import OllamaImageGenLLM

    empty_resp = MagicMock()
    empty_resp.raise_for_status = MagicMock()
    empty_resp.json.return_value = {"model": "flux"}

    with patch("src.impl.impl_ollama.requests.post", return_value=empty_resp):
        llm = OllamaImageGenLLM(model="flux", ollama_url=_OLLAMA_URL)
        with pytest.raises(ValueError):
            llm.generate("a cat", max_retries=2)


def test_ollama_image_gen_width_height_seed():
    from src.impl.impl_ollama import OllamaImageGenLLM

    b64 = base64.b64encode(b"img").decode()

    with patch("src.impl.impl_ollama.requests.post", return_value=_image_gen_response(b64)) as mock_post:
        llm = OllamaImageGenLLM(model="flux", ollama_url=_OLLAMA_URL)
        llm.generate("x", max_retries=1, width=256, height=256, seed=42)

    payload = mock_post.call_args[1]["json"]
    assert payload["options"]["width"] == 256
    assert payload["options"]["height"] == 256
    assert payload["options"]["seed"] == 42


# ---------------------------------------------------------------------------
# OllamaImageInspectorLLM
# ---------------------------------------------------------------------------

def test_ollama_image_inspector_happy_path():
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    with patch("src.impl.impl_ollama.requests.post",
               return_value=_mock_generate_response("I see a cat")):
        llm = OllamaImageInspectorLLM(model="llava", ollama_url=_OLLAMA_URL)
        result = llm.inspect(b"imgdata", "You are an analyst.", "Describe the image.")

    assert result.content == "I see a cat"


def test_ollama_image_inspector_passes_image_b64():
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    raw = b"raw image bytes"
    expected_b64 = base64.b64encode(raw).decode("ascii")

    with patch("src.impl.impl_ollama.requests.post",
               return_value=_mock_generate_response("desc")) as mock_post:
        llm = OllamaImageInspectorLLM(model="llava", ollama_url=_OLLAMA_URL)
        llm.inspect(raw, "sys", "describe")

    assert mock_post.call_args[1]["json"]["images"] == [expected_b64]


# ---------------------------------------------------------------------------
# OllamaToolsLLM
# ---------------------------------------------------------------------------

_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def test_ollama_tools_happy_path():
    from src.impl.impl_ollama import OllamaToolsLLM

    tool_calls_data = [{"function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}]
    mock_resp = _mock_chat_response("", tool_calls=tool_calls_data)

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp):
        llm = OllamaToolsLLM(model="phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete(
            [{"role": "user", "content": "Weather in Berlin?"}],
            _SAMPLE_TOOLS,
        )

    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Berlin"}
    assert result.attempts == 1


def test_ollama_tools_passes_tools_in_payload():
    from src.impl.impl_ollama import OllamaToolsLLM

    tool_calls_data = [{"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]
    mock_resp = _mock_chat_response("", tool_calls=tool_calls_data)

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp) as mock_post:
        llm = OllamaToolsLLM(model="phi3", ollama_url=_OLLAMA_URL)
        llm.complete([{"role": "user", "content": "x"}], _SAMPLE_TOOLS)

    assert mock_post.call_args[1]["json"]["tools"] == _SAMPLE_TOOLS


def test_ollama_tools_no_tool_calls_returns_empty_list():
    from src.impl.impl_ollama import OllamaToolsLLM

    mock_resp = _mock_chat_response("I can answer directly.", tool_calls=[])

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp):
        llm = OllamaToolsLLM(model="phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete([{"role": "user", "content": "x"}], _SAMPLE_TOOLS)

    assert result.tool_calls == []
    assert result.content == "I can answer directly."


def test_ollama_tools_generates_id_when_missing():
    from src.impl.impl_ollama import OllamaToolsLLM

    # Ollama sometimes omits the id field
    tool_calls_data = [{"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}]
    mock_resp = _mock_chat_response("", tool_calls=tool_calls_data)

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp):
        llm = OllamaToolsLLM(model="phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete([{"role": "user", "content": "x"}], _SAMPLE_TOOLS)

    assert result.tool_calls[0].id != ""
