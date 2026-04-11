"""Tests for Ollama implementations of all LLM interface types."""

import base64
from unittest.mock import MagicMock, patch

import pytest

_OLLAMA_URL = "http://localhost:11434"


def _mock_chat_response(
    content: str, model: str = "test-model", tool_calls: list | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    data: dict = {"message": {"content": content, "role": "assistant"}, "model": model}
    if tool_calls is not None:
        data["message"]["tool_calls"] = tool_calls
    resp.json.return_value = data
    return resp


def _mock_generate_response(
    response: str = "", model: str = "test-model", images: list | None = None
) -> MagicMock:
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

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("hi")
    ) as mock_post:
        llm = OllamaGeneralLLM(model="ollama/phi3", ollama_url=_OLLAMA_URL)
        result = llm.complete([{"role": "user", "content": "hello"}])

    assert result.content == "hi"
    assert result.attempts == 1
    assert mock_post.call_args[1]["json"]["model"] == "phi3"


def test_ollama_general_strips_prefix():
    from src.impl.impl_ollama import OllamaGeneralLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("ok")
    ) as mock_post:
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

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_image_gen_response(b64)
    ) as mock_post:
        llm = OllamaImageGenLLM(model="flux", ollama_url=_OLLAMA_URL)
        llm.generate("x", max_retries=1, width=256, height=512, seed=42, num_inference_steps=2)

    payload = mock_post.call_args[1]["json"]
    assert payload["width"] == 256
    assert payload["height"] == 512
    assert payload["seed"] == 42
    assert payload["steps"] == 2


# ---------------------------------------------------------------------------
# OllamaImageInspectorLLM
# ---------------------------------------------------------------------------


def test_ollama_image_inspector_happy_path():
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_generate_response("I see a cat")
    ):
        llm = OllamaImageInspectorLLM(model="llava", ollama_url=_OLLAMA_URL)
        result = llm.inspect(b"imgdata", "You are an analyst.", "Describe the image.")

    assert result.content == "I see a cat"


def test_ollama_image_inspector_passes_image_b64():
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    raw = b"raw image bytes"
    expected_b64 = base64.b64encode(raw).decode("ascii")

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_generate_response("desc")
    ) as mock_post:
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


# ---------------------------------------------------------------------------
# options parameter forwarding
# ---------------------------------------------------------------------------


def test_ollama_general_options_merged_into_payload():
    from src.impl.impl_ollama import OllamaGeneralLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("ok")
    ) as mock_post:
        llm = OllamaGeneralLLM(model="phi3", ollama_url=_OLLAMA_URL)
        llm.complete([{"role": "user", "content": "x"}], options={"num_ctx": 4096})

    assert mock_post.call_args[1]["json"]["options"]["num_ctx"] == 4096


def test_ollama_text_gen_options_merged_into_payload():
    from src.impl.impl_ollama import OllamaTextGenLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("ok")
    ) as mock_post:
        llm = OllamaTextGenLLM(model="phi3", ollama_url=_OLLAMA_URL)
        llm.complete([{"role": "user", "content": "x"}], options={"num_ctx": 2048})

    assert mock_post.call_args[1]["json"]["options"]["num_ctx"] == 2048


def test_ollama_reasoning_options_merged_into_payload():
    from src.impl.impl_ollama import OllamaReasoningLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("ok")
    ) as mock_post:
        llm = OllamaReasoningLLM(model="phi3", ollama_url=_OLLAMA_URL)
        llm.complete([{"role": "user", "content": "x"}], options={"top_k": 10})

    assert mock_post.call_args[1]["json"]["options"]["top_k"] == 10


def test_ollama_image_gen_options_merged_into_payload():
    from src.impl.impl_ollama import OllamaImageGenLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_image_gen_response(b64)
    ) as mock_post:
        llm = OllamaImageGenLLM(model="flux", ollama_url=_OLLAMA_URL)
        llm.generate("x", max_retries=1, options={"num_ctx": 512})

    assert mock_post.call_args[1]["json"]["options"]["num_ctx"] == 512


def test_ollama_image_inspector_options_merged_into_payload():
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_generate_response("desc")
    ) as mock_post:
        llm = OllamaImageInspectorLLM(model="llava", ollama_url=_OLLAMA_URL)
        llm.inspect(b"img", "sys", "describe", options={"num_ctx": 1024})

    assert mock_post.call_args[1]["json"]["options"]["num_ctx"] == 1024


def test_ollama_tools_options_merged_into_payload():
    from src.impl.impl_ollama import OllamaToolsLLM

    tool_calls_data = [{"function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]
    mock_resp = _mock_chat_response("", tool_calls=tool_calls_data)

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp) as mock_post:
        llm = OllamaToolsLLM(model="phi3", ollama_url=_OLLAMA_URL)
        llm.complete([{"role": "user", "content": "x"}], _SAMPLE_TOOLS, options={"num_ctx": 256})

    assert mock_post.call_args[1]["json"]["options"]["num_ctx"] == 256


def test_ollama_options_coexist_with_temperature():
    """options dict is merged with temperature/max_tokens, not replacing them."""
    from src.impl.impl_ollama import OllamaGeneralLLM

    with patch(
        "src.impl.impl_ollama.requests.post", return_value=_mock_chat_response("ok")
    ) as mock_post:
        llm = OllamaGeneralLLM(model="phi3", ollama_url=_OLLAMA_URL, temperature=0.5)
        llm.complete([{"role": "user", "content": "x"}], options={"num_ctx": 4096})

    opts = mock_post.call_args[1]["json"]["options"]
    assert opts["temperature"] == 0.5
    assert opts["num_ctx"] == 4096


# ---------------------------------------------------------------------------
# OllamaImageGenLLM — integration tests (require live Ollama)
# ---------------------------------------------------------------------------


def test_ollama_image_gen_size_and_steps_integration(ollama_image_model, ollama_url):
    """Verify that width/height and steps (optimize) are actually honoured by Ollama."""
    import struct
    import time

    from src.impl.impl_ollama import OllamaImageGenLLM

    def png_dimensions(data: bytes) -> tuple[int, int]:
        """Read width/height from PNG IHDR chunk (bytes 16-24)."""
        assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a PNG"
        w, h = struct.unpack(">II", data[16:24])
        return w, h

    llm = OllamaImageGenLLM(model=ollama_image_model, ollama_url=ollama_url, timeout=120)

    # --- size ---
    resp = llm.generate("a red circle", width=512, height=256, num_inference_steps=4)
    w, h = png_dimensions(resp.image)
    assert (w, h) == (512, 256), f"Expected 512x256, got {w}x{h}"

    # --- steps affect speed ---
    timings = {}
    for steps, label in [(4, "quality"), (3, "normal"), (2, "fast")]:
        t0 = time.monotonic()
        llm.generate("a red circle", width=256, height=256, num_inference_steps=steps)
        timings[label] = time.monotonic() - t0

    assert timings["fast"] < timings["quality"], (
        f"fast ({timings['fast']:.2f}s) should be faster than quality ({timings['quality']:.2f}s)"
    )


# ---------------------------------------------------------------------------
# _ollama_generate options (width, height, seed) and response_schema
# ---------------------------------------------------------------------------


def test_ollama_generate_sets_width_height_seed_in_options():
    """_ollama_generate passes width, height, and seed into the options payload."""
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


def test_ollama_generate_response_schema_sets_format():
    """_ollama_generate adds the schema as the 'format' key when response_schema is given."""
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    schema = {"type": "object", "properties": {"label": {"type": "string"}}}

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": "a label", "model": "llava"}

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp) as mock_post:
        llm = OllamaImageInspectorLLM(model="llava", ollama_url=_OLLAMA_URL)
        llm.inspect(b"imgdata", "sys", "describe", response_schema=schema)

    payload = mock_post.call_args[1]["json"]
    assert payload["format"] == schema


# ---------------------------------------------------------------------------
# OllamaImageGenLLM — reference images and 'image' fallback key
# ---------------------------------------------------------------------------


def test_ollama_image_gen_sends_reference_images():
    """OllamaImageGenLLM base64-encodes reference_images into the payload."""
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


def test_ollama_image_gen_fallback_to_image_key():
    """If 'images' is absent, OllamaImageGenLLM falls back to the 'image' key."""
    import base64

    from src.impl.impl_ollama import OllamaImageGenLLM

    raw = b"\x89PNG\r\n"
    b64 = base64.b64encode(raw).decode()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    # Return 'image' (singular) instead of 'images' (list)
    mock_resp.json.return_value = {"image": b64, "model": "flux"}

    with patch("src.impl.impl_ollama.requests.post", return_value=mock_resp):
        llm = OllamaImageGenLLM(model="flux", ollama_url=_OLLAMA_URL)
        result = llm.generate("a cat", max_retries=1)

    assert result.image == raw
