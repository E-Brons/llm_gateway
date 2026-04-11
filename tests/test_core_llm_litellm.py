"""Tests for LiteLLM implementations of all LLM interface types."""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_completion(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    return resp


def _mock_tool_completion(tool_calls: list[dict]) -> MagicMock:
    """Build a mock litellm response that contains tool calls."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = None

    mock_calls = []
    for tc in tool_calls:
        m = MagicMock()
        m.id = tc["id"]
        m.function.name = tc["name"]
        m.function.arguments = json.dumps(tc["arguments"])
        mock_calls.append(m)
    resp.choices[0].message.tool_calls = mock_calls
    return resp


def _mock_image_generation(b64: str) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock()]
    resp.data[0].b64_json = b64
    return resp


# ---------------------------------------------------------------------------
# LiteLLMGeneralLLM
# ---------------------------------------------------------------------------


def test_litellm_general_happy_path():
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("hello")
        ):
            llm = LiteLLMGeneralLLM(model="gpt-4o")
            result = llm.complete([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    assert result.attempts == 1


def test_litellm_general_api_base_passed():
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("ok")
        ) as mock_c:
            llm = LiteLLMGeneralLLM(model="ollama/phi3", api_base="http://localhost:11434")
            llm.complete([{"role": "user", "content": "x"}])

    assert mock_c.call_args[1]["api_base"] == "http://localhost:11434"


# ---------------------------------------------------------------------------
# LiteLLMTextGenLLM
# ---------------------------------------------------------------------------


def test_litellm_text_gen_happy_path():
    from src.impl.impl_litellm import LiteLLMTextGenLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("result")
        ):
            llm = LiteLLMTextGenLLM(model="gpt-4o")
            result = llm.complete([{"role": "user", "content": "x"}])

    assert result.content == "result"


def test_litellm_text_gen_transfer_encoding_triggers_reset():
    from src.impl.impl_litellm import LiteLLMTextGenLLM

    calls = [0]
    reset_calls = [0]

    def mock_reset():
        reset_calls[0] += 1

    def mock_completion(**kwargs):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("Transfer-Encoding header error")
        return _mock_completion("fixed")

    with patch("src.impl.impl_litellm.reset_litellm_client", mock_reset):
        with patch("src.impl.impl_litellm.litellm.completion", side_effect=mock_completion):
            llm = LiteLLMTextGenLLM(model="gpt-4o")
            result = llm.complete([{"role": "user", "content": "x"}], max_retries=3)

    assert result.content == "fixed"
    assert reset_calls[0] >= 1


# ---------------------------------------------------------------------------
# LiteLLMReasoningLLM
# ---------------------------------------------------------------------------


def test_litellm_reasoning_no_thinking_budget():
    from src.impl.impl_litellm import LiteLLMReasoningLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("deep")
        ) as mock_c:
            llm = LiteLLMReasoningLLM(model="claude-opus-4-6")
            result = llm.complete([{"role": "user", "content": "x"}])

    assert "thinking" not in mock_c.call_args[1]
    assert result.content == "deep"


def test_litellm_reasoning_with_thinking_budget():
    from src.impl.impl_litellm import LiteLLMReasoningLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("thought")
        ) as mock_c:
            llm = LiteLLMReasoningLLM(model="claude-opus-4-6")
            result = llm.complete([{"role": "user", "content": "x"}], thinking_budget=2048)

    assert mock_c.call_args[1]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert result.content == "thought"


# ---------------------------------------------------------------------------
# LiteLLMImageGenLLM
# ---------------------------------------------------------------------------


def test_litellm_image_gen_happy_path():
    from src.impl.impl_litellm import LiteLLMImageGenLLM

    raw = b"\x89PNG"
    b64 = base64.b64encode(raw).decode()

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.image_generation",
            return_value=_mock_image_generation(b64),
        ):
            llm = LiteLLMImageGenLLM(model="dall-e-3")
            result = llm.generate("a sunset", max_retries=1)

    assert result.image == raw
    assert result.attempts == 1


def test_litellm_image_gen_no_data_retries_raises():
    from src.impl.impl_litellm import LiteLLMImageGenLLM

    empty_resp = MagicMock()
    empty_resp.data = []

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch("src.impl.impl_litellm.litellm.image_generation", return_value=empty_resp):
            llm = LiteLLMImageGenLLM(model="dall-e-3")
            with pytest.raises(ValueError):
                llm.generate("x", max_retries=2)


# ---------------------------------------------------------------------------
# LiteLLMImageInspectorLLM
# ---------------------------------------------------------------------------


def test_litellm_image_inspector_multimodal_format():
    from src.impl.impl_litellm import LiteLLMImageInspectorLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("a face")
        ) as mock_c:
            llm = LiteLLMImageInspectorLLM(model="claude-sonnet-4-6")
            result = llm.inspect(b"imgdata", "You are an analyst.", "Describe.")

    assert result.content == "a face"
    messages = mock_c.call_args[1]["messages"]
    user_msg = next(m for m in messages if m["role"] == "user")
    content = user_msg["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert "base64" in content[0]["image_url"]["url"]


# ---------------------------------------------------------------------------
# LiteLLMToolsLLM
# ---------------------------------------------------------------------------

_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def test_litellm_tools_happy_path():
    from src.impl.impl_litellm import LiteLLMToolsLLM

    mock_resp = _mock_tool_completion(
        [
            {"id": "call_abc", "name": "get_weather", "arguments": {"city": "Paris"}},
        ]
    )

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch("src.impl.impl_litellm.litellm.completion", return_value=mock_resp):
            llm = LiteLLMToolsLLM(model="gpt-4o")
            result = llm.complete([{"role": "user", "content": "Weather in Paris?"}], _SAMPLE_TOOLS)

    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}
    assert result.content is None
    assert result.attempts == 1


def test_litellm_tools_passes_tools_to_litellm():
    from src.impl.impl_litellm import LiteLLMToolsLLM

    mock_resp = _mock_tool_completion(
        [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "NYC"}},
        ]
    )

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch("src.impl.impl_litellm.litellm.completion", return_value=mock_resp) as mock_c:
            llm = LiteLLMToolsLLM(model="gpt-4o")
            llm.complete([{"role": "user", "content": "x"}], _SAMPLE_TOOLS)

    assert mock_c.call_args[1]["tools"] == _SAMPLE_TOOLS


def test_litellm_tools_text_response_alongside_tool_calls():
    """Model may return both content and tool calls."""
    from src.impl.impl_litellm import LiteLLMToolsLLM

    mock_resp = _mock_tool_completion(
        [
            {"id": "c1", "name": "get_weather", "arguments": {"city": "Rome"}},
        ]
    )
    mock_resp.choices[0].message.content = "Sure, let me check."

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch("src.impl.impl_litellm.litellm.completion", return_value=mock_resp):
            llm = LiteLLMToolsLLM(model="gpt-4o")
            result = llm.complete([{"role": "user", "content": "x"}], _SAMPLE_TOOLS)

    assert result.content == "Sure, let me check."
    assert len(result.tool_calls) == 1


def test_litellm_tools_no_tool_calls_returns_empty_list():
    from src.impl.impl_litellm import LiteLLMToolsLLM

    mock_resp = _mock_completion("I don't need a tool for that.")

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch("src.impl.impl_litellm.litellm.completion", return_value=mock_resp):
            llm = LiteLLMToolsLLM(model="gpt-4o")
            result = llm.complete([{"role": "user", "content": "x"}], _SAMPLE_TOOLS)

    assert result.tool_calls == []
    assert result.content == "I don't need a tool for that."


# ---------------------------------------------------------------------------
# options parameter forwarding
# ---------------------------------------------------------------------------


def test_litellm_general_options_forwarded_as_extra_body():
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("ok")
        ) as mock_c:
            llm = LiteLLMGeneralLLM(model="gpt-4o")
            llm.complete([{"role": "user", "content": "x"}], options={"num_ctx": 4096})

    assert mock_c.call_args[1]["extra_body"] == {"options": {"num_ctx": 4096}}


def test_litellm_text_gen_options_forwarded_as_extra_body():
    from src.impl.impl_litellm import LiteLLMTextGenLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("ok")
        ) as mock_c:
            llm = LiteLLMTextGenLLM(model="gpt-4o")
            llm.complete([{"role": "user", "content": "x"}], options={"num_ctx": 2048})

    assert mock_c.call_args[1]["extra_body"] == {"options": {"num_ctx": 2048}}


def test_litellm_reasoning_options_forwarded_as_extra_body():
    from src.impl.impl_litellm import LiteLLMReasoningLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("ok")
        ) as mock_c:
            llm = LiteLLMReasoningLLM(model="claude-opus-4-6")
            llm.complete([{"role": "user", "content": "x"}], options={"top_k": 10})

    assert mock_c.call_args[1]["extra_body"] == {"options": {"top_k": 10}}


def test_litellm_image_inspector_options_forwarded_as_extra_body():
    from src.impl.impl_litellm import LiteLLMImageInspectorLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("desc")
        ) as mock_c:
            llm = LiteLLMImageInspectorLLM(model="gpt-4o")
            llm.inspect(b"img", "sys", "describe", options={"temperature": 0.1})

    assert mock_c.call_args[1]["extra_body"] == {"options": {"temperature": 0.1}}


def test_litellm_tools_options_forwarded_as_extra_body():
    from src.impl.impl_litellm import LiteLLMToolsLLM

    mock_resp = _mock_completion("ok")

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch("src.impl.impl_litellm.litellm.completion", return_value=mock_resp) as mock_c:
            llm = LiteLLMToolsLLM(model="gpt-4o")
            llm.complete(
                [{"role": "user", "content": "x"}], _SAMPLE_TOOLS, options={"num_ctx": 512}
            )

    assert mock_c.call_args[1]["extra_body"] == {"options": {"num_ctx": 512}}


def test_litellm_no_extra_body_when_options_none():
    from src.impl.impl_litellm import LiteLLMGeneralLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion", return_value=_mock_completion("ok")
        ) as mock_c:
            llm = LiteLLMGeneralLLM(model="gpt-4o")
            llm.complete([{"role": "user", "content": "x"}])

    assert "extra_body" not in mock_c.call_args[1]


# ---------------------------------------------------------------------------
# api_base forwarding
# ---------------------------------------------------------------------------


def test_litellm_text_gen_api_base_passed():
    from src.impl.impl_litellm import LiteLLMTextGenLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion",
            return_value=_mock_completion("ok"),
        ) as mock_c:
            llm = LiteLLMTextGenLLM(model="ollama/phi3", api_base="http://localhost:11434")
            llm.complete([{"role": "user", "content": "x"}])

    assert mock_c.call_args[1]["api_base"] == "http://localhost:11434"


def test_litellm_reasoning_api_base_passed():
    from src.impl.impl_litellm import LiteLLMReasoningLLM

    with patch("src.impl.impl_litellm.reset_litellm_client"):
        with patch(
            "src.impl.impl_litellm.litellm.completion",
            return_value=_mock_completion("reasoning"),
        ) as mock_c:
            llm = LiteLLMReasoningLLM(model="claude-3-opus", api_base="http://custom:9000")
            llm.complete([{"role": "user", "content": "think"}])

    assert mock_c.call_args[1]["api_base"] == "http://custom:9000"


def test_litellm_image_gen_api_base_passed():
    import base64

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
            return_value=_mock_completion("I see a cat"),
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


# ---------------------------------------------------------------------------
# Invalid JSON in tool-call arguments
# ---------------------------------------------------------------------------


def test_litellm_tools_invalid_json_args_become_empty_dict():
    """When tool_calls arguments contain invalid JSON, they fall back to {}."""
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
