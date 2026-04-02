"""Tests for TextResponse, ImageResponse, ToolCall, and ToolCallResponse."""
import pytest


def test_text_response_creation():
    from src import TextResponse

    r = TextResponse(content="hello", model="gpt-4", duration_ms=42.0, attempts=1)
    assert r.content == "hello"
    assert r.model == "gpt-4"
    assert r.duration_ms == 42.0
    assert r.attempts == 1
    assert r.last_error is None


def test_text_response_with_last_error():
    from src import TextResponse

    r = TextResponse(content="ok", model="m", duration_ms=1.0, attempts=2,
                     last_error="previous attempt failed")
    assert r.last_error == "previous attempt failed"
    assert r.attempts == 2


def test_text_response_immutable():
    from src import TextResponse

    r = TextResponse(content="x", model="m", duration_ms=1.0, attempts=1)
    with pytest.raises(Exception):
        r.content = "y"  # type: ignore[misc]


def test_image_response_creation():
    from src import ImageResponse

    img = b"\x89PNG"
    r = ImageResponse(image=img, model="dall-e", duration_ms=100.0, attempts=1)
    assert r.image == img
    assert r.model == "dall-e"
    assert r.last_error is None


def test_image_response_with_last_error():
    from src import ImageResponse

    r = ImageResponse(image=b"data", model="m", duration_ms=5.0, attempts=3,
                      last_error="bad image on attempt 1")
    assert r.last_error == "bad image on attempt 1"
    assert r.attempts == 3


def test_image_response_immutable():
    from src import ImageResponse

    r = ImageResponse(image=b"x", model="m", duration_ms=1.0, attempts=1)
    with pytest.raises(Exception):
        r.image = b"y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------

def test_tool_call_creation():
    from src import ToolCall

    tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    assert tc.id == "call_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}


def test_tool_call_immutable():
    from src import ToolCall

    tc = ToolCall(id="call_1", name="fn", arguments={})
    with pytest.raises(Exception):
        tc.name = "other"  # type: ignore[misc]


def test_tool_call_empty_arguments():
    from src import ToolCall

    tc = ToolCall(id="c", name="noop", arguments={})
    assert tc.arguments == {}


# ---------------------------------------------------------------------------
# ToolCallResponse
# ---------------------------------------------------------------------------

def test_tool_call_response_with_tool_calls():
    from src import ToolCall, ToolCallResponse

    tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Berlin"})
    r = ToolCallResponse(content=None, tool_calls=[tc], model="gpt-4o",
                         duration_ms=55.0, attempts=1)
    assert r.content is None
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "get_weather"
    assert r.last_error is None


def test_tool_call_response_with_content_and_tools():
    from src import ToolCall, ToolCallResponse

    tc = ToolCall(id="call_2", name="search", arguments={"query": "llm"})
    r = ToolCallResponse(content="Let me search for that.", tool_calls=[tc],
                         model="claude-sonnet-4-6", duration_ms=120.0, attempts=1)
    assert r.content == "Let me search for that."
    assert len(r.tool_calls) == 1


def test_tool_call_response_empty_tool_calls():
    from src import ToolCallResponse

    r = ToolCallResponse(content="No tools needed.", tool_calls=[],
                         model="gpt-4o", duration_ms=30.0, attempts=1)
    assert r.tool_calls == []
    assert r.content == "No tools needed."


def test_tool_call_response_immutable():
    from src import ToolCallResponse

    r = ToolCallResponse(content=None, tool_calls=[], model="m",
                         duration_ms=1.0, attempts=1)
    with pytest.raises(Exception):
        r.content = "x"  # type: ignore[misc]


def test_tool_call_response_with_last_error():
    from src import ToolCallResponse

    r = ToolCallResponse(content=None, tool_calls=[], model="m",
                         duration_ms=1.0, attempts=2, last_error="timeout on attempt 1")
    assert r.last_error == "timeout on attempt 1"
    assert r.attempts == 2
