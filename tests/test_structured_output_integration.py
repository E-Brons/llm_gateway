"""Acceptance tests for structured output — require live backends.

Each test makes a real LLM call and asserts:
  1. The response is valid JSON (parseable).
  2. All required schema fields are present with the correct types.

Run with:
    pytest tests/test_structured_output_integration.py -m integration -v

Backends:
  - CLI   : needs `claude` CLI installed and authenticated.
  - Ollama: needs Ollama server running with at least one text model and one vision model.
"""

from __future__ import annotations

import json
import struct
import zlib

import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}

_COLOR_SCHEMA = {
    "type": "object",
    "properties": {
        "dominant_color": {"type": "string"},
        "is_single_color": {"type": "boolean"},
    },
    "required": ["dominant_color", "is_single_color"],
    "additionalProperties": False,
}

_REASONING_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "explanation": {"type": "string"},
    },
    "required": ["answer", "explanation"],
    "additionalProperties": False,
}


def _assert_json_matches_schema(content: str, schema: dict) -> dict:
    """Parse *content* as JSON and check all required fields exist with the right types."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        pytest.fail(f"Response is not valid JSON: {exc}\nContent: {content!r}")

    type_map = {"string": str, "integer": int, "boolean": bool, "number": (int, float)}
    props = schema.get("properties", {})
    required = schema.get("required", [])

    for field in required:
        assert field in data, f"Required field {field!r} missing from response: {data}"
        expected_type = props[field]["type"]
        python_type = type_map[expected_type]
        assert isinstance(data[field], python_type), (
            f"Field {field!r} expected {expected_type}, got {type(data[field]).__name__}: {data[field]!r}"
        )

    return data


def _minimal_red_png() -> bytes:
    """Return a 1×1 red PNG for vision tests."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    iend = chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


_PERSON_PROMPT = (
    "Extract the person's details from this sentence: 'Alice is 30 years old and lives in Paris.'"
)

# ---------------------------------------------------------------------------
# CLI backend
# ---------------------------------------------------------------------------


def test_cli_text_gen_structured_output(cli_available):
    """CLITextGenLLM returns valid JSON matching the schema when response_schema is given."""
    from src.impl.impl_cli import CLITextGenLLM

    llm = CLITextGenLLM(model="claude", timeout=60)
    result = llm.complete(
        [{"role": "user", "content": _PERSON_PROMPT}],
        response_schema=_PERSON_SCHEMA,
    )
    data = _assert_json_matches_schema(result.content, _PERSON_SCHEMA)
    assert data["name"] == "Alice"
    assert data["age"] == 30
    assert "paris" in data["city"].lower()


def test_cli_reasoning_structured_output(cli_available):
    """CLIReasoningLLM returns valid JSON matching the schema when response_schema is given."""
    from src.impl.impl_cli import CLIReasoningLLM

    llm = CLIReasoningLLM(model="claude", timeout=120)
    result = llm.complete(
        [{"role": "user", "content": "What is 17 multiplied by 6? Show your working."}],
        response_schema=_REASONING_SCHEMA,
    )
    data = _assert_json_matches_schema(result.content, _REASONING_SCHEMA)
    assert data["answer"] == 102


def test_cli_image_inspector_structured_output(cli_available):
    """CLIImageInspectorLLM returns valid JSON matching the schema when response_schema is given."""
    from src.impl.impl_cli import CLIImageInspectorLLM

    llm = CLIImageInspectorLLM(model="claude", timeout=60)
    result = llm.inspect(
        _minimal_red_png(),
        system="You are a precise image analyst.",
        prompt="Analyse this image.",
        response_schema=_COLOR_SCHEMA,
    )
    data = _assert_json_matches_schema(result.content, _COLOR_SCHEMA)
    assert "red" in data["dominant_color"].lower()
    assert data["is_single_color"] is True


def test_cli_general_structured_output(cli_available):
    """CLIGeneralLLM returns valid JSON matching the schema when response_schema is given."""
    from src.impl.impl_cli import CLIGeneralLLM

    llm = CLIGeneralLLM(model="claude", timeout=60)
    result = llm.complete(
        [{"role": "user", "content": _PERSON_PROMPT}],
        response_schema=_PERSON_SCHEMA,
    )
    _assert_json_matches_schema(result.content, _PERSON_SCHEMA)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------


def test_ollama_text_gen_structured_output(ollama_text_model, ollama_url):
    """OllamaTextGenLLM returns valid JSON matching the schema when response_schema is given."""
    from src.impl.impl_ollama import OllamaTextGenLLM

    llm = OllamaTextGenLLM(model=ollama_text_model, ollama_url=ollama_url, timeout=120)
    result = llm.complete(
        [{"role": "user", "content": _PERSON_PROMPT}],
        response_schema=_PERSON_SCHEMA,
    )
    data = _assert_json_matches_schema(result.content, _PERSON_SCHEMA)
    assert "alice" in data["name"].lower()
    assert data["age"] == 30


def test_ollama_reasoning_structured_output(ollama_text_model, ollama_url):
    """OllamaReasoningLLM returns valid JSON matching the schema when response_schema is given."""
    from src.impl.impl_ollama import OllamaReasoningLLM

    llm = OllamaReasoningLLM(model=ollama_text_model, ollama_url=ollama_url, timeout=120)
    result = llm.complete(
        [{"role": "user", "content": "What is 12 multiplied by 9? Show your working."}],
        response_schema=_REASONING_SCHEMA,
    )
    data = _assert_json_matches_schema(result.content, _REASONING_SCHEMA)
    assert data["answer"] == 108


def test_ollama_image_inspector_structured_output(ollama_vision_model, ollama_url):
    """OllamaImageInspectorLLM returns valid JSON matching the schema when response_schema is given."""
    from src.impl.impl_ollama import OllamaImageInspectorLLM

    llm = OllamaImageInspectorLLM(model=ollama_vision_model, ollama_url=ollama_url, timeout=120)
    result = llm.inspect(
        _minimal_red_png(),
        system="You are a precise image analyst.",
        prompt="Analyse this image.",
        response_schema=_COLOR_SCHEMA,
    )
    data = _assert_json_matches_schema(result.content, _COLOR_SCHEMA)
    assert "red" in data["dominant_color"].lower()


def test_ollama_general_structured_output(ollama_text_model, ollama_url):
    """OllamaGeneralLLM returns valid JSON matching the schema when response_schema is given."""
    from src.impl.impl_ollama import OllamaGeneralLLM

    llm = OllamaGeneralLLM(model=ollama_text_model, ollama_url=ollama_url, timeout=120)
    result = llm.complete(
        [{"role": "user", "content": _PERSON_PROMPT}],
        response_schema=_PERSON_SCHEMA,
    )
    _assert_json_matches_schema(result.content, _PERSON_SCHEMA)


# ---------------------------------------------------------------------------
# Schema override: per-call schema takes precedence over config-level schema
# ---------------------------------------------------------------------------


def test_cli_per_call_schema_overrides_config_schema(cli_available):
    """A per-call schema replaces the constructor schema in the real CLI response."""
    from src.impl.impl_cli import CLITextGenLLM

    # Construction-time schema asks for a single "value" field
    config_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    # Per-call schema asks for person data — this should win
    llm = CLITextGenLLM(model="claude", timeout=60, response_schema=config_schema)
    result = llm.complete(
        [{"role": "user", "content": _PERSON_PROMPT}],
        response_schema=_PERSON_SCHEMA,
    )
    # Result must match _PERSON_SCHEMA, not config_schema
    data = _assert_json_matches_schema(result.content, _PERSON_SCHEMA)
    assert "alice" in data["name"].lower()


def test_ollama_per_call_schema_overrides_config_schema(ollama_text_model, ollama_url):
    """A per-call schema replaces the constructor schema in the real Ollama response."""
    from src.impl.impl_ollama import OllamaTextGenLLM

    config_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    llm = OllamaTextGenLLM(
        model=ollama_text_model,
        ollama_url=ollama_url,
        timeout=120,
        response_schema=config_schema,
    )
    result = llm.complete(
        [{"role": "user", "content": _PERSON_PROMPT}],
        response_schema=_PERSON_SCHEMA,
    )
    data = _assert_json_matches_schema(result.content, _PERSON_SCHEMA)
    assert "alice" in data["name"].lower()
