"""Tests for Claude CLI subprocess implementations."""
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


def _mock_proc(stdout: str, returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = ""
    return proc


def _json_result(result: str) -> str:
    return json.dumps({"result": result})


def _ndjson_result(result: str) -> str:
    return json.dumps({"type": "result", "result": result})


# ---------------------------------------------------------------------------
# CLIGeneralLLM
# ---------------------------------------------------------------------------

def test_cli_general_happy_path():
    from src.impl.impl_cli import CLIGeneralLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_json_result("hello"))):
        llm = CLIGeneralLLM()
        result = llm.complete([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    assert result.attempts == 1


def test_cli_general_system_prompt_passed():
    from src.impl.impl_cli import CLIGeneralLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_json_result("ok"))) as mock_run:
        llm = CLIGeneralLLM()
        llm.complete([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hello"},
        ])

    cmd = mock_run.call_args[0][0]
    assert "--system-prompt" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "Be helpful."


def test_cli_general_non_zero_exit_raises():
    from src.impl.impl_cli import CLIGeneralLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc("", returncode=1)):
        llm = CLIGeneralLLM()
        with pytest.raises(subprocess.CalledProcessError):
            llm.complete([{"role": "user", "content": "x"}])


# ---------------------------------------------------------------------------
# CLITextGenLLM
# ---------------------------------------------------------------------------

def test_cli_text_gen_happy_path():
    from src.impl.impl_cli import CLITextGenLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_ndjson_result("response"))):
        llm = CLITextGenLLM()
        result = llm.complete([{"role": "user", "content": "x"}])

    assert result.content == "response"


def test_cli_text_gen_stream_json_format():
    from src.impl.impl_cli import CLITextGenLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_ndjson_result("ok"))) as mock_run:
        llm = CLITextGenLLM()
        llm.complete([{"role": "user", "content": "x"}])

    cmd = mock_run.call_args[0][0]
    assert "--input-format" in cmd
    assert "stream-json" in cmd


def test_cli_text_gen_empty_retries_then_raises():
    from src.impl.impl_cli import CLITextGenLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_ndjson_result(""))):
        llm = CLITextGenLLM()
        with pytest.raises(ValueError):
            llm.complete([{"role": "user", "content": "x"}], max_retries=2)


# ---------------------------------------------------------------------------
# CLIReasoningLLM
# ---------------------------------------------------------------------------

def test_cli_reasoning_uses_effort_high():
    from src.impl.impl_cli import CLIReasoningLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_ndjson_result("deep"))) as mock_run:
        llm = CLIReasoningLLM()
        result = llm.complete([{"role": "user", "content": "think"}])

    cmd = mock_run.call_args[0][0]
    assert "--effort" in cmd
    assert "high" in cmd
    assert result.content == "deep"


def test_cli_reasoning_thinking_budget_accepted():
    from src.impl.impl_cli import CLIReasoningLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_ndjson_result("ok"))):
        llm = CLIReasoningLLM()
        result = llm.complete([{"role": "user", "content": "x"}], thinking_budget=1024)

    assert result.content == "ok"


# ---------------------------------------------------------------------------
# CLIImageInspectorLLM
# ---------------------------------------------------------------------------

def test_cli_image_inspector_happy_path():
    from src.impl.impl_cli import CLIImageInspectorLLM

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_ndjson_result("a dog"))):
        llm = CLIImageInspectorLLM()
        result = llm.inspect(b"imgdata", "You are a visual analyst.", "What do you see?")

    assert result.content == "a dog"


def test_cli_image_inspector_embeds_image_in_stream_json():
    import base64
    from src.impl.impl_cli import CLIImageInspectorLLM

    raw = b"fake png data"

    with patch("src.impl.impl_cli.subprocess.run", return_value=_mock_proc(_ndjson_result("ok"))) as mock_run:
        llm = CLIImageInspectorLLM()
        llm.inspect(raw, "sys", "describe")

    stdin = mock_run.call_args[1].get("input", "")
    assert base64.b64encode(raw).decode("ascii") in stdin
