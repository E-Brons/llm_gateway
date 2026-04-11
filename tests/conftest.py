"""Shared pytest fixtures for llm_gateway tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests

# Make the project root importable so tests can use `import src` or `from src import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_VISION_KEYWORDS = ("llava", "moondream", "vision", "minicpm-v", "bakllava", "vl")
_IMAGE_KEYWORDS = ("flux", "diffusion", "sdxl", "sd3")


def _get_ollama_models(url: str) -> list[str]:
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


@pytest.fixture(scope="session")
def ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434")


@pytest.fixture(scope="session")
def _ollama_model_list(ollama_url: str) -> list[str]:
    return _get_ollama_models(ollama_url)


@pytest.fixture(scope="session")
def ollama_text_model(ollama_url: str, _ollama_model_list: list[str]) -> str:
    if not _ollama_model_list:
        pytest.skip("Ollama server not available")
    override = os.environ.get("OLLAMA_TEXT_MODEL")
    if override:
        return override
    for name in _ollama_model_list:
        if not any(kw in name.lower() for kw in _VISION_KEYWORDS + _IMAGE_KEYWORDS):
            return name
    pytest.skip("No text model available on Ollama")


@pytest.fixture(scope="session")
def ollama_vision_model(ollama_url: str, _ollama_model_list: list[str]) -> str:
    if not _ollama_model_list:
        pytest.skip("Ollama server not available")
    override = os.environ.get("OLLAMA_VISION_MODEL")
    if override:
        return override
    for name in _ollama_model_list:
        if any(kw in name.lower() for kw in _VISION_KEYWORDS):
            return name
    pytest.skip("No vision model available on Ollama")


@pytest.fixture(scope="session")
def ollama_image_model(ollama_url: str, _ollama_model_list: list[str]) -> str:
    if not _ollama_model_list:
        pytest.skip("Ollama server not available")
    override = os.environ.get("OLLAMA_IMAGE_MODEL")
    if override:
        return override
    for name in _ollama_model_list:
        if any(kw in name.lower() for kw in _IMAGE_KEYWORDS):
            return name
    pytest.skip("No image generation model available on Ollama")


@pytest.fixture(scope="session")
def diffusion_server_url() -> str:
    """Base URL for IP-Adapter diffusion server tests.

    Skip the test if the server is not reachable.
    Override with env var DIFFUSION_SERVER_URL (default http://localhost:7860).
    """
    url = os.environ.get("DIFFUSION_SERVER_URL", "http://localhost:7860")
    try:
        r = requests.get(f"{url}/health", timeout=5)
        if r.status_code not in (200, 404):  # 404 = server running but no /health route
            pytest.skip(f"Diffusion server not healthy at {url}")
    except Exception:
        pytest.skip(f"Diffusion server not reachable at {url}")
    return url


@pytest.fixture(scope="session")
def cli_available() -> None:
    """Skip if the `claude` CLI is not installed or not authenticated."""
    try:
        proc = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            pytest.skip("claude CLI not available")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("claude CLI not found")
