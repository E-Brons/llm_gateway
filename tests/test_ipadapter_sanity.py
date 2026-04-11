"""Sanity / integration tests for IP-Adapter diffusion server implementations.

These tests require a running diffusion server that implements:
  POST {api_base}/ipadapter
  POST {api_base}/ipadapter_faceid

Configure the server URL with env var DIFFUSION_SERVER_URL (default http://localhost:7860).
Tests are skipped automatically if the server is not reachable.

Run with:
    pytest tests/test_ipadapter_sanity.py -m integration -v
"""

from __future__ import annotations

import os
import struct
import zlib

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_png(r: int = 200, g: int = 100, b: int = 50) -> bytes:
    """Return a minimal 1×1 PNG with the given RGB colour."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(bytes([0, r, g, b])))
    iend = chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def _is_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


_DIFFUSION_MODEL = os.environ.get("IPADAPTER_MODEL", "ip-adapter_sd15_light_v11")
_FACEID_MODEL = os.environ.get("IPADAPTER_FACEID_MODEL", "ip-adapter-faceid_sd15")
_FACE_IMAGE_PATH = os.environ.get("IPADAPTER_FACE_IMAGE")


def _face_image_bytes() -> bytes:
    """Return a real face image for FaceID tests, or skip if none provided."""
    if not _FACE_IMAGE_PATH:
        pytest.skip("Set IPADAPTER_FACE_IMAGE=/path/to/face.jpg to run FaceID inference tests")
    with open(_FACE_IMAGE_PATH, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# ipadapter
# ---------------------------------------------------------------------------


def test_ipadapter_returns_image_gen_llm(diffusion_server_url):
    """factory.ipadapter() returns an ImageGenLLM — not a bespoke type."""
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory
    from src.types import IPAdapterLLM

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="x"),
        text_gen=LLMTypeConfig(implementation="ollama", model="x"),
        reasoning=LLMTypeConfig(implementation="ollama", model="x"),
        image_gen=LLMTypeConfig(implementation="ollama", model="x"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="x"),
        tools=LLMTypeConfig(implementation="ollama", model="x"),
        ipadapter=LLMTypeConfig(
            implementation="diffusion_server",
            model=_DIFFUSION_MODEL,
            api_base=diffusion_server_url,
        ),
    )
    factory = LLMFactory(cfg)
    llm = factory.ipadapter()
    assert isinstance(llm, IPAdapterLLM)


def test_ipadapter_generate_returns_image(diffusion_server_url):
    """Real call to /ipadapter returns PNG bytes."""
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    llm = DiffusionServerIPAdapterLLM(
        model=_DIFFUSION_MODEL,
        api_base=diffusion_server_url,
        timeout=300,
    )
    result = llm.generate(
        "a cat sitting on a wooden bench",
        _minimal_png(),
        width=256,
        height=256,
        max_retries=1,
    )
    assert result.image, "Expected image bytes, got empty"
    assert _is_png(result.image), "Response is not a valid PNG"
    assert result.attempts == 1


def test_ipadapter_weight_param_accepted(diffusion_server_url):
    """Explicit weight param is forwarded without error and image is returned."""
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    llm = DiffusionServerIPAdapterLLM(
        model=_DIFFUSION_MODEL,
        api_base=diffusion_server_url,
        timeout=300,
    )
    result = llm.generate(
        "a landscape",
        _minimal_png(),
        ip_adapter_scale=0.8,
        width=256,
        height=256,
        max_retries=1,
    )
    assert result.image, "Expected image bytes with explicit ip_adapter_scale=0.8"


def test_ipadapter_validator_accepted(diffusion_server_url):
    """A passing validator results in attempts == 1."""
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    llm = DiffusionServerIPAdapterLLM(
        model=_DIFFUSION_MODEL,
        api_base=diffusion_server_url,
        timeout=300,
    )
    result = llm.generate(
        "a flower",
        _minimal_png(),
        width=256,
        height=256,
        max_retries=3,
        validator=lambda img: len(img) > 0,
    )
    assert result.image
    assert result.attempts == 1


def test_ipadapter_seed_produces_deterministic_output(diffusion_server_url):
    """Two calls with the same seed should return identical images."""
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    llm = DiffusionServerIPAdapterLLM(
        model=_DIFFUSION_MODEL,
        api_base=diffusion_server_url,
        timeout=300,
    )
    kwargs = dict(
        width=256,
        height=256,
        seed=1234,
        max_retries=1,
    )
    r1 = llm.generate("a house", _minimal_png(), **kwargs)
    r2 = llm.generate("a house", _minimal_png(), **kwargs)
    assert r1.image == r2.image, "Same seed should produce the same image"


# ---------------------------------------------------------------------------
# ipadapter_faceid
# ---------------------------------------------------------------------------


def test_ipadapter_faceid_returns_image_gen_llm(diffusion_server_url):
    """factory.ipadapter_faceid() returns an ImageGenLLM — not a bespoke type."""
    from src.config import LLMConfig, LLMTypeConfig
    from src.factory import LLMFactory
    from src.types import IPAdapterFaceIDLLM

    cfg = LLMConfig(
        general=LLMTypeConfig(implementation="ollama", model="x"),
        text_gen=LLMTypeConfig(implementation="ollama", model="x"),
        reasoning=LLMTypeConfig(implementation="ollama", model="x"),
        image_gen=LLMTypeConfig(implementation="ollama", model="x"),
        image_inspector=LLMTypeConfig(implementation="ollama", model="x"),
        tools=LLMTypeConfig(implementation="ollama", model="x"),
        ipadapter_faceid=LLMTypeConfig(
            implementation="diffusion_server",
            model=_FACEID_MODEL,
            api_base=diffusion_server_url,
        ),
    )
    factory = LLMFactory(cfg)
    llm = factory.ipadapter_faceid()
    assert isinstance(llm, IPAdapterFaceIDLLM)


def test_ipadapter_faceid_generate_returns_image(diffusion_server_url):
    """Real call to /ipadapter_faceid returns PNG bytes."""
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    llm = DiffusionServerIPAdapterFaceIDLLM(
        model=_FACEID_MODEL,
        api_base=diffusion_server_url,
        timeout=300,
    )
    result = llm.generate(
        "a person in a forest, natural lighting",
        _face_image_bytes(),
        width=256,
        height=256,
        max_retries=1,
    )
    assert result.image, "Expected image bytes, got empty"
    assert _is_png(result.image), "Response is not a valid PNG"
    assert result.attempts == 1


def test_ipadapter_faceid_weight_param_accepted(diffusion_server_url):
    """Explicit weight param is forwarded without error."""
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    llm = DiffusionServerIPAdapterFaceIDLLM(
        model=_FACEID_MODEL,
        api_base=diffusion_server_url,
        timeout=300,
    )
    result = llm.generate(
        "a portrait, studio lighting",
        _face_image_bytes(),
        ip_adapter_scale=0.75,
        width=256,
        height=256,
        max_retries=1,
    )
    assert result.image, "Expected image bytes with explicit ip_adapter_scale=0.75"
