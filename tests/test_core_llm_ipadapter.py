"""Tests for DiffusionServer IP-Adapter implementations."""

import base64
from unittest.mock import MagicMock, patch

import pytest

_API_BASE = "http://localhost:7860"


def _mock_image_response(img_b64: str, model: str = "ip-adapter") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"image": img_b64, "model": model}
    return resp


def _mock_empty_response(model: str = "ip-adapter") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"model": model}
    return resp


# ---------------------------------------------------------------------------
# DiffusionServerIPAdapterLLM
# ---------------------------------------------------------------------------


def test_ipadapter_happy_path():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    raw = b"\x89PNG\r\n"
    b64 = base64.b64encode(raw).decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ):
        llm = DiffusionServerIPAdapterLLM(model="ip-adapter_sd15", api_base=_API_BASE)
        result = llm.generate("a cat", b"ref_image", max_retries=1)

    assert result.image == raw
    assert result.attempts == 1


def test_ipadapter_posts_to_correct_endpoint():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="ip-adapter_sd15", api_base=_API_BASE)
        llm.generate("a cat", b"ref_image", max_retries=1)

    assert mock_post.call_args[0][0] == f"{_API_BASE}/ipadapter"


def test_ipadapter_sends_reference_image_b64():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    ref_bytes = b"raw ref image data"
    expected_b64 = base64.b64encode(ref_bytes).decode("ascii")
    b64_result = base64.b64encode(b"result").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64_result),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="ip-adapter_sd15", api_base=_API_BASE)
        llm.generate("a cat", ref_bytes, max_retries=1)

    payload = mock_post.call_args[1]["json"]
    assert payload["reference_image"] == expected_b64


def test_ipadapter_sends_all_generation_params():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="ip-adapter_sd15", api_base=_API_BASE)
        llm.generate(
            "a cat",
            b"ref",
            max_retries=1,
            ip_adapter_scale=0.8,
            width=512,
            height=256,
            seed=42,
            num_inference_steps=10,
            negative_prompt="blurry",
            cfg_scale=9.0,
            lora="my/lora",
            lora_weight=0.7,
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["prompt"] == "a cat"
    assert payload["ip_adapter_scale"] == 0.8
    assert payload["width"] == 512
    assert payload["height"] == 256
    assert payload["seed"] == 42
    assert payload["steps"] == 10
    assert payload["negative_prompt"] == "blurry"
    assert payload["cfg_scale"] == 9.0
    assert payload["lora"] == "my/lora"
    assert payload["lora_weight"] == 0.7


def test_ipadapter_strips_model_prefix():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="diffusion/ip-adapter_sd15", api_base=_API_BASE)
        llm.generate("x", b"ref", max_retries=1)

    assert mock_post.call_args[1]["json"]["model"] == "ip-adapter_sd15"


def test_ipadapter_no_seed_omits_key():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="m", api_base=_API_BASE)
        llm.generate("x", b"ref", max_retries=1)

    assert "seed" not in mock_post.call_args[1]["json"]


def test_ipadapter_default_weight():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="m", api_base=_API_BASE)
        llm.generate("x", b"ref", max_retries=1)

    assert mock_post.call_args[1]["json"]["ip_adapter_scale"] == 0.5


def test_ipadapter_empty_response_retries_then_raises():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_empty_response(),
    ):
        llm = DiffusionServerIPAdapterLLM(model="m", api_base=_API_BASE)
        with pytest.raises(ValueError):
            llm.generate("x", b"ref", max_retries=2)


def test_ipadapter_validator_triggers_retry():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    good_b64 = base64.b64encode(b"good_image").decode()
    bad_b64 = base64.b64encode(b"bad_image").decode()
    responses = [_mock_image_response(bad_b64), _mock_image_response(good_b64)]

    with patch("src.impl.impl_ipadapter.requests.post", side_effect=responses):
        llm = DiffusionServerIPAdapterLLM(model="m", api_base=_API_BASE)
        result = llm.generate(
            "x",
            b"ref",
            max_retries=2,
            validator=lambda img: img == b"good_image",
        )

    assert result.image == b"good_image"
    assert result.attempts == 2


def test_ipadapter_default_api_base():
    from src.impl.impl_ipadapter import (
        _DEFAULT_API_BASE,
        DiffusionServerIPAdapterLLM,
    )

    llm = DiffusionServerIPAdapterLLM(model="m")
    assert llm.api_base == _DEFAULT_API_BASE


def test_ipadapter_optional_params_omitted_when_none():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="m", api_base=_API_BASE)
        llm.generate("x", b"ref", max_retries=1)

    payload = mock_post.call_args[1]["json"]
    assert "negative_prompt" not in payload
    assert "lora" not in payload
    assert "lora_weight" not in payload


def test_ipadapter_cfg_scale_default_omitted():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="m", api_base=_API_BASE)
        llm.generate("x", b"ref", max_retries=1)

    assert "cfg_scale" not in mock_post.call_args[1]["json"]


def test_ipadapter_cfg_scale_forwarded_when_set():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterLLM(model="m", api_base=_API_BASE)
        llm.generate("x", b"ref", max_retries=1, cfg_scale=12.0)

    assert mock_post.call_args[1]["json"]["cfg_scale"] == 12.0


# ---------------------------------------------------------------------------
# DiffusionServerIPAdapterFaceIDLLM
# ---------------------------------------------------------------------------


def test_ipadapter_faceid_happy_path():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    raw = b"\x89PNG\r\n"
    b64 = base64.b64encode(raw).decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ):
        llm = DiffusionServerIPAdapterFaceIDLLM(model="ip-faceid", api_base=_API_BASE)
        result = llm.generate("a portrait", b"face_image", max_retries=1)

    assert result.image == raw
    assert result.attempts == 1


def test_ipadapter_faceid_posts_to_correct_endpoint():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterFaceIDLLM(model="ip-faceid", api_base=_API_BASE)
        llm.generate("a portrait", b"face_image", max_retries=1)

    assert mock_post.call_args[0][0] == f"{_API_BASE}/ipadapter_faceid"


def test_ipadapter_faceid_sends_face_image_b64():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    face_bytes = b"raw face image data"
    expected_b64 = base64.b64encode(face_bytes).decode("ascii")
    b64_result = base64.b64encode(b"result").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64_result),
    ) as mock_post:
        llm = DiffusionServerIPAdapterFaceIDLLM(model="ip-faceid", api_base=_API_BASE)
        llm.generate("a portrait", face_bytes, max_retries=1)

    payload = mock_post.call_args[1]["json"]
    assert payload["face_image"] == expected_b64


def test_ipadapter_faceid_sends_all_generation_params():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterFaceIDLLM(model="ip-faceid", api_base=_API_BASE)
        llm.generate(
            "a portrait",
            b"face",
            max_retries=1,
            ip_adapter_scale=0.75,
            width=512,
            height=512,
            seed=7,
            num_inference_steps=20,
            negative_prompt="ugly",
            cfg_scale=8.0,
            lora="my/faceid-lora",
            lora_weight=0.5,
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["prompt"] == "a portrait"
    assert payload["ip_adapter_scale"] == 0.75
    assert payload["width"] == 512
    assert payload["height"] == 512
    assert payload["seed"] == 7
    assert payload["steps"] == 20
    assert payload["negative_prompt"] == "ugly"
    assert payload["cfg_scale"] == 8.0
    assert payload["lora"] == "my/faceid-lora"
    assert payload["lora_weight"] == 0.5


def test_ipadapter_faceid_empty_response_retries_then_raises():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_empty_response(),
    ):
        llm = DiffusionServerIPAdapterFaceIDLLM(model="m", api_base=_API_BASE)
        with pytest.raises(ValueError):
            llm.generate("x", b"face", max_retries=2)


def test_ipadapter_faceid_default_api_base():
    from src.impl.impl_ipadapter import (
        _DEFAULT_API_BASE,
        DiffusionServerIPAdapterFaceIDLLM,
    )

    llm = DiffusionServerIPAdapterFaceIDLLM(model="m")
    assert llm.api_base == _DEFAULT_API_BASE


def test_ipadapter_faceid_optional_params_omitted_when_none():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterFaceIDLLM(model="m", api_base=_API_BASE)
        llm.generate("x", b"face", max_retries=1)

    payload = mock_post.call_args[1]["json"]
    assert "negative_prompt" not in payload
    assert "lora" not in payload
    assert "lora_weight" not in payload


def test_ipadapter_faceid_default_ip_adapter_scale():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterFaceIDLLM(model="m", api_base=_API_BASE)
        llm.generate("x", b"face", max_retries=1)

    assert mock_post.call_args[1]["json"]["ip_adapter_scale"] == 0.5


def test_ipadapter_faceid_lora_forwarded():
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM

    b64 = base64.b64encode(b"img").decode()

    with patch(
        "src.impl.impl_ipadapter.requests.post",
        return_value=_mock_image_response(b64),
    ) as mock_post:
        llm = DiffusionServerIPAdapterFaceIDLLM(model="m", api_base=_API_BASE)
        llm.generate(
            "x",
            b"face",
            max_retries=1,
            lora="civitai/my-lora",
            lora_weight=0.6,
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["lora"] == "civitai/my-lora"
    assert payload["lora_weight"] == 0.6


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_ipadapter_returns_image_gen_llm(tmp_path):
    import textwrap

    from src.config import load_llm_config
    from src.factory import LLMFactory
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterLLM
    from src.types import IPAdapterLLM

    yaml = textwrap.dedent("""\
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
    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(yaml)
    cfg = load_llm_config(cfg_file)
    factory = LLMFactory(cfg)
    obj = factory.ipadapter()
    assert isinstance(obj, IPAdapterLLM)
    assert isinstance(obj, DiffusionServerIPAdapterLLM)
    assert obj.api_base == "http://localhost:7860"


def test_factory_ipadapter_faceid_returns_image_gen_llm(tmp_path):
    import textwrap

    from src.config import load_llm_config
    from src.factory import LLMFactory
    from src.impl.impl_ipadapter import DiffusionServerIPAdapterFaceIDLLM
    from src.types import IPAdapterFaceIDLLM

    yaml = textwrap.dedent("""\
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
        ipadapter_faceid:
          implementation: diffusion_server
          model: ip-faceid-plus_sd15
          api_base: http://localhost:7860
    """)
    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(yaml)
    cfg = load_llm_config(cfg_file)
    factory = LLMFactory(cfg)
    obj = factory.ipadapter_faceid()
    assert isinstance(obj, IPAdapterFaceIDLLM)
    assert isinstance(obj, DiffusionServerIPAdapterFaceIDLLM)


def test_factory_ipadapter_not_configured_raises(tmp_path):
    import textwrap

    from src.config import load_llm_config
    from src.factory import LLMFactory

    yaml = textwrap.dedent("""\
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
    """)
    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(yaml)
    cfg = load_llm_config(cfg_file)
    factory = LLMFactory(cfg)
    with pytest.raises(ValueError, match="ipadapter is not configured"):
        factory.ipadapter()


def test_factory_ipadapter_faceid_not_configured_raises(tmp_path):
    import textwrap

    from src.config import load_llm_config
    from src.factory import LLMFactory

    yaml = textwrap.dedent("""\
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
    """)
    cfg_file = tmp_path / "llm.yml"
    cfg_file.write_text(yaml)
    cfg = load_llm_config(cfg_file)
    factory = LLMFactory(cfg)
    with pytest.raises(ValueError, match="ipadapter_faceid is not configured"):
        factory.ipadapter_faceid()
