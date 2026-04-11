"""Diffusion-server REST implementations for IP-Adapter and IP-Adapter FaceID.

Both classes implement ImageGenLLM and POST to a lightweight HTTP server that
wraps a local diffusion pipeline.  The server is expected to expose two endpoints:

  POST {api_base}/ipadapter
    Request  : {"model", "prompt", "reference_image" (b64), "width", "height",
                "steps", "ip_adapter_scale", "seed", "negative_prompt",
                "cfg_scale", "lora", "lora_weight"}
    Response : {"image" (b64), "model"}

  POST {api_base}/ipadapter_faceid
    Request  : {"model", "prompt", "face_image" (b64), "width", "height",
                "steps", "ip_adapter_scale", "seed", "negative_prompt",
                "cfg_scale", "lora", "lora_weight"}
    Response : {"image" (b64), "model"}

Pass the reference/face image via ``reference_images=[img_bytes]``.
Pass the adapter conditioning strength via the ``ip_adapter_scale`` parameter (default 0.5).
"""

from __future__ import annotations

import base64
from typing import Callable

import requests

from .._retry import retry_image_generation
from ..responses import ImageResponse
from ..types import ImageGenLLM

_DEFAULT_API_BASE = "http://localhost:7860"


def _bare_model(model: str) -> str:
    """Strip common provider prefixes for the diffusion server."""
    for prefix in ("ollama/", "diffusion/"):
        model = model.removeprefix(prefix)
    return model


class DiffusionServerIPAdapterLLM(ImageGenLLM):
    """IP-Adapter image generation via a local diffusion REST server."""

    def __init__(
        self,
        model: str,
        timeout: int = 300,
        api_base: str = _DEFAULT_API_BASE,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )
        self.api_base = api_base.rstrip("/")

    def generate(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        validator: Callable[[bytes], bool] | None = None,
        reference_images: list[bytes] | None = None,
        ip_adapter_scale: float | None = None,
        width: int = 256,
        height: int = 256,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        negative_prompt: str | None = None,
        cfg_scale: float | None = None,
        lora: str | None = None,
        lora_weight: float = 1.0,
        options: dict | None = None,
    ) -> ImageResponse:
        reference_image = (reference_images or [b""])[0]
        ref_b64 = base64.b64encode(reference_image).decode("ascii")
        effective_scale = ip_adapter_scale if ip_adapter_scale is not None else 0.5

        def call_fn() -> tuple[bytes, str]:
            payload: dict = {
                "model": _bare_model(self.model),
                "prompt": prompt,
                "reference_image": ref_b64,
                "width": width,
                "height": height,
                "ip_adapter_scale": effective_scale,
            }
            if seed is not None:
                payload["seed"] = seed
            if num_inference_steps is not None:
                payload["steps"] = num_inference_steps
            if negative_prompt is not None:
                payload["negative_prompt"] = negative_prompt
            if cfg_scale is not None:
                payload["cfg_scale"] = cfg_scale
            if lora is not None:
                payload["lora"] = lora
                payload["lora_weight"] = lora_weight

            resp = requests.post(f"{self.api_base}/ipadapter", json=payload, timeout=self.timeout)
            if not resp.ok:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} {resp.reason}: {detail}", response=resp
                )
            data = resp.json()
            img_b64 = data.get("image")
            if not img_b64:
                return b"", data.get("model", self.model)
            return base64.b64decode(img_b64), data.get("model", self.model)

        return retry_image_generation(call_fn, max_retries, self.model, validator=validator)


class DiffusionServerIPAdapterFaceIDLLM(ImageGenLLM):
    """IP-Adapter FaceID image generation via a local diffusion REST server."""

    def __init__(
        self,
        model: str,
        timeout: int = 300,
        api_base: str = _DEFAULT_API_BASE,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict | None = None,
    ) -> None:
        super().__init__(
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )
        self.api_base = api_base.rstrip("/")

    def generate(
        self,
        prompt: str,
        *,
        max_retries: int = 3,
        validator: Callable[[bytes], bool] | None = None,
        reference_images: list[bytes] | None = None,
        ip_adapter_scale: float | None = None,
        width: int = 256,
        height: int = 256,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        negative_prompt: str | None = None,
        cfg_scale: float | None = None,
        lora: str | None = None,
        lora_weight: float = 1.0,
        options: dict | None = None,
    ) -> ImageResponse:
        face_image = (reference_images or [b""])[0]
        face_b64 = base64.b64encode(face_image).decode("ascii")
        effective_scale = ip_adapter_scale if ip_adapter_scale is not None else 0.5

        def call_fn() -> tuple[bytes, str]:
            payload: dict = {
                "model": _bare_model(self.model),
                "prompt": prompt,
                "face_image": face_b64,
                "width": width,
                "height": height,
                "ip_adapter_scale": effective_scale,
            }
            if seed is not None:
                payload["seed"] = seed
            if num_inference_steps is not None:
                payload["steps"] = num_inference_steps
            if negative_prompt is not None:
                payload["negative_prompt"] = negative_prompt
            if cfg_scale is not None:
                payload["cfg_scale"] = cfg_scale
            if lora is not None:
                payload["lora"] = lora
                payload["lora_weight"] = lora_weight

            resp = requests.post(
                f"{self.api_base}/ipadapter_faceid", json=payload, timeout=self.timeout
            )
            if not resp.ok:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} {resp.reason}: {detail}", response=resp
                )
            data = resp.json()
            img_b64 = data.get("image")
            if not img_b64:
                return b"", data.get("model", self.model)
            return base64.b64decode(img_b64), data.get("model", self.model)

        return retry_image_generation(call_fn, max_retries, self.model, validator=validator)
