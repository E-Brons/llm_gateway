"""Pipeline management for the diffusion server.

Pipelines are loaded lazily on first request and cached in memory.
Only one pipeline is kept loaded at a time — if a new model is requested
the old one is offloaded to free GPU/MPS memory.
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("diffusion_server")


# ---------------------------------------------------------------------------
# Public exception types
# ---------------------------------------------------------------------------


class PipelineLoadError(RuntimeError):
    """Raised when a model pipeline fails to load (download, weight, OOM, …)."""


class BadImageError(ValueError):
    """Raised when the provided image bytes cannot be decoded."""


# ---------------------------------------------------------------------------
# Model registry
#
# Maps the bare model name (as written in llm_route.yml) to the HuggingFace
# repo and weight file needed to load it.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict[str, Any]] = {
    "ip-adapter_sd15_light_v11": {
        "base": "runwayml/stable-diffusion-v1-5",
        "adapter_repo": "h94/IP-Adapter",
        "adapter_subfolder": "models",
        "adapter_weight": "ip-adapter_sd15_light_v11.bin",
        "mode": "style",
    },
    "ip-adapter_sd15": {
        "base": "runwayml/stable-diffusion-v1-5",
        "adapter_repo": "h94/IP-Adapter",
        "adapter_subfolder": "models",
        "adapter_weight": "ip-adapter_sd15.bin",
        "mode": "style",
    },
    "ip-adapter-faceid-plus_sd15": {
        "base": "runwayml/stable-diffusion-v1-5",
        "adapter_repo": "h94/IP-Adapter-FaceID",
        "adapter_subfolder": None,
        "adapter_weight": "ip-adapter-faceid-plus_sd15.bin",
        "image_encoder_folder": None,
        "mode": "faceid",
    },
    "ip-adapter-faceid_sd15": {
        "base": "runwayml/stable-diffusion-v1-5",
        "adapter_repo": "h94/IP-Adapter-FaceID",
        "adapter_subfolder": None,
        "adapter_weight": "ip-adapter-faceid_sd15.bin",
        "image_encoder_folder": None,
        "mode": "faceid",
    },
}

KNOWN_MODELS = list(_REGISTRY.keys())


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(device: str) -> torch.dtype:
    return torch.float16 if device in ("cuda", "mps") else torch.float32


# ---------------------------------------------------------------------------
# Pipeline cache — one pipeline kept loaded at a time
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}  # {"model": pipe}
_face_app: Any = None  # insightface FaceAnalysis singleton
_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialize inference — MPS is not thread-safe


def _load_pipeline(model: str) -> Any:
    from diffusers import StableDiffusionPipeline

    cfg = _REGISTRY[model]
    dev = _device()
    dt = _dtype(dev)

    try:
        logger.info("Loading base model %s onto %s …", cfg["base"], dev)
        pipe = StableDiffusionPipeline.from_pretrained(cfg["base"], torch_dtype=dt)
        pipe = pipe.to(dev)
    except Exception as exc:
        raise PipelineLoadError(f"Failed to load base model {cfg['base']!r}: {exc}") from exc

    try:
        logger.info(
            "Loading IP-Adapter weights %s/%s …",
            cfg["adapter_repo"],
            cfg["adapter_weight"],
        )
        load_kwargs: dict = {
            "subfolder": cfg["adapter_subfolder"] or "",
            "weight_name": cfg["adapter_weight"],
        }
        if "image_encoder_folder" in cfg:
            load_kwargs["image_encoder_folder"] = cfg["image_encoder_folder"]
        pipe.load_ip_adapter(cfg["adapter_repo"], **load_kwargs)
    except Exception as exc:
        raise PipelineLoadError(
            f"Failed to load adapter weights {cfg['adapter_weight']!r}: {exc}"
        ) from exc

    # diffusers' prepare_ip_adapter_image_embeds accesses self.image_encoder.dtype
    # even when using pre-computed embeds. Patch a dtype proxy when no encoder is loaded.
    if getattr(pipe, "image_encoder", None) is None:
        class _DtypeProxy:
            def __init__(self, dtype: torch.dtype) -> None:
                self.dtype = dtype
        pipe.image_encoder = _DtypeProxy(dt)

    # Disable safety checker — local inference server, no NSFW filtering needed.
    # Without this, SD v1.5 returns black images for any face-conditioned output.
    pipe.safety_checker = None

    logger.info("Pipeline ready: %s", model)
    return pipe


def _get_pipeline(model: str) -> Any:
    with _lock:
        if model not in _cache:
            # Offload any previously cached pipeline to free memory
            for m, p in list(_cache.items()):
                logger.info("Offloading pipeline %s", m)
                del p
            _cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _cache[model] = _load_pipeline(model)
        return _cache[model]


def _get_face_app() -> Any:
    global _face_app
    with _lock:
        if _face_app is None:
            from insightface.app import FaceAnalysis

            logger.info("Loading InsightFace buffalo_l …")
            _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
        return _face_app


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------


def generate_ipadapter(
    model: str,
    prompt: str,
    reference_image: bytes,
    *,
    weight: float = 0.5,
    width: int = 256,
    height: int = 256,
    seed: int | None = None,
    steps: int = 20,
) -> bytes:
    """Generate an image conditioned on a reference style image."""
    if model not in _REGISTRY:
        raise ValueError(f"Unknown model {model!r}. Known: {KNOWN_MODELS}")
    if _REGISTRY[model]["mode"] != "style":
        raise ValueError(f"Model {model!r} is a FaceID model — use /ipadapter_faceid")

    pipe = _get_pipeline(model)
    dev = _device()

    try:
        ref_img = Image.open(io.BytesIO(reference_image)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise BadImageError(f"Cannot decode reference image: {exc}") from exc

    pipe.set_ip_adapter_scale(weight)
    generator = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
    try:
        with _infer_lock:
            images = pipe(
                prompt=prompt,
                ip_adapter_image=ref_img,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
            ).images
    except Exception as exc:
        raise RuntimeError(f"Image generation failed: {exc}") from exc

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


def generate_ipadapter_faceid(
    model: str,
    prompt: str,
    face_image: bytes,
    *,
    weight: float = 0.5,
    width: int = 256,
    height: int = 256,
    seed: int | None = None,
    steps: int = 20,
) -> bytes:
    """Generate an image conditioned on a face image (identity-preserving)."""
    import cv2
    import numpy as np

    if model not in _REGISTRY:
        raise ValueError(f"Unknown model {model!r}. Known: {KNOWN_MODELS}")
    if _REGISTRY[model]["mode"] != "faceid":
        raise ValueError(f"Model {model!r} is a style model — use /ipadapter")

    pipe = _get_pipeline(model)
    dev = _device()

    face_app = _get_face_app()
    try:
        pil_img = Image.open(io.BytesIO(face_image)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise BadImageError(f"Cannot decode face image: {exc}") from exc

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = face_app.get(bgr)
    if not faces:
        raise ValueError("No face detected in the provided image")

    faceid_embeds = (
        torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(1).to(dev, dtype=_dtype(dev))
    )
    # For CFG: concatenate negative (zeros) and positive embeds along dim 0.
    # diffusers requires 3D tensors in ip_adapter_image_embeds and calls .chunk(2) on each.
    neg_embeds = torch.zeros_like(faceid_embeds)
    faceid_embeds_cfg = torch.cat([neg_embeds, faceid_embeds], dim=0)  # [2, 1, 512]

    pipe.set_ip_adapter_scale(weight)
    generator = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
    try:
        with _infer_lock:
            images = pipe(
                prompt=prompt,
                ip_adapter_image_embeds=[faceid_embeds_cfg],
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
            ).images
    except Exception as exc:
        raise RuntimeError(f"Image generation failed: {exc}") from exc

    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()
