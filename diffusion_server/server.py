"""Diffusion server — exposes IP-Adapter image generation over HTTP.

Endpoints:
  GET  /health
  GET  /models
  POST /ipadapter
  POST /ipadapter_faceid

Start with:
  ./scripts/run_diffusion.sh
"""

from __future__ import annotations

import base64
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import (
    KNOWN_MODELS,
    generate_ipadapter,
    generate_ipadapter_faceid,
    _REGISTRY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("diffusion_server")

app = FastAPI(title="Diffusion Server — IP-Adapter")


# ── Request models ──────────────────────────────────────────────────────────


class IPAdapterRequest(BaseModel):
    model: str
    prompt: str
    reference_image: str   # base64 PNG
    weight: float = 0.5
    width: int = 256
    height: int = 256
    seed: int | None = None
    steps: int = 20


class IPAdapterFaceIDRequest(BaseModel):
    model: str
    prompt: str
    face_image: str        # base64 PNG
    weight: float = 0.5
    width: int = 256
    height: int = 256
    seed: int | None = None
    steps: int = 20


# ── Routes ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    """List all models this server knows how to serve."""
    return {
        "models": [
            {"name": name, "mode": cfg["mode"], "base": cfg["base"]}
            for name, cfg in _REGISTRY.items()
        ]
    }


@app.post("/ipadapter")
def ipadapter(req: IPAdapterRequest):
    try:
        ref_bytes = base64.b64decode(req.reference_image)
        img_bytes = generate_ipadapter(
            req.model,
            req.prompt,
            ref_bytes,
            weight=req.weight,
            width=req.width,
            height=req.height,
            seed=req.seed,
            steps=req.steps,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("ipadapter generation failed")
        raise HTTPException(500, str(exc)) from exc

    return {
        "image": base64.b64encode(img_bytes).decode(),
        "model": req.model,
    }


@app.post("/ipadapter_faceid")
def ipadapter_faceid(req: IPAdapterFaceIDRequest):
    try:
        face_bytes = base64.b64decode(req.face_image)
        img_bytes = generate_ipadapter_faceid(
            req.model,
            req.prompt,
            face_bytes,
            weight=req.weight,
            width=req.width,
            height=req.height,
            seed=req.seed,
            steps=req.steps,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("ipadapter_faceid generation failed")
        raise HTTPException(500, str(exc)) from exc

    return {
        "image": base64.b64encode(img_bytes).decode(),
        "model": req.model,
    }
