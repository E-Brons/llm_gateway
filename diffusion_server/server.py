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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pipeline import (
    _REGISTRY,
    BadImageError,
    NoFaceDetectedError,
    PipelineLoadError,
    generate_ipadapter,
    generate_ipadapter_faceid,
)
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("diffusion_server")

app = FastAPI(title="Diffusion Server — IP-Adapter")


@app.middleware("http")
async def _exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.error("Error on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Request models ──────────────────────────────────────────────────────────


class IPAdapterRequest(BaseModel):
    model: str
    prompt: str
    reference_image: str  # base64 PNG
    weight: float = 0.5
    width: int = 256
    height: int = 256
    seed: int | None = None
    steps: int = 20


class IPAdapterFaceIDRequest(BaseModel):
    model: str
    prompt: str
    face_image: str  # base64 PNG
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
    except BadImageError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except PipelineLoadError as exc:
        logger.error("Pipeline load failed: %s", exc)
        raise HTTPException(503, str(exc)) from exc

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
    except NoFaceDetectedError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except BadImageError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except PipelineLoadError as exc:
        logger.error("Pipeline load failed: %s", exc)
        raise HTTPException(503, str(exc)) from exc

    return {
        "image": base64.b64encode(img_bytes).decode(),
        "model": req.model,
    }
