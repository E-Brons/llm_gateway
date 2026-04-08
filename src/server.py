"""FastAPI gateway server — exposes all LLM types over HTTP."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import zlib
from contextlib import asynccontextmanager
from typing import Any, Literal

import requests as _requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import LLMConfig, load_llm_config
from .factory import LLMFactory

logger = logging.getLogger("llm_gateway")

_factory: LLMFactory | None = None
_config: LLMConfig | None = None

_TASK_CAPABILITY: dict[str, str] = {
    "general": "text",
    "text_gen": "text",
    "reasoning": "reasoning",
    "image_gen": "image_gen",
    "image_inspector": "visual",
    "tools": "tools",
    "ipadapter": "image_gen",
    "ipadapter_faceid": "image_gen",
}

_SANITY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Return the answer to the question.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
]


def _minimal_png() -> bytes:
    """1×1 red PNG for image inspector sanity check."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))  # filter=0, R=255 G=0 B=0
    iend = chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def _run_sanity_checks(factory: LLMFactory) -> None:
    """Run a minimal live call for each factory type and log the result."""
    sep = "─" * 56
    prompt = [{"role": "user", "content": "Reply with the single word: OK"}]

    checks: list[tuple[str, Any]] = [
        (
            "general",
            lambda: factory.general().complete(prompt),
        ),
        (
            "text_gen",
            lambda: factory.text_gen().complete(prompt, max_retries=1),
        ),
        (
            "reasoning",
            lambda: factory.reasoning().complete(prompt),
        ),
        (
            "image_gen",
            lambda: factory.image_gen().generate(
                "a solid red square",
                reference_images=[_minimal_png()],
                width=32,
                height=32,
                max_retries=1,
            ),
        ),
        (
            "image_inspector",
            lambda: factory.image_inspector().inspect(
                _minimal_png(),
                "You are a concise image analyst.",
                "What is the dominant color? Reply in one word.",
                max_retries=1,
            ),
        ),
        (
            "tools",
            lambda: factory.tools().complete(
                [{"role": "user", "content": "Use the answer tool: what is 1+1?"}],
                _SANITY_TOOLS,
                max_retries=1,
            ),
        ),
    ]

    if _config is not None and _config.ipadapter is not None:
        checks.append(
            (
                "ipadapter",
                lambda: factory.ipadapter().generate(
                    "a solid red square",
                    reference_images=[_minimal_png()],
                    width=32,
                    height=32,
                    num_inference_steps=1,
                    max_retries=1,
                ),
            )
        )

    if _config is not None and _config.ipadapter_faceid is not None:
        checks.append(
            (
                "ipadapter_faceid",
                lambda: factory.ipadapter_faceid().generate(
                    "a portrait",
                    reference_images=[_minimal_png()],
                    width=32,
                    height=32,
                    num_inference_steps=1,
                    max_retries=1,
                ),
            )
        )

    logger.info(sep)
    logger.info("  Sanity checks")
    all_ok = True
    for name, fn in checks:
        try:
            result = fn()
            # Summarise the result
            if hasattr(result, "image"):
                summary = f"{len(result.image)} bytes"
            elif hasattr(result, "tool_calls") and result.tool_calls:
                tc = result.tool_calls[0]
                summary = f"tool_call: {tc.name}({tc.arguments})"
            else:
                summary = (result.content or "").strip()[:60]
            logger.info("    ✓  %-16s %s", name, summary)
        except Exception as exc:
            logger.warning("    ✗  %-16s FAILED: %s", name, exc)
            all_ok = False

    if all_ok:
        logger.info("  all checks passed")
    else:
        logger.warning("  some checks failed — gateway may be partially functional")
    logger.info(sep)


def _load_settings(base: str = "settings.json", override: str = "local/settings.json") -> dict:
    """Merge settings.json with optional local/settings.json override."""
    import json as _json
    from pathlib import Path

    def _deep_merge(b: dict, o: dict) -> dict:
        r = {**b}
        for k, v in o.items():
            r[k] = (
                _deep_merge(r[k], v)
                if k in r and isinstance(r[k], dict) and isinstance(v, dict)
                else v
            )
        return r

    data: dict = {}
    if Path(base).exists():
        data = _json.loads(Path(base).read_text())
    if Path(override).exists():
        data = _deep_merge(data, _json.loads(Path(override).read_text()))
    return data


def _log_startup(
    config: LLMConfig,
    settings: dict,
    config_path: str,
    override_path: str,
    override_active: bool,
    host: str,
    port: str,
) -> None:
    """Log the fully resolved configuration at startup."""
    sep = "─" * 56

    lines = [
        "",
        "  LLM Gateway  starting up",
        sep,
        f"  Listen      {host}:{port}",
        f"  Config      {config_path}",
    ]

    if override_active:
        lines.append(f"  Override    {override_path}  (active)")
    else:
        lines.append(f"  Override    {override_path}  (not found — using base only)")

    lines += [
        sep,
        "  LLM Config",
        f"    general        {config.general.implementation} / {config.general.model}",
        f"    text_gen       {config.text_gen.implementation} / {config.text_gen.model}",
        f"    reasoning      {config.reasoning.implementation} / {config.reasoning.model}",
        f"    image_gen      {config.image_gen.implementation} / {config.image_gen.model}",
        f"    image_inspector {config.image_inspector.implementation} / {config.image_inspector.model}",
        f"    tools          {config.tools.implementation} / {config.tools.model}",
    ]

    if config.ipadapter is not None:
        lines.append(
            f"    ipadapter      {config.ipadapter.implementation} / {config.ipadapter.model}"
        )
    if config.ipadapter_faceid is not None:
        lines.append(
            f"    ipadapter_faceid {config.ipadapter_faceid.implementation}"
            f" / {config.ipadapter_faceid.model}"
        )

    if settings:
        lines += [
            sep,
            "  Settings",
        ]
        for line in json.dumps(settings, indent=4).splitlines():
            lines.append(f"    {line}")

    lines.append(sep)

    for line in lines:
        logger.info(line)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _factory, _config

    config_path = os.environ.get("LLM_GATEWAY_ROUTE", "llm_route.yml")
    override_path = os.environ.get("LLM_GATEWAY_ROUTE_LOCAL", "local/llm_route.yml")
    host = os.environ.get("LLM_GATEWAY_HOST", "127.0.0.1")
    port = os.environ.get("LLM_GATEWAY_PORT", "4096")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    if not os.path.exists(config_path):
        raise RuntimeError(
            f"Base config not found: {config_path}\n"
            "Ensure llm_route.yml exists at the project root or pass --config_file <PATH>."
        )

    override_active = os.path.exists(override_path)
    _config = load_llm_config(config_path, override_path if override_active else None)
    _factory = LLMFactory(_config)

    settings = _load_settings()
    _log_startup(_config, settings, config_path, override_path, override_active, host, port)

    # Run sanity checks in a background thread — server starts serving immediately.
    loop = asyncio.get_event_loop()
    sanity_task = loop.run_in_executor(None, _run_sanity_checks, _factory)

    yield

    await sanity_task  # ensure clean shutdown if checks are still running


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


def _f() -> LLMFactory:
    if _factory is None:
        raise HTTPException(503, "Factory not initialised")
    return _factory


# ── Request models ─────────────────────────────────────────────────────────


class MessageRequest(BaseModel):
    messages: list[dict[str, Any]]
    max_retries: int = 3
    temperature: float | None = None
    response_schema: dict[str, Any] | None = None


class ReasoningRequest(BaseModel):
    messages: list[dict[str, Any]]
    thinking_budget: int | None = None
    temperature: float | None = None
    response_schema: dict[str, Any] | None = None


class ImageGenRequest(BaseModel):
    prompt: str
    reference_images_b64: list[str] | None = None  # base64-encoded reference PNGs
    width: int = 256
    height: int = 256
    seed: int | None = None
    optimize: Literal["quality", "normal", "fast"] = "normal"
    max_retries: int = 3


class ImageInspectRequest(BaseModel):
    image_b64: str
    system: str
    prompt: str
    max_retries: int = 3
    temperature: float | None = None
    response_schema: dict[str, Any] | None = None


class ToolsRequest(BaseModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_retries: int = 3


class IPAdapterRequest(BaseModel):
    prompt: str
    reference_image_b64: str  # base64-encoded reference PNG
    weight: float = 0.5
    width: int = 256
    height: int = 256
    seed: int | None = None
    optimize: Literal["quality", "normal", "fast"] = "normal"
    max_retries: int = 3


class IPAdapterFaceIDRequest(BaseModel):
    prompt: str
    face_image_b64: str  # base64-encoded face PNG
    weight: float = 0.5
    width: int = 256
    height: int = 256
    seed: int | None = None
    optimize: Literal["quality", "normal", "fast"] = "normal"
    max_retries: int = 3


# ── Routes ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"name": "llm_gateway", "version": "1.0.0", "docs": "/docs"}


# ── Compatibility shims ────────────────────────────────────────────────────────
# Ollama and OpenAI-compatible discovery endpoints so local LLM clients
# (OpenWebUI, Continue.dev, LM Studio, etc.) can auto-discover the gateway.


@app.get("/api/tags")
def ollama_tags():
    """Ollama-compatible model list."""
    if _config is None:
        return {"models": []}
    seen: set[str] = set()
    models = []
    for cfg in [
        _config.general,
        _config.text_gen,
        _config.reasoning,
        _config.image_gen,
        _config.image_inspector,
        _config.tools,
        *([_config.ipadapter] if _config.ipadapter is not None else []),
        *([_config.ipadapter_faceid] if _config.ipadapter_faceid is not None else []),
    ]:
        name = cfg.model.removeprefix("ollama/")
        if name not in seen:
            seen.add(name)
            models.append(
                {
                    "name": name,
                    "modified_at": "2026-01-01T00:00:00Z",
                    "size": 0,
                    "digest": "",
                    "details": {
                        "family": "llm_gateway",
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                }
            )
    return {"models": models}


@app.get("/v1/models")
def openai_models():
    """OpenAI-compatible model list."""
    if _config is None:
        return {"object": "list", "data": []}
    seen: set[str] = set()
    data = []
    for task, cfg in [
        ("general", _config.general),
        ("text_gen", _config.text_gen),
        ("reasoning", _config.reasoning),
        ("image_gen", _config.image_gen),
        ("image_inspector", _config.image_inspector),
        ("tools", _config.tools),
        *([("ipadapter", _config.ipadapter)] if _config.ipadapter is not None else []),
        *(
            [("ipadapter_faceid", _config.ipadapter_faceid)]
            if _config.ipadapter_faceid is not None
            else []
        ),
    ]:
        model_id = cfg.model.removeprefix("ollama/")
        if model_id not in seen:
            seen.add(model_id)
            data.append(
                {"id": model_id, "object": "model", "created": 0, "owned_by": "llm_gateway"}
            )
    return {"object": "list", "data": data}


@app.get("/models")
def list_models():
    if _config is None:
        raise HTTPException(503, "Config not loaded")

    tasks = [
        ("general", _config.general),
        ("text_gen", _config.text_gen),
        ("reasoning", _config.reasoning),
        ("image_gen", _config.image_gen),
        ("image_inspector", _config.image_inspector),
        ("tools", _config.tools),
        *([("ipadapter", _config.ipadapter)] if _config.ipadapter is not None else []),
        *(
            [("ipadapter_faceid", _config.ipadapter_faceid)]
            if _config.ipadapter_faceid is not None
            else []
        ),
    ]

    groups: dict[tuple[str, str], dict] = {}
    ollama_url = "http://localhost:11434"
    for task, cfg in tasks:
        key = (cfg.model, cfg.implementation)
        if key not in groups:
            groups[key] = {
                "model": cfg.model,
                "implementation": cfg.implementation,
                "tasks": [],
                "capabilities": set(),
            }
        groups[key]["tasks"].append(task)
        groups[key]["capabilities"].add(_TASK_CAPABILITY[task])
        if cfg.implementation == "ollama" and cfg.ollama_url:
            ollama_url = cfg.ollama_url

    configured = [
        {
            "model": v["model"],
            "implementation": v["implementation"],
            "tasks": sorted(v["tasks"]),
            "capabilities": sorted(v["capabilities"]),
        }
        for v in groups.values()
    ]

    ollama_pulled: list[str] = []
    try:
        resp = _requests.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.ok:
            ollama_pulled = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass

    diffusion_available: list[str] = []
    if _config is not None and (
        _config.ipadapter is not None or _config.ipadapter_faceid is not None
    ):
        # Try to query the diffusion server for its known models
        diff_bases: list[str] = []
        if _config.ipadapter is not None and _config.ipadapter.api_base:
            diff_bases.append(_config.ipadapter.api_base)
        if _config.ipadapter_faceid is not None and _config.ipadapter_faceid.api_base:
            diff_bases.append(_config.ipadapter_faceid.api_base)
        for base in dict.fromkeys(diff_bases):  # deduplicate, preserve order
            try:
                r = _requests.get(f"{base}/models", timeout=5)
                if r.ok:
                    diffusion_available += [m["name"] for m in r.json().get("models", [])]
            except Exception:
                pass

    for entry in configured:
        if entry["implementation"] == "ollama":
            bare = entry["model"].removeprefix("ollama/")
            entry["available"] = any(
                p == bare or p.startswith(bare + ":") or p == entry["model"] for p in ollama_pulled
            )
        elif entry["implementation"] == "diffusion_server":
            bare = entry["model"].removeprefix("diffusion/")
            entry["available"] = bare in diffusion_available

    return {"configured": configured, "ollama_available": ollama_pulled, "diffusion_available": diffusion_available}


@app.post("/general")
def general(req: MessageRequest):
    return (
        _f()
        .general()
        .complete(req.messages, temperature=req.temperature, response_schema=req.response_schema)
        .model_dump()
    )


@app.post("/text_gen")
def text_gen(req: MessageRequest):
    return (
        _f()
        .text_gen()
        .complete(
            req.messages,
            max_retries=req.max_retries,
            temperature=req.temperature,
            response_schema=req.response_schema,
        )
        .model_dump()
    )


@app.post("/reasoning")
def reasoning(req: ReasoningRequest):
    return (
        _f()
        .reasoning()
        .complete(
            req.messages,
            thinking_budget=req.thinking_budget,
            temperature=req.temperature,
            response_schema=req.response_schema,
        )
        .model_dump()
    )


@app.post("/image_gen")
def image_gen(req: ImageGenRequest):
    _OPTIMIZE_STEPS = {"quality": 4, "normal": 3, "fast": 2}
    reference_images = (
        [base64.b64decode(b) for b in req.reference_images_b64]
        if req.reference_images_b64
        else None
    )
    resp = (
        _f()
        .image_gen()
        .generate(
            req.prompt,
            reference_images=reference_images,
            width=req.width,
            height=req.height,
            seed=req.seed,
            num_inference_steps=_OPTIMIZE_STEPS[req.optimize],
            max_retries=req.max_retries,
        )
    )
    return {
        "image_b64": base64.b64encode(resp.image).decode(),
        "model": resp.model,
        "duration_ms": resp.duration_ms,
        "attempts": resp.attempts,
        "last_error": resp.last_error,
    }


@app.post("/image_inspector")
def image_inspector(req: ImageInspectRequest):
    image = base64.b64decode(req.image_b64)
    return (
        _f()
        .image_inspector()
        .inspect(
            image,
            req.system,
            req.prompt,
            max_retries=req.max_retries,
            temperature=req.temperature,
            response_schema=req.response_schema,
        )
        .model_dump()
    )


@app.post("/tools")
def tools(req: ToolsRequest):
    return _f().tools().complete(req.messages, req.tools, max_retries=req.max_retries).model_dump()


@app.post("/ipadapter")
def ipadapter(req: IPAdapterRequest):
    _OPTIMIZE_STEPS = {"quality": 4, "normal": 3, "fast": 2}
    reference_image = base64.b64decode(req.reference_image_b64)
    resp = (
        _f()
        .ipadapter()
        .generate(
            req.prompt,
            reference_images=[reference_image],
            weight=req.weight,
            width=req.width,
            height=req.height,
            seed=req.seed,
            num_inference_steps=_OPTIMIZE_STEPS[req.optimize],
            max_retries=req.max_retries,
        )
    )
    return {
        "image_b64": base64.b64encode(resp.image).decode(),
        "model": resp.model,
        "duration_ms": resp.duration_ms,
        "attempts": resp.attempts,
        "last_error": resp.last_error,
    }


@app.post("/ipadapter_faceid")
def ipadapter_faceid(req: IPAdapterFaceIDRequest):
    _OPTIMIZE_STEPS = {"quality": 4, "normal": 3, "fast": 2}
    face_image = base64.b64decode(req.face_image_b64)
    resp = (
        _f()
        .ipadapter_faceid()
        .generate(
            req.prompt,
            reference_images=[face_image],
            weight=req.weight,
            width=req.width,
            height=req.height,
            seed=req.seed,
            num_inference_steps=_OPTIMIZE_STEPS[req.optimize],
            max_retries=req.max_retries,
        )
    )
    return {
        "image_b64": base64.b64encode(resp.image).decode(),
        "model": resp.model,
        "duration_ms": resp.duration_ms,
        "attempts": resp.attempts,
        "last_error": resp.last_error,
    }
