# Service Setup — Diffusion Server (IP-Adapter)

**Role in llm_gateway**: Runs the `ipadapter` and `ipadapter_faceid` routes — reference-image-conditioned and face-conditioned image generation.

The diffusion server is a FastAPI process bundled in this repo (`diffusion_server/`). It wraps HuggingFace `diffusers` pipelines and exposes a REST API that the gateway routes to.

---

## Prerequisites

- Python ≥ 3.10 (a separate venv from the main gateway — `diffusion_server/.venv`)
- ~5–6 GB free disk space (base SD1.5 model + adapter weights)
- GPU strongly recommended: Apple Silicon (MPS), NVIDIA CUDA, or CPU fallback (~10× slower)

No separate binary to install — everything is Python.

---

## Install

```sh
./scripts/install_diffusion.sh
```

This will:
1. Create `diffusion_server/.venv` with all Python dependencies (`diffusers`, `torch`, `insightface`, etc.)
2. Download the following models into `~/.cache/huggingface/`:

| Download | Size | Purpose |
|----------|------|---------|
| `runwayml/stable-diffusion-v1-5` | ~4 GB | Base diffusion model |
| `h94/IP-Adapter` → `ip-adapter_sd15_light_v11.bin` | ~300 MB | Style adapter weights |
| `h94/IP-Adapter-FaceID` → `ip-adapter-faceid_sd15.bin` | ~500 MB | FaceID adapter weights |
| InsightFace `buffalo_l` | ~300 MB | Face detection model for FaceID |

To install Python deps without downloading models (e.g. in CI):
```sh
./scripts/install_diffusion.sh --skip-models
```

---

## Start the Server

```sh
./scripts/run_diffusion.sh
```

Default address: `http://localhost:7860`

To use a different port or host:
```sh
./scripts/run_diffusion.sh --port 7861 --host 0.0.0.0
```

Update `llm_route.yml` to match:
```yaml
ipadapter:
  api_base: http://localhost:7861
```

---

## Stop the Server

```sh
./scripts/stop_diffusion.sh
```

---

## Available Models

The server knows the following model names out of the box:

| Model name | Mode | Adapter weights |
|------------|------|----------------|
| `ip-adapter_sd15_light_v11` | style | `h94/IP-Adapter` — lighter, faster |
| `ip-adapter_sd15` | style | `h94/IP-Adapter` — full weight |
| `ip-adapter-faceid-plus_sd15` | faceid | `h94/IP-Adapter-FaceID` — identity + quality |
| `ip-adapter-faceid_sd15` | faceid | `h94/IP-Adapter-FaceID` — identity only |

Check what the running server knows:
```sh
curl http://localhost:7860/models
```

---

## Configuration in llm_route.yml

```yaml
ipadapter:
  implementation: diffusion_server
  model: ip-adapter_sd15_light_v11
  api_base: http://localhost:7860

ipadapter_faceid:
  implementation: diffusion_server
  model: ip-adapter-faceid_sd15
  api_base: http://localhost:7860
```

Both sections are **optional**. If omitted, calling `factory.ipadapter()` raises `ValueError`.

---

## REST API

The server exposes these endpoints (also callable directly, bypassing the gateway):

### `GET /health`
```json
{"status": "ok"}
```

### `GET /models`
```json
{
  "models": [
    {"name": "ip-adapter_sd15_light_v11", "mode": "style", "base": "runwayml/stable-diffusion-v1-5"},
    {"name": "ip-adapter-faceid-plus_sd15", "mode": "faceid", "base": "runwayml/stable-diffusion-v1-5"}
  ]
}
```

### `POST /ipadapter`

**Request fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | — | Model name from the registry |
| `prompt` | string | — | Scene / style description |
| `reference_image` | string | — | Base64-encoded reference image (PNG/JPEG) |
| `ip_adapter_scale` | float | `0.5` | How strongly the reference image guides generation (0–1) |
| `width` | int | `256` | Output width in pixels |
| `height` | int | `256` | Output height in pixels |
| `steps` | int | `20` | Number of diffusion inference steps |
| `cfg_scale` | float | `7.5` | Classifier-free guidance scale |
| `negative_prompt` | string | `null` | Text describing what to avoid |
| `seed` | int | `null` | RNG seed for reproducibility |
| `lora` | string | `null` | HuggingFace repo ID or local path to LoRA weights |
| `lora_weight` | float | `1.0` | LoRA conditioning strength (0–1); only used when `lora` is set |

**Example request:**
```json
{
  "model": "ip-adapter_sd15_light_v11",
  "prompt": "a cat on a wooden table, studio lighting",
  "reference_image": "<base64 PNG>",
  "ip_adapter_scale": 0.6,
  "width": 512,
  "height": 512,
  "cfg_scale": 7.5,
  "negative_prompt": "blurry, low quality",
  "seed": 42,
  "steps": 20
}
```

**Response:** `{"image": "<base64 PNG>", "model": "ip-adapter_sd15_light_v11"}`

---

### `POST /ipadapter_faceid`

Same fields as `/ipadapter` except `reference_image` is replaced by `face_image`:

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | — | Model name from the registry |
| `prompt` | string | — | Scene / style description |
| `face_image` | string | — | Base64-encoded portrait image (PNG/JPEG) — the identity reference |
| `ip_adapter_scale` | float | `0.5` | How strongly face identity is preserved (0–1) |
| `width` | int | `256` | Output width in pixels |
| `height` | int | `256` | Output height in pixels |
| `steps` | int | `20` | Number of diffusion inference steps |
| `cfg_scale` | float | `7.5` | Classifier-free guidance scale |
| `negative_prompt` | string | `null` | Text describing what to avoid |
| `seed` | int | `null` | RNG seed for reproducibility |
| `lora` | string | `null` | HuggingFace repo ID or local path to LoRA weights |
| `lora_weight` | float | `1.0` | LoRA conditioning strength (0–1); only used when `lora` is set |

**Example request:**
```json
{
  "model": "ip-adapter-faceid_sd15",
  "prompt": "a person in a space suit, cinematic lighting",
  "face_image": "<base64 PNG>",
  "ip_adapter_scale": 0.75,
  "width": 512,
  "height": 512,
  "cfg_scale": 7.5,
  "negative_prompt": "cartoon, anime, illustration",
  "seed": 7
}
```

**Response:** `{"image": "<base64 PNG>", "model": "ip-adapter-faceid_sd15"}`

**Error responses:**

| Status | Condition |
|---|---|
| 400 | Unknown model or wrong endpoint for model type |
| 422 | Image bytes cannot be decoded, or no face detected |
| 503 | Pipeline failed to load (OOM, missing weights, etc.) |

---

## Verify

```sh
# Diffusion server health
curl http://localhost:7860/health

# Gateway model status (shows diffusion_available)
curl http://localhost:4096/models
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `connection refused` on port 7860 | Run `./scripts/run_diffusion.sh` |
| `diffusion_server/.venv not found` | Run `./scripts/install_diffusion.sh` |
| `No face detected` | Input image has no clearly visible face; use a front-facing photo |
| Very slow generation | No GPU/MPS detected — running on CPU. Check `torch.backends.mps.is_available()` |
| Out of memory on MPS | Reduce `width`/`height` or `steps`; restart the server to clear the pipeline cache |
| Model not in registry | Add an entry to `diffusion_server/pipeline.py` `_REGISTRY` dict |

---

## Notes

- **Lazy loading**: the pipeline is loaded on the first request, not at startup. The first call will be slow (~30–60 s on Apple Silicon).
- **One model at a time**: switching between models offloads the previous pipeline to free memory. Avoid mixing `ipadapter` and `ipadapter_faceid` calls in rapid alternation.
- **HuggingFace cache**: models are stored in `~/.cache/huggingface/hub`. Re-running `install_diffusion.sh` after models are downloaded is safe (no re-download).
