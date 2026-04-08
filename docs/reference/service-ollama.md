# Service Setup — Ollama

**Role in llm_gateway**: Runs all local text and image routes (`general`, `text_gen`, `reasoning`, `image_gen`, `image_inspector`, `tools`).

---

## Install

### macOS
```sh
brew install ollama
```

### Linux
```sh
curl -fsSL https://ollama.com/install.sh | sh
```

Minimum version: **0.19** (required for MLX/Apple Silicon support). Check with `ollama --version`.

---

## Start the Server

Ollama starts automatically as a background service after install. If it is not running:

```sh
ollama serve
```

Default address: `http://localhost:11434`

To use a different address, set `OLLAMA_HOST`:
```sh
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Update `llm_route.yml` to match:
```yaml
text_gen:
  ollama_url: http://192.168.1.10:11434
```

---

## Pull Models

Models must be pulled before the gateway can use them. Use `ollama pull <model>`:

```sh
# Text models
ollama pull phi3
ollama pull qwen2.5:7b

# Image generation (diffusion)
ollama pull x/flux2-klein:4b

# Vision (image inspector)
ollama pull qwen2.5vl:7b
ollama pull llava

# Reasoning / tool-use capable
ollama pull qwen3.5:9b
```

Check what is currently pulled:
```sh
ollama list
```

The gateway's `GET /models` endpoint also shows pulled vs. configured models.

---

## Configuration in llm_route.yml

```yaml
general:
  implementation: ollama
  model: phi3
  ollama_url: http://localhost:11434

text_gen:
  implementation: ollama
  model: qwen2.5:7b
  ollama_url: http://localhost:11434
  temperature: 0.3
  max_tokens: 1024

image_gen:
  implementation: ollama
  model: x/flux2-klein:4b
  ollama_url: http://localhost:11434

image_inspector:
  implementation: ollama
  model: qwen2.5vl:7b
  ollama_url: http://localhost:11434
```

Model names are passed as-is to Ollama — the `ollama/` prefix is stripped automatically if present.

---

## Verify

```sh
# Ollama health
curl http://localhost:11434/api/tags

# Gateway model status
curl http://localhost:4096/models
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `connection refused` on port 11434 | `ollama serve` is not running |
| `model not found` | `ollama pull <model>` |
| Slow first response | Model is loading into memory — normal for large models |
| `ollama --version` shows < 0.19 | Upgrade: `brew upgrade ollama` or re-run the install script |
