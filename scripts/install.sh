#!/usr/bin/env bash
# Install llm_gateway and all dependencies.
#
# Usage:
#   ./scripts/install.sh                   # full install: core + IP-Adapter diffusion server + models
#   ./scripts/install.sh --skip-models     # install deps only, skip model download
#   ./scripts/install.sh --no-diffusion    # core only (text + image routes via Ollama / LiteLLM)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
WITH_DIFFUSION=1
SKIP_MODELS=0

for arg in "$@"; do
  case "$arg" in
    --no-diffusion) WITH_DIFFUSION=0 ;;
    --skip-models)  SKIP_MODELS=1 ;;
    *) echo "Unknown argument: $arg"; echo "Usage: install.sh [--no-diffusion] [--skip-models]"; exit 1 ;;
  esac
done

# ── Prerequisites ────────────────────────────────────────────────────────────
echo "Checking prerequisites…"

missing=0
for cmd in ollama python3; do
  if command -v "$cmd" &>/dev/null; then
    printf "  %-10s %s\n" "$cmd" "$($cmd --version 2>&1 | head -1)"
  else
    echo "  ✗ $cmd not found"
    missing=1
  fi
done
[ "$missing" -eq 1 ] && { echo "Install missing prerequisites and re-run."; exit 1; }

# python3 >= 3.14
py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
py_major=$(echo "$py_ver" | cut -d. -f1)
py_minor=$(echo "$py_ver" | cut -d. -f2)
if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 14 ]; }; then
  echo "  ✗ python3 $py_ver found, but >= 3.14 is required."
  exit 1
fi

# ollama >= 0.19
ollama_ver=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
ollama_major=$(echo "$ollama_ver" | cut -d. -f1)
ollama_minor=$(echo "$ollama_ver" | cut -d. -f2)
if [ "$ollama_major" -eq 0 ] && [ "$ollama_minor" -lt 19 ]; then
  echo "  ✗ ollama $ollama_ver found, but >= 0.19 is required."
  exit 1
fi

# ── Python virtual environment ───────────────────────────────────────────────
echo ""
if [ ! -d "$VENV" ]; then
  echo "Creating Python venv at $VENV …"
  python3 -m venv "$VENV"
fi

echo "Installing core Python dependencies …"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$ROOT[dev]"

# ── Diffusion server (optional) ──────────────────────────────────────────────
if [ "$WITH_DIFFUSION" -eq 1 ]; then
  echo ""
  echo "Installing diffusion dependencies …"
  "$VENV/bin/pip" install -e "$ROOT[diffusion]"

  if [ "$SKIP_MODELS" -eq 0 ]; then
    echo ""
    echo "Downloading IP-Adapter models from HuggingFace …"
    echo "  (Base SD1.5 model ~4 GB; adapter weights ~300–600 MB each)"
    echo "  Models are cached in ~/.cache/huggingface — re-running is safe."
    echo ""

    "$VENV/bin/python" - <<'PYEOF'
from huggingface_hub import hf_hub_download, snapshot_download

print("  Downloading base model: runwayml/stable-diffusion-v1-5 …")
snapshot_download("runwayml/stable-diffusion-v1-5", ignore_patterns=["*.ckpt"])

print("  Downloading IP-Adapter style weights (ip-adapter_sd15_light_v11) …")
hf_hub_download("h94/IP-Adapter", filename="models/ip-adapter_sd15_light_v11.bin")

print("  Downloading IP-Adapter FaceID weights (ip-adapter-faceid-plus_sd15) …")
hf_hub_download("h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plus_sd15.bin")

print("  Downloading InsightFace buffalo_l face detection model …")
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

print("  All models downloaded.")
PYEOF

  else
    echo "  Skipping model download (--skip-models)."
  fi
else
  echo ""
  echo "Skipping diffusion install (--no-diffusion)."
  echo "Tip: re-run without --no-diffusion to install IP-Adapter diffusion server."
fi

# ── Git hooks ─────────────────────────────────────────────────────────────────
echo ""
echo "Installing git hooks …"
git -C "$ROOT" config core.hooksPath .githooks
echo "  pre-push hook active (ruff check + format)"

echo ""
echo "Installation complete."
[ "$WITH_DIFFUSION" -eq 1 ] && echo "  Start diffusion server: ./scripts/run_diffusion.sh"
echo "  Start gateway:          ./scripts/run.sh"
