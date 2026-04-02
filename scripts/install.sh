#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Prerequisites ────────────────────────────────────────────────────
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
if [ "$missing" -eq 1 ]; then
  echo "Install missing prerequisites and re-run."
  exit 1
fi

# ── Version checks ────────────────────────────────────────────────────
# Require python3 >= 3.14
py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
py_major=$(echo "$py_ver" | cut -d. -f1)
py_minor=$(echo "$py_ver" | cut -d. -f2)
if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 14 ]; }; then
  echo "  ✗ python3 $py_ver found, but >= 3.14 is required."
  exit 1
fi

# Require ollama >= 0.19
ollama_ver=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
ollama_major=$(echo "$ollama_ver" | cut -d. -f1)
ollama_minor=$(echo "$ollama_ver" | cut -d. -f2)
if [ "$ollama_major" -eq 0 ] && [ "$ollama_minor" -lt 19 ]; then
  echo "  ✗ ollama $ollama_ver found, but >= 0.19 is required."
  exit 1
fi

# ── Python virtual environment ───────────────────────────────────────
echo ""
VENV="$ROOT/.venv"
if [ ! -d "$VENV" ]; then
  echo "Creating Python venv at $VENV …"
  python3 -m venv "$VENV"
fi

echo "Installing Python dependencies …"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$ROOT[dev]"

# ── Git hooks ─────────────────────────────────────────────────────────
echo ""
echo "Installing git hooks …"
git -C "$ROOT" config core.hooksPath .githooks
echo "  pre-push hook active (ruff check + format)"
