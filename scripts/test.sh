#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"

echo "── lint ────────────────────────────────────"
"$VENV/bin/ruff" check src/ tests/
"$VENV/bin/ruff" format --check src/ tests/

echo "── tests ───────────────────────────────────"
"$VENV/bin/python" -m pytest "$ROOT/tests/" -m "not integration" "$@"
