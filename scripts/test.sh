#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"

"$VENV/bin/python" -m pytest "$ROOT/tests/" -m "not integration" "$@"
