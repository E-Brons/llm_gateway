#!/usr/bin/env bash
# Start the diffusion server (IP-Adapter pipeline).
#
# Usage:
#   ./scripts/run_diffusion.sh [--port PORT] [--host HOST]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIFFUSION_DIR="$ROOT/diffusion_server"
VENV="$ROOT/.venv"
PID_FILE="$ROOT/local/diffusion.pid"
PORT=7860
HOST="127.0.0.1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; echo "Usage: run_diffusion.sh [--port PORT] [--host HOST]"; exit 1 ;;
  esac
done

if [ ! -d "$VENV" ]; then
  echo "Virtual environment not found. Run: ./scripts/install.sh --with-diffusion"
  exit 1
fi

mkdir -p "$ROOT/local"

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Diffusion server already running (PID $PID). Run stop_diffusion.sh first."
    exit 1
  fi
  rm -f "$PID_FILE"
fi

echo "Starting diffusion server on $HOST:$PORT …"

cd "$DIFFUSION_DIR"
"$VENV/bin/uvicorn" server:app --host "$HOST" --port "$PORT" &
echo $! > "$PID_FILE"

# Wait for ready
echo -n "Waiting for diffusion server"
for i in $(seq 1 60); do
  if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo " ready."
    echo "Diffusion server running at http://$HOST:$PORT"
    echo "  Models: http://$HOST:$PORT/models"
    exit 0
  fi
  sleep 1
  [ $((i % 10)) -eq 0 ] && echo -n " ${i}s" || echo -n "."
done

echo " timed out — check logs."
exit 1
