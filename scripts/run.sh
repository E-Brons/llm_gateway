#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
PID_FILE="$ROOT/local/gateway.pid"

# Defaults
PORT=4096
HOST="127.0.0.1"
CONFIG_FILE="$ROOT/llm_route.yml"          # global base config (tracked)
CONFIG_LOCAL="$ROOT/local/llm_route.yml"   # optional user override (gitignored)

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      KEY="${2%%:*}"
      VAL="${2#*:}"
      case "$KEY" in
        port) PORT="$VAL" ;;
        host) HOST="$VAL" ;;
        *) echo "Unknown --config key: $KEY (supported: port, host)"; exit 1 ;;
      esac
      shift 2
      ;;
    --config_file)
      CONFIG_FILE="$2"
      CONFIG_LOCAL=""   # explicit base — no local override
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: run.sh [--config port:PORT] [--config host:HOST] [--config_file PATH]"
      exit 1
      ;;
  esac
done

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Base config not found: $CONFIG_FILE"
  exit 1
fi

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Gateway already running (PID $PID). Run stop.sh first."
    exit 1
  fi
  rm -f "$PID_FILE"
fi

LLM_GATEWAY_ROUTE="$CONFIG_FILE" \
LLM_GATEWAY_ROUTE_LOCAL="$CONFIG_LOCAL" \
LLM_GATEWAY_HOST="$HOST" \
LLM_GATEWAY_PORT="$PORT" \
  "$VENV/bin/uvicorn" src.server:app \
  --host "$HOST" \
  --port "$PORT" \
  --app-dir "$ROOT" &

echo $! > "$PID_FILE"

# Wait for the server to be ready
echo -n "Waiting for gateway"
for i in $(seq 1 30); do
  if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo " ready."
    echo "LLM Gateway running at http://$HOST:$PORT"
    exit 0
  fi
  sleep 1
  echo -n "."
done

echo " timed out."
exit 1
