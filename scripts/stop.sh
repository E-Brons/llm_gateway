#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/local/gateway.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found — gateway may not be running."
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  rm -f "$PID_FILE"
  echo "Stopped gateway (PID $PID)."
else
  echo "Process $PID not found — already stopped."
  rm -f "$PID_FILE"
fi
