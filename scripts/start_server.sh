#!/usr/bin/env bash
# Start the pinned local server without tying its lifetime to a terminal tab.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f results/live/server.pid ]]; then
    pid="$(cat results/live/server.pid)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && \
        ps -p "$pid" -o args= | grep -q 'local-jev serve --model llm-qwen3.5-4b'; then
        echo "The Qwen server is already running (PID $pid); check results/live/server.log."
        exit 1
    fi
fi
if curl --silent --fail --max-time 2 http://127.0.0.1:8765/healthz >/dev/null; then
    echo "A server is already responding on port 8765; inspect its /healthz before restarting."
    exit 1
fi
mkdir -p results/live
nohup env HF_HOME="$PWD/.cache/huggingface" PYTHONUNBUFFERED=1 \
    .venv-server/bin/local-jev serve --model llm-qwen3.5-4b \
    >results/live/server.log 2>&1 </dev/null &
echo "$!" >results/live/server.pid
echo "Server PID: $(cat results/live/server.pid); log: results/live/server.log"
