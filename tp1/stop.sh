#!/usr/bin/env bash
# tp1/stop.sh — stop the single-Spark vLLM container and its memory watchdog.
set -euo pipefail
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 && echo "stopped $CONTAINER_NAME" || echo "$CONTAINER_NAME was not running"
# pkill pattern is split so this script's own command line never matches it.
pkill -f "memwatch.sh ${CONTAINER_NAME}" 2>/dev/null && echo "watchdog stopped" || true
