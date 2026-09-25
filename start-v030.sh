#!/usr/bin/env bash
# start-v030.sh — same 2-node launch as start.sh on stock vLLM 0.30.0.
# Usage: ./start-v030.sh [--launch|--no-download]   (both nodes need the image)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export V030=true
export OVERRIDE_IMAGE="${OVERRIDE_IMAGE:-vllm/vllm-openai:v0.30.0}"
export OVERRIDE_KV_CACHE_DTYPE="${OVERRIDE_KV_CACHE_DTYPE:-fp8}"
export OVERRIDE_GPU_MEMORY_UTILIZATION="${OVERRIDE_GPU_MEMORY_UTILIZATION:-0.80}"
export SKIP_PLE_PATCH=true

exec "$SCRIPT_DIR/start.sh" "$@"
