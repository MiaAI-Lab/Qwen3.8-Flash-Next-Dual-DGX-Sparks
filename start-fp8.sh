#!/usr/bin/env bash
# ============================================================================
# start-fp8.sh — Same 2-node vLLM launch as start.sh, but serve
#                 Qwen/Qwen3.8-Flash-Next-FP8 instead of NVFP4.
#
# Official FP8 is fine-grained (block-128) over the whole checkpoint, so the
# NVFP4 PLE resolver shim is not applied. Context is native 262144 (no YaRN):
# FP8 weights leave too little KV for 1M on this 2×Spark kit.
#
# Weights stay on the head HuggingFace cache and are exported over NFS
# (ConnectX) — the worker does not download or rsync a local copy.
#
# Cluster IPs, TP/EP/MTP, image, NFS, and ports still come from .env.
#
# Usage: same flags as start.sh
#   ./start-fp8.sh
#   ./start-fp8.sh --no-download
#   ./start-fp8.sh --launch
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OVERRIDE_MODEL_ID="${OVERRIDE_MODEL_ID:-Qwen/Qwen3.8-Flash-Next-FP8}"
export OVERRIDE_SERVED_MODEL_NAME="${OVERRIDE_SERVED_MODEL_NAME:-qwen3.8-flash-next-fp8}"
export OVERRIDE_MAX_MODEL_LEN="${OVERRIDE_MAX_MODEL_LEN:-262144}"
export OVERRIDE_YARN_ENABLE="${OVERRIDE_YARN_ENABLE:-false}"
export SKIP_PLE_PATCH=true

exec "$SCRIPT_DIR/start.sh" "$@"
