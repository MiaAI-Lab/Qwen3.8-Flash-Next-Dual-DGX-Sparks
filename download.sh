#!/usr/bin/env bash
# ============================================================================
# download.sh — Fetch HuggingFace weights onto the HEAD node only.
#
# The worker never gets a local copy. After this, ./start.sh --launch (or
# ./start-fp8.sh --launch) exports the head cache over NFS on ConnectX.
#
# Usage:
#   ./download.sh                 # MODEL_ID from .env (NVFP4)
#   ./download.sh --fp8           # Qwen/Qwen3.8-Flash-Next-FP8
#   ./download.sh org/repo        # explicit HuggingFace repo
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.sample to .env and edit it."
    echo "  cp .env.sample .env"
    exit 1
fi
# shellcheck source=.env
source .env

HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME="$HF_CACHE_DIR"
HUB_PATH="$HF_CACHE_DIR/hub"
mkdir -p "$HUB_PATH"

MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
FP8_MODEL_ID="Qwen/Qwen3.8-Flash-Next-FP8"

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--fp8 | org/repo]"
            echo ""
            echo "  (default)  Download MODEL_ID from .env onto this node (head HF cache)"
            echo "  --fp8      Download $FP8_MODEL_ID"
            echo "  org/repo   Download that HuggingFace repo"
            echo ""
            echo "Does not copy weights to the worker. NFS share happens at launch."
            exit 0
            ;;
        --fp8) MODEL_ID="$FP8_MODEL_ID" ;;
        -*) err "Unknown argument: $arg (try --help)" ;;
        *)  MODEL_ID="$arg" ;;
    esac
done

ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"

info "Downloading $MODEL_ID"
info "Head cache: $HF_CACHE_DIR"
info "Worker:     not updated (NFS from head at launch)"

if command -v uvx &>/dev/null; then
    info "Using uvx..."
    HF_HOME="$HF_CACHE_DIR" uvx hf download "$MODEL_ID" --cache-dir "$HUB_PATH"
elif command -v huggingface-cli &>/dev/null; then
    info "Using huggingface-cli..."
    HF_HOME="$HF_CACHE_DIR" huggingface-cli download "$MODEL_ID" --cache-dir "$HUB_PATH"
elif command -v hf &>/dev/null; then
    info "Using hf CLI..."
    HF_HOME="$HF_CACHE_DIR" hf download "$MODEL_ID" --cache-dir "$HUB_PATH"
else
    err "No HuggingFace download tool found. Install one of:\n  pip install huggingface_hub\n  pip install uv"
fi

if [[ -d "$MODEL_PATH" ]]; then
    SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    ok "Download complete: $MODEL_PATH ($SIZE)"
else
    err "Download finished but $MODEL_PATH was not found"
fi
