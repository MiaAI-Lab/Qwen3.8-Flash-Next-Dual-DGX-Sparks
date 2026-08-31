#!/usr/bin/env bash
# check-weights.sh — Verify model weights exist on both head and worker nodes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found."
    exit 1
fi

source .env

WORKER_USER="${WORKER_USER:-}"
WORKER_IP="${WORKER_IP:?WORKER_IP not set in .env}"
MODEL_ID="${MODEL_ID:?MODEL_ID not set in .env}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"
ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"

ssh_cmd() {
    local user_prefix=""
    [[ -n "$WORKER_USER" ]] && user_prefix="${WORKER_USER}@"
    ssh -o StrictHostKeyChecking=no "${user_prefix}$WORKER_IP" "$@"
}

check_node() {
    local label="$1"
    local path="$2"
    local run_cmd="${3:-local}"

    if [[ "$run_cmd" == "local" ]]; then
        if [[ -d "$path" ]]; then
            local size
            size=$(du -sh "$path" 2>/dev/null | cut -f1)
            local shards
            shards=$(find "$path" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | wc -l)
            echo "  $label ✅  $path"
            echo "         Size: $size | Shards: $shards"
            return 0
        else
            echo "  $label ❌  $path — NOT FOUND"
            return 1
        fi
    else
        # Remote check via SSH
        if ssh_cmd "test -d '$path'" 2>/dev/null; then
            local size shards
            size=$(ssh_cmd "du -sh '$path' 2>/dev/null" | cut -f1)
            shards=$(ssh_cmd "find '$path' -name '*.safetensors' -o -name '*.bin' 2>/dev/null | wc -l")
            echo "  $label ✅  $path"
            echo "         Size: $size | Shards: $shards"
            return 0
        else
            echo "  $label ❌  $path — NOT FOUND"
            return 1
        fi
    fi
}

echo "Checking weights for: $MODEL_ID"
echo "Cache path: $MODEL_PATH"
echo ""

HEAD_OK=false
WORKER_OK=false

check_node "HEAD  ($HEAD_IP)" "$MODEL_PATH" "local" && HEAD_OK=true
check_node "WORKER ($WORKER_IP)" "$MODEL_PATH" "ssh" && WORKER_OK=true

echo ""

if $HEAD_OK && $WORKER_OK; then
    echo "✅ Weights present on both nodes."
    exit 0
else
    echo "❌ Missing weights on one or more nodes."
    [[ "$HEAD_OK" == "false" ]] && echo "   → Run: ./start.sh (no --no-download) to fetch to head"
    [[ "$WORKER_OK" == "false" ]] && echo "   → Run: ./start.sh (no --no-launch) to rsync to worker"
    exit 1
fi
