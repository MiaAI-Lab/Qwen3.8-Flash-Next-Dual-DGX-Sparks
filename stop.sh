#!/usr/bin/env bash
# stop.sh — Stop the vLLM container on both head and worker nodes.
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
CONTAINER_NAME="vllm-fn"

ssh_cmd() {
    local user_prefix=""
    [[ -n "$WORKER_USER" ]] && user_prefix="${WORKER_USER}@"
    ssh -o StrictHostKeyChecking=no "${user_prefix}$WORKER_IP" "$@"
}

echo "Stopping $CONTAINER_NAME on worker ($WORKER_IP)..."
ssh_cmd "docker rm -f $CONTAINER_NAME 2>/dev/null && echo '  Worker: stopped.' || echo '  Worker: not running.'"

echo "Stopping $CONTAINER_NAME on head..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null && echo "  Head: stopped." || echo "  Head: not running."

echo "Done."
