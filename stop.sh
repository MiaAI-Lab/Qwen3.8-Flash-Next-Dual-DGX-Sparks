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
NFS_CONTAINER="${NFS_CONTAINER:-vllm-fn-nfs}"
NFS_VOLUME="${NFS_VOLUME:-vllm-fn-hf}"
STOP_NFS=false

for arg in "$@"; do
    case "$arg" in
        --nfs|--all) STOP_NFS=true ;;
        -h|--help)
            echo "Usage: $0 [--nfs]"
            echo "  (default)  Stop vLLM on worker then head"
            echo "  --nfs      Also stop the head NFS share and remove the worker volume"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg (try --help)"
            exit 1
            ;;
    esac
done

ssh_cmd() {
    local user_prefix=""
    [[ -n "$WORKER_USER" ]] && user_prefix="${WORKER_USER}@"
    ssh -o StrictHostKeyChecking=no "${user_prefix}$WORKER_IP" "$@"
}

echo "Stopping $CONTAINER_NAME on worker ($WORKER_IP)..."
ssh_cmd "docker rm -f $CONTAINER_NAME 2>/dev/null && echo '  Worker: stopped.' || echo '  Worker: not running.'"

echo "Stopping $CONTAINER_NAME on head..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null && echo "  Head: stopped." || echo "  Head: not running."

if $STOP_NFS; then
    echo "Stopping NFS share ($NFS_CONTAINER) on head..."
    echo "  (kernel NFS in Docker can ignore SIGKILL if rpcbind is in D-state; Ctrl-C and reboot if this hangs)"
    if timeout 15 docker rm -f "$NFS_CONTAINER" >/dev/null 2>&1; then
        echo "  NFS server: stopped."
    else
        echo "  NFS server: still running (could not kill). Leave it — start.sh will reuse it."
    fi
    echo "Removing worker NFS volume ($NFS_VOLUME)..."
    ssh_cmd "docker volume rm $NFS_VOLUME 2>/dev/null && echo '  Worker volume: removed.' || echo '  Worker volume: not present.'"
fi

echo "Done."
