#!/usr/bin/env bash
# check-weights.sh — Verify the checkpoint on the head and on the worker.
#
# Default (NFS_SHARE=false): the worker keeps its own copy (HF_HOME / WORKER_HF_HOME aware).
# NFS_SHARE=true: the worker has no local copy and is checked through the NFS volume.
#
# Usage:
#   ./check-weights.sh          # presence + size on both nodes (fast)
#   ./check-weights.sh --verify # per-file SHA-256 against the HF manifest
#
# --verify streams every shard and compares its SHA-256 with the checksum the
# Hugging Face API reports for the repo. A size-only check does not catch a
# shard corrupted mid-download that kept roughly the wrong size (see issue #30);
# hashing does. It fetches the manifest once on the head, saves it locally, and
# ships it to the worker so both nodes validate against the same manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DO_VERIFY=false
if [[ "${1:-}" == "--verify" ]]; then
    DO_VERIFY=true
elif [[ -n "${1:-}" ]]; then
    echo "Usage: $0 [--verify]" >&2
    exit 2
fi

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found."
    exit 1
fi

source .env

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

WORKER_USER="${WORKER_USER:-}"
WORKER_IP="${WORKER_IP:?WORKER_IP not set in .env}"
MODEL_ID="${MODEL_ID:?MODEL_ID not set in .env}"
IFACE="${IFACE:?IFACE not set in .env}"
NFS_SHARE="${NFS_SHARE:-false}"
NFS_SERVER_IP="${NFS_SERVER_IP:-}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"
ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"
MODEL_REL="hub/models--${ORG}--${NAME}"

ssh_worker() {
    local user_prefix=""
    [[ -n "$WORKER_USER" ]] && user_prefix="${WORKER_USER}@"
    ssh -o StrictHostKeyChecking=no "${user_prefix}$WORKER_IP" "$@"
}

REMOTE_HOME=$(ssh_worker "echo \"\$HOME\"")
if [[ -n "${WORKER_HF_HOME:-}" ]]; then
    REMOTE_HF="$WORKER_HF_HOME"
elif [[ "$HF_CACHE_DIR" == "$HOME" || "$HF_CACHE_DIR" == "$HOME/"* ]]; then
    REMOTE_HF="${REMOTE_HOME}${HF_CACHE_DIR#"$HOME"}"
else
    REMOTE_HF="$HF_CACHE_DIR"
fi
WORKER_MODEL_PATH="${REMOTE_HF}/hub/models--${ORG}--${NAME}"

check_node() {
    local label="$1"
    local path="$2"
    local run_cmd="${3:-local}"
    local size shards

    if [[ "$run_cmd" == "local" ]]; then
        if [[ -d "$path" ]]; then
            size=$(du -sh "$path" 2>/dev/null | cut -f1)
            shards=$(find "$path" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | wc -l)
            echo "  $label ✅  $path"
            echo "         Size: $size | Shards: $shards"
            if $DO_VERIFY; then
                echo "         Verifying SHA-256 against the HF manifest ..."
                if ! python3 "$SCRIPT_DIR/verify-weights.py" --path "$path" --repo "$MODEL_ID" \
                        --manifest "$VERIFY_MANIFEST"; then
                    echo "  $label ❌  SHA-256 mismatch in $path"
                    return 1
                fi
            fi
            return 0
        else
            echo "  $label ❌  $path — NOT FOUND"
            return 1
        fi
    else
        # Remote check via SSH
        if ssh_worker "test -d '$path'" 2>/dev/null; then
            size=$(ssh_worker "du -sh '$path' 2>/dev/null" | cut -f1)
            shards=$(ssh_worker "find '$path' -name '*.safetensors' -o -name '*.bin' 2>/dev/null | wc -l")
            echo "  $label ✅  $path"
            echo "         Size: $size | Shards: $shards"
            if $DO_VERIFY; then
                echo "         Verifying SHA-256 against the HF manifest ..."
                # The worker cache does not have the repo checkout, so ship the
                # verifier (stdlib-only, single file) and the manifest there.
                scp -q "$SCRIPT_DIR/verify-weights.py" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/verify-weights.py"
                scp -q "$VERIFY_MANIFEST" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/verify-manifest.json"
                if ! ssh_cmd "python3 /tmp/verify-weights.py --path '$path' --repo '$MODEL_ID' --manifest /tmp/verify-manifest.json"; then
                    echo "  $label ❌  SHA-256 mismatch in $path"
                    return 1
                fi
            fi
            return 0
        else
            echo "  $label ❌  $path — NOT FOUND"
            return 1
        fi
    fi
}

# shellcheck source=files/nfs-share.sh
source "$SCRIPT_DIR/files/nfs-share.sh"

echo "Checking weights for: $MODEL_ID"
echo "Head cache:   $MODEL_PATH"
echo "Worker cache: $WORKER_MODEL_PATH"
echo ""

# Export for verify-weights.py, which resolves the default model path from the
# environment (same convention as start.sh).
export HF_HOME="$HF_CACHE_DIR"
export HF_TOKEN="${HF_TOKEN:-}"
VERIFY_MANIFEST=""
if $DO_VERIFY; then
    VERIFY_MANIFEST="${TMPDIR:-/tmp}/verify-manifest.json"
    echo "Fetching manifest for $MODEL_ID ..."
    if ! python3 "$SCRIPT_DIR/verify-weights.py" --repo "$MODEL_ID" \
            --save-manifest "$VERIFY_MANIFEST" --fetch-only --quiet; then
        echo "ERROR: could not fetch manifest for $MODEL_ID" >&2
        exit 1
    fi
fi

HEAD_OK=false
WORKER_OK=false

check_node "HEAD  ($HEAD_IP)" "$MODEL_PATH" "local" && HEAD_OK=true

if [[ "$NFS_SHARE" == "true" ]]; then
    # Worker keeps no local copy; verify through the NFS volume instead.
    if docker ps --format '{{.Names}}' | grep -qx "$NFS_CONTAINER"; then
        nfs_detect_server_ip
        nfs_ensure_worker_volume
        if nfs_worker_has_model "$MODEL_REL"; then
            echo "  WORKER ($WORKER_IP) ✅  nfs://${NFS_SERVER_IP}/$MODEL_REL  (volume $NFS_VOLUME)"
            WORKER_OK=true
        else
            echo "  WORKER ($WORKER_IP) ❌  NFS volume $NFS_VOLUME is up but $MODEL_REL is missing"
        fi
    else
        echo "  WORKER ($WORKER_IP) ❌  NFS server ($NFS_CONTAINER) is not running on head"
        echo "         → Run: ./start.sh --no-launch --nfs   (exports head cache over NFS, no vLLM)"
    fi
else
    check_node "WORKER ($WORKER_IP)" "$WORKER_MODEL_PATH" "ssh" && WORKER_OK=true
fi

echo ""

if $HEAD_OK && $WORKER_OK; then
    if [[ "$NFS_SHARE" == "true" ]]; then
        echo "✅ Weights on head, visible to worker over NFS (no local worker copy required)."
    else
        echo "✅ Weights present on both nodes."
    fi
    exit 0
else
    echo "❌ Weights not available on both sides."
    [[ "$HEAD_OK" == "false" ]] && echo "   → Run: ./start.sh (no --no-download) to fetch to head"
    if [[ "$WORKER_OK" == "false" ]]; then
        if [[ "$NFS_SHARE" == "true" ]]; then
            echo "   → Run: ./start.sh --no-launch --nfs to export the head cache over NFS"
        else
            echo "   → Run: ./start.sh --no-launch to rsync the checkpoint to the worker"
        fi
    fi
    exit 1
fi
