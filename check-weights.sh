#!/usr/bin/env bash
# check-weights.sh — Verify model weights exist on both head and worker nodes.
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

REMOTE_HOME=$(ssh_cmd "echo \"\$HOME\"")
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
        if ssh_cmd "test -d '$path'" 2>/dev/null; then
            size=$(ssh_cmd "du -sh '$path' 2>/dev/null" | cut -f1)
            shards=$(ssh_cmd "find '$path' -name '*.safetensors' -o -name '*.bin' 2>/dev/null | wc -l")
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
check_node "WORKER ($WORKER_IP)" "$WORKER_MODEL_PATH" "ssh" && WORKER_OK=true

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
