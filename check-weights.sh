#!/usr/bin/env bash
# check-weights.sh — Verify the checkpoint on the head and on the worker.
#
# Default (NFS_SHARE=false): the worker keeps its own copy (HF_HOME / WORKER_HF_HOME aware).
# NFS_SHARE=true: the worker has no local copy and is checked through the NFS volume.
#
# Usage:
#   ./check-weights.sh                    # presence + size on both nodes (fast)
#   ./check-weights.sh --verify           # per-file SHA-256 against the HF manifest
#   ./check-weights.sh --dry-run          # plan --verify without hashing or scp
#   ./check-weights.sh --manifest FILE    # verify against a saved manifest (no API call)
#
# --verify streams every shard and compares its content hash with the manifest
# the Hugging Face API reports for the repo (SHA-256 for LFS shards, git blob
# SHA-1 for plain files). A size-only check does not catch a
# shard corrupted mid-download that kept roughly the wrong size (see issue #30);
# hashing does. It fetches the manifest once on the head, saves it locally, and
# ships it to the worker so both nodes validate against the same manifest.
#
# --dry-run (alias -n) does the same planning and the cheap presence/size checks
# on both nodes, but skips the hashing pass and the scp of the verifier to the
# worker. It is safe to run while the cluster is busy: no 135 GB reads, no
# copies, only a manifest fetch and stat calls.
#
# --manifest FILE verifies against a manifest saved earlier
# (`python3 verify-weights.py --repo ID --save-manifest FILE --fetch-only`) instead
# of calling the Hugging Face API, so nodes without direct access to huggingface.co
# can still verify. It implies --verify and combines with --dry-run. HF_API_BASE
# in .env points the fetch at a mirror.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DO_VERIFY=false
DO_DRY_RUN=false
SAVED_MANIFEST=""
usage() {
    echo "Usage: $0 [--verify|--dry-run] [--manifest FILE]" >&2
    exit 2
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify)
            DO_VERIFY=true
            ;;
        --dry-run|-n)
            DO_VERIFY=true
            DO_DRY_RUN=true
            ;;
        --manifest)
            [[ $# -ge 2 ]] || usage
            DO_VERIFY=true
            SAVED_MANIFEST="$2"
            shift
            ;;
        --manifest=*)
            DO_VERIFY=true
            SAVED_MANIFEST="${1#--manifest=}"
            ;;
        *)
            usage
            ;;
    esac
    shift
done

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found."
    exit 1
fi

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

_CLI_ABLIT="${ABLIT:-}"
_CLI_OVERRIDE_MODEL_ID="${OVERRIDE_MODEL_ID:-}"
_CLI_FP8_DENSE="${FP8_DENSE:-}"
source .env
[[ -n "$_CLI_ABLIT" ]] && ABLIT="$_CLI_ABLIT"
ABLIT="${ABLIT:-0}"
[[ "$ABLIT" == "0" || "$ABLIT" == "1" ]] || err "ABLIT must be 0 or 1 (got: '$ABLIT')"
[[ -n "$_CLI_OVERRIDE_MODEL_ID" ]] && OVERRIDE_MODEL_ID="$_CLI_OVERRIDE_MODEL_ID"
[[ -n "$_CLI_FP8_DENSE" ]] && FP8_DENSE="$_CLI_FP8_DENSE"

WORKER_USER="${WORKER_USER:-}"
WORKER_IP="${WORKER_IP:?WORKER_IP not set in .env}"
MODEL_ID="${MODEL_ID:?MODEL_ID not set in .env}"
ABLIT_MODEL_ID="drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47"
FP8_DENSE="${FP8_DENSE:-false}"
FP8_DENSE_MODEL_ID="${FP8_DENSE_MODEL_ID:-MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense}"
if [[ -n "${OVERRIDE_MODEL_ID:-}" ]]; then
    MODEL_ID="$OVERRIDE_MODEL_ID"
fi
if [[ "$FP8_DENSE" == "true" ]]; then
    MODEL_ID="$FP8_DENSE_MODEL_ID"
fi
if [[ "$ABLIT" == "1" ]]; then
    if [[ "$FP8_DENSE" == "true" ]]; then
        warn "ABLIT=1 ignored for checkpoint selection: FP8_DENSE=true (MODEL_ID=$MODEL_ID)"
    elif [[ -n "${OVERRIDE_MODEL_ID:-}" && "$MODEL_ID" != "$ABLIT_MODEL_ID" ]]; then
        warn "ABLIT=1 ignored for checkpoint selection: OVERRIDE_MODEL_ID=$MODEL_ID"
    else
        MODEL_ID="$ABLIT_MODEL_ID"
    fi
fi
IFACE="${IFACE:?IFACE not set in .env}"
NFS_SHARE="${NFS_SHARE:-false}"
NFS_SERVER_IP="${NFS_SERVER_IP:-}"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"
ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
MODEL_REL="hub/models--${ORG}--${NAME}"

# Resolve the local model directory exactly like start.sh does: the standard
# hub path (blobs/refs/snapshots) first, then the `hf path` CLI if the hub dir
# does not exist. check-weights.sh must agree with the launcher about where
# the weights live, or a real --verify run would hash nothing. It deliberately
# does NOT guess beyond that: walking up the tree looking for any config.json
# can resolve to an unrelated checkpoint, and a verifier that checks the wrong
# tree is worse than one that reports it missing.
resolve_model_dir() {
    local hub="$HUB_PATH/models--${ORG}--${NAME}"
    if [[ -d "$hub" && -n "$(ls -A "$hub" 2>/dev/null)" ]]; then
        echo "$hub"
        return 0
    fi
    local guess=""
    for tool_cmd in "hf path" "huggingface-cli path"; do
        local first_word="${tool_cmd%% *}"
        if command -v "$first_word" &>/dev/null; then
            guess=$($tool_cmd "$MODEL_ID" 2>/dev/null || true)
            [[ -n "$guess" && -d "$guess" ]] && break
            guess=""
        fi
    done
    if [[ -n "$guess" && -d "$guess" ]]; then
        case "$guess" in
            *"/models--${ORG}--${NAME}"*)
                guess="${guess%%/models--${ORG}--${NAME}*}/models--${ORG}--${NAME}"
                ;;
        esac
        echo "$guess"
        return 0
    fi
    return 1
}

MODEL_PATH="$(resolve_model_dir 2>/dev/null || echo "$HUB_PATH/models--${ORG}--${NAME}")"

ssh_worker() {
    local user_prefix=""
    [[ -n "$WORKER_USER" ]] && user_prefix="${WORKER_USER}@"
    # Fail fast when the worker is unreachable (e.g. cluster in use elsewhere)
    # instead of hanging on the default TCP timeout.
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${user_prefix}$WORKER_IP" "$@"
}

# Resolve the worker's model directory. Mirrored clusters often use different
# absolute roots per node (e.g. /home/user/models-gigabyte on head vs
# /home/user/models on worker) with the same layout, so an explicit
# WORKER_MODEL_PATH override in .env is the reliable way to pin it; without it
# we mirror the head's HF_HOME under the worker's $HOME. The worker is
# expected to keep the standard hub path there. No guessing beyond that, for
# the same reason as resolve_model_dir above: check_node reports a missing
# worker copy as NOT FOUND, which is honest, while a guessed dir would not be.
WORKER_MODEL_PATH="${WORKER_MODEL_PATH:-}"
if [[ -z "$WORKER_MODEL_PATH" ]]; then
    REMOTE_HOME="$HOME"
    REMOTE_HOME=$(ssh_worker "echo \"\$HOME\"" 2>/dev/null || echo "$HOME")
    if [[ -n "${WORKER_HF_HOME:-}" ]]; then
        REMOTE_HF="$WORKER_HF_HOME"
    elif [[ "$HF_CACHE_DIR" == "$HOME" || "$HF_CACHE_DIR" == "$HOME/"* ]]; then
        REMOTE_HF="${REMOTE_HOME}${HF_CACHE_DIR#"$HOME"}"
    else
        REMOTE_HF="$HF_CACHE_DIR"
    fi
    WORKER_MODEL_PATH="$REMOTE_HF/hub/models--${ORG}--${NAME}"
fi

# Translate the verifier's exit status into a message. verify-weights.py
# returns 1 when the weights failed verification and 2 when the check itself
# broke (unresolvable manifest, missing python3, ...). Those are different
# answers: the first means corrupt weights, the second means try again.
# Printing "mismatch" for both would send an operator after weights when the
# tool is what broke.
report_verify_rc() {  # report_verify_rc <rc> <label> <path>
    local vrc="$1" vlabel="$2" vpath="$3"
    if [[ "$vrc" -eq 1 ]]; then
        echo "  $vlabel ❌  verification FAILED for $vpath (does not match the manifest)"
        return 1
    elif [[ "$vrc" -ne 0 ]]; then
        echo "  $vlabel ❌  could not verify $vpath (checker exit $vrc; tool failure, not a weight mismatch)"
        return 1
    fi
    return 0
}

check_node() {
    local label="$1"
    local path="$2"
    local run_cmd="${3:-local}"
    local size shards rc worker_tmp

    if [[ "$run_cmd" == "local" ]]; then
        if [[ -d "$path" ]]; then
            size=$(du -sh "$path" 2>/dev/null | cut -f1)
            shards=$(find "$path" -name "*.safetensors" -o -name "*.bin" 2>/dev/null | wc -l)
            echo "  $label ✅  $path"
            echo "         Size: $size | Shards: $shards"
            if $DO_VERIFY; then
                if $DO_DRY_RUN; then
                    echo "         [dry run] presence + size check only (no hashing)"
                    rc=0
                    python3 "$SCRIPT_DIR/verify-weights.py" --path "$path" --repo "$MODEL_ID" \
                            --manifest "$VERIFY_MANIFEST" --dry-run || rc=$?
                    report_verify_rc "$rc" "$label" "$path" || return 1
                else
                    echo "         Verifying content hashes against the HF manifest ..."
                    rc=0
                    python3 "$SCRIPT_DIR/verify-weights.py" --path "$path" --repo "$MODEL_ID" \
                            --manifest "$VERIFY_MANIFEST" || rc=$?
                    report_verify_rc "$rc" "$label" "$path" || return 1
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
                if $DO_DRY_RUN; then
                    echo "         [dry run] would stage verifier + manifest and hash $path"
                else
                    echo "         Verifying content hashes against the HF manifest ..."
                    # The worker cache does not have the repo checkout, so ship the
                    # verifier (stdlib-only, single file) and the manifest there.
                    # A fresh private temp dir, not a fixed /tmp path: predictable
                    # names are a symlink-attack surface and leak between runs.
                    worker_tmp=""
                    if ! worker_tmp=$(ssh_worker "mktemp -d /tmp/verify-weights.XXXXXX" 2>/dev/null); then
                        echo "  $label ❌  could not stage verifier on worker"
                        return 1
                    fi
                    scp -q "$SCRIPT_DIR/verify-weights.py" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${worker_tmp}/verify-weights.py"
                    scp -q "$VERIFY_MANIFEST" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${worker_tmp}/manifest.json"
                    rc=0
                    ssh_worker "python3 '${worker_tmp}/verify-weights.py' --path '$path' --repo '$MODEL_ID' --manifest '${worker_tmp}/manifest.json'" || rc=$?
                    ssh_worker "rm -rf '$worker_tmp'" 2>/dev/null || true
                    report_verify_rc "$rc" "$label" "$path" || return 1
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
if [[ "$ABLIT" == "1" && "$MODEL_ID" == "$ABLIT_MODEL_ID" ]]; then
    echo "ABLIT=1 (gated Keys house QSA L3-47)"
fi
echo "Head cache:   $MODEL_PATH"
echo "Worker cache: $WORKER_MODEL_PATH"
echo ""

# Export for verify-weights.py, which resolves the default model path from the
# environment (same convention as start.sh).
export HF_HOME="$HF_CACHE_DIR"
export HF_TOKEN="${HF_TOKEN:-}"
# A mirror set in .env must reach the verifier too, on both nodes.
[[ -n "${HF_API_BASE:-}" ]] && export HF_API_BASE
VERIFY_MANIFEST=""
# Temp manifest created by this run. Tracked separately from VERIFY_MANIFEST
# so the EXIT trap only removes what we created, never a user --manifest file.
VERIFY_MANIFEST_TMP=""
cleanup_verify_manifest() {
    if [[ -n "$VERIFY_MANIFEST_TMP" ]]; then
        rm -f "$VERIFY_MANIFEST_TMP"
    fi
}
trap cleanup_verify_manifest EXIT
if $DO_VERIFY; then
    if [[ -n "$SAVED_MANIFEST" ]]; then
        if [[ ! -f "$SAVED_MANIFEST" ]]; then
            echo "ERROR: manifest not found: $SAVED_MANIFEST" >&2
            exit 1
        fi
        VERIFY_MANIFEST="$SAVED_MANIFEST"
        echo "Using saved manifest: $VERIFY_MANIFEST"
    else
        VERIFY_MANIFEST="$(mktemp "${TMPDIR:-/tmp}/verify-manifest.XXXXXX.json")"
        VERIFY_MANIFEST_TMP="$VERIFY_MANIFEST"
        echo "Fetching manifest for $MODEL_ID ..."
        if ! python3 "$SCRIPT_DIR/verify-weights.py" --repo "$MODEL_ID" \
                --save-manifest "$VERIFY_MANIFEST" --fetch-only --quiet; then
            echo "ERROR: could not fetch manifest for $MODEL_ID" >&2
            echo "       Offline or behind a proxy? Save it once where the API is reachable:" >&2
            echo "       python3 verify-weights.py --repo $MODEL_ID --save-manifest m.json --fetch-only" >&2
            echo "       then run: $0 --verify --manifest m.json" >&2
            exit 1
        fi
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
