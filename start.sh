#!/usr/bin/env bash
# ============================================================================
# start.sh — Serve RadixArk/Qwen3.8-Flash-Next-NVFP4 across a 2-node
#             DGX Spark cluster with vLLM TP2+EP+MTP3.
#
# Based on: https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark
#
# Weight distribution (default: rsync). Each node keeps its own copy of the
# checkpoint in ~/.cache/huggingface; the worker copy is rsync'd from the head
# once and reused. Set NFS_SHARE=true (or pass --nfs) to instead export the head
# cache over NFS on the ConnectX link, so the worker keeps no local copy — see
# "NFS weight sharing (optional)" in the README for the trade-offs.
#
# Usage:
#   ./download.sh             # fetch NVFP4 onto the head (optional; start.sh can too)
#   ./start.sh                # download on head if needed → sync to worker → patch → launch
#   ./start.sh --no-download  # skip download (weights already cached on head)
#   ./start.sh --no-launch    # download + sync only, don't start server
#   ./start.sh --launch       # skip download/sync; apply patch + launch
#   ./start.sh --nfs          # distribute weights over NFS instead of rsync
#   ./start.sh --no-nfs       # force rsync distribution (overrides NFS_SHARE=true)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Copy .env.sample to .env and edit it."
    echo "  cp .env.sample .env"
    exit 1
fi

# shellcheck source=.env
source .env

# Validate required variables
for var in HEAD_IP WORKER_IP IFACE IB_HCA IB_GID_INDEX MODEL_ID \
           MAX_MODEL_LEN GPU_MEMORY_UTILIZATION MAX_NUM_SEQS \
           MAX_NUM_BATCHED_TOKENS PORT TENSOR_PARALLEL_SIZE IMAGE \
           MASTER_PORT; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Required variable $var is not set in .env"
        exit 1
    fi
done

WORKER_USER="${WORKER_USER:-}"
# Numeric sanity: the YaRN guard below does an arithmetic comparison on MAX_MODEL_LEN
if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
    exit 1
fi
# Per-node overrides — the two nodes may be cross-wired (head port f1 ↔ worker port f0),
# so the connected interface/HCA can have different names on each node.
WORKER_IFACE="${WORKER_IFACE:-$IFACE}"
WORKER_IB_HCA="${WORKER_IB_HCA:-$IB_HCA}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-true}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"   # fp8 needs files/patch_qsa_fp8_kv.py, applied automatically in step 4f; auto = bf16
PLE_OFFLOAD="${PLE_OFFLOAD:-false}"
# Vision MLP intermediate_size=4304 is not divisible by 16 after TP split (4304/2=2152).
# NVFP4 kernels require input features % 16 == 0, so replicate the encoder on each GPU.
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
HF_TOKEN="${HF_TOKEN:-}"
# Weight distribution. false (default) = each node keeps its own ~/.cache/huggingface
# copy, worker seeded by rsync from the head. true = head exports its cache over NFS
# on ConnectX and the worker mounts it read-only, keeping no local copy.
NFS_SHARE="${NFS_SHARE:-false}"
# Optional: head ConnectX address used as the NFS server (auto-detected from IFACE).
NFS_SERVER_IP="${NFS_SERVER_IP:-}"
# Optional overrides from start-fp8.sh (applied after .env so FP8 can share
# the same cluster config). Weights still live on the head and are NFS-mounted.
if [[ -n "${OVERRIDE_MODEL_ID:-}" ]]; then
    MODEL_ID="$OVERRIDE_MODEL_ID"
fi
if [[ -n "${OVERRIDE_SERVED_MODEL_NAME:-}" ]]; then
    SERVED_MODEL_NAME="$OVERRIDE_SERVED_MODEL_NAME"
fi
if [[ -n "${OVERRIDE_MAX_MODEL_LEN:-}" ]]; then
    MAX_MODEL_LEN="$OVERRIDE_MAX_MODEL_LEN"
fi
if [[ -n "${OVERRIDE_YARN_ENABLE:-}" ]]; then
    YARN_ENABLE="$OVERRIDE_YARN_ENABLE"
fi
SKIP_PLE_PATCH="${SKIP_PLE_PATCH:-false}"
# FP8-dense hybrid checkpoint (NVFP4 experts + FP8 per-channel dense projections,
# built by files/fp8dense/make_fp8_dense_checkpoint.py). Needs the vLLM overlay
# patches in files/overlay (bind-mounted, no image rebuild).
FP8_DENSE="${FP8_DENSE:-false}"
FP8_DENSE_MODEL_ID="${FP8_DENSE_MODEL_ID:-MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense}"
if [[ "$FP8_DENSE" == "true" ]]; then
    MODEL_ID="$FP8_DENSE_MODEL_ID"
    DO_DOWNLOAD_DEFAULT=false   # local-only checkpoint, never on the Hub
fi
# Reduced-vocabulary MTP drafting: path to a token-id list from
# files/build_draft_vocab.py, or empty to draft over the full 248,320 vocabulary.
MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-}"
# QSA Triton launch profile: stock | gb10 | path to JSON from files/qsa_gb10/bench_qsa_kernels.py
QSA_PROFILE="${QSA_PROFILE:-stock}"
# Refuse to launch when another process already holds the GPU (both nodes).
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"

# YaRN only makes sense ABOVE the native 262144 context. At or below native,
# rope scaling degrades quality for zero benefit — force it off.
YARN_ENABLE="${YARN_ENABLE:-false}"
if [[ "$YARN_ENABLE" == "true" && "$MAX_MODEL_LEN" -le 262144 ]]; then
    echo "NOTE: MAX_MODEL_LEN=$MAX_MODEL_LEN <= native 262144 — YaRN force-disabled."
    YARN_ENABLE=false
fi

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
DO_DOWNLOAD="${DO_DOWNLOAD_DEFAULT:-true}"
DO_LAUNCH=true
DO_SYNC=true

for arg in "$@"; do
    case "$arg" in
        --no-download)  DO_DOWNLOAD=false ;;
        --no-launch)    DO_LAUNCH=false ;;
        --launch)       DO_DOWNLOAD=false; DO_SYNC=false ;;
        --nfs)          NFS_SHARE=true ;;
        --no-nfs)       NFS_SHARE=false ;;
        -h|--help)
            echo "Usage: $0 [--no-download] [--no-launch] [--launch] [--nfs|--no-nfs]"
            echo ""
            echo "  (default)      Download weights on head, rsync to worker, apply patch, launch"
            echo "  --no-download  Skip HF download (weights already cached on head)"
            echo "  --no-launch    Download + sync weights only, don't start vLLM"
            echo "  --launch       Skip download + sync; apply patch and launch"
            echo "  --nfs          Share the head cache over NFS instead of rsync (no worker copy)"
            echo "  --no-nfs       Force rsync distribution even if NFS_SHARE=true in .env"
            exit 0
            ;;
        *)
            err "Unknown argument: $arg (try --help)"
            ;;
    esac
done

# shellcheck source=files/nfs-share.sh
source "$SCRIPT_DIR/files/nfs-share.sh"

# ---------------------------------------------------------------------------
# Worker SSH helper
# ---------------------------------------------------------------------------
ssh_worker() {
    local user_prefix=""
    if [[ -n "$WORKER_USER" ]]; then
        user_prefix="${WORKER_USER}@"
    fi
    ssh -o StrictHostKeyChecking=no "${user_prefix}${WORKER_IP}" "$@"
}

# ---------------------------------------------------------------------------
# 1. Download the model weights (head node)
# ---------------------------------------------------------------------------
if $DO_DOWNLOAD; then
    "$SCRIPT_DIR/download.sh" "$MODEL_ID"
fi

# ---------------------------------------------------------------------------
# 2. Resolve local cache path
# ---------------------------------------------------------------------------
info "=== Step 2: Resolve cache path ==="

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_PATH="$HF_CACHE_DIR/hub"

ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
# Hub repo dir (blobs/refs/snapshots). Prefer this over `hf path`, which is not
# a real CLI command and (when it exists) often returns a snapshot subdir.
MODEL_DIR="$HUB_PATH/models--${ORG}--${NAME}"

if [[ ! -d "$MODEL_DIR" ]]; then
    local_guess=""
    for tool_cmd in "uvx hf path" "hf path" "huggingface-cli path"; do
        first_word="${tool_cmd%% *}"
        if command -v "$first_word" &>/dev/null; then
            local_guess=$($tool_cmd "$MODEL_ID" 2>/dev/null || true)
            [[ -n "$local_guess" && -d "$local_guess" ]] && break
            local_guess=""
        fi
    done
    if [[ -n "$local_guess" && -d "$local_guess" ]]; then
        # Walk up to models--org--name if the CLI pointed at a snapshot.
        case "$local_guess" in
            *"/models--${ORG}--${NAME}"*)
                MODEL_DIR="${local_guess%%/models--${ORG}--${NAME}*}/models--${ORG}--${NAME}"
                ;;
            *)
                MODEL_DIR="$local_guess"
                ;;
        esac
    fi
fi

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
    err "Could not resolve local cache path for $MODEL_ID under $HUB_PATH"
fi
case "$MODEL_DIR" in
    "$HF_CACHE_DIR"|"$HF_CACHE_DIR"/*) ;;
    *) err "snapshot ${MODEL_DIR} is not under HF_HOME=${HF_CACHE_DIR}" ;;
esac

ok "Model cache: $MODEL_DIR"

ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
HEAD_MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"
if [[ ! -d "$HEAD_MODEL_PATH" ]]; then
    err "Could not find HF repo cache at $HEAD_MODEL_PATH (resolved snapshot: ${MODEL_DIR:-none})"
fi

# Checkpoints disagree about declaring text_config.ple_embedding_dtype, which is
# what the patched ple_layer.py dispatches on (nvidia/... omits it and declares
# the FP8 PLE table only in quantization_config.config_groups). Recover it from
# the checkpoint and feed it back via --hf-overrides below. Empty = already
# declared, or no quantized PLE table.
PLE_CONFIG_DIR="$MODEL_DIR"
if [[ ! -f "$PLE_CONFIG_DIR/config.json" ]]; then
    PLE_CONFIG_DIR=$(ls -d "$HEAD_MODEL_PATH"/snapshots/*/ 2>/dev/null | head -1)
fi
PLE_EMBEDDING_DTYPE="${PLE_EMBEDDING_DTYPE:-}"
if [[ -z "$PLE_EMBEDDING_DTYPE" && -f "$PLE_CONFIG_DIR/config.json" ]]; then
    PLE_EMBEDDING_DTYPE=$(python3 "$SCRIPT_DIR/files/detect_ple_dtype.py" "$PLE_CONFIG_DIR")
fi
if [[ -n "$PLE_EMBEDDING_DTYPE" ]]; then
    ok "PLE table dtype not declared by checkpoint — overriding to $PLE_EMBEDDING_DTYPE"
fi
SNAPSHOT_SHA=$(basename "${PLE_CONFIG_DIR%/}")

# Resolve the worker's HF cache. It mirrors the head's absolute path unless that
# path lives under $HOME (then the prefix is rewritten to the worker's $HOME), or
# WORKER_HF_HOME overrides it outright.
REMOTE_HOME=$(ssh_worker "echo \"\$HOME\"")
[[ -n "$REMOTE_HOME" ]] || err "Could not resolve \$HOME on worker ($WORKER_IP). Check SSH / WORKER_USER."
if [[ -n "${WORKER_HF_HOME:-}" ]]; then
    REMOTE_HF="$WORKER_HF_HOME"
elif [[ "$HF_CACHE_DIR" == "$HOME" || "$HF_CACHE_DIR" == "$HOME/"* ]]; then
    REMOTE_HF="${REMOTE_HOME}${HF_CACHE_DIR#"$HOME"}"
else
    REMOTE_HF="$HF_CACHE_DIR"
fi
info "Head HF cache:   $HF_CACHE_DIR"
info "Worker HF cache: $REMOTE_HF"

# ---------------------------------------------------------------------------
# 3. Distribute weights to the worker.
#    Default: rsync into the worker's own HF cache (worker keeps a local copy).
#    NFS_SHARE=true: export the head cache over NFS (ConnectX); no worker copy.
# ---------------------------------------------------------------------------
REMOTE_HUB="${REMOTE_HF}/hub"

if [[ "$NFS_SHARE" == "true" ]]; then
    info "=== Step 3: NFS-share weights from head ==="
    nfs_ensure_server
else
    info "=== Step 3: Sync weights to worker ($WORKER_IP) ==="
    if ! $DO_SYNC; then
        info "  --launch: skipping sync (assuming worker cache is current)"
    elif ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>/dev/null; then
        ok "Worker already has models--${ORG}--${NAME} — skipping rsync."
        info "  (delete $REMOTE_HUB/models--${ORG}--${NAME} on the worker to force a re-sync)"
    else
        [[ -d "$HEAD_MODEL_PATH" ]] || err "HEAD ($HEAD_IP): $HEAD_MODEL_PATH — NOT FOUND. Nothing to sync; run ./download.sh first."
        info "  Worker cache: $REMOTE_HUB"
        warn "  This copies the full checkpoint over the network and needs the same"
        warn "  free space on the worker. NFS_SHARE=true avoids both — see the README."
        ssh_worker "mkdir -p '$REMOTE_HUB'"
        rsync -av --progress --partial \
            "${HEAD_MODEL_PATH}/" \
            "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${REMOTE_HUB}/models--${ORG}--${NAME}/"
        ok "Rsync complete."
    fi
fi

# ---------------------------------------------------------------------------
# 4. Verify weights on head (the worker is checked once its cache is in place)
# ---------------------------------------------------------------------------
info "=== Step 4: Verify weights on head ==="
if [[ -d "$HEAD_MODEL_PATH" ]]; then
    HEAD_SIZE=$(du -sh "$HEAD_MODEL_PATH" 2>/dev/null | cut -f1)
    ok "HEAD  ($HEAD_IP): $HEAD_MODEL_PATH ($HEAD_SIZE)"
else
    err "HEAD  ($HEAD_IP): $HEAD_MODEL_PATH — NOT FOUND"
fi

if ! $DO_LAUNCH; then
    if [[ "$NFS_SHARE" == "true" ]]; then
        nfs_ensure_worker_volume
        if nfs_worker_has_model "hub/models--${ORG}--${NAME}"; then
            ok "WORKER ($WORKER_IP): nfs $NFS_VOLUME → hub/models--${ORG}--${NAME}"
        else
            err "WORKER cannot see hub/models--${ORG}--${NAME} over NFS. Check: docker logs $NFS_CONTAINER"
        fi
    elif ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>/dev/null; then
        WORKER_SIZE=$(ssh_worker "du -sh '$REMOTE_HUB/models--${ORG}--${NAME}' 2>/dev/null | cut -f1" || true)
        ok "WORKER ($WORKER_IP): $REMOTE_HUB/models--${ORG}--${NAME} (${WORKER_SIZE:-?})"
    else
        err "WORKER ($WORKER_IP): $REMOTE_HUB/models--${ORG}--${NAME} — NOT FOUND. Re-run without --launch to sync."
    fi
fi

# ---------------------------------------------------------------------------
# 4b. Preflight: both GPUs must be free (another vLLM/SGLang tenant would OOM us
#     ten minutes into weight loading).
# ---------------------------------------------------------------------------
gpu_tenants() {  # prints "pid,name,mem" lines for compute apps, empty if idle
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^$/d'
}
if $DO_LAUNCH && [[ "$REQUIRE_IDLE_GPU" == "true" ]]; then
    info "=== Step 4b: GPU preflight ==="
    HEAD_TENANTS=$(gpu_tenants || true)
    WORKER_TENANTS=$(ssh_worker "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed '/^\$/d'" || true)
    if [[ -n "$HEAD_TENANTS" || -n "$WORKER_TENANTS" ]]; then
        echo "HEAD:   ${HEAD_TENANTS:-idle}"
        echo "WORKER: ${WORKER_TENANTS:-idle}"
        err "GPU is in use on at least one node (set REQUIRE_IDLE_GPU=false to override)."
    fi
    ok "Both GPUs idle."
fi

# ---------------------------------------------------------------------------
# 4c. vLLM overlay patches (bind-mounted files, no image rebuild).
#     files/overlay/*.py   -> FP8-dense loader support (only with FP8_DENSE=true)
#     files/qsa_gb10/qsa.py -> QSA launch-profile override (only with QSA_PROFILE != stock)
# ---------------------------------------------------------------------------
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
OVERLAY_MOUNTS=()        # head-side "-v host:container:ro"
OVERLAY_FILES=()         # host files to scp to the worker (/tmp/vllm-overlay/<name>)
OVERLAY_ENV=()
add_overlay() {          # add_overlay <host file> <container path>
    [[ -f "$1" ]] || err "overlay file missing: $1"
    # Two overlays on one container path would silently race in docker run, and
    # the worker copies land in a flat /tmp/vllm-overlay keyed by basename, so a
    # basename clash would have one file quietly overwrite the other there.
    for existing in "${OVERLAY_FILES[@]:-}"; do
        if [[ "${existing#*|}" == "$2" ]]; then
            err "overlay conflict on $2: already claimed by ${existing%%|*}, now $1"
        fi
        if [[ "$(basename "${existing%%|*}")" == "$(basename "$1")" ]]; then
            err "overlay basename clash on $(basename "$1"): ${existing%%|*} vs $1
       (worker overlays share a flat /tmp/vllm-overlay directory)"
        fi
    done
    OVERLAY_MOUNTS+=("-v $1:$2:ro")
    OVERLAY_FILES+=("$1|$2")
}
extract_from_image() {   # extract_from_image <container path> <host dest>
    [[ -f "$2" ]] && return 0
    info "  Extracting $(basename "$1") from image..."
    local c; c=$(docker create "$IMAGE" /bin/true)
    docker cp "$c:$1" "$2" >/dev/null
    docker rm "$c" >/dev/null 2>&1
    [[ -f "$2" ]] || err "Failed to extract $1 from image."
}
if $DO_LAUNCH && [[ "$FP8_DENSE" == "true" ]]; then
    info "=== Step 4c: FP8-dense overlay ==="
    OV="$SCRIPT_DIR/files/overlay"
    [[ -f "$OV/modelopt.py" ]] || python3 "$OV/apply_patches.py"
    add_overlay "$OV/modelopt.py"        "$VLLM_PKG/model_executor/layers/quantization/modelopt.py"
    add_overlay "$OV/model.py"           "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/model.py"
    add_overlay "$OV/hyperconnection.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/hyperconnection.py"
    add_overlay "$OV/mtp.py"             "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
    ok "FP8-dense overlay: 4 files"
fi
if $DO_LAUNCH && [[ "$QSA_PROFILE" != "stock" ]]; then
    info "=== Step 4d: QSA profile overlay ($QSA_PROFILE) ==="
    QO="$SCRIPT_DIR/files/qsa_gb10"
    [[ -f "$QO/qsa.py" ]] || python3 "$QO/apply_patch.py"
    add_overlay "$QO/qsa.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
    if [[ -f "$QSA_PROFILE" ]]; then
        add_overlay "$QSA_PROFILE" "/etc/vllm-qsa-profile.json"
        OVERLAY_ENV+=("-e VLLM_QSA_PROFILE_JSON=/etc/vllm-qsa-profile.json")
    else
        OVERLAY_ENV+=("-e VLLM_QSA_PROFILE=$QSA_PROFILE")
    fi
fi

# ---------------------------------------------------------------------------
# 4d. Reduced-vocabulary MTP drafting (FR-Spec style).
#     The drafter owns a full 248,320-row BF16 lm_head that is read once per
#     draft step; slicing it to a frequency-ranked subset is the single largest
#     bandwidth lever in a decode step. Output-safe: a draft outside the subset
#     is rejected at verification, never emitted. See files/patch_mtp_draft_vocab.py.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ -n "$MTP_DRAFT_VOCAB" ]]; then
    info "=== Step 4e: MTP reduced draft vocabulary ==="
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" == "0" ]]; then
        err "MTP_DRAFT_VOCAB is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts."
    fi
    [[ -f "$MTP_DRAFT_VOCAB" ]] || err "MTP_DRAFT_VOCAB file not found: $MTP_DRAFT_VOCAB
       Build one with: python3 files/build_draft_vocab.py <corpus.jsonl> --out draft_vocab.txt --size 65536"
    if [[ "$FP8_DENSE" == "true" ]]; then
        err "MTP_DRAFT_VOCAB and FP8_DENSE both overlay nvidia/mtp.py - pick one."
    fi
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py" \
                       "$SCRIPT_DIR/files/mtp_patched.py.orig"
    python3 "$SCRIPT_DIR/files/patch_mtp_draft_vocab.py"
    add_overlay "$SCRIPT_DIR/files/mtp_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
    add_overlay "$MTP_DRAFT_VOCAB" "/etc/vllm-draft-vocab.txt"
    OVERLAY_ENV+=("-e VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt")
    ok "Draft vocab: $(wc -l < "$MTP_DRAFT_VOCAB") ids from $MTP_DRAFT_VOCAB"
fi

# ---------------------------------------------------------------------------
# 4e. FP8 KV cache. The stock QSA kernels hard-refuse anything but BF16 KV
#     (supported_kv_cache_dtypes = ["auto","bfloat16"]); this teaches them to
#     read an FP8-e4m3 cache with per-tensor scales applied after the dots.
#     A capacity trade, not a free win - see the README before enabling.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ "$KV_CACHE_DTYPE" == fp8* ]]; then
    info "=== Step 4f: FP8 KV cache patch ($KV_CACHE_DTYPE) ==="
    if [[ "$QSA_PROFILE" != "stock" ]]; then
        err "KV_CACHE_DTYPE=$KV_CACHE_DTYPE and QSA_PROFILE=$QSA_PROFILE both overlay ops/qsa.py - pick one."
    fi
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py" \
                       "$SCRIPT_DIR/files/qsa_ops_patched.py.orig"
    extract_from_image "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py" \
                       "$SCRIPT_DIR/files/qsa_nvidia_patched.py.orig"
    python3 "$SCRIPT_DIR/files/patch_qsa_fp8_kv.py"
    add_overlay "$SCRIPT_DIR/files/qsa_ops_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
    add_overlay "$SCRIPT_DIR/files/qsa_nvidia_patched.py" \
                "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py"
    warn "FP8 KV is a quality trade on sparse attention - validate reasoning on your workload."
fi

# ---------------------------------------------------------------------------
# 4d. MTP layer-index alias overlay.
#     vLLM builds the MTP draft layer at the absolute index that continues the
#     main stack (mtp.layers.48 for num_hidden_layers=48) and matches that
#     prefix against quantization_config.quantized_layers by exact string.
#     nvidia/... records only mtp.layers.0, so the lookup misses, the MTP MoE is
#     built unquantized, and its FP8 weight_scale_inv tensors fail to load. We
#     bind-mount a config.json carrying both names (what the known-good
#     local-inference-lab checkpoint ships) — the HF cache is left untouched.
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ -n "${SNAPSHOT_SHA:-}" ]]; then
    info "=== Step 4g: MTP layer-index alias ==="
    CONTAINER_SNAPSHOT="/root/.cache/huggingface/hub/models--${ORG}--${NAME}/snapshots/${SNAPSHOT_SHA}"
    rm -f "$SCRIPT_DIR/files/config_patched.json" \
          "$SCRIPT_DIR/files/hf_quant_config_patched.json"
    PATCHED_FILES=$(python3 "$SCRIPT_DIR/files/patch_checkpoint_config.py" \
        "$PLE_CONFIG_DIR" "$SCRIPT_DIR/files")
    if [[ -z "$PATCHED_FILES" ]]; then
        ok "Checkpoint already declares absolute MTP layer indices"
    else
        # quantized_layers lives in BOTH config.json and the legacy
        # hf_quant_config.json, and vLLM reads the legacy file when present —
        # mounting only config.json leaves the stale mapping in play.
        for cfg_name in $PATCHED_FILES; do
            case "$cfg_name" in
                config.json)         host_file="$SCRIPT_DIR/files/config_patched.json" ;;
                hf_quant_config.json) host_file="$SCRIPT_DIR/files/hf_quant_config_patched.json" ;;
                *) err "unexpected patched config: $cfg_name" ;;
            esac
            add_overlay "$host_file" "$CONTAINER_SNAPSHOT/$cfg_name"
        done
        ok "MTP experts alias added to: $PATCHED_FILES"
    fi

    # Speculative decoding needs the MTP routed experts to be built with a
    # quantization method the mixed-precision dispatch actually implements.
    # ModelOptMixedPrecisionConfig.get_quant_method covers FP8 / NVFP4 /
    # W4A16_NVFP4 / MXFP8 for RoutedExperts and returns None for anything else,
    # which yields a silently *unquantized* MoE that then dies ~7 min into the
    # load. Fail fast here instead.
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        MTP_ALGO=$(python3 "$SCRIPT_DIR/files/patch_checkpoint_config.py" \
            --mtp-moe-algo "$PLE_CONFIG_DIR") && MTP_RC=0 || MTP_RC=$?
        if [[ "$MTP_RC" -eq 3 ]]; then
            err "MTP experts are ${MTP_ALGO}, which this image's mixed-precision MoE
       dispatch cannot build (supports FP8 / NVFP4 / W4A16_NVFP4 / MXFP8).
       Set MTP_NUM_SPECULATIVE_TOKENS=0 in .env to serve without speculative
       decoding, or use a checkpoint whose MTP experts are NVFP4."
        fi
        ok "MTP experts quantization: ${MTP_ALGO:-unquantized} (supported)"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Ensure Docker image on both nodes
# ---------------------------------------------------------------------------
if $DO_LAUNCH; then
    info "=== Step 5: Docker image '$IMAGE' ==="

    if ! docker image inspect "$IMAGE" &>/dev/null; then
        info "Pulling $IMAGE ..."
        docker pull "$IMAGE"
    fi
    ok "Image ready on head."

    LOCAL_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || echo "")
    REMOTE_ID=$(ssh_worker "docker image inspect --format '{{.Id}}' '$IMAGE' 2>/dev/null" || echo "")

    if [[ "$LOCAL_ID" != "$REMOTE_ID" ]]; then
        info "Pulling image on worker..."
        ssh_worker "docker pull '$IMAGE'"
        ok "Image ready on worker."
    else
        ok "Image already on worker."
    fi

    # ---------------------------------------------------------------------------
    # 6. Prepare the PLE patch
    #    Mixed-quant NVFP4 checkpoints declare ple_embedding_dtype in config:
    #      nvfp4          -> packed uint8 PLE table (this checkpoint)
    #      float8_e4m3fn  -> FP8 PLE excluded from the parent ModelOpt config
    #    files/patch_ple_layer.py adds NVFP4 + mixed dispatch (vLLM PR #53899 logic).
    # ---------------------------------------------------------------------------
    PATCHED_PLE="$SCRIPT_DIR/files/ple_layer_patched.py"
    PLE_ORIG="$SCRIPT_DIR/files/ple_layer_patched.py.orig"
    HEAD_PLE_MOUNT=""
    WORKER_PLE_MOUNT=""

    if [[ "$SKIP_PLE_PATCH" == "true" ]]; then
        info "=== Step 6: PLE patch skipped (native FP8 checkpoint) ==="
    else
    info "=== Step 6: Prepare PLE patch ==="

    if [[ ! -f "$PLE_ORIG" ]]; then
        info "Extracting ple_layer.py from image..."
        mkdir -p "$SCRIPT_DIR/files"
        tmp_container=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp_container:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py" "$PLE_ORIG"
        docker rm "$tmp_container" >/dev/null 2>&1
        [[ -f "$PLE_ORIG" ]] || err "Failed to extract ple_layer.py from image. Is the image pulled?"
    fi

    python3 "$SCRIPT_DIR/files/patch_ple_layer.py"
    [[ -f "$PATCHED_PLE" ]] || err "PLE patch file not found after patch_ple_layer.py"

    ok "PLE patch ready: $PATCHED_PLE"
    HEAD_PLE_MOUNT="-v $PATCHED_PLE:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro"
    WORKER_PLE_MOUNT="-v /tmp/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro"
    fi

    # ---------------------------------------------------------------------------
    # 6b. MXFP8 kernel-fallback patch.
    #     FlashInfer mm_mxfp8 needs N,K >= 128 and both divisible by 32. Two
    #     shapes in this checkpoint miss that — linear_attn.in_proj_a/b [48,2560]
    #     (fatal at engine start) and the vision MLP fc1 [4304,1152] — so those
    #     layers are routed to the BF16 emulation kernel. visual.* stays fully
    #     emulated (verified multimodal path; global dequant would OOM).
    # ---------------------------------------------------------------------------
    PATCHED_MODELOPT="$SCRIPT_DIR/files/modelopt_patched.py"
    MODELOPT_ORIG="$SCRIPT_DIR/files/modelopt_patched.py.orig"
    HEAD_MODELOPT_MOUNT=""
    WORKER_MODELOPT_MOUNT=""
    MODEL_OPT_PKG="$VLLM_PKG/model_executor/layers/quantization/modelopt.py"

    info "=== Step 6b: Prepare MXFP8 kernel-fallback patch ==="
    if [[ ! -f "$MODELOPT_ORIG" ]]; then
        info "Extracting modelopt.py from image..."
        tmp_container=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp_container:$MODEL_OPT_PKG" "$MODELOPT_ORIG"
        docker rm "$tmp_container" >/dev/null 2>&1
        [[ -f "$MODELOPT_ORIG" ]] || err "Failed to extract modelopt.py from image."
    fi
    python3 "$SCRIPT_DIR/files/patch_modelopt_mxfp8.py"
    [[ -f "$PATCHED_MODELOPT" ]] || err "modelopt patch missing after patch_modelopt_mxfp8.py"
    # Stacks on top: adds the FP8_BLOCK_SCALES routed-expert branch that neither
    # this image nor upstream vLLM has, which is what MTP needs on this checkpoint.
    python3 "$SCRIPT_DIR/files/patch_modelopt_fp8_block_moe.py"
    ok "MXFP8 fallback patch ready: $PATCHED_MODELOPT"
    HEAD_MODELOPT_MOUNT="-v $PATCHED_MODELOPT:$MODEL_OPT_PKG:ro"
    WORKER_MODELOPT_MOUNT="-v /tmp/modelopt_patched.py:$MODEL_OPT_PKG:ro"

    # ---------------------------------------------------------------------------
    # 7. Build vLLM args (shared between head and worker)
    # ---------------------------------------------------------------------------
    info "=== Step 7: Launch vLLM ==="

    VLLM_ARGS=()
    VLLM_ARGS+=("--served-model-name" "$SERVED_MODEL_NAME")
    VLLM_ARGS+=("--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE")
    VLLM_ARGS+=("--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION")
    VLLM_ARGS+=("--max-num-seqs" "$MAX_NUM_SEQS")
    VLLM_ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
    VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
    VLLM_ARGS+=("--kv-cache-dtype" "$KV_CACHE_DTYPE")
    VLLM_ARGS+=("--load-format" "safetensors")
    VLLM_ARGS+=("--safetensors-load-strategy" "lazy")
    VLLM_ARGS+=("--enable-chunked-prefill")
    VLLM_ARGS+=("--reasoning-parser" "qwen3")
    VLLM_ARGS+=("--enable-auto-tool-choice")
    VLLM_ARGS+=("--tool-call-parser" "qwen3_coder")
    VLLM_ARGS+=("--distributed-executor-backend" "mp")
    VLLM_ARGS+=("--mm-encoder-tp-mode" "$MM_ENCODER_TP_MODE")
    VLLM_ARGS+=("--nnodes" "2")
    VLLM_ARGS+=("--master-addr" "$HEAD_IP")
    VLLM_ARGS+=("--master-port" "$MASTER_PORT")

    if [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]]; then
        VLLM_ARGS+=("--enable-expert-parallel")
        VLLM_ARGS+=("--all2all-backend" "allgather_reducescatter")
    fi

    # JSON args: use printf to build properly quoted strings for the heredoc
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
            # get_top_tokens (added by patch_mtp_draft_vocab.py) is only reached
            # through this flag; it also cuts the draft all-gather from
            # O(vocab_size) to O(2*tp_size) per token.
            VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s,\"use_local_argmax_reduction\":true}'" "$MTP_NUM_SPECULATIVE_TOKENS")")
        else
            VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s}'" "$MTP_NUM_SPECULATIVE_TOKENS")")
        fi
    fi

    VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'")")

    # hf-overrides: ONE merged payload, nested under "text_config".
    # vLLM's ModelConfig._apply_dict_overrides only recurses into keys that are
    # themselves nested configs. For qwen4_exp the parent config also exposes a
    # plain `rope_parameters` dict, so a top-level {"rope_parameters":...} is
    # setattr'd onto the parent and NEVER reaches text_config -- i.e. YaRN was
    # silently a no-op. Everything the model reads lives under text_config, so
    # nest both the rope override and the PLE dtype there.
    HF_OVERRIDES_JSON=$(
        PLE_DTYPE="$PLE_EMBEDDING_DTYPE" \
        YARN="$YARN_ENABLE" YARN_FACTOR="${YARN_FACTOR:-}" \
        python3 -c '
import json, os
tc = {}
if os.environ.get("PLE_DTYPE"):
    tc["ple_embedding_dtype"] = os.environ["PLE_DTYPE"]
if os.environ.get("YARN") == "true":
    tc["rope_parameters"] = {
        "rope_type": "yarn",
        "factor": float(os.environ["YARN_FACTOR"]),
        "original_max_position_embeddings": 262144,
    }
print(json.dumps({"text_config": tc}, separators=(",", ":")) if tc else "")
'
    )
    if [[ -n "$HF_OVERRIDES_JSON" ]]; then
        VLLM_ARGS+=("--hf-overrides" "'$HF_OVERRIDES_JSON'")
    fi

    # EXTRA_VLLM_ARGS is appended LAST and verbatim (quote JSON values with single
    # quotes exactly as you would on a shell command line).
    if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
        VLLM_ARGS+=("$EXTRA_VLLM_ARGS")
    fi
    # One rendered string shared by the worker and head launch scripts. JSON
    # values already carry their own single quotes (see printf above).
    VLLM_ARGS_STR="${VLLM_ARGS[*]}"
    OVERLAY_ENV_STR="${OVERLAY_ENV[*]:-}"

    # Build docker run args (base, without node-specific VLLM_HOST_IP)
    DOCKER_ARGS=()
    DOCKER_ARGS+=(-d --name vllm-fn)
    DOCKER_ARGS+=(--gpus all --network host --ipc host)
    DOCKER_ARGS+=(--cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864)
    DOCKER_ARGS+=(--device /dev/infiniband:/dev/infiniband)
    # NCCL / fabric env (VLLM_HOST_IP set per-node below)
    DOCKER_ARGS+=(-e "GLOO_SOCKET_IFNAME=$IFACE")
    DOCKER_ARGS+=(-e "NCCL_SOCKET_IFNAME=$IFACE")
    DOCKER_ARGS+=(-e "TP_SOCKET_IFNAME=$IFACE")
    DOCKER_ARGS+=(-e "NCCL_IB_DISABLE=0")
    DOCKER_ARGS+=(-e "NCCL_IB_HCA=$IB_HCA")
    DOCKER_ARGS+=(-e "NCCL_IB_GID_INDEX=$IB_GID_INDEX")
    DOCKER_ARGS+=(-e "NCCL_IB_AUTO_DETECT=0")
    DOCKER_ARGS+=(-e "NCCL_DEBUG=WARN")
    # Offline mode
    DOCKER_ARGS+=(-e "HF_HUB_OFFLINE=1")
    DOCKER_ARGS+=(-e "TRANSFORMERS_OFFLINE=1")
    # YaRN: allow max_model_len > 262K
    if [[ -n "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" ]]; then
        DOCKER_ARGS+=(-e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN")
    fi
    if [[ "$PLE_OFFLOAD" == "true" ]]; then
        DOCKER_ARGS+=(-e "VLLM_PLE_CPU_OFFLOAD=1")
    fi
    # Volumes — single elements (flag + value together) for eval to parse correctly
    # NOTE: the container runs as root (HOME=/root), so cache mounts must target /root,
    # not the host user's $HOME — otherwise offline HF lookups fail.
    if [[ -n "$HEAD_PLE_MOUNT" ]]; then
        DOCKER_ARGS+=("$HEAD_PLE_MOUNT")
    fi
    if [[ -n "$HEAD_MODELOPT_MOUNT" ]]; then
        DOCKER_ARGS+=("$HEAD_MODELOPT_MOUNT")
    fi
    DOCKER_ARGS+=("-e HF_HOME=/root/.cache/huggingface")
    DOCKER_ARGS+=("-v $HF_CACHE_DIR:/root/.cache/huggingface")
    DOCKER_ARGS+=("-v $HOME/.cache/vllm:/root/.cache/vllm")
    # HF token
    if [[ -n "$HF_TOKEN" ]]; then
        DOCKER_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
    fi
    if [[ -n "$EXTRA_DOCKER_ARGS" ]]; then
        # shellcheck disable=SC2206
        DOCKER_ARGS+=($EXTRA_DOCKER_ARGS)
    fi

    # -----------------------------------------------------------------------
    # Launch: worker (rank 1) first, then head (rank 0).
    # The head node serves the API; the worker runs headless.
    # -----------------------------------------------------------------------

    info ""
    info "Config:"
    info "  Model:      $MODEL_ID"
    if [[ "$NFS_SHARE" == "true" ]]; then
        info "  Weights:    NFS from $NFS_SERVER_IP (head cache, no worker copy)"
    else
        info "  Weights:    local HF cache on each node (worker copy synced from head)"
    fi
    info "  Image:      $IMAGE"
    info "  Nodes:      $HEAD_IP (head, rank 0) + $WORKER_IP (worker, rank 1)"
    info "  TP=$TENSOR_PARALLEL_SIZE  EP=$( [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]] && echo on || echo off )  MTP=$MTP_NUM_SPECULATIVE_TOKENS"
    info "  Context:    $MAX_MODEL_LEN tokens"
    info "  GMU:        $GPU_MEMORY_UTILIZATION"
    info "  Max seqs:   $MAX_NUM_SEQS"
    info "  Port:       $PORT"
    info "  IFACE:      $IFACE"
    info "  IB_HCA:     $IB_HCA"
    info "  FP8 dense:  $FP8_DENSE   QSA profile: $QSA_PROFILE"
    info "  KV dtype:   $KV_CACHE_DTYPE   Draft vocab: ${MTP_DRAFT_VOCAB:-full}"
    info "  MM encoder: $MM_ENCODER_TP_MODE tp-mode"
    [[ -n "$EXTRA_VLLM_ARGS" ]] && info "  Extra args: $EXTRA_VLLM_ARGS"
    info ""

    # ---- Worker (rank 1) ----
    info "--- Launching worker (rank 1) on $WORKER_IP ---"
    ssh_worker "docker rm -f vllm-fn >/dev/null 2>&1 || true"
    ssh_worker "mkdir -p '$REMOTE_HF' ~/.cache/vllm"
    if [[ "$NFS_SHARE" == "true" ]]; then
        nfs_ensure_worker_volume recreate
        if nfs_worker_has_model "hub/models--${ORG}--${NAME}"; then
            ok "Worker sees checkpoint over NFS"
        else
            err "WORKER cannot see hub/models--${ORG}--${NAME} over NFS. Check: docker logs $NFS_CONTAINER"
        fi
        WORKER_HF_MOUNT="-v $NFS_VOLUME:/root/.cache/huggingface:ro"
    else
        if ! ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}'" 2>/dev/null; then
            err "WORKER is missing $REMOTE_HUB/models--${ORG}--${NAME}. Re-run ./start.sh without --launch to sync, or use --nfs."
        fi
        ok "Worker has a local checkpoint copy"
        WORKER_HF_MOUNT="-v $REMOTE_HF:/root/.cache/huggingface"
    fi

    # Worker can't mount head's filesystem — copy the patched file over
    if [[ -n "$WORKER_PLE_MOUNT" ]]; then
        info "  Copying PLE patch to worker..."
        PLE_DEST="/tmp/ple_layer_patched.py"
        scp -q "$PATCHED_PLE" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${PLE_DEST}"
    fi
    if [[ -n "$WORKER_MODELOPT_MOUNT" ]]; then
        info "  Copying MXFP8 fallback patch to worker..."
        scp -q "$PATCHED_MODELOPT" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/modelopt_patched.py"
    fi
    # Overlay files likewise: copy to /tmp/vllm-overlay on the worker
    WORKER_OVERLAY_MOUNTS=""
    if [[ ${#OVERLAY_FILES[@]} -gt 0 ]]; then
        info "  Copying ${#OVERLAY_FILES[@]} overlay file(s) to worker..."
        ssh_worker "mkdir -p /tmp/vllm-overlay"
        for entry in "${OVERLAY_FILES[@]}"; do
            host_file="${entry%%|*}"; container_path="${entry##*|}"
            scp -q "$host_file" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/vllm-overlay/$(basename "$host_file")"
            WORKER_OVERLAY_MOUNTS+=" -v /tmp/vllm-overlay/$(basename "$host_file"):$container_path:ro"
        done
    fi
    HEAD_OVERLAY_MOUNTS="${OVERLAY_MOUNTS[*]:-}"

    # PLE offload env flag (only set when explicitly true — avoids the ${VAR:+}
    # pitfall where "false" is non-empty and would wrongly enable the flag)
    PLE_OFFLOAD_ENV=""
    [[ "$PLE_OFFLOAD" == "true" ]] && PLE_OFFLOAD_ENV="-e VLLM_PLE_CPU_OFFLOAD=1"

    # Write worker launch script to a temp file and scp it (avoids SSH JSON quoting issues)
    WORKER_SCRIPT=$(mktemp /tmp/vllm_worker_XXXXXX.sh)
    cat > "$WORKER_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \
    -d --name vllm-fn \
    --gpus all --network host --ipc host \
    --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    -e GLOO_SOCKET_IFNAME=$WORKER_IFACE \
    -e NCCL_SOCKET_IFNAME=$WORKER_IFACE \
    -e TP_SOCKET_IFNAME=$WORKER_IFACE \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA=$WORKER_IB_HCA \
    -e NCCL_IB_GID_INDEX=$IB_GID_INDEX \
    -e NCCL_IB_AUTO_DETECT=0 \
    -e NCCL_DEBUG=WARN \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_HOST_IP=$WORKER_IP \
    ${VLLM_ALLOW_LONG_MAX_MODEL_LEN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
    $PLE_OFFLOAD_ENV \
    -e HF_HOME=/root/.cache/huggingface \
    $WORKER_PLE_MOUNT \
    $WORKER_MODELOPT_MOUNT \
    $WORKER_OVERLAY_MOUNTS \
    $OVERLAY_ENV_STR \
    $WORKER_HF_MOUNT \
    -v $REMOTE_HOME/.cache/vllm:/root/.cache/vllm \
    $IMAGE \
    $MODEL_ID \
    $VLLM_ARGS_STR \
    --node-rank 1 \
    --headless
LAUNCH_EOF
    chmod +x "$WORKER_SCRIPT"
    scp -q "$WORKER_SCRIPT" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:/tmp/vllm_worker_launch.sh"
    rm -f "$WORKER_SCRIPT"

    info "  (starting worker container...)"
    ssh_worker "bash /tmp/vllm_worker_launch.sh"
    ok "Worker container started."
    info "  Waiting 15s for worker to initialize..."
    sleep 15

    # ---- Head (rank 0) ----
    info "--- Launching head (rank 0) on $HEAD_IP ---"
    docker rm -f vllm-fn >/dev/null 2>&1 || true
    mkdir -p "$HOME/.cache/vllm"

    # Write head launch script (same approach as worker — avoids eval JSON issues)
    HEAD_SCRIPT=$(mktemp /tmp/vllm_head_XXXXXX.sh)
    cat > "$HEAD_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \
    -d --name vllm-fn \
    --gpus all --network host --ipc host \
    --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    -e GLOO_SOCKET_IFNAME=$IFACE \
    -e NCCL_SOCKET_IFNAME=$IFACE \
    -e TP_SOCKET_IFNAME=$IFACE \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA=$IB_HCA \
    -e NCCL_IB_GID_INDEX=$IB_GID_INDEX \
    -e NCCL_IB_AUTO_DETECT=0 \
    -e NCCL_DEBUG=WARN \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_HOST_IP=$HEAD_IP \
    ${VLLM_ALLOW_LONG_MAX_MODEL_LEN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
    $PLE_OFFLOAD_ENV \
    -e HF_HOME=/root/.cache/huggingface \
    $HEAD_PLE_MOUNT \
    $HEAD_MODELOPT_MOUNT \
    $HEAD_OVERLAY_MOUNTS \
    $OVERLAY_ENV_STR \
    -v $HF_CACHE_DIR:/root/.cache/huggingface \
    -v $HOME/.cache/vllm:/root/.cache/vllm \
    $IMAGE \
    $MODEL_ID \
    $VLLM_ARGS_STR \
    --node-rank 0 \
    --host 0.0.0.0 \
    --port $PORT
LAUNCH_EOF
    chmod +x "$HEAD_SCRIPT"
    cp "$HEAD_SCRIPT" "$SCRIPT_DIR/.last_head_launch.sh"   # for inspection (gitignored)

    info "  (starting head container...)"
    bash "$HEAD_SCRIPT"
    rm -f "$HEAD_SCRIPT"
    ok "Head container started."
    info ""
    info "vLLM is loading (~6-7 min). Following logs until ready..."
    info ""

    # Follow logs in background, poll /health until 200, then return to shell
    docker logs -f vllm-fn &
    LOGPID=$!

    info "Waiting for /health to return 200..."
    while true; do
        sleep 10
        # Check if container is still running
        if ! docker ps --format '{{.Names}}' | grep -q '^vllm-fn$'; then
            kill $LOGPID 2>/dev/null || true
            err "Container vllm-fn exited unexpectedly. Check: docker logs vllm-fn"
        fi
        # Check health endpoint
        HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo "000")
        if [[ "$HTTP_CODE" == "200" ]]; then
            kill $LOGPID 2>/dev/null || true
            echo ""
            ok "vLLM is ready and serving on port $PORT!"
            info ""
            info "Test with:"
            info "  curl http://localhost:$PORT/v1/chat/completions \\"
            info "    -H 'Content-Type: application/json' \\"
            info "    -d '{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
            info ""
            info "View logs: docker logs -f vllm-fn"
            info "Stop:      ./stop.sh"
            break
        fi
    done
fi

ok "Done."
