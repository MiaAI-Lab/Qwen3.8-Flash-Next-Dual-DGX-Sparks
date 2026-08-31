#!/usr/bin/env bash
# ============================================================================
# start.sh — Download, distribute, and serve RadixArk/Qwen3.8-Flash-Next-NVFP4
#             across a 2-node DGX Spark cluster with vLLM TP2+EP+MTP3.
#
# Based on: https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark
#
# Usage:
#   ./start.sh                # download weights → rsync → apply patch → launch
#   ./start.sh --no-download  # skip download (weights already cached locally)
#   ./start.sh --no-launch    # download + rsync + patch only, don't start server
#   ./start.sh --launch       # skip download + rsync, just apply patch + launch
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
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"   # auto = model dtype (bf16); this checkpoint only supports bf16 KV cache
PLE_OFFLOAD="${PLE_OFFLOAD:-false}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
HF_TOKEN="${HF_TOKEN:-}"

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
DO_DOWNLOAD=true
DO_RSYNC=true
DO_LAUNCH=true

for arg in "$@"; do
    case "$arg" in
        --no-download)  DO_DOWNLOAD=false ;;
        --no-launch)    DO_LAUNCH=false ;;
        --launch)       DO_DOWNLOAD=false; DO_RSYNC=false ;;
        -h|--help)
            echo "Usage: $0 [--no-download] [--no-launch] [--launch]"
            echo ""
            echo "  (default)      Download weights, rsync to worker, apply patch, launch"
            echo "  --no-download  Skip HF download (weights already cached locally)"
            echo "  --no-launch    Download + rsync only, don't start vLLM"
            echo "  --launch       Apply patch + launch only (weights already on both nodes)"
            exit 0
            ;;
        *)
            err "Unknown argument: $arg (try --help)"
            ;;
    esac
done

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
# HF cache on both nodes
# WORKER_HF_HOME wins. Otherwise mirror HF_HOME: a path under the head's $HOME
# is rewritten onto the worker's $HOME; any other absolute path (/data/hf, …)
# is used as-is so both Sparks mount the same layout.
# ---------------------------------------------------------------------------
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME="$HF_CACHE_DIR"
HUB_PATH="$HF_CACHE_DIR/hub"
mkdir -p "$HF_CACHE_DIR"

REMOTE_HOME=$(ssh_worker "echo \"\$HOME\"")
if [[ -n "${WORKER_HF_HOME:-}" ]]; then
    REMOTE_HF="$WORKER_HF_HOME"
elif [[ "$HF_CACHE_DIR" == "$HOME" || "$HF_CACHE_DIR" == "$HOME/"* ]]; then
    REMOTE_HF="${REMOTE_HOME}${HF_CACHE_DIR#"$HOME"}"
else
    REMOTE_HF="$HF_CACHE_DIR"
fi
REMOTE_HUB="${REMOTE_HF}/hub"
info "Head HF cache:   $HF_CACHE_DIR"
info "Worker HF cache: $REMOTE_HF"

# Auto-detect: skip download/rsync if weights already on both nodes
ORG="${MODEL_ID%%/*}"
NAME="${MODEL_ID##*/}"
if $DO_DOWNLOAD && $DO_RSYNC; then
    HEAD_HAS=$( [[ -d "$HUB_PATH/models--${ORG}--${NAME}" ]] && echo 1 || echo 0 )
    WORKER_HAS=$(ssh_worker "test -d '$REMOTE_HUB/models--${ORG}--${NAME}' && echo 1 || echo 0" 2>/dev/null || echo 0)
    if [[ "$HEAD_HAS" == "1" && "$WORKER_HAS" == "1" ]]; then
        ok "Weights already present on both nodes — skipping download + rsync."
        DO_DOWNLOAD=false
        DO_RSYNC=false
    fi
fi

# ---------------------------------------------------------------------------
# 1. Download the model weights (head node)
# ---------------------------------------------------------------------------
if $DO_DOWNLOAD; then
    info "=== Step 1: Download $MODEL_ID ==="

    if command -v uvx &>/dev/null; then
        info "Using uvx to download..."
        HF_HOME="$HF_CACHE_DIR" uvx hf download "$MODEL_ID"
    elif command -v huggingface-cli &>/dev/null; then
        info "Using huggingface-cli to download..."
        HF_HOME="$HF_CACHE_DIR" huggingface-cli download "$MODEL_ID"
    elif command -v hf &>/dev/null; then
        info "Using hf CLI to download..."
        HF_HOME="$HF_CACHE_DIR" hf download "$MODEL_ID"
    else
        err "No HuggingFace download tool found. Install one of:\n  pip install huggingface_hub\n  pip install uv"
    fi
    ok "Download complete."
fi

# ---------------------------------------------------------------------------
# 2. Resolve local cache path
# ---------------------------------------------------------------------------
info "=== Step 2: Resolve cache path ==="

# Try CLI tools first, then guess from cache layout
MODEL_DIR=""
for tool_cmd in "uvx hf path" "hf path" "huggingface-cli path"; do
    first_word="${tool_cmd%% *}"
    if command -v "$first_word" &>/dev/null; then
        MODEL_DIR=$($tool_cmd "$MODEL_ID" 2>/dev/null || true)
        [[ -n "$MODEL_DIR" && -d "$MODEL_DIR" ]] && break
        MODEL_DIR=""
    fi
done

# Fallback: guess from standard HF cache layout
if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
    ORG="${MODEL_ID%%/*}"
    NAME="${MODEL_ID##*/}"
    CANDIDATE="$HUB_PATH/models--${ORG}--${NAME}"
    [[ -d "$CANDIDATE" ]] && MODEL_DIR="$CANDIDATE"
fi

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
    err "Could not resolve local cache path for $MODEL_ID under $HUB_PATH"
fi

ok "Model cache: $MODEL_DIR"

# ---------------------------------------------------------------------------
# 3. Rsync weights to worker node
# ---------------------------------------------------------------------------
if $DO_RSYNC; then
    info "=== Step 3: Sync weights to worker ($WORKER_IP) ==="
    info "  Worker cache: $REMOTE_HUB"

    ssh_worker "mkdir -p '$REMOTE_HUB'"

    rsync -av --progress --partial \
        "${MODEL_DIR}/" \
        "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${REMOTE_HUB}/"

    ok "Rsync complete."
fi

# ---------------------------------------------------------------------------
# 4. Verify weights on both nodes
# ---------------------------------------------------------------------------
if $DO_LAUNCH; then
    info "=== Step 4: Verify weights ==="

    ORG="${MODEL_ID%%/*}"
    NAME="${MODEL_ID##*/}"
    HEAD_MODEL_PATH="$HUB_PATH/models--${ORG}--${NAME}"
    WORKER_MODEL_PATH="$REMOTE_HUB/models--${ORG}--${NAME}"

    # Check head
    if [[ -d "$HEAD_MODEL_PATH" ]]; then
        HEAD_SIZE=$(du -sh "$HEAD_MODEL_PATH" 2>/dev/null | cut -f1)
        ok "HEAD  ($HEAD_IP): $HEAD_MODEL_PATH ($HEAD_SIZE)"
    else
        err "HEAD  ($HEAD_IP): $HEAD_MODEL_PATH — NOT FOUND"
    fi

    # Check worker
    if ssh_worker "test -d '$WORKER_MODEL_PATH'" 2>/dev/null; then
        WORKER_SIZE=$(ssh_worker "du -sh '$WORKER_MODEL_PATH' 2>/dev/null" | cut -f1)
        ok "WORKER ($WORKER_IP): $WORKER_MODEL_PATH ($WORKER_SIZE)"
    else
        err "WORKER ($WORKER_IP): $WORKER_MODEL_PATH — NOT FOUND. Run: ./start.sh --no-download"
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
    # 5. Prepare the PLE patch
    #    The NVFP4 checkpoint stores PLE as FP8 shards, but declares them excluded
    #    in a ModelOpt-NVFP4 quant config. vLLM's PLE resolver only enables FP8-PLE
    #    when the whole checkpoint is FP8, so it builds BF16 and crashes.
    #    The patch installs a resolver shim: with PLE_QUANT_OVERRIDE=fp8 in the
    #    environment, the PLE quant-method lookup short-circuits to the FP8 method.
    # ---------------------------------------------------------------------------
    info "=== Step 6: Prepare PLE patch ==="

    PATCHED_PLE="$SCRIPT_DIR/files/ple_layer_patched.py"

    if [[ ! -f "$PATCHED_PLE" ]]; then
        info "Extracting ple_layer.py from image..."
        mkdir -p "$SCRIPT_DIR/files"
        tmp_container=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp_container:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py" "$PATCHED_PLE.orig"
        docker rm "$tmp_container" >/dev/null 2>&1

        if [[ -f "$PATCHED_PLE.orig" ]]; then
            # Apply patch: wrap the stock resolver with the PLE_QUANT_OVERRIDE shim
            python3 -c "
import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# Shim: rebind the resolver so PLE_QUANT_OVERRIDE=fp8 selects the global-scale
# FP8 method regardless of the parent quant config (ModelOpt NVFP4 excludes *.ple.*).
anchor = 'class Qwen3_8FlashNextNGramEmbedding(PleOffloadLayer):'

if '_ORIG_PLE_QUANT_RESOLVER' in content:
    print('WARNING: source already carries an override shim; using it as-is.')
    import shutil
    shutil.copy(sys.argv[1], sys.argv[2])
elif content.find(anchor) < 0:
    print('WARNING: Could not find patch target. Image may have changed.')
    import shutil
    shutil.copy(sys.argv[1], sys.argv[2])
else:
    shim = (
        '_ORIG_PLE_QUANT_RESOLVER = _get_ple_embedding_quant_method\n\n'
        'def _get_ple_embedding_quant_method(quant_config, prefix):\n'
        '    \"\"\"Re-resolve the PLE embedding method under an override.\"\"\"\n'
        '    from os import getenv as _ple_getenv\n'
        '    if _ple_getenv(\"PLE_QUANT_OVERRIDE\", \"\").strip().lower() == \"fp8\":\n'
        '        # NVFP4 checkpoints ship PLE as FP8 shards + one global weight_scale\n'
        '        # yet exclude *.ple.* from the parent quant config; honor the shards.\n'
        '        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n'
        '    return _ORIG_PLE_QUANT_RESOLVER(quant_config, prefix)\n\n\n'
    )
    pos = content.find(anchor)
    patched = content[:pos] + shim + content[pos:]
    with open(sys.argv[2], 'w') as f:
        f.write(patched)
    print('Patch applied successfully.')
" "$PATCHED_PLE.orig" "$PATCHED_PLE"
            rm -f "$PATCHED_PLE.orig"
        else
            err "Failed to extract ple_layer.py from image. Is the image pulled?"
        fi
    fi

    if [[ -f "$PATCHED_PLE" ]]; then
        ok "PLE patch ready: $PATCHED_PLE"
    else
        err "PLE patch file not found."
    fi

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
    VLLM_ARGS+=("--nnodes" "2")
    VLLM_ARGS+=("--master-addr" "$HEAD_IP")
    VLLM_ARGS+=("--master-port" "$MASTER_PORT")

    if [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]]; then
        VLLM_ARGS+=("--enable-expert-parallel")
        VLLM_ARGS+=("--all2all-backend" "allgather_reducescatter")
    fi

    # JSON args: use printf to build properly quoted strings for the heredoc
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s}'" "$MTP_NUM_SPECULATIVE_TOKENS")")
    fi

    VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'")")

    # YaRN rope scaling: extend 262K → ~1M context
    if [[ "$YARN_ENABLE" == "true" ]]; then
        VLLM_ARGS+=("--hf-overrides" "$(printf "'{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":%s,\"original_max_position_embeddings\":262144}}'" "$YARN_FACTOR")")
    fi

    if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
        # shellcheck disable=SC2206
        VLLM_ARGS+=($EXTRA_VLLM_ARGS)
    fi

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
    # PLE FP8 fix
    DOCKER_ARGS+=(-e "PLE_QUANT_OVERRIDE=fp8")
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
    DOCKER_ARGS+=("-v $PATCHED_PLE:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro")
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
    info "  Image:      $IMAGE"
    info "  Nodes:      $HEAD_IP (head, rank 0) + $WORKER_IP (worker, rank 1)"
    info "  TP=$TENSOR_PARALLEL_SIZE  EP=$( [[ "$ENABLE_EXPERT_PARALLEL" == "true" ]] && echo on || echo off )  MTP=$MTP_NUM_SPECULATIVE_TOKENS"
    info "  Context:    $MAX_MODEL_LEN tokens"
    info "  GMU:        $GPU_MEMORY_UTILIZATION"
    info "  Max seqs:   $MAX_NUM_SEQS"
    info "  Port:       $PORT"
    info "  IFACE:      $IFACE"
    info "  IB_HCA:     $IB_HCA"
    info ""

    # ---- Worker (rank 1) ----
    info "--- Launching worker (rank 1) on $WORKER_IP ---"
    ssh_worker "docker rm -f vllm-fn >/dev/null 2>&1 || true"
    ssh_worker "mkdir -p '$REMOTE_HF' ~/.cache/vllm"

    # Worker can't mount head's filesystem — copy the patched file over
    info "  Copying PLE patch to worker..."
    PLE_DEST="/tmp/ple_layer_patched.py"
    scp -q "$PATCHED_PLE" "${WORKER_USER:+${WORKER_USER}@}${WORKER_IP}:${PLE_DEST}"

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
    -e PLE_QUANT_OVERRIDE=fp8 \
    -e VLLM_HOST_IP=$WORKER_IP \
    ${VLLM_ALLOW_LONG_MAX_MODEL_LEN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
    $PLE_OFFLOAD_ENV \
    -e HF_HOME=/root/.cache/huggingface \
    -v /tmp/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
    -v $REMOTE_HF:/root/.cache/huggingface \
    -v $REMOTE_HOME/.cache/vllm:/root/.cache/vllm \
    $IMAGE \
    $MODEL_ID \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-dtype $KV_CACHE_DTYPE \
    --load-format safetensors \
    --safetensors-load-strategy lazy \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --distributed-executor-backend mp \
    --nnodes 2 \
    --master-addr $HEAD_IP \
    --master-port $MASTER_PORT \
    --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --speculative-config '{"method":"mtp","num_speculative_tokens":$MTP_NUM_SPECULATIVE_TOKENS}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --node-rank 1 \
    --headless
LAUNCH_EOF
    # Append YaRN hf-overrides if enabled (can't use ${} inside heredoc with JSON)
    if [[ "$YARN_ENABLE" == "true" ]]; then
        # Add continuation to last line, then append hf-overrides
        sed -i '$ s/$/ \\/' "$WORKER_SCRIPT"
        echo "    --hf-overrides '{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":$YARN_FACTOR,\"original_max_position_embeddings\":262144}}'"  >> "$WORKER_SCRIPT"
    fi
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
    -e PLE_QUANT_OVERRIDE=fp8 \
    -e VLLM_HOST_IP=$HEAD_IP \
    ${VLLM_ALLOW_LONG_MAX_MODEL_LEN:+-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN} \
    $PLE_OFFLOAD_ENV \
    -e HF_HOME=/root/.cache/huggingface \
    -v $PATCHED_PLE:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
    -v $HF_CACHE_DIR:/root/.cache/huggingface \
    -v $HOME/.cache/vllm:/root/.cache/vllm \
    $IMAGE \
    $MODEL_ID \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-num-seqs $MAX_NUM_SEQS \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-dtype $KV_CACHE_DTYPE \
    --load-format safetensors \
    --safetensors-load-strategy lazy \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --distributed-executor-backend mp \
    --nnodes 2 \
    --master-addr $HEAD_IP \
    --master-port $MASTER_PORT \
    --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --speculative-config '{"method":"mtp","num_speculative_tokens":$MTP_NUM_SPECULATIVE_TOKENS}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --node-rank 0 \
    --host 0.0.0.0 \
    --port $PORT
LAUNCH_EOF
    # Append YaRN hf-overrides if enabled
    if [[ "$YARN_ENABLE" == "true" ]]; then
        sed -i '$ s/$/ \\/' "$HEAD_SCRIPT"
        echo "    --hf-overrides '{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":$YARN_FACTOR,\"original_max_position_embeddings\":262144}}'"  >> "$HEAD_SCRIPT"
    fi
    chmod +x "$HEAD_SCRIPT"

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
