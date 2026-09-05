#!/usr/bin/env bash
# ============================================================================
# start-tp1.sh — Single-node, single-GPU (TP=1) vLLM launch on ONE DGX Spark.
#
# Serves the *smaller* NVFP4 checkpoint (local-inference-lab, 98.6 GiB on disk)
# — MXFP8 attention + a 4-bit NVFP4 PLE table. The RadixArk build (125.9 GiB)
# cannot fit one Spark and is not offered here.
#
# ---------------------------------------------------------------------------
# HOW IT FITS (measured on this box — see docs/HANDOFF-single-spark.md)
#
#   unified pool ............ 121.69 GiB   (LPDDR5X; CPU and GPU share it)
#   checkpoint on disk ......  98.57 GiB
#     of which PLE table ....  26.82 GiB   -> NOT on the GPU (see below)
#   weights on GPU ..........  71.75 GiB
#   runtime overhead ........   5.6  GiB   (non-torch 3.37 + activation 1.92
#                                          + graphs 0.12, all measured at TP1)
#   KV cache ................  whatever GMU leaves (~7-15 GiB => 250-500k tok)
#
# The PLE n-gram table is served by vLLM's CPU-offload worker from a
# MEMORY-MAPPED pre-packed file (files/build_ple_packed_table.py, built on
# first launch, ~40 s). File-backed pages are evictable page cache, so the
# non-evictable footprint of the whole deployment is ~78 GiB + KV instead of
# ~104 GiB + KV. That margin is what keeps the host alive: exhausting the
# unified pool hangs the kernel (no OOM, no logs — three times last session).
#
# Two GB10-specific bugs in vLLM's offload path are patched in
# files/patch_ple_offload.py (CUDA stream memory ops are unsupported on GB10,
# which deadlocked the GPU worker after graph capture) and
# files/patch_ple_layer.py (offload rows must carry codes AND scales).
#
# SAFETY (no sudo needed):
#   * The container runs under a hard cgroup memory cap. Measured: GPU
#     parameter allocations are NOT charged to it on GB10, so the cap bounds
#     the host-side footprint (Python procs, pinned buffers, page cache) while
#     vLLM's own --gpu-memory-utilization budget bounds the GPU side.
#   * A background watchdog (files/memwatch.sh) kills the container if host
#     MemAvailable drops below MEMWATCH_MIN_GIB.
#   * comfy-h3.service is a bash loop that launches ComfyUI (a GPU co-tenant)
#     the moment *anything* answers on port 8888. The launcher refuses 8888
#     while that service is active (disable it: sudo systemctl disable --now
#     comfy-h3.service); with it disabled the default port is 8888.
#
# For >262144 (YaRN 1M) you still need the 2-node ./start.sh.
# ---------------------------------------------------------------------------
#
# Usage:
#   ./start-tp1.sh                  # 65536 context, MTP off, port 8888
#   ./start-tp1.sh --no-launch      # patch + print the command, don't start
#   MAX_MODEL_LEN=262144 ./start-tp1.sh
#   MTP_NUM_SPECULATIVE_TOKENS=3 ./start-tp1.sh   # re-enable MTP (1.5 GiB)
#   GPU_MEMORY_UTILIZATION=0.75 ./start-tp1.sh    # pin the budget yourself
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

# Capture caller-supplied overrides BEFORE sourcing .env — .env sets the
# 2-node values (MAX_MODEL_LEN=1000000 etc.) and would otherwise clobber them.
_CLI_MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
_CLI_GMU="${GPU_MEMORY_UTILIZATION:-}"
_CLI_MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
_CLI_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
_CLI_MTP="${MTP_NUM_SPECULATIVE_TOKENS:-}"
_CLI_REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-}"
_CLI_PLE_OFFLOAD="${PLE_OFFLOAD:-}"
_CLI_PORT="${PORT:-}"
_CLI_KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

[[ -f .env ]] || err ".env not found. Copy .env.sample to .env and edit it."
# shellcheck source=.env
source .env

# ---------------------------------------------------------------------------
# TP1 defaults — deliberately override the 2-node .env values.
# ---------------------------------------------------------------------------
MODEL_ID="${TP1_MODEL_ID:-local-inference-lab/Qwen3.8-Flash-Next-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
PORT="${_CLI_PORT:-8888}"            # 8888 is safe only while comfy-h3.service is disabled (it watches this port)
IMAGE="${IMAGE:?IMAGE not set in .env}"

MAX_MODEL_LEN="${_CLI_MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${_CLI_GMU:-}"          # empty => derived in Step 2
MAX_NUM_SEQS="${_CLI_MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${_CLI_MAX_NUM_BATCHED_TOKENS:-2048}"
MTP_NUM_SPECULATIVE_TOKENS="${_CLI_MTP:-0}"
KV_CACHE_DTYPE="${_CLI_KV_CACHE_DTYPE:-auto}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"          # optional hard pin, bytes
# Runtime overhead on top of weights, GiB (measured at TP1: 3.37+1.92+0.12).
OVERHEAD_GIB="${OVERHEAD_GIB:-5.6}"
# KV the derived budget targets when GMU is not pinned. More KV = more UVM.
KV_TARGET_GIB="${KV_TARGET_GIB:-8.0}"
# Host-side memory the container needs beyond the GPU budget: three Python
# processes, pinned staging buffers, CPU-side torch, page cache slack.
HOST_SLACK_GIB="${HOST_SLACK_GIB:-10.0}"
# Never let the container cgroup cap come within this much of the pool.
OS_RESERVE_GIB="${OS_RESERVE_GIB:-16.0}"
# Watchdog: kill the container if host MemAvailable drops below this.
MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-6}"
PLE_OFFLOAD="${_CLI_PLE_OFFLOAD:-true}"
PLE_GIB="${PLE_GIB:-26.82}"
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
REQUIRE_IDLE_GPU="${_CLI_REQUIRE_IDLE_GPU:-${REQUIRE_IDLE_GPU:-true}}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
HF_TOKEN="${HF_TOKEN:-}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"   # NONE for eager debug

DO_LAUNCH=true
for arg in "$@"; do
    case "$arg" in
        --no-launch) DO_LAUNCH=false ;;
        -h|--help)   sed -n '1,52p' "$0"; exit 0 ;;
        *)           err "Unknown argument: $arg (try --help)" ;;
    esac
done

if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    err "MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
fi
if [[ "$MAX_MODEL_LEN" -gt 262144 ]]; then
    err "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds native 262144 and would need YaRN.
       Use the 2-node ./start.sh for that."
fi
[[ "$PLE_OFFLOAD" == "true" ]] || err "PLE_OFFLOAD=false cannot fit one Spark (98.6 GiB of weights through UVM hung the host last session). Refusing."

# ---------------------------------------------------------------------------
# 1. Resolve the checkpoint in the local HF cache (no download, no NFS).
# ---------------------------------------------------------------------------
info "=== Step 1: Resolve checkpoint ==="
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
ORG="${MODEL_ID%%/*}"; NAME="${MODEL_ID##*/}"
MODEL_PATH="$HF_CACHE_DIR/hub/models--${ORG}--${NAME}"
[[ -d "$MODEL_PATH" ]] || err "Checkpoint not in cache: $MODEL_PATH
       Fetch it first:  ./download.sh $MODEL_ID"
SNAPSHOT_REL="snapshots/$(ls "$MODEL_PATH/snapshots" | head -1)"
[[ -f "$MODEL_PATH/$SNAPSHOT_REL/config.json" ]] || err "No snapshot under $MODEL_PATH/snapshots"
ok "$MODEL_ID  ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"

# ---------------------------------------------------------------------------
# 2. Co-tenant guard + memory budget.
# ---------------------------------------------------------------------------
COTENANT=$(systemctl is-active comfy-h3.service 2>/dev/null || true)
if pgrep -f "ComfyUI/main.py" >/dev/null 2>&1; then
    err "ComfyUI (comfy-h3) is RUNNING and holds GPU memory. It cannot coexist
       with this deployment on unified memory. Stop it:
         sudo systemctl stop comfy-h3.service"
fi
if [[ "$COTENANT" == "active" && "$PORT" == "8888" ]]; then
    err "comfy-h3.service is active: its launcher polls http://127.0.0.1:8888/v1/models
       and starts ComfyUI (a GPU co-tenant) as soon as it answers. Serving on
       8888 would trigger it. Either use PORT=8890 or disable the service:
         sudo systemctl disable --now comfy-h3.service"
fi
if [[ "$COTENANT" == "active" ]]; then
    warn "comfy-h3.service is active but idle (waiting on port 8888). Serving on $PORT keeps it asleep; disable it to use 8888."
fi

info "=== Step 2: Memory budget ==="
KV_BYTES_PER_TOKEN=29482          # measured: 28.8 KiB/token, bf16 KV, this arch
WEIGHT_BYTES=$(du -sb "$MODEL_PATH/$SNAPSHOT_REL/" -L | cut -f1)

read -r MEM_TOTAL_GIB MEM_AVAIL_GIB <<<"$(python3 -c "
m={l.split(':')[0]:int(l.split()[1]) for l in open('/proc/meminfo') if ':' in l}
print(m['MemTotal']/1048576, m['MemAvailable']/1048576)")"

MTP_GIB=0
[[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]] && MTP_GIB=1.49
KV_MULT=1.0
[[ "$KV_CACHE_DTYPE" == fp8* ]] && KV_MULT=0.5

read -r WEIGHTS_GPU_GIB KV_NEED_GIB BUDGET_GIB DERIVED_GMU KV_EXPECT_GIB KV_EXPECT_TOK <<<"$(python3 -c "
w=$WEIGHT_BYTES/2**30-$PLE_GIB
kv_need=$MAX_MODEL_LEN*$KV_BYTES_PER_TOKEN*$KV_MULT/2**30
budget=w+$OVERHEAD_GIB+$MTP_GIB+max(kv_need,$KV_TARGET_GIB)
gmu=budget/$MEM_TOTAL_GIB
kv_exp=budget-w-$OVERHEAD_GIB-$MTP_GIB
print(f'{w:.2f} {kv_need:.2f} {budget:.2f} {gmu:.3f} {kv_exp:.2f} {int(kv_exp*2**30/($KV_BYTES_PER_TOKEN*$KV_MULT))}')")"

if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    warn "  caller-pinned GMU=$GPU_MEMORY_UTILIZATION (derived would be $DERIVED_GMU)"
    read -r BUDGET_GIB KV_EXPECT_GIB KV_EXPECT_TOK <<<"$(python3 -c "
b=$GPU_MEMORY_UTILIZATION*$MEM_TOTAL_GIB
kv=b-$WEIGHTS_GPU_GIB-$OVERHEAD_GIB-$MTP_GIB
print(f'{b:.2f} {kv:.2f} {int(max(kv,0)*2**30/($KV_BYTES_PER_TOKEN*$KV_MULT))}')")"
else
    GPU_MEMORY_UTILIZATION="$DERIVED_GMU"
fi
CONTAINER_MEM_GIB="${CONTAINER_MEM_GIB:-$(python3 -c "print(int($BUDGET_GIB+$HOST_SLACK_GIB))")}"
MAX_CONTAINER_GIB=$(python3 -c "print(int($MEM_TOTAL_GIB-$OS_RESERVE_GIB))")

info "  unified pool ............. ${MEM_TOTAL_GIB%.*} GiB total, ${MEM_AVAIL_GIB%.*} GiB available now"
info "  weights on GPU ........... ${WEIGHTS_GPU_GIB} GiB  (checkpoint minus ${PLE_GIB} GiB PLE table)"
info "  PLE table ................ ${PLE_GIB} GiB  memory-mapped in the CPU offload worker"
info "  runtime overhead ......... ${OVERHEAD_GIB} GiB"
[[ "$MTP_GIB" != 0 ]] && info "  MTP draft model .......... ${MTP_GIB} GiB"
info "  KV needed for ${MAX_MODEL_LEN} ...... ${KV_NEED_GIB} GiB  (kv dtype ${KV_CACHE_DTYPE})"
info "  GPU budget (GMU ${GPU_MEMORY_UTILIZATION}) ... ${BUDGET_GIB} GiB  => ~${KV_EXPECT_GIB} GiB KV (~${KV_EXPECT_TOK} tokens)"
info "  container cgroup cap ..... ${CONTAINER_MEM_GIB} GiB  (hard ceiling ${MAX_CONTAINER_GIB})"

if python3 -c "import sys; sys.exit(0 if $KV_EXPECT_GIB < $KV_NEED_GIB else 1)"; then
    err "Budget leaves ${KV_EXPECT_GIB} GiB for KV but ${MAX_MODEL_LEN} tokens need ${KV_NEED_GIB} GiB.
       Lower MAX_MODEL_LEN or raise GPU_MEMORY_UTILIZATION (FP8 KV is unsupported by this model)."
fi
if [[ "$CONTAINER_MEM_GIB" -gt "$MAX_CONTAINER_GIB" ]]; then
    err "Container cap ${CONTAINER_MEM_GIB} GiB exceeds the hard ceiling ${MAX_CONTAINER_GIB} GiB
       (pool ${MEM_TOTAL_GIB%.*} GiB minus OS_RESERVE_GIB=${OS_RESERVE_GIB}). On unified memory
       this is the line between a killed container and a hung host. Lower the budget."
fi
if python3 -c "import sys; sys.exit(0 if $MEM_AVAIL_GIB < $CONTAINER_MEM_GIB+4 else 1)"; then
    err "Only ${MEM_AVAIL_GIB%.*} GiB available now but the container may use ${CONTAINER_MEM_GIB} GiB.
       Something else is holding memory (docker ps; ps --sort=-rss)."
fi
ok "  budget fits."

# ---------------------------------------------------------------------------
# 3. GPU preflight
# ---------------------------------------------------------------------------
if $DO_LAUNCH && [[ "$REQUIRE_IDLE_GPU" == "true" ]]; then
    info "=== Step 3: GPU preflight ==="
    TENANTS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
              --format=csv,noheader 2>/dev/null | sed '/^$/d' || true)
    if [[ -n "$TENANTS" ]]; then
        echo "$TENANTS"
        err "GPU is in use. Stop the 2-node server first (./stop.sh), or set REQUIRE_IDLE_GPU=false."
    fi
    ok "GPU idle."
fi

# ---------------------------------------------------------------------------
# 4. Patches + packed PLE table
# ---------------------------------------------------------------------------
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
PLE_PKG="$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ple_layer.py"
MODELOPT_PKG="$VLLM_PKG/model_executor/layers/quantization/modelopt.py"

info "=== Step 4: Prepare patches ==="
if ! docker image inspect "$IMAGE" &>/dev/null; then
    info "Pulling $IMAGE ..."
    docker pull "$IMAGE"
fi

extract() {  # <path-in-image> <dest>
    if [[ ! -f "$2" ]]; then
        info "Extracting $(basename "$1") from image..."
        local tmp; tmp=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp:$1" "$2"
        docker rm "$tmp" >/dev/null 2>&1
    fi
}
PATCHED_PLE="$SCRIPT_DIR/files/ple_layer_patched.py"
extract "$PLE_PKG" "$SCRIPT_DIR/files/ple_layer_patched.py.orig"
python3 "$SCRIPT_DIR/files/patch_ple_layer.py"
[[ -f "$PATCHED_PLE" ]] || err "PLE patch missing after patch_ple_layer.py"

PATCHED_MODELOPT="$SCRIPT_DIR/files/modelopt_patched.py"
extract "$MODELOPT_PKG" "$SCRIPT_DIR/files/modelopt_patched.py.orig"
python3 "$SCRIPT_DIR/files/patch_modelopt_mxfp8.py"
[[ -f "$PATCHED_MODELOPT" ]] || err "modelopt patch missing after patch_modelopt_mxfp8.py"

OFFLOAD_DIR="$SCRIPT_DIR/files/ple_offload"
mkdir -p "$OFFLOAD_DIR/orig"
extract "$VLLM_PKG/model_executor/layers/ple_offload_layer.py" "$OFFLOAD_DIR/orig/ple_offload_layer.py"
for f in connector worker protocol; do
    extract "$VLLM_PKG/v1/ple_offload/$f.py" "$OFFLOAD_DIR/orig/$f.py"
done
python3 "$SCRIPT_DIR/files/patch_ple_offload.py"
for f in ple_offload_layer connector worker protocol; do
    [[ -f "$OFFLOAD_DIR/$f.py" ]] || err "offload patch missing: $f.py"
done
ok "Patches ready."

PLE_CACHE_HOST="$HOME/.cache/vllm/ple_cache/${ORG}--${NAME}"
PLE_CACHE_CTR="/root/.cache/vllm/ple_cache/${ORG}--${NAME}"
if ! ls "$PLE_CACHE_HOST"/*.packed_u8 >/dev/null 2>&1; then
    info "Building packed PLE table (one-time, ~40 s, <1 GiB RAM, no GPU)..."
    mkdir -p "$PLE_CACHE_HOST"
    docker run --rm --name "${CONTAINER_NAME}-plebuild" --memory 6g --cpus 8 \
        -v "$MODEL_PATH:/m:ro" -v "$HOME/.cache/vllm/ple_cache:/out" \
        -v "$SCRIPT_DIR/files/build_ple_packed_table.py:/b.py:ro" \
        --entrypoint python3 "$IMAGE" -u /b.py "/m/$SNAPSHOT_REL" "/out/${ORG}--${NAME}"
fi
ok "Packed PLE table: $(ls "$PLE_CACHE_HOST"/*.packed_u8 | head -1) ($(du -sh "$PLE_CACHE_HOST" | cut -f1))"

# ---------------------------------------------------------------------------
# 5. Build vLLM args.
# ---------------------------------------------------------------------------
VLLM_ARGS=()
VLLM_ARGS+=("--served-model-name" "$SERVED_MODEL_NAME")
VLLM_ARGS+=("--tensor-parallel-size" "1")
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
# REQUIRED for PLE offload: only multiproc_executor spawns the offload worker.
VLLM_ARGS+=("--distributed-executor-backend" "mp")
[[ -n "$KV_CACHE_MEMORY" ]] && VLLM_ARGS+=("--kv-cache-memory" "$KV_CACHE_MEMORY")
if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
    VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s}'" "$MTP_NUM_SPECULATIVE_TOKENS")")
fi
VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":0,\"cudagraph_mode\":\"%s\"}'" "$CUDAGRAPH_MODE")")
[[ -n "$EXTRA_VLLM_ARGS" ]] && VLLM_ARGS+=("$EXTRA_VLLM_ARGS")
VLLM_ARGS_STR="${VLLM_ARGS[*]}"

info ""
info "Config (single Spark, TP=1):"
info "  Model:      $MODEL_ID"
info "  Image:      $IMAGE"
info "  Context:    $MAX_MODEL_LEN tokens (native rope, no YaRN)"
info "  GMU:        $GPU_MEMORY_UTILIZATION  (budget ${BUDGET_GIB} GiB, cgroup cap ${CONTAINER_MEM_GIB} GiB)"
info "  Max seqs:   $MAX_NUM_SEQS   Batched tokens: $MAX_NUM_BATCHED_TOKENS   KV dtype: $KV_CACHE_DTYPE"
info "  MTP:        $MTP_NUM_SPECULATIVE_TOKENS $( [[ "$MTP_NUM_SPECULATIVE_TOKENS" -eq 0 ]] && echo '(disabled)')"
info "  Graphs:     $CUDAGRAPH_MODE"
info "  Port:       $PORT"
info ""

LAUNCH_SCRIPT=$(mktemp /tmp/vllm_tp1_XXXXXX.sh)
cat > "$LAUNCH_SCRIPT" <<LAUNCH_EOF
#!/bin/bash
docker run \\
    -d --name $CONTAINER_NAME \\
    --gpus all --network host --ipc host \\
    --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \\
    --memory ${CONTAINER_MEM_GIB}g --memory-swap ${CONTAINER_MEM_GIB}g \\
    -e HF_HUB_OFFLINE=1 \\
    -e TRANSFORMERS_OFFLINE=1 \\
    -e VLLM_PLE_CPU_OFFLOAD=1 \\
    -e VLLM_PLE_PACKED_TABLE_DIR=$PLE_CACHE_CTR \\
    -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 \\
    -e HF_HOME=/root/.cache/huggingface \\
    ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \\
    -v $PATCHED_PLE:$PLE_PKG:ro \\
    -v $PATCHED_MODELOPT:$MODELOPT_PKG:ro \\
    -v $OFFLOAD_DIR/ple_offload_layer.py:$VLLM_PKG/model_executor/layers/ple_offload_layer.py:ro \\
    -v $OFFLOAD_DIR/connector.py:$VLLM_PKG/v1/ple_offload/connector.py:ro \\
    -v $OFFLOAD_DIR/worker.py:$VLLM_PKG/v1/ple_offload/worker.py:ro \\
    -v $OFFLOAD_DIR/protocol.py:$VLLM_PKG/v1/ple_offload/protocol.py:ro \\
    -v $HF_CACHE_DIR:/root/.cache/huggingface \\
    -v $HOME/.cache/vllm:/root/.cache/vllm \\
    $EXTRA_DOCKER_ARGS \\
    $IMAGE \\
    $MODEL_ID \\
    $VLLM_ARGS_STR \\
    --host 0.0.0.0 \\
    --port $PORT
LAUNCH_EOF
chmod +x "$LAUNCH_SCRIPT"
cp "$LAUNCH_SCRIPT" "$SCRIPT_DIR/.last_tp1_launch.sh"

if ! $DO_LAUNCH; then
    info "--no-launch: command written to .last_tp1_launch.sh"
    cat "$SCRIPT_DIR/.last_tp1_launch.sh"
    rm -f "$LAUNCH_SCRIPT"
    exit 0
fi

# ---------------------------------------------------------------------------
# 6. Launch + watchdog
# ---------------------------------------------------------------------------
info "=== Step 6: Launch ==="
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$HOME/.cache/vllm"
bash "$LAUNCH_SCRIPT"
rm -f "$LAUNCH_SCRIPT"
ok "Container $CONTAINER_NAME started."

# Kill the previous watchdog (if any) and start a fresh one.
pkill -f "memwatch.sh $CONTAINER_NAME" 2>/dev/null || true
mkdir -p "$SCRIPT_DIR/logs"
nohup bash "$SCRIPT_DIR/files/memwatch.sh" "$CONTAINER_NAME" "$MEMWATCH_MIN_GIB" \
    > "$SCRIPT_DIR/logs/memwatch-${CONTAINER_NAME}.log" 2>&1 &
ok "Watchdog running (kills container if MemAvailable < ${MEMWATCH_MIN_GIB} GiB): logs/memwatch-${CONTAINER_NAME}.log"
info "Loading weights (~3-4 min). Following logs until ready..."

docker logs -f "$CONTAINER_NAME" &
LOGPID=$!
while true; do
    sleep 10
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
        kill $LOGPID 2>/dev/null || true
        echo ""
        REASON=$(docker logs "$CONTAINER_NAME" 2>&1 \
                 | grep -oE "(ValueError|RuntimeError|TimeoutError|torch\.[A-Za-z]*Error): .*" \
                 | grep -viE "min_frames|max_frames" | tail -1 | cut -c1-400)
        [[ -n "$REASON" ]] && { echo "  vLLM reported:"; echo "    $REASON"; }
        if docker inspect "$CONTAINER_NAME" --format '{{.State.OOMKilled}}' 2>/dev/null | grep -q true; then
            echo "  Container was OOM-killed by its cgroup cap (${CONTAINER_MEM_GIB} GiB) — the host survived as designed."
        fi
        err "Container exited. Full logs: docker logs $CONTAINER_NAME"
    fi
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || echo "000")
    if [[ "$CODE" == "200" ]]; then
        kill $LOGPID 2>/dev/null || true
        echo ""
        ok "vLLM ready on port $PORT (TP=1, single Spark)."
        docker logs "$CONTAINER_NAME" 2>&1 | grep -iE "GPU KV cache size|Available KV cache|Maximum concurrency" | tail -3 || true
        info ""
        info "Stop:  docker rm -f $CONTAINER_NAME; pkill -f 'memwatch.sh $CONTAINER_NAME'"
        break
    fi
done
