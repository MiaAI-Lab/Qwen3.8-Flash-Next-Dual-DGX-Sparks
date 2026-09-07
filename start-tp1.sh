#!/usr/bin/env bash
# ============================================================================
# start-tp1.sh — Single-node, single-GPU (TP=1) vLLM launch on ONE DGX Spark.
#
# Two ways the checkpoint can fit one Spark (121.69 GiB unified LPDDR5X):
#
#   PLE_OFFLOAD=true  (default, the 2026-09-04 known-good path)
#     The PLE n-gram table is served by vLLM's CPU-offload worker from a
#     MEMORY-MAPPED pre-packed file (files/build_ple_packed_table.py, built on
#     first launch, ~40 s). File-backed pages are evictable page cache, so the
#     non-evictable footprint is (checkpoint - PLE) + ~5.6 GiB + KV. Costs a
#     host-side handshake per PLE lookup.
#
#   PLE_OFFLOAD=false  (resident; needs an NVFP4 PLE table, e.g. the
#                       MiaAI-Lab/...-FP8dense-MTP-PLE4 checkpoint from
#                       files/ple_nvfp4/build.sh, 98.7 GiB total)
#     Everything lives on the GPU: 98.7 GiB weights + ~5.6 GiB runtime + KV.
#     Margin is a few GiB. Exhausting the unified pool HANGS the kernel (no OOM,
#     no logs), so this mode insists on the VM tunables from
#     docs/HANDOFF-single-spark.md §6 unless TP1_ALLOW_DEFAULT_VM_TUNABLES=true.
#
# Levers shared with the 2-node start.sh (all opt-in here, TP1_* prefixed so the
# 2-node values in .env never leak into a single-Spark launch):
#   TP1_FP8_DENSE=true        FP8-dense loader overlay (files/overlay), needed for
#                             the MiaAI-Lab FP8dense* checkpoints
#   TP1_MTP_DRAFT_VOCAB=file  reduced-vocabulary MTP drafting (+ local argmax)
#   TP1_KV_CACHE_DTYPE=fp8    FP8 KV cache (files/patch_qsa_fp8_kv.py)
#   TP1_MTP_NUM_SPECULATIVE_TOKENS=3
#   TP1_LANGUAGE_MODEL_ONLY=true   skip the 0.84 GiB vision tower (text-only API)
#   TP1_SKIP_MM_PROFILING=true     skip the multimodal profile run
# Profiles: ./start-tp1.sh --profile tp1/resident.env   (or tp1/offload.env)
# A profile is sourced AFTER .env; caller environment beats both.
# VLLM_MTP_DRAFT_VOCAB_BALANCE is a no-op at TP=1 (files/overlay/mtp_draftvocab.py
# gates it on tp_size > 1) and is deliberately not set.
# torch.compile is pinned to mode 0: Inductor hangs GB10 (RESULTS.md).
#
# ---------------------------------------------------------------------------
# HOW IT FITS (measured on this box — see docs/HANDOFF-single-spark.md)
#
#   unified pool ............ 121.69 GiB   (LPDDR5X; CPU and GPU share it)
#   runtime overhead ........   5.6  GiB   (non-torch 3.37 + activation 1.92
#                                          @2048 batched tokens + graphs 0.12)
#   KV cache ................  whatever GMU leaves
#   GPU parameter allocations are NOT charged to the container cgroup on GB10
#   (a TP=2 container holding ~60 GiB of weights shows 9 GiB in docker stats),
#   so --memory bounds host-side usage only; vLLM's GMU bounds the GPU side.
#
# SAFETY (no sudo needed):
#   * container cgroup memory cap; files/memwatch.sh kills the container if host
#     MemAvailable < MEMWATCH_MIN_GIB.
#   * comfy-h3.service launches a GPU co-tenant the moment anything answers on
#     port 8888; the launcher refuses 8888 while that service is active.
#
# For >262144 (YaRN 1M) you still need the 2-node ./start.sh.
# ---------------------------------------------------------------------------
#
# Usage:
#   ./start-tp1.sh                              # 65536 context, MTP off, offload PLE
#   ./start-tp1.sh --profile tp1/resident.env   # single-Spark FP8dense+MTP+NVFP4-PLE
#   ./start-tp1.sh --no-launch                  # patch + print the command, don't start
#   ./start-tp1.sh --dry-run                    # budget + command only, touches nothing
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

# --profile must be known before .env is sourced.
TP1_PROFILE="${TP1_PROFILE:-}"
_args=()
while (($#)); do
    case "$1" in
        --profile)   [[ $# -ge 2 ]] || err "--profile needs a file"; TP1_PROFILE="$2"; shift 2 ;;
        --profile=*) TP1_PROFILE="${1#*=}"; shift ;;
        *)           _args+=("$1"); shift ;;
    esac
done
set -- "${_args[@]+"${_args[@]}"}"

DO_LAUNCH=true
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --no-launch) DO_LAUNCH=false ;;
        --dry-run)   DO_LAUNCH=false; DRY_RUN=true ;;
        -h|--help)   sed -n '1,66p' "$0"; exit 0 ;;
        *)           err "Unknown argument: $arg (try --help)" ;;
    esac
done

# Capture caller-supplied overrides BEFORE sourcing .env / the profile — .env
# sets the 2-node values (MAX_MODEL_LEN, FP8_DENSE, EXTRA_VLLM_ARGS, ...) and
# would otherwise clobber them.
_CLI_MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
_CLI_GMU="${GPU_MEMORY_UTILIZATION:-}"
_CLI_MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
_CLI_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
_CLI_MTP="${MTP_NUM_SPECULATIVE_TOKENS:-}"
_CLI_REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-}"
_CLI_PLE_OFFLOAD="${PLE_OFFLOAD:-}"
_CLI_PORT="${PORT:-}"
_CLI_KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
_CLI_KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"
_CLI_FP8_DENSE="${FP8_DENSE:-}"
_CLI_MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-}"
_CLI_VLLM_EXTRA_ENV="${VLLM_EXTRA_ENV:-}"
_CLI_EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
_CLI_EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"
_CLI_KV_TARGET_GIB="${KV_TARGET_GIB:-}"
_CLI_HOST_SLACK_GIB="${HOST_SLACK_GIB:-}"
_CLI_OVERHEAD_GIB="${OVERHEAD_GIB:-}"
_CLI_CONTAINER_MEM_GIB="${CONTAINER_MEM_GIB:-}"
_CLI_MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-}"
_CLI_LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-}"
_CLI_SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-}"
_CLI_CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-}"

[[ -f .env ]] || err ".env not found. Copy .env.sample to .env and edit it."
# shellcheck source=.env
source .env
_ENV_EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
if [[ -n "$TP1_PROFILE" ]]; then
    [[ -f "$TP1_PROFILE" ]] || err "profile not found: $TP1_PROFILE"
    # shellcheck disable=SC1090
    source "$TP1_PROFILE"
fi

# ---------------------------------------------------------------------------
# TP1 defaults — deliberately override the 2-node .env values. Precedence:
# caller environment > --profile (TP1_*) > built-in default.
# ---------------------------------------------------------------------------
MODEL_ID="${TP1_MODEL_ID:-local-inference-lab/Qwen3.8-Flash-Next-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"
PORT="${_CLI_PORT:-${TP1_PORT:-8888}}"   # 8888 is safe only while comfy-h3.service is disabled (it watches this port)
IMAGE="${IMAGE:?IMAGE not set in .env}"

MAX_MODEL_LEN="${_CLI_MAX_MODEL_LEN:-${TP1_MAX_MODEL_LEN:-65536}}"
GPU_MEMORY_UTILIZATION="${_CLI_GMU:-${TP1_GPU_MEMORY_UTILIZATION:-}}"   # empty => derived in Step 2
MAX_NUM_SEQS="${_CLI_MAX_NUM_SEQS:-${TP1_MAX_NUM_SEQS:-4}}"
MAX_NUM_BATCHED_TOKENS="${_CLI_MAX_NUM_BATCHED_TOKENS:-${TP1_MAX_NUM_BATCHED_TOKENS:-2048}}"
MTP_NUM_SPECULATIVE_TOKENS="${_CLI_MTP:-${TP1_MTP_NUM_SPECULATIVE_TOKENS:-0}}"
KV_CACHE_DTYPE="${_CLI_KV_CACHE_DTYPE:-${TP1_KV_CACHE_DTYPE:-auto}}"
KV_CACHE_MEMORY="${_CLI_KV_CACHE_MEMORY:-${TP1_KV_CACHE_MEMORY:-}}"     # optional hard pin, bytes
# Runtime overhead on top of weights, GiB (measured at TP1: 3.37+1.92+0.12).
OVERHEAD_GIB="${_CLI_OVERHEAD_GIB:-${TP1_OVERHEAD_GIB:-5.6}}"
# KV the derived budget targets when GMU is not pinned. More KV = more UVM.
KV_TARGET_GIB="${_CLI_KV_TARGET_GIB:-${TP1_KV_TARGET_GIB:-8.0}}"
# Host-side memory the container needs beyond the GPU budget (offload mode):
# three Python processes, pinned staging buffers, CPU-side torch, page cache slack.
HOST_SLACK_GIB="${_CLI_HOST_SLACK_GIB:-${TP1_HOST_SLACK_GIB:-10.0}}"
# Never let the container cgroup cap come within this much of the pool.
OS_RESERVE_GIB="${TP1_OS_RESERVE_GIB:-16.0}"
# Watchdog: kill the container if host MemAvailable drops below this.
MEMWATCH_MIN_GIB="${_CLI_MEMWATCH_MIN_GIB:-${TP1_MEMWATCH_MIN_GIB:-6}}"
PLE_OFFLOAD="${_CLI_PLE_OFFLOAD:-${TP1_PLE_OFFLOAD:-true}}"
PLE_GIB="${TP1_PLE_GIB:-auto}"           # auto = summed from the checkpoint's PLE shards
CONTAINER_NAME="${TP1_CONTAINER_NAME:-vllm-fn-tp1}"
REQUIRE_IDLE_GPU="${_CLI_REQUIRE_IDLE_GPU:-${TP1_REQUIRE_IDLE_GPU:-${REQUIRE_IDLE_GPU:-true}}}"
FP8_DENSE="${_CLI_FP8_DENSE:-${TP1_FP8_DENSE:-false}}"
MTP_DRAFT_VOCAB="${_CLI_MTP_DRAFT_VOCAB:-${TP1_MTP_DRAFT_VOCAB:-}}"
VLLM_EXTRA_ENV="${_CLI_VLLM_EXTRA_ENV:-${TP1_VLLM_EXTRA_ENV:-}}"
EXTRA_VLLM_ARGS="${_CLI_EXTRA_VLLM_ARGS:-${TP1_EXTRA_VLLM_ARGS:-}}"
EXTRA_DOCKER_ARGS="${_CLI_EXTRA_DOCKER_ARGS:-${TP1_EXTRA_DOCKER_ARGS:-}}"
LANGUAGE_MODEL_ONLY="${_CLI_LANGUAGE_MODEL_ONLY:-${TP1_LANGUAGE_MODEL_ONLY:-false}}"
SKIP_MM_PROFILING="${_CLI_SKIP_MM_PROFILING:-${TP1_SKIP_MM_PROFILING:-false}}"
ALLOW_DEFAULT_VM_TUNABLES="${TP1_ALLOW_DEFAULT_VM_TUNABLES:-false}"
CONTAINER_MEM_GIB="${_CLI_CONTAINER_MEM_GIB:-${TP1_CONTAINER_MEM_GIB:-}}"   # empty => derived in Step 2
HF_TOKEN="${HF_TOKEN:-}"
CUDAGRAPH_MODE="${_CLI_CUDAGRAPH_MODE:-${TP1_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}}"   # NONE for eager debug
if [[ -n "$_ENV_EXTRA_VLLM_ARGS" && "$_ENV_EXTRA_VLLM_ARGS" != "$EXTRA_VLLM_ARGS" ]]; then
    warn ".env EXTRA_VLLM_ARGS='$_ENV_EXTRA_VLLM_ARGS' is a 2-node setting and is IGNORED here (use TP1_EXTRA_VLLM_ARGS)"
fi

if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]]; then
    err "MAX_MODEL_LEN must be a positive integer (got: '$MAX_MODEL_LEN')"
fi
if [[ "$MAX_MODEL_LEN" -gt 262144 ]]; then
    err "MAX_MODEL_LEN=$MAX_MODEL_LEN exceeds native 262144 and would need YaRN.
       Use the 2-node ./start.sh for that."
fi
if [[ -n "$MTP_DRAFT_VOCAB" && "$MTP_NUM_SPECULATIVE_TOKENS" == "0" ]]; then
    err "MTP_DRAFT_VOCAB is set but MTP_NUM_SPECULATIVE_TOKENS=0 - nothing drafts."
fi
[[ -z "$MTP_DRAFT_VOCAB" || -f "$MTP_DRAFT_VOCAB" ]] || err "MTP_DRAFT_VOCAB file not found: $MTP_DRAFT_VOCAB"

# ---------------------------------------------------------------------------
# 1. Resolve the checkpoint in the local HF cache (no download, no NFS).
# ---------------------------------------------------------------------------
info "=== Step 1: Resolve checkpoint ==="
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
ORG="${MODEL_ID%%/*}"; NAME="${MODEL_ID##*/}"
MODEL_PATH="$HF_CACHE_DIR/hub/models--${ORG}--${NAME}"
[[ -d "$MODEL_PATH" ]] || err "Checkpoint not in cache: $MODEL_PATH
       Fetch it first:  ./download.sh $MODEL_ID   (or build it: files/ple_nvfp4/build.sh)"
if [[ -f "$MODEL_PATH/refs/main" ]]; then
    SNAPSHOT_SHA="$(cat "$MODEL_PATH/refs/main")"
else
    SNAPSHOT_SHA="$(ls "$MODEL_PATH/snapshots" | head -1)"
fi
SNAPSHOT_REL="snapshots/$SNAPSHOT_SHA"
SNAPSHOT_DIR="$MODEL_PATH/$SNAPSHOT_REL"
[[ -f "$SNAPSHOT_DIR/config.json" ]] || err "No snapshot under $MODEL_PATH/snapshots"
CONTAINER_SNAPSHOT="/root/.cache/huggingface/hub/models--${ORG}--${NAME}/snapshots/${SNAPSHOT_SHA}"
ok "$MODEL_ID @ $SNAPSHOT_SHA  ($(du -sh -L "$SNAPSHOT_DIR" 2>/dev/null | cut -f1))"

# PLE table format + size, from the checkpoint itself (pure-python header scan).
read -r PLE_DTYPE_DECL PLE_TABLE_GIB PLE_SHARD_DTYPE <<<"$(python3 - "$SNAPSHOT_DIR" <<'PY'
import json, os, struct, sys
snap = sys.argv[1]
cfg = json.load(open(os.path.join(snap, "config.json")))
decl = cfg.get("text_config", cfg).get("ple_embedding_dtype") or "-"
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
hdrs = {}
total = 0
dtypes = set()
for name, f in idx.items():
    if ".ngram_embedding." not in name:
        continue
    if f not in hdrs:
        with open(os.path.join(snap, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdrs[f] = json.loads(fh.read(n))
    m = hdrs[f][name]
    total += m["data_offsets"][1] - m["data_offsets"][0]
    if ".shard_" in name and name.endswith(".weight"):
        dtypes.add(m["dtype"])
print(decl, f"{total / 2**30:.2f}", "/".join(sorted(dtypes)) or "-")
PY
)"
if [[ "$PLE_GIB" == "auto" ]]; then PLE_GIB="$PLE_TABLE_GIB"; fi
info "  PLE table: ${PLE_TABLE_GIB} GiB, shard dtype ${PLE_SHARD_DTYPE}, declared ple_embedding_dtype=${PLE_DTYPE_DECL}"

if [[ "$PLE_OFFLOAD" != "true" ]]; then
    if [[ "$PLE_SHARD_DTYPE" != "U8" ]]; then
        err "PLE_OFFLOAD=false (resident PLE) needs an NVFP4 (U8-packed) PLE table; this checkpoint's is
       ${PLE_SHARD_DTYPE} (${PLE_TABLE_GIB} GiB) and cannot fit one Spark resident.
       Build one: files/ple_nvfp4/build.sh   or keep PLE_OFFLOAD=true."
    fi
    if [[ "$ALLOW_DEFAULT_VM_TUNABLES" != "true" ]]; then
        MFK=$(cat /proc/sys/vm/min_free_kbytes); WSF=$(cat /proc/sys/vm/watermark_scale_factor)
        if (( MFK < 1048576 || WSF < 100 )); then
            $DRY_RUN && warn "(dry run) VM tunables are at defaults (min_free_kbytes=$MFK, watermark_scale_factor=$WSF); a real launch refuses without them"
            $DRY_RUN || err "Resident PLE loads ~99 GiB through UVM with a few GiB of margin. With the default VM
       tunables (min_free_kbytes=$MFK, watermark_scale_factor=$WSF) that hung this host
       (docs/HANDOFF-single-spark.md §6). Apply first (resets on reboot):
         sudo sysctl -w vm.min_free_kbytes=4194304 vm.watermark_scale_factor=300 vm.swappiness=30 vm.vfs_cache_pressure=200
       or set TP1_ALLOW_DEFAULT_VM_TUNABLES=true to proceed anyway."
        fi
    fi
fi

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
KV_BYTES_PER_TOKEN=29482          # measured: 28.8 KiB/token, bf16 KV, this arch (TP-summed)
WEIGHT_BYTES=$(du -sb -L "$SNAPSHOT_DIR/" | cut -f1)

read -r MEM_TOTAL_GIB MEM_AVAIL_GIB <<<"$(python3 -c "
m={l.split(':')[0]:int(l.split()[1]) for l in open('/proc/meminfo') if ':' in l}
print(m['MemTotal']/1048576, m['MemAvailable']/1048576)")"

# Draft-model allowance on top of the checkpoint bytes. 1.49 was local-inference-lab's
# whole MTP draft (its weights are already inside the du figure, so it is conservative);
# runtime extras are really the dequantized draft-head slice (0.16 GiB @32k), draft KV
# and graphs. Checkpoints whose MTP is already counted can lower it (TP1_MTP_GIB).
MTP_GIB=0
[[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]] && MTP_GIB="${TP1_MTP_GIB:-1.49}"
KV_MULT=1.0
# fp8 halves the K/V blocks; the QSA indexer caches stay bf16 (indexer_qsa.py), so
# 0.5 is slightly optimistic — the budget below only sizes GMU, vLLM measures the rest.
[[ "$KV_CACHE_DTYPE" == fp8* ]] && KV_MULT=0.5
PLE_ON_GPU_GIB=0
[[ "$PLE_OFFLOAD" == "true" ]] || PLE_ON_GPU_GIB="$PLE_GIB"
VISUAL_GIB=0
[[ "$LANGUAGE_MODEL_ONLY" == "true" ]] && VISUAL_GIB=0.84   # vision tower skipped (StageMissingLayer)

read -r WEIGHTS_GPU_GIB KV_NEED_GIB BUDGET_GIB DERIVED_GMU KV_EXPECT_GIB KV_EXPECT_TOK <<<"$(python3 -c "
w=$WEIGHT_BYTES/2**30-($PLE_GIB-$PLE_ON_GPU_GIB)-$VISUAL_GIB
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
if [[ -z "$CONTAINER_MEM_GIB" ]]; then
    if [[ "$PLE_OFFLOAD" == "true" ]]; then
        CONTAINER_MEM_GIB=$(python3 -c "print(int($BUDGET_GIB+$HOST_SLACK_GIB))")
    else
        # Resident: GPU parameters are not cgroup-charged; cap host-side usage only.
        CONTAINER_MEM_GIB=40
    fi
fi
MAX_CONTAINER_GIB=$(python3 -c "print(int($MEM_TOTAL_GIB-$OS_RESERVE_GIB))")

info "  unified pool ............. ${MEM_TOTAL_GIB%.*} GiB total, ${MEM_AVAIL_GIB%.*} GiB available now"
if [[ "$PLE_OFFLOAD" == "true" ]]; then
    info "  weights on GPU ........... ${WEIGHTS_GPU_GIB} GiB  (checkpoint minus ${PLE_GIB} GiB PLE table)"
    info "  PLE table ................ ${PLE_GIB} GiB  memory-mapped in the CPU offload worker"
else
    info "  weights on GPU ........... ${WEIGHTS_GPU_GIB} GiB  (whole checkpoint RESIDENT incl. ${PLE_GIB} GiB NVFP4 PLE table)"
fi
[[ "$LANGUAGE_MODEL_ONLY" == "true" ]] && info "  vision tower ............. skipped (--language-model-only, -${VISUAL_GIB} GiB)"
info "  runtime overhead ......... ${OVERHEAD_GIB} GiB"
[[ "$MTP_GIB" != 0 ]] && info "  MTP draft model .......... ${MTP_GIB} GiB"
info "  KV needed for ${MAX_MODEL_LEN} ...... ${KV_NEED_GIB} GiB  (kv dtype ${KV_CACHE_DTYPE})"
info "  GPU budget (GMU ${GPU_MEMORY_UTILIZATION}) ... ${BUDGET_GIB} GiB  => ~${KV_EXPECT_GIB} GiB KV (~${KV_EXPECT_TOK} tokens)"
info "  container cgroup cap ..... ${CONTAINER_MEM_GIB} GiB  (hard ceiling ${MAX_CONTAINER_GIB}; host-side only on GB10)"

if python3 -c "import sys; sys.exit(0 if $KV_EXPECT_GIB < $KV_NEED_GIB else 1)"; then
    err "Budget leaves ${KV_EXPECT_GIB} GiB for KV but ${MAX_MODEL_LEN} tokens need ${KV_NEED_GIB} GiB.
       Lower MAX_MODEL_LEN, use KV_CACHE_DTYPE=fp8, or raise GPU_MEMORY_UTILIZATION."
fi
if [[ "$CONTAINER_MEM_GIB" -gt "$MAX_CONTAINER_GIB" ]]; then
    err "Container cap ${CONTAINER_MEM_GIB} GiB exceeds the hard ceiling ${MAX_CONTAINER_GIB} GiB
       (pool ${MEM_TOTAL_GIB%.*} GiB minus OS_RESERVE_GIB=${OS_RESERVE_GIB}). On unified memory
       this is the line between a killed container and a hung host. Lower the budget."
fi
if python3 -c "import sys; sys.exit(0 if $MEM_AVAIL_GIB < $BUDGET_GIB+4 else 1)"; then
    $DRY_RUN && warn "(dry run) only ${MEM_AVAIL_GIB%.*} GiB available now vs a ${BUDGET_GIB} GiB budget; a real launch refuses"
    $DRY_RUN || err "Only ${MEM_AVAIL_GIB%.*} GiB available now but the GPU budget is ${BUDGET_GIB} GiB (+4 slack).
       vLLM refuses to start when free device memory < GMU x total (v1/worker/utils.py
       request_memory). Something else is holding memory (docker ps; ps --sort=-rss),
       or drop page cache: sudo sysctl -w vm.drop_caches=3"
fi
if [[ "$PLE_OFFLOAD" == "true" ]] && python3 -c "import sys; sys.exit(0 if $MEM_AVAIL_GIB < $CONTAINER_MEM_GIB+4 else 1)"; then
    $DRY_RUN && warn "(dry run) only ${MEM_AVAIL_GIB%.*} GiB available now vs a ${CONTAINER_MEM_GIB} GiB container cap"
    $DRY_RUN || err "Only ${MEM_AVAIL_GIB%.*} GiB available now but the container may use ${CONTAINER_MEM_GIB} GiB.
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
# 4. Patches, overlays, packed PLE table
# ---------------------------------------------------------------------------
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
PLE_PKG="$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ple_layer.py"
MODELOPT_PKG="$VLLM_PKG/model_executor/layers/quantization/modelopt.py"
PATCHED_PLE="$SCRIPT_DIR/files/ple_layer_patched.py"
PATCHED_MODELOPT="$SCRIPT_DIR/files/modelopt_patched.py"
OFFLOAD_DIR="$SCRIPT_DIR/files/ple_offload"
OVERLAY_MOUNTS=()
OVERLAY_ENV=()
OVERLAY_DESTS=()
add_overlay() {          # add_overlay <host file> <container path>
    $DRY_RUN || [[ -f "$1" ]] || err "overlay file missing: $1"
    for existing in "${OVERLAY_DESTS[@]+"${OVERLAY_DESTS[@]}"}"; do
        [[ "$existing" != "$2" ]] || err "overlay conflict on $2 (two files claim it)"
    done
    OVERLAY_MOUNTS+=("-v $1:$2:ro")
    OVERLAY_DESTS+=("$2")
}
extract() {  # <path-in-image> <dest>
    if [[ ! -f "$2" ]]; then
        info "Extracting $(basename "$1") from image..."
        local tmp; tmp=$(docker create "$IMAGE" /bin/true)
        docker cp "$tmp:$1" "$2"
        docker rm "$tmp" >/dev/null 2>&1
    fi
}

PLE_CACHE_HOST="$HOME/.cache/vllm/ple_cache/${ORG}--${NAME}"
PLE_CACHE_CTR="/root/.cache/vllm/ple_cache/${ORG}--${NAME}"
if $DRY_RUN; then
    info "=== Step 4: (dry run) patches/overlays are not regenerated; mounts listed as they would be ==="
else
    info "=== Step 4: Prepare patches ==="
    if ! docker image inspect "$IMAGE" &>/dev/null; then
        info "Pulling $IMAGE ..."
        docker pull "$IMAGE"
    fi
    # 4a. PLE layer: NVFP4 / FP8 table dispatch keyed on text_config.ple_embedding_dtype.
    extract "$PLE_PKG" "$SCRIPT_DIR/files/ple_layer_patched.py.orig"
    python3 "$SCRIPT_DIR/files/patch_ple_layer.py"
    [[ -f "$PATCHED_PLE" ]] || err "PLE patch missing after patch_ple_layer.py"
    # 4b. modelopt.py: MXFP8 fallback -> FP8_BLOCK_SCALES MoE (quantized MTP experts)
    #     -> FP8-dense per-channel dispatch. Same stacking order as start.sh 6b.
    extract "$MODELOPT_PKG" "$SCRIPT_DIR/files/modelopt_patched.py.orig"
    python3 "$SCRIPT_DIR/files/patch_modelopt_mxfp8.py"
    python3 "$SCRIPT_DIR/files/patch_modelopt_fp8_block_moe.py"
    if [[ "$FP8_DENSE" == "true" ]]; then
        python3 "$SCRIPT_DIR/files/stack_modelopt_fp8dense.py" "$SCRIPT_DIR"
    fi
    [[ -f "$PATCHED_MODELOPT" ]] || err "modelopt patch missing"
    # 4c. FP8-dense loader overlay (model / hyperconnection / mtp), as start.sh 4c.
    if [[ "$FP8_DENSE" == "true" ]]; then
        OV="$SCRIPT_DIR/files/overlay"
        python3 "$OV/apply_patches.py"
        if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
            python3 "$SCRIPT_DIR/files/stack_mtp_fp8_draftvocab.py" "$SCRIPT_DIR"
        fi
    elif [[ -n "$MTP_DRAFT_VOCAB" ]]; then
        extract "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py" "$SCRIPT_DIR/files/mtp_patched.py.orig"
        python3 "$SCRIPT_DIR/files/patch_mtp_draft_vocab.py"
    fi
    # 4d. FP8 KV cache (QSA kernels), as start.sh 4f.
    if [[ "$KV_CACHE_DTYPE" == fp8* ]]; then
        extract "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py" "$SCRIPT_DIR/files/qsa_ops_patched.py.orig"
        extract "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py" "$SCRIPT_DIR/files/qsa_nvidia_patched.py.orig"
        python3 "$SCRIPT_DIR/files/patch_qsa_fp8_kv.py"
    fi
    # 4e. PLE offload worker patches (GB10 has no CUDA stream memory ops) — offload mode only.
    if [[ "$PLE_OFFLOAD" == "true" ]]; then
        mkdir -p "$OFFLOAD_DIR/orig"
        extract "$VLLM_PKG/model_executor/layers/ple_offload_layer.py" "$OFFLOAD_DIR/orig/ple_offload_layer.py"
        for f in connector worker protocol; do
            extract "$VLLM_PKG/v1/ple_offload/$f.py" "$OFFLOAD_DIR/orig/$f.py"
        done
        python3 "$SCRIPT_DIR/files/patch_ple_offload.py"
        for f in ple_offload_layer connector worker protocol; do
            [[ -f "$OFFLOAD_DIR/$f.py" ]] || err "offload patch missing: $f.py"
        done
    fi
    # 4f. MTP layer-index alias in the checkpoint config (start.sh 4g); no-op for
    #     checkpoints that already carry mtp.layers.<num_hidden_layers>.* entries.
    rm -f "$SCRIPT_DIR/files/config_patched.json" "$SCRIPT_DIR/files/hf_quant_config_patched.json"
    PATCHED_FILES=$(python3 "$SCRIPT_DIR/files/patch_checkpoint_config.py" "$SNAPSHOT_DIR" "$SCRIPT_DIR/files")
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
        MTP_ALGO=$(python3 "$SCRIPT_DIR/files/patch_checkpoint_config.py" --mtp-moe-algo "$SNAPSHOT_DIR") && MTP_RC=0 || MTP_RC=$?
        if [[ "$MTP_RC" -eq 3 ]]; then
            err "MTP experts are ${MTP_ALGO}, which this image's mixed-precision MoE dispatch cannot build."
        fi
        ok "MTP experts quantization: ${MTP_ALGO:-unquantized} (supported)"
    fi
    ok "Patches ready."
fi
# Mounts (both modes)
add_overlay "$PATCHED_PLE" "$PLE_PKG"
add_overlay "$PATCHED_MODELOPT" "$MODELOPT_PKG"
if [[ "$FP8_DENSE" == "true" ]]; then
    OV="$SCRIPT_DIR/files/overlay"
    add_overlay "$OV/model.py"           "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/model.py"
    add_overlay "$OV/hyperconnection.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/hyperconnection.py"
    if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
        add_overlay "$OV/mtp_draftvocab.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
    else
        add_overlay "$OV/mtp.py"            "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
    fi
elif [[ -n "$MTP_DRAFT_VOCAB" ]]; then
    add_overlay "$SCRIPT_DIR/files/mtp_patched.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
fi
if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
    add_overlay "$MTP_DRAFT_VOCAB" "/etc/vllm-draft-vocab.txt"
    OVERLAY_ENV+=("-e VLLM_MTP_DRAFT_VOCAB=/etc/vllm-draft-vocab.txt")
fi
if [[ "$KV_CACHE_DTYPE" == fp8* ]]; then
    add_overlay "$SCRIPT_DIR/files/qsa_ops_patched.py"    "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
    add_overlay "$SCRIPT_DIR/files/qsa_nvidia_patched.py" "$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py"
fi
if [[ "$PLE_OFFLOAD" == "true" ]]; then
    add_overlay "$OFFLOAD_DIR/ple_offload_layer.py" "$VLLM_PKG/model_executor/layers/ple_offload_layer.py"
    add_overlay "$OFFLOAD_DIR/connector.py"         "$VLLM_PKG/v1/ple_offload/connector.py"
    add_overlay "$OFFLOAD_DIR/worker.py"            "$VLLM_PKG/v1/ple_offload/worker.py"
    add_overlay "$OFFLOAD_DIR/protocol.py"          "$VLLM_PKG/v1/ple_offload/protocol.py"
    OVERLAY_ENV+=("-e VLLM_PLE_CPU_OFFLOAD=1" "-e VLLM_PLE_PACKED_TABLE_DIR=$PLE_CACHE_CTR" "-e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300")
fi
if ! $DRY_RUN; then
    for cfg_name in ${PATCHED_FILES:-}; do
        case "$cfg_name" in
            config.json)          add_overlay "$SCRIPT_DIR/files/config_patched.json" "$CONTAINER_SNAPSHOT/config.json" ;;
            hf_quant_config.json) add_overlay "$SCRIPT_DIR/files/hf_quant_config_patched.json" "$CONTAINER_SNAPSHOT/hf_quant_config.json" ;;
            *) err "unexpected patched config: $cfg_name" ;;
        esac
    done
    [[ -z "${PATCHED_FILES:-}" ]] || ok "MTP experts alias added to: $PATCHED_FILES"
fi
for _kv in $VLLM_EXTRA_ENV; do OVERLAY_ENV+=("-e $_kv"); done

if [[ "$PLE_OFFLOAD" == "true" ]]; then
    if ! ls "$PLE_CACHE_HOST"/*.packed_u8 >/dev/null 2>&1; then
        if $DRY_RUN; then
            info "(dry run) packed PLE table would be built into $PLE_CACHE_HOST"
        else
            info "Building packed PLE table (one-time, ~40 s, <1 GiB RAM, no GPU)..."
            mkdir -p "$PLE_CACHE_HOST"
            docker run --rm --name "${CONTAINER_NAME}-plebuild" --memory 6g --cpus 8 \
                -v "$MODEL_PATH:/m:ro" -v "$HOME/.cache/vllm/ple_cache:/out" \
                -v "$SCRIPT_DIR/files/build_ple_packed_table.py:/b.py:ro" \
                --entrypoint python3 "$IMAGE" -u /b.py "/m/$SNAPSHOT_REL" "/out/${ORG}--${NAME}"
        fi
    fi
    if ls "$PLE_CACHE_HOST"/*.packed_u8 >/dev/null 2>&1; then
        ok "Packed PLE table: $(ls "$PLE_CACHE_HOST"/*.packed_u8 | head -1) ($(du -sh "$PLE_CACHE_HOST" | cut -f1))"
    fi
fi

# hf-overrides: only the PLE dtype, and only when the checkpoint omits it (nvidia/...).
PLE_EMBEDDING_DTYPE="${PLE_EMBEDDING_DTYPE:-$(python3 "$SCRIPT_DIR/files/detect_ple_dtype.py" "$SNAPSHOT_DIR")}"
[[ -z "$PLE_EMBEDDING_DTYPE" ]] || ok "PLE table dtype not declared by checkpoint — overriding to $PLE_EMBEDDING_DTYPE"

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
[[ "$LANGUAGE_MODEL_ONLY" == "true" ]] && VLLM_ARGS+=("--language-model-only")
[[ "$SKIP_MM_PROFILING" == "true" ]] && VLLM_ARGS+=("--skip-mm-profiling")
[[ -n "$KV_CACHE_MEMORY" ]] && VLLM_ARGS+=("--kv-cache-memory-bytes" "$KV_CACHE_MEMORY")
if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then
    if [[ -n "$MTP_DRAFT_VOCAB" ]]; then
        # get_top_tokens (draft-vocab patch) is only reached through use_local_argmax_reduction.
        VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s,\"use_local_argmax_reduction\":true}'" "$MTP_NUM_SPECULATIVE_TOKENS")")
    else
        VLLM_ARGS+=("--speculative-config" "$(printf "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s}'" "$MTP_NUM_SPECULATIVE_TOKENS")")
    fi
fi
# mode 0 (eager): torch.compile/Inductor hangs GB10 for 100+ minutes (RESULTS.md). Never 3.
VLLM_ARGS+=("--compilation-config" "$(printf "'{\"mode\":0,\"cudagraph_mode\":\"%s\"}'" "$CUDAGRAPH_MODE")")
if [[ -n "$PLE_EMBEDDING_DTYPE" ]]; then
    VLLM_ARGS+=("--hf-overrides" "$(printf "'{\"text_config\":{\"ple_embedding_dtype\":\"%s\"}}'" "$PLE_EMBEDDING_DTYPE")")
fi
[[ -n "$EXTRA_VLLM_ARGS" ]] && VLLM_ARGS+=("$EXTRA_VLLM_ARGS")
VLLM_ARGS_STR="${VLLM_ARGS[*]}"
OVERLAY_MOUNTS_STR="${OVERLAY_MOUNTS[*]+"${OVERLAY_MOUNTS[*]}"}"
OVERLAY_ENV_STR="${OVERLAY_ENV[*]+"${OVERLAY_ENV[*]}"}"

info ""
info "Config (single Spark, TP=1):"
info "  Model:      $MODEL_ID"
info "  Image:      $IMAGE"
info "  Context:    $MAX_MODEL_LEN tokens (native rope, no YaRN)"
info "  GMU:        $GPU_MEMORY_UTILIZATION  (budget ${BUDGET_GIB} GiB, cgroup cap ${CONTAINER_MEM_GIB} GiB)"
info "  Max seqs:   $MAX_NUM_SEQS   Batched tokens: $MAX_NUM_BATCHED_TOKENS   KV dtype: $KV_CACHE_DTYPE"
info "  MTP:        $MTP_NUM_SPECULATIVE_TOKENS $( [[ "$MTP_NUM_SPECULATIVE_TOKENS" -eq 0 ]] && echo '(disabled)')  Draft vocab: ${MTP_DRAFT_VOCAB:-full}"
info "  PLE:        $( [[ "$PLE_OFFLOAD" == "true" ]] && echo 'offload (mmap packed table)' || echo 'RESIDENT on GPU')   FP8 dense: $FP8_DENSE   LM-only: $LANGUAGE_MODEL_ONLY"
info "  Graphs:     $CUDAGRAPH_MODE   compile mode 0"
[[ -n "$VLLM_EXTRA_ENV" ]] && info "  Extra env:  $VLLM_EXTRA_ENV"
[[ -n "$EXTRA_VLLM_ARGS" ]] && info "  Extra args: $EXTRA_VLLM_ARGS"
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
    -e HF_HOME=/root/.cache/huggingface \\
    ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \\
    $OVERLAY_ENV_STR \\
    $OVERLAY_MOUNTS_STR \\
    -v $HF_CACHE_DIR:/root/.cache/huggingface \\
    -v $HOME/.cache/vllm:/root/.cache/vllm \\
    $EXTRA_DOCKER_ARGS \\
    $IMAGE \\
    $CONTAINER_SNAPSHOT \\
    $VLLM_ARGS_STR \\
    --host 0.0.0.0 \\
    --port $PORT
LAUNCH_EOF
chmod +x "$LAUNCH_SCRIPT"
cp "$LAUNCH_SCRIPT" "$SCRIPT_DIR/.last_tp1_launch.sh"

if ! $DO_LAUNCH; then
    info "$( $DRY_RUN && echo '--dry-run' || echo '--no-launch' ): command written to .last_tp1_launch.sh"
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
info "Loading weights (~3-12 min). Following logs until ready..."

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
