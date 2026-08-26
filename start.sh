#!/usr/bin/env bash
# =============================================================================
#  start.sh — RadixArk/Qwen3.8-Flash-Next-NVFP4 on 2× DGX Spark (GB10 / SM121)
#
#  Serves https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 with SGLang
#  as an OpenAI-compatible endpoint on 0.0.0.0:8888 (reachable from any LAN
#  interface). TP=2 across two Sparks over the ConnectX-7 RoCE link.
#
#  Model: Qwen3.8-Flash-Next (qwen4_exp — Qwen4-architecture preview)
#    176B total / 6B active · 48 layers (GDN 3 : QSA sparse full-attn 1)
#    · 512 routed experts (top-10) + shared expert · multimodal (vision tower)
#    · 262144-token native context · always-thinking.
#    NVFP4 checkpoint = 135.25 GB:
#      68.0 GB  routed experts, NVFP4 W4A4 (ModelOpt, group 16)
#      52.0 GB  PLE n-gram embedding tables, FP8-E4M3 (fp8-resident at load)
#      16.0 GB  BF16 dense (attn/GDN/routers/shared expert/vision/MTP/lm_head)
#
#  Why two nodes: 135 GB of weights does not fit one 128 GB Spark. With TP=2
#  each node holds ~34 GB (experts) + ~26 GB (PLE, TP-sharded, fp8) + ~8 GB
#  (dense) ≈ 68 GB inside the 0.80 mem-fraction budget (~97 GB of ~121 GB
#  visible), leaving ~29 GB for the QSA KV / GDN state / CUDA-graph pools.
#
#  Draft (speculative decoding) — NOTHING EXTRA TO DOWNLOAD: the multi-step
#  MTP draft layer ships inside this checkpoint (BF16 "mtp.*" tensors, byte-
#  identical to the source), so NEXTN drafting uses the weights this script
#  already downloads. There is no separate DSpark/EAGLE repo for Flash-Next.
#  PLE constraints from the SGLang qwen4_exp implementation: spec decode must
#  be topk=1 (enforced below), NGRAM speculation unsupported, two-batch
#  overlap must stay off (it is).
#
#  Recipe provenance (SGLang cookbook for DGX Spark + house champions):
#   * Cookbook "Qwen3.8-Flash-Next" page — NVFP4 cells:
#       https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
#     NEXTN 3 steps / topk 1 / 4 draft tokens, --mamba-ssm-dtype bfloat16,
#     --reasoning-parser auto, docker image lmsysorg/sglang:qwen38flashnext
#     (model support = sgl-project/sglang PR #36497, baked into the image;
#     arm64 build verified on SM121).
#   * Model card serving recipe (validated TP=2 on GB300/B300):
#     --quantization modelopt_fp4, --page-size 64, mamba extra_buffer +
#     --mamba-track-interval 64, --chunked-prefill-size 4096,
#     --mem-fraction-static 0.80, --context-length 262144, --allow-auto-truncate.
#   * Cookbook "Qwen3.8-27B" DGX Spark cell (GB10/SM121 specifics):
#     --attention-backend flashinfer is required on SM120/SM121 (trtllm_mha is
#     SM100-only); the SM100-gated fast paths (trtllm sparse decode, BF16
#     split-K GEMM) auto-disable here and fall back to triton/cuBLAS.
#   * Cookbook Deployment panel, DGX Spark multi-node docker flags:
#     --ulimit memlock=-1 --cap-add IPC_LOCK --device /dev/infiniband, plus
#     --nnodes/--node-rank/--dist-init-addr for the 2-node rendezvous.
#   * House 2-node SGLang champion (Inkling-Small-NVFP4-Dual-DGX-Sparks):
#     per-node CX7 iface/HCA pinning, NCCL RoCEv2 env, worker-first launch,
#     pinned decode CUDA-graph sizes (default list OOMs the pool),
#     --disable-prefill-cuda-graph on SM121, TORCH_CUDA_ARCH_LIST=12.1a.
#   * NCCL: image bundles 2.29.7; the house GLM-5.2 runbook pins >= 2.30.7 for
#     CUDA-graph + TP on GB10+CX7, so USE_HOST_NCCL=1 LD_PRELOADs the staged
#     ~/nccl-2.30.7 on BOTH nodes by default (set 0 to use the bundled one).
#
#  Cluster assumed (matches this pair of Sparks):
#    head    spark1  10.0.22.1  (CX7 rocep1s0f1 / enp1s0f1np1)  ← run here
#    worker  spark2  10.0.22.2  (CX7 rocep1s0f0 / enp1s0f0np0)
#  All of HEAD_CX7_IP, WORKER_CX7_IP and the worker ssh target are overridable
#  via .env (see .env.example). The worker login user defaults to the SAME user
#  as the head; set WORKER_USER only if the worker uses a different account.
#
#  First boot takes a while: 135 GB download (head) + rsync to the worker,
#  weight load on both nodes, QSA/GDN Triton JIT compilation (cached under
#  .cache/triton for later boots) and CUDA-graph capture. Default readiness
#  timeout is 90 minutes (WAIT_TIMEOUT_MIN).
#
#  SM121 kernel work (DSpark patch, validated on this cluster):
#    The stock lmsysorg/sglang:qwen38flashnext CANNOT serve this model on
#    GB10: the QSA (Qwen sparse attention) backend resolves its packed
#    varlen attention to flash-attn-4's CuTe-DLS kernels, which fail MLIR
#    compilation on SM121. This script builds a derivative image
#    (qwen38-flashnext-dspark:local) that swaps in a Triton
#    FlashDecoding-style fallback — CUDA-graph-safe (cu_seqlens are read
#    on-device, so graph replay works). Validated end-to-end: single-node
#    and 2-node TP2 dummy boots with prefill, decode CUDA graphs and NEXTN
#    speculative decoding. KERNEL_PATCH=0 disables the patch (the stock
#    image then dies at warmup with an MLIRError).
#
#  GB10 WARNING — never probe this model with --load-format dummy:
#    initialize_dummy_weights() materializes an fp16 COPY of the ~26 GB/rank
#    fp8 PLE table, which overcommits unified memory and hard-freezes the
#    whole GB10 node (both machines, twice, empirically — full reboots).
#
#  Usage:
#    ./start.sh                 # preflight → image → download → sync → serve
#    ./start.sh download        # (or --download-only) fetch weights, then exit
#    ./start.sh stop            # tear down both nodes
#    ./start.sh status          # containers / API state on both nodes
#    ./start.sh logs [head|worker]
#    ./start.sh smoke           # one chat completion against :8888
#    ./start.sh doctor          # preflight checks only
#
#  Tunables (env or .env in this directory; shell env wins):
#    PORT=8888                  OpenAI API port (bound on 0.0.0.0, all LANs)
#    KERNEL_PATCH=1             build+use the SM121 QSA fallback image
#    BASE_IMAGE=lmsysorg/sglang:qwen38flashnext    upstream image to patch
#    PATCHED_IMAGE=qwen38-flashnext-dspark:local   derivative image name
#    IMAGE=<name>               use this image verbatim (no patch build)
#    PLE_OFFLOAD=               empty = auto (offload; recommended on GB10)
#                               1 = --ple-offload-embedding, 0 = GPU-resident
#    WORKER_SSH=spark2          ssh target for the worker (user@host form).
#                               Defaults to WORKER_HOST (alias) under the head
#                               user; set WORKER_USER only for a different login.
#    WORKER_USER=                empty = reuse the head login user (most setups)
#    WORKER_HOST=spark2          worker hostname/IP for ssh (or a ~/.ssh/config alias)
#    HEAD_CX7_IP=10.0.22.1       head's CX7 RoCE IP (rendezvous + NCCL rail)
#    WORKER_CX7_IP=10.0.22.2     worker's CX7 RoCE IP
#    HF_HOME=~/.cache/huggingface        head-side HF cache
#    WORKER_HF_HOME=<auto>      worker-side HF cache (default $HOME/.cache/…)
#    DIST_PORT=26400            2-node rendezvous port on the head
#    DOWNLOAD_MODE=rsync        rsync (head→worker) | direct (worker pulls too)
#    MEM_FRACTION_STATIC=0.70   GB10 math: 0.70×122 GB GPU + 26 GB pinned PLE
#                               + OS ≈ 119 of 122 GB — do not raise past ~0.75
#                               unless PLE_OFFLOAD=0
#    CONTEXT_LENGTH=262144      (hard-capped at native 262144; YaRN = EXTRA_ARGS)
#    CHUNKED_PREFILL_SIZE=4096
#    MAX_RUNNING_REQUESTS=16
#    SPEC_STEPS=3 SPEC_TOPK=1 SPEC_DRAFT=4     NEXTN chain (draft = steps + 1)
#    ENABLE_DECODE_GRAPHS=1     0 → --cuda-graph-backend-decode=disabled
#    CUDA_GRAPH_BS="1 2 3 4 5 6 7 8 10 12 14 16"
#    ATTENTION_BACKEND=flashinfer
#    KV_CACHE_DTYPE=            empty = auto (bf16); e.g. fp8_e4m3 (experimental)
#    FP4_GEMM_BACKEND=          empty = auto (proven on SM121 for the 27B)
#    LINEAR_ATTN_PREFILL_BACKEND=  empty = default (triton on SM121)
#    LINEAR_ATTN_DECODE_BACKEND=   empty = default (triton on SM121)
#    MAMBA_RADIX_CACHE_STRATEGY=extra_buffer   (must stay extra_buffer*: page 64)
#    MAMBA_TRACK_INTERVAL=64    (must be a multiple of page size, >= draft 4)
#    MAX_MAMBA_CACHE_SIZE=      empty = derive from MAX_RUNNING_REQUESTS
#    MAMBA_FULL_MEMORY_RATIO=   empty = default
#    SERVED_MODEL_NAME=Qwen3.8-Flash-Next-NVFP4
#    REASONING_PARSER=auto  TOOL_CALL_PARSER=auto   (auto = from chat template)
#    CPUSET=5-9,15-19           GB10 big cores; empty = no pinning
#    USE_HOST_NCCL=1            LD_PRELOAD staged ~/nccl-2.30.7 on both nodes
#    HF_TOKEN=                  picked up from ~/.bashrc if unset
#    HF_REVISION=               empty = main
#    WAIT_TIMEOUT_MIN=90
#    NCCL_DEBUG=WARN
#    EXTRA_ARGS="…"             appended last (argparse last-wins → overrides)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---- .env (shell env wins; .env fills the gaps) ----------------------------
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    if [[ -z "${!key:-}" ]]; then
      # shellcheck disable=SC2163
      export "${key}=${value}"
    fi
  done < "${SCRIPT_DIR}/.env"
fi

# ---- Model / image / cluster ------------------------------------------------
MODEL_ID="RadixArk/Qwen3.8-Flash-Next-NVFP4"
REPO_CACHE_DIRNAME="models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:qwen38flashnext}"
PATCHED_IMAGE="${PATCHED_IMAGE:-qwen38-flashnext-dspark:local}"
KERNEL_PATCH="${KERNEL_PATCH:-1}"
IMAGE="${IMAGE:-}"   # empty = resolve from KERNEL_PATCH; set = verbatim

# 2-node rendezvous over the direct CX7 link (spark1 rocep1s0f1 ↔ spark2 rocep1s0f0)
HEAD_CX7_IP="${HEAD_CX7_IP:-10.0.22.1}"
WORKER_CX7_IP="${WORKER_CX7_IP:-10.0.22.2}"
HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
# Second rail (spark1 roceP2p1s0f1 ↔ spark2 roceP2p1s0f0, 10.0.22.101↔.102) is
# wired on this pair; to use it set e.g.:
#   HEAD_CX7_IB="rocep1s0f1,roceP2p1s0f1" WORKER_CX7_IB="rocep1s0f0,roceP2p1s0f0" NCCL_CROSS_NIC=1

# Worker SSH target. Most setups run the SAME user on both nodes, so the
# worker user is optional — leave WORKER_USER empty to reuse the head login
# user (ssh's own default). Set WORKER_HOST to the worker's hostname/IP for
# ssh; leave it at the alias "spark2" if you have a ~/.ssh/config entry.
# WORKER_SSH, if set explicitly, overrides everything (use user@host form).
WORKER_USER="${WORKER_USER:-}"
WORKER_HOST="${WORKER_HOST:-spark2}"
if [[ -n "${WORKER_SSH:-}" ]]; then
  : # explicit override wins
elif [[ -n "${WORKER_USER}" ]]; then
  WORKER_SSH="${WORKER_USER}@${WORKER_HOST}"
else
  WORKER_SSH="${WORKER_HOST}"   # ssh alias — uses the head user by default
fi
SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o BatchMode=yes)

HEAD_CONTAINER="qwen38-flash-next-head"
WORKER_CONTAINER="qwen38-flash-next-worker"

# ---- Serving knobs ----------------------------------------------------------
PORT="${PORT:-8888}"
HOST_BIND="${HOST_BIND:-0.0.0.0}"
DIST_PORT="${DIST_PORT:-26400}"
TP_SIZE=2
NNODES=2
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next-NVFP4}"

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"   # GB10: leaves DRAM for the ~26 GB/rank pinned PLE table
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"
ENABLE_DECODE_GRAPHS="${ENABLE_DECODE_GRAPHS:-1}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-"1 2 3 4 5 6 7 8 10 12 14 16"}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
PLE_OFFLOAD="${PLE_OFFLOAD:-}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-}"
LINEAR_ATTN_PREFILL_BACKEND="${LINEAR_ATTN_PREFILL_BACKEND:-}"
LINEAR_ATTN_DECODE_BACKEND="${LINEAR_ATTN_DECODE_BACKEND:-}"
MAMBA_RADIX_CACHE_STRATEGY="${MAMBA_RADIX_CACHE_STRATEGY:-extra_buffer}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-64}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-}"
REASONING_PARSER="${REASONING_PARSER:-auto}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-auto}"
CPUSET="${CPUSET:-5-9,15-19}"
USE_HOST_NCCL="${USE_HOST_NCCL:-1}"
DOWNLOAD_MODE="${DOWNLOAD_MODE:-rsync}"
HF_REVISION="${HF_REVISION:-}"
WAIT_TIMEOUT_MIN="${WAIT_TIMEOUT_MIN:-90}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---- Paths / logs -----------------------------------------------------------
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRIPT_DIR}/.cache/triton}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-${SCRIPT_DIR}/.cache/flashinfer}"
LOG_FILE="${SCRIPT_DIR}/.sglang.log"
WORKER_LOG_FILE="${SCRIPT_DIR}/.sglang-worker.log"
PID_FILE="${SCRIPT_DIR}/.sglang.pid"
READY_URL="http://127.0.0.1:${PORT}/v1/models"
NCCL_SO_NAME="libnccl.so.2.30.7"

# ---- Logging helpers --------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()   { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()     { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header() { echo -e "\n${BOLD}${CYAN}━━━ $* ━━━${NC}"; }

# ---- Remote helpers (worker = spark2 via ssh alias) -------------------------
wrun() { # remote shell snippet
  ssh "${SSH_OPTS[@]}" "${WORKER_SSH}" "$1"
}
worker_docker() { # docker <argv…> on the worker, each arg shell-quoted
  local q=() a
  for a in "$@"; do q+=("$(printf '%q' "$a")"); done
  wrun "docker ${q[*]}"
}
remote_home() { wrun 'echo $HOME'; }

# ---- HF token (from ~/.bashrc if not exported; public repo, rate limits) ----
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.bashrc" ]]; then
  HF_TOKEN="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?HF_TOKEN=["'"'"']\?\([A-Za-z0-9_-]\+\).*/\2/p' "${HOME}/.bashrc" | head -1)"
fi
export HF_TOKEN="${HF_TOKEN:-}"

# ---- Argument dispatch ------------------------------------------------------
ACTION="${1:-serve}"
case "${ACTION}" in
  serve|"") ;;
  download|--download-only) ACTION="download" ;;
  stop|status|logs|smoke|doctor|-h|--help) ;;
  *) echo "Unknown argument: ${ACTION}"; echo "Usage: $0 [serve|download|stop|status|logs|smoke|doctor]"; exit 1 ;;
esac
if [[ "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  sed -n '2,120p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

# =============================================================================
# Preflight
# =============================================================================
preflight() { # $1 = action; port checks only matter when serving
  header "Preflight"

  command -v docker >/dev/null 2>&1 || { error "docker not on PATH"; exit 1; }
  command -v curl   >/dev/null 2>&1 || { error "curl not on PATH";   exit 1; }
  docker info >/dev/null 2>&1        || { error "Docker daemon unreachable"; exit 1; }
  ok "docker (head)"

  nvidia-smi -L >/dev/null 2>&1      || { error "nvidia-smi failed on head"; exit 1; }
  local gpu; gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  [[ "${gpu}" == *"GB10"* ]] || warn "head GPU is '${gpu}' — this recipe targets DGX Spark (GB10/SM121)"
  ok "head GPU: ${gpu}"

  local mem_gib=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
  if (( mem_gib < 110 )); then
    error "head has only ${mem_gib} GiB RAM — 2×TP shard needs ~68 GB weights + pools"
    exit 1
  fi
  ok "head RAM: ${mem_gib} GiB (unified)"

  # Worker reachability + docker + GPU
  wrun "echo ssh-ok" >/dev/null 2>&1 || { error "cannot SSH to worker '${WORKER_SSH}' (see ~/.ssh/config; override with WORKER_SSH=user@host)"; exit 1; }
  ok "SSH to worker ${WORKER_SSH}"
  worker_docker info >/dev/null 2>&1 || { error "docker over SSH on worker failed"; exit 1; }
  ok "docker (worker)"
  local wgpu; wgpu="$(wrun "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1" || true)"
  [[ "${wgpu}" == *"GB10"* ]] || warn "worker GPU is '${wgpu}' — expected GB10"
  ok "worker GPU: ${wgpu:-unknown}"
  local wmem; wmem="$(wrun "awk '/MemTotal/ {print \$2}' /proc/meminfo")"
  wmem=$(( wmem / 1024 / 1024 ))
  if (( wmem < 110 )); then error "worker has only ${wmem} GiB RAM"; exit 1; fi
  ok "worker RAM: ${wmem} GiB (unified)"

  # Direct CX7 link (rendezvous + NCCL rail)
  if ping -c 1 -W 2 "${WORKER_CX7_IP}" >/dev/null 2>&1; then
    ok "CX7 link head ${HEAD_CX7_IP} ↔ worker ${WORKER_CX7_IP} reachable"
  else
    warn "cannot ping ${WORKER_CX7_IP} — 2-node rendezvous may fail"
  fi

  # Disk: needs depend on what is already cached (weights are ~135 GB)
  mkdir -p "${HF_HOME}"
  local need=150 warn_at=170 free
  resolve_snapshot >/dev/null 2>&1 && need=20
  free=$(( $(df -Pk "${HF_HOME}" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
  if (( free < need )); then error "head: ${free} GiB free under ${HF_HOME} — need >= ${need} GiB"; exit 1; fi
  (( free < warn_at )) && warn "head: only ${free} GiB free (weights are ~135 GiB)"
  ok "head disk: ${free} GiB free (need ${need})"
  local wfree wneed=150
  wrun "test -f '$(worker_hf_home)/hub/${REPO_CACHE_DIRNAME}/snapshots'/*/config.json" 2>/dev/null && wneed=20
  wfree="$(wrun "df -Pk \$(echo \$HOME)/.cache 2>/dev/null || df -Pk \$HOME" | awk 'NR==2 {print $4}')"
  wfree=$(( wfree / 1024 / 1024 ))
  if (( wfree < wneed )); then error "worker: ${wfree} GiB free — need >= ${wneed} GiB"; exit 1; fi
  (( wfree < warn_at )) && warn "worker: only ${wfree} GiB free"
  ok "worker disk: ${wfree} GiB free (need ${wneed})"

  # Ports on the head (host network) — only when we are about to serve
  if [[ "${1:-serve}" == "serve" ]]; then
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") >/dev/null 2>&1; then
      error "port ${PORT} already in use on head — run './start.sh stop' first or set PORT"
      exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${DIST_PORT}") >/dev/null 2>&1; then
      warn "dist port ${DIST_PORT} in use on head — override with DIST_PORT"
    fi
    ok "ports ${PORT} (API) and ${DIST_PORT} (rendezvous) free on head"
  fi

  # Config sanity
  if (( CONTEXT_LENGTH > 262144 )); then
    error "CONTEXT_LENGTH=${CONTEXT_LENGTH} exceeds the native 262144 (YaRN for this model is untested here — use EXTRA_ARGS if you want to experiment)"
    exit 1
  fi
  if [[ "${SPEC_TOPK}" != "1" ]]; then
    error "SPEC_TOPK must be 1: Qwen4-Exp PLE speculative decoding supports only topk=1"
    exit 1
  fi
  if (( SPEC_DRAFT != SPEC_STEPS + 1 )); then
    error "SPEC_DRAFT must equal SPEC_STEPS + 1 for topk=1 (got steps=${SPEC_STEPS}, draft=${SPEC_DRAFT})"
    exit 1
  fi
  if (( MAMBA_TRACK_INTERVAL % 64 != 0 )) || (( MAMBA_TRACK_INTERVAL < SPEC_DRAFT )); then
    error "MAMBA_TRACK_INTERVAL must be a multiple of page size 64 and >= SPEC_DRAFT (${SPEC_DRAFT})"
    exit 1
  fi
  case "${MAMBA_RADIX_CACHE_STRATEGY}" in
    extra_buffer|extra_buffer_lazy) ;;
    *) error "MAMBA_RADIX_CACHE_STRATEGY must be extra_buffer or extra_buffer_lazy (compressed QSA pins page-size 64; MambaRadixCache needs the extra-buffer strategy)"; exit 1 ;;
  esac
  ok "recipe constraints valid (NEXTN ${SPEC_STEPS}/${SPEC_TOPK}/${SPEC_DRAFT}, page 64, track ${MAMBA_TRACK_INTERVAL})"
}

# =============================================================================
# Images: base pull (head + worker) and the SM121 kernel-patch derivative
# =============================================================================
ensure_base_image() {
  header "Base image: ${BASE_IMAGE}"
  if docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    ok "base image present on head"
  else
    info "pulling ${BASE_IMAGE} on head (arm64, ~30 GB)…"
    docker pull "${BASE_IMAGE}" || { error "pull failed on head"; exit 1; }
    ok "base image pulled on head"
  fi
  if worker_docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    ok "base image present on worker"
  else
    info "base image missing on worker — trying 'docker load' there…"
    if docker save "${BASE_IMAGE}" | worker_docker load >/dev/null 2>&1; then
      ok "base image loaded on worker"
    else
      warn "worker load failed — syncing image head → worker via docker save/load"
    fi
  fi
}

# SM121 QSA fallback kernel (see header). Written as a docker build context
# under .patch/ and built identically on both nodes.
write_patch_context() {
  mkdir -p "${SCRIPT_DIR}/.patch"
  cat > "${SCRIPT_DIR}/.patch/qsa_fa_fallback.py" <<'QSA_EOF'
"""SM121 (DGX Spark / GB10) fallback varlen attention for Qwen sparse attention.

Upstream `qwen_sparse_attn_backend._resolve_flash_attn_varlen_func` prefers
classic FA2 and otherwise falls back to flash-attn-4's CuTe DSL interface.
FA4's cute kernels do not compile on SM121 (MLIR layout-congruence error),
so this module provides a drop-in `flash_attn_varlen_func` backed by a Triton
FlashDecoding-style kernel specialized for the QSA call contract:

  * every "sequence" has exactly ONE query row (cu_seqlens_q = arange),
  * variable numbers of gathered KV rows per sequence (<= topk),
  * GQA (num_q_heads a multiple of num_kv_heads),
  * cu_seqlens_* are read on-device, so the kernel is CUDA-graph replay safe
    (the QSA backend rewrites cu_seqlens_k contents during graph replay).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _varlen_one_q_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    sm_scale,
    HQ: tl.constexpr, HKV: tl.constexpr,
    D: tl.constexpr, D_PAD: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    q_stride_t: tl.constexpr, q_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr, k_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr, v_stride_h: tl.constexpr,
    o_stride_t: tl.constexpr, o_stride_h: tl.constexpr,
):
    """One program per (query row, q head): online-softmax attention over the
    gathered KV rows of that query, 1-token query, causal is a no-op."""
    row = tl.program_id(0)
    head = tl.program_id(1)

    # QSA always emits one query per varlen sequence.
    q_start = tl.load(cu_seqlens_q_ptr + row)
    k_start = tl.load(cu_seqlens_k_ptr + row)
    k_end = tl.load(cu_seqlens_k_ptr + row + 1)

    offs_d = tl.arange(0, D_PAD)
    mask_d = offs_d < D

    q = tl.load(
        q_ptr + (q_start * q_stride_t) + (head * q_stride_h) + offs_d,
        mask=mask_d, other=0.0,
    ).to(tl.float32)

    kh = head // (HQ // HKV)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D_PAD], dtype=tl.float32)

    for k0 in range(k_start, k_end, BLOCK_KV):
        offs_kv = k0 + tl.arange(0, BLOCK_KV)
        mask_kv = offs_kv < k_end
        kv_ptrs = (offs_kv * k_stride_t) + (kh * k_stride_h)
        k_blk = tl.load(
            k_ptr + kv_ptrs[:, None] + offs_d[None, :],
            mask=mask_kv[:, None] & mask_d[None, :], other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * k_blk, axis=1) * sm_scale
        scores = tl.where(mask_kv, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        v_ptrs = (offs_kv * v_stride_t) + (kh * v_stride_h)
        v_blk = tl.load(
            v_ptr + v_ptrs[:, None] + offs_d[None, :],
            mask=mask_kv[:, None] & mask_d[None, :], other=0.0,
        ).to(tl.float32)
        acc += tl.sum(p[:, None] * v_blk, axis=0)
        m_i = m_new

    out = acc / tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(
        o_ptr + (row * o_stride_t) + (head * o_stride_h) + offs_d,
        out.to(o_ptr.dtype.element_ty),
        mask=mask_d,
    )


def _next_pow2(n: int) -> int:
    return triton.next_power_of_2(max(n, 16))


def triton_varlen_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 0,
    softmax_scale: float = 1.0,
    causal: bool = True,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for flash_attn.flash_attn_varlen_func (QSA calls).

    Supports the exact shape the QSA sparse backend issues: one query row per
    varlen sequence, GQA, matching q/k/v dtypes (bf16/fp16), any head_dim.
    """
    if not q.is_cuda:
        raise RuntimeError("qsa_fa_fallback requires CUDA tensors")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise RuntimeError(f"expected 3D q/k/v, got {q.shape}/{k.shape}/{v.shape}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise RuntimeError(
            f"qsa_fa_fallback: q/k/v dtypes must match "
            f"({q.dtype}/{k.dtype}/{v.dtype}); keep KV cache in model dtype"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"qsa_fa_fallback: unsupported dtype {q.dtype}")

    total_q, hq, d = q.shape
    total_k, hkv, dk = k.shape
    if dk != d or v.shape[2] != d:
        raise RuntimeError("head_dim mismatch")
    if hq % hkv != 0:
        raise RuntimeError(f"GQA mismatch: {hq} q heads vs {hkv} kv heads")

    num_seqs = total_q
    if cu_seqlens_k.numel() != num_seqs + 1:
        raise RuntimeError("cu_seqlens_k size mismatch")

    # Host-side syncs (torch.equal) are illegal inside CUDA graph capture;
    # the eager path has already validated the one-query-per-sequence contract
    # before any graph is captured, so skip the check while capturing.
    if not torch.cuda.is_current_stream_capturing():
        q_lens_ok = bool(
            torch.equal(
                cu_seqlens_q[1:] - cu_seqlens_q[:-1],
                torch.ones_like(cu_seqlens_q[1:]),
            )
        )
        if not q_lens_ok:
            raise RuntimeError(
                "qsa_fa_fallback only supports q_len==1 per varlen sequence "
                "(the QSA backend's only call shape)"
            )

    if cu_seqlens_q.dtype != torch.int32:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)

    q_c = q if q.is_contiguous() else q.contiguous()
    k_c = k if k.is_contiguous() else k.contiguous()
    v_c = v if v.is_contiguous() else v.contiguous()
    out = torch.empty_like(q_c)

    BLOCK_KV = 64
    grid = (num_seqs, hq)
    _varlen_one_q_attn_kernel[grid](
        q_c, k_c, v_c, out,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        HQ=hq, HKV=hkv,
        D=d, D_PAD=_next_pow2(d),
        BLOCK_KV=BLOCK_KV,
        q_stride_t=q_c.stride(0), q_stride_h=q_c.stride(1),
        k_stride_t=k_c.stride(0), k_stride_h=k_c.stride(1),
        v_stride_t=v_c.stride(0), v_stride_h=v_c.stride(1),
        o_stride_t=out.stride(0), o_stride_h=out.stride(1),
        num_warps=4,
    )
    return out
QSA_EOF

  cat > "${SCRIPT_DIR}/.patch/Dockerfile" <<DKR_EOF
# DSpark kernel-work image for Qwen3.8-Flash-Next-NVFP4 on DGX Spark (SM121/GB10).
# See start.sh header for the full story.
FROM ${BASE_IMAGE}

COPY qsa_fa_fallback.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_fa_fallback.py
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
s = open(p).read()
anchor = "    try:\n        from flash_attn import flash_attn_varlen_func"
patch = (
    "    from sglang.srt.utils import is_sm100_supported\n"
    "    if not is_sm100_supported():\n"
    "        from sglang.srt.layers.attention.qsa_fa_fallback import triton_varlen_attn_func\n"
    "        return triton_varlen_attn_func\n"
) + anchor
assert anchor in s, "anchor not found — upstream image layout changed"
assert "qsa_fa_fallback" not in s, "already patched"
open(p, "w").write(s.replace(anchor, patch, 1))
print("qwen_sparse_attn_backend.py patched for SM121")
PYEOF
DKR_EOF
}

ensure_patched_image() {
  header "SM121 kernel patch → ${PATCHED_IMAGE}"
  write_patch_context
  local stamp
  stamp="$(cat "${SCRIPT_DIR}/.patch/qsa_fa_fallback.py" "${SCRIPT_DIR}/.patch/Dockerfile" | sha256sum | cut -d' ' -f1)"

  if docker image inspect "${PATCHED_IMAGE}" >/dev/null 2>&1 \
     && [[ "$(cat "${SCRIPT_DIR}/.patch/.stamp" 2>/dev/null)" == "${stamp}" ]]; then
    ok "patched image present on head (context unchanged)"
  else
    docker build -t "${PATCHED_IMAGE}" "${SCRIPT_DIR}/.patch" || { error "failed to build ${PATCHED_IMAGE} on head"; exit 1; }
    echo "${stamp}" > "${SCRIPT_DIR}/.patch/.stamp"
    ok "patched image built on head"
  fi

  rsync -a -e "ssh ${SSH_OPTS[*]}" "${SCRIPT_DIR}/.patch/" "${WORKER_SSH}:/tmp/qwen38-dspark-patch/"
  if worker_docker image inspect "${PATCHED_IMAGE}" >/dev/null 2>&1 \
     && [[ "$(wrun "cat /tmp/qwen38-dspark-patch/.stamp 2>/dev/null")" == "${stamp}" ]]; then
    ok "patched image present on worker (context unchanged)"
  else
    wrun "cd /tmp/qwen38-dspark-patch && docker build -t '${PATCHED_IMAGE}' ." \
      || { error "failed to build ${PATCHED_IMAGE} on worker"; exit 1; }
    echo "${stamp}" | ssh "${SSH_OPTS[@]}" "${WORKER_SSH}" "cat > /tmp/qwen38-dspark-patch/.stamp"
    ok "patched image built on worker"
  fi
}

ensure_images() {
  if [[ -n "${IMAGE}" ]]; then
    header "Docker image: ${IMAGE} (explicit override — no patch build)"
    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
      docker pull "${IMAGE}" || { error "pull failed on head"; exit 1; }
    fi
    if ! worker_docker image inspect "${IMAGE}" >/dev/null 2>&1; then
      worker_docker pull "${IMAGE}" || { error "image missing on worker and pull failed"; exit 1; }
    fi
    ok "image ready on both nodes"
    return
  fi
  ensure_base_image
  if [[ "${KERNEL_PATCH}" == "1" ]]; then
    ensure_patched_image
    IMAGE="${PATCHED_IMAGE}"
  else
    warn "KERNEL_PATCH=0 — the stock image cannot serve Qwen4Exp on SM121 (QSA FA4-cute MLIR failure)"
    IMAGE="${BASE_IMAGE}"
  fi
}

# =============================================================================
# Weights: download on head, verify, sync to worker
# =============================================================================
snapshot_root() { echo "${HF_HOME}/hub/${REPO_CACHE_DIRNAME}"; }

resolve_snapshot() { # → absolute host path of the snapshot (or fail)
  local base snap
  base="$(snapshot_root)"
  [[ -d "${base}/snapshots" ]] || return 1
  if [[ -n "${HF_REVISION}" && -f "${base}/snapshots/${HF_REVISION}/config.json" ]]; then
    echo "${base}/snapshots/${HF_REVISION}"; return 0
  fi
  if [[ -f "${base}/refs/main" ]]; then
    local hash; hash="$(tr -d ' \n' <"${base}/refs/main")"
    snap="${base}/snapshots/${hash}"
    [[ -f "${snap}/config.json" ]] && { echo "${snap}"; return 0; }
  fi
  for snap in "${base}"/snapshots/*/; do
    [[ -f "${snap}config.json" ]] && { echo "${snap%/}"; return 0; }
  done
  return 1
}

verify_snapshot() { # <snapshot_dir> <label> — byte-size check against the HF API
  local dir="$1" label="$2"
  python3 - "${MODEL_ID}" "${dir}" "${label}" <<'PY' || return 1
import json, os, sys, urllib.request
model, snap, label = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"https://huggingface.co/api/models/{model}?blobs=true"
try:
    with urllib.request.urlopen(url, timeout=30) as r:
        files = [(s["rfilename"], s.get("size") or 0) for s in json.load(r)["siblings"]]
except Exception as e:
    print(f"[WARN]  HF API unreachable ({e}); falling back to heuristic check")
    need = ("config.json", "model.safetensors.index.json")
    st = sum(1 for f in os.listdir(snap) if f.endswith(".safetensors"))
    if all(os.path.exists(os.path.join(snap, n)) for n in need) and st >= 400:
        print(f"[OK]    {label}: heuristic check passed ({st} safetensors)")
        sys.exit(0)
    print(f"[ERROR] {label}: heuristic check failed ({st} safetensors)")
    sys.exit(1)
missing, bad = [], []
for name, size in files:
    p = os.path.join(snap, name)
    if not os.path.exists(p):
        missing.append(name); continue
    try:
        actual = os.stat(p).st_size  # snapshot files are symlinks → blobs
    except OSError:
        bad.append(name); continue
    if size and actual != size:
        bad.append(f"{name} ({actual} != {size})")
if missing or bad:
    for n in missing[:5]: print(f"[ERROR] {label}: missing {n}" + (" …" if len(missing) > 5 else ""))
    for n in bad[:5]:    print(f"[ERROR] {label}: size mismatch {n}" + (" …" if len(bad) > 5 else ""))
    sys.exit(1)
total = sum(s for _, s in files)
print(f"[OK]    {label}: all {len(files)} files verified ({total/1e9:.2f} GB)")
PY
}

download_on_head() {
  local snap
  if snap="$(resolve_snapshot)" && verify_snapshot "${snap}" "head cache"; then
    ok "weights already cached on head: ${snap}"
    return 0
  fi
  info "downloading ${MODEL_ID} (~135 GB) into ${HF_HOME} …"
  mkdir -p "${HF_HOME}"
  local rev=()
  [[ -n "${HF_REVISION}" ]] && rev=(--revision "${HF_REVISION}")
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" hf download "${MODEL_ID}" "${rev[@]}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="${HF_HOME}" huggingface-cli download "${MODEL_ID}" "${rev[@]}"
  else
    docker run --rm --network host \
      -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" \
      -v "${HF_HOME}:/root/.cache/huggingface" \
      "${IMAGE}" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID}', token=None, revision='${HF_REVISION}' or None)
"
  fi
  snap="$(resolve_snapshot)" || { error "download finished but no snapshot found"; exit 1; }
  verify_snapshot "${snap}" "head cache" || { error "download incomplete — rerun (hf download resumes)"; exit 1; }
  ok "weights downloaded on head"
}

WORKER_HF_HOME_RESOLVED=""
worker_hf_home() {
  if [[ -z "${WORKER_HF_HOME_RESOLVED}" ]]; then
    if [[ -n "${WORKER_HF_HOME:-}" ]]; then
      WORKER_HF_HOME_RESOLVED="${WORKER_HF_HOME}"
    else
      WORKER_HF_HOME_RESOLVED="$(remote_home)/.cache/huggingface"
    fi
  fi
  echo "${WORKER_HF_HOME_RESOLVED}"
}

sync_weights_to_worker() {
  header "Weights on worker (${DOWNLOAD_MODE})"
  local whf wrepo
  whf="$(worker_hf_home)"
  wrepo="${whf}/hub/${REPO_CACHE_DIRNAME}"
  wrun "mkdir -p '${wrepo}'"

  if [[ "${DOWNLOAD_MODE}" == "direct" ]]; then
    local have=0
    wrun "test -f '${wrepo}/snapshots'/*/config.json" 2>/dev/null && have=1 || true
    if (( have )) && worker_docker run --rm --network host \
        -e HF_HOME=/root/.cache/huggingface \
        -v "${whf}:/root/.cache/huggingface" \
        "${IMAGE}" python3 -c "
import glob, sys
sys.exit(0 if glob.glob('/root/.cache/huggingface/hub/${REPO_CACHE_DIRNAME}/snapshots/*/model.safetensors.index.json') else 1)
" >/dev/null 2>&1; then
      ok "weights already on worker"
      return 0
    fi
    info "downloading ${MODEL_ID} on worker (docker, ~135 GB) …"
    worker_docker run --rm --network host \
      -e HF_HOME=/root/.cache/huggingface -e HF_TOKEN="${HF_TOKEN:-}" \
      -v "${whf}:/root/.cache/huggingface" \
      "${IMAGE}" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID}', token=None, revision='${HF_REVISION}' or None)
" || { error "worker download failed — try DOWNLOAD_MODE=rsync"; exit 1; }
    ok "weights downloaded on worker"
    return 0
  fi

  # rsync head → worker over the CX7 link (single internet download; resumable)
  # Count under snapshots/ only — .locks lives in the repo dir on the head but
  # is excluded from the rsync, so counting the whole repo would never match.
  local hcount wcount
  hcount="$(find "$(snapshot_root)/snapshots" \( -type f -o -type l \) 2>/dev/null | wc -l)"
  wcount="$(wrun "find '${wrepo}/snapshots' \( -type f -o -type l \) 2>/dev/null | wc -l" || echo 0)"
  if [[ "${hcount}" == "${wcount}" ]] && wrun "test -f '${wrepo}/snapshots'/*/config.json" 2>/dev/null; then
    ok "weights already synced to worker (${wcount} files)"
    return 0
  fi
  info "rsyncing weights head → worker (~135 GB over the CX7 link, resumable) …"
  rsync -a --partial --info=progress2,stats1 --exclude '.locks' \
    -e "ssh ${SSH_OPTS[*]}" \
    "$(snapshot_root)/" "${WORKER_SSH}:${wrepo}/" \
    || { error "rsync failed — rerun to resume, or set DOWNLOAD_MODE=direct"; exit 1; }
  wcount="$(wrun "find '${wrepo}/snapshots' \( -type f -o -type l \) 2>/dev/null | wc -l" || echo 0)"
  if [[ "${hcount}" != "${wcount}" ]]; then
    error "worker file count ${wcount} != head ${hcount} — rerun to resume"
    exit 1
  fi
  wrun "test -f '${wrepo}/snapshots'/*/config.json" 2>/dev/null || { error "worker snapshot incomplete"; exit 1; }
  ok "weights synced to worker (${wcount} files)"
}

# =============================================================================
# Launch
# =============================================================================
container_path() { # host snapshot path → path inside the container
  local p="$1" mapped
  mapped="${p/#${HF_HOME}//root/.cache/huggingface}"
  [[ "${mapped}" == "${p}" ]] && { error "snapshot ${p} is not under HF_HOME=${HF_HOME}"; exit 1; }
  echo "${mapped}"
}

build_sglang_args() { # $1 = container model path
  local model_path="$1"
  SGLANG_ARGS=(
    --model-path "${model_path}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --trust-remote-code
    --tp-size "${TP_SIZE}"
    --quantization modelopt_fp4
    --attention-backend "${ATTENTION_BACKEND}"
    # page 64 is pinned unconditionally by SGLang for compressed QSA
    # (full_slot // compress_ratio addressing); explicit here for clarity.
    --page-size 64
    --mamba-ssm-dtype bfloat16
    --mamba-radix-cache-strategy "${MAMBA_RADIX_CACHE_STRATEGY}"
    --mamba-track-interval "${MAMBA_TRACK_INTERVAL}"
    # PLE placement: the checkpoint's ~51 GB fp8 n-gram table is CPU-offloaded
    # by SGLang's auto-rule (CUDA + bf16). At TP=2 that is ~26 GB of pinned
    # host memory per node — validated safe on GB10. PLE_OFFLOAD=0 forces the
    # table GPU-resident (~+26 GB GPU weights/rank; only viable with a lower
    # mem-fraction), PLE_OFFLOAD=1 forces offload.
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --context-length "${CONTEXT_LENGTH}"
    # NEXTN speculative decoding — draft = the in-checkpoint MTP layer.
    --speculative-algorithm NEXTN
    --speculative-num-steps "${SPEC_STEPS}"
    --speculative-eagle-topk "${SPEC_TOPK}"
    --speculative-num-draft-tokens "${SPEC_DRAFT}"
    --reasoning-parser "${REASONING_PARSER}"
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --allow-auto-truncate
    --enable-metrics
    --enable-cache-report
    # SM121 house rule (27B Spark cell / Inkling champion): prefill CUDA
    # graphs are unreliable on GB10.
    --disable-prefill-cuda-graph
    --host "${HOST_BIND}"
    --port "${PORT}"
  )
  [[ -n "${KV_CACHE_DTYPE}" ]]              && SGLANG_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
  [[ "${PLE_OFFLOAD}" == "1" ]]             && SGLANG_ARGS+=(--ple-offload-embedding)
  [[ "${PLE_OFFLOAD}" == "0" ]]             && SGLANG_ARGS+=(--no-ple-offload-embedding)
  [[ -n "${FP4_GEMM_BACKEND}" ]]            && SGLANG_ARGS+=(--fp4-gemm-backend "${FP4_GEMM_BACKEND}")
  [[ -n "${LINEAR_ATTN_PREFILL_BACKEND}" ]] && SGLANG_ARGS+=(--linear-attn-prefill-backend "${LINEAR_ATTN_PREFILL_BACKEND}")
  [[ -n "${LINEAR_ATTN_DECODE_BACKEND}" ]]  && SGLANG_ARGS+=(--linear-attn-decode-backend "${LINEAR_ATTN_DECODE_BACKEND}")
  [[ -n "${MAX_MAMBA_CACHE_SIZE}" ]]        && SGLANG_ARGS+=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
  [[ -n "${MAMBA_FULL_MEMORY_RATIO}" ]]     && SGLANG_ARGS+=(--mamba-full-memory-ratio "${MAMBA_FULL_MEMORY_RATIO}")
  if [[ "${ENABLE_DECODE_GRAPHS}" == "1" ]]; then
    # Pin the decode-graph batch list (the default list OOMs the graph pool
    # on GB10). Extend alongside MAX_RUNNING_REQUESTS.
    read -ra _bs <<< "${CUDA_GRAPH_BS}"
    SGLANG_ARGS+=(--cuda-graph-bs-decode "${_bs[@]}")
  else
    SGLANG_ARGS+=(--cuda-graph-backend-decode=disabled)
  fi
  # Free-form overrides go last (argparse last-wins).
  read -ra _extra <<< "${EXTRA_ARGS}"
  if [[ ${#_extra[@]} -gt 0 ]]; then SGLANG_ARGS+=("${_extra[@]}"); fi
  return 0
}

common_docker_env() { # echoes "-e K=V" pairs shared by both nodes (callers word-split)
  local vars=(
    "NCCL_IB_DISABLE=0"
    "NCCL_IB_ROCE_VERSION_NUM=2"
    "NCCL_IB_GID_INDEX=3"
    "NCCL_NET=IB"
    "NCCL_NET_PLUGIN=none"
    "NCCL_NVLS_ENABLE=0"
    "NCCL_CUMEM_ENABLE=0"
    "NCCL_CROSS_NIC=${NCCL_CROSS_NIC}"
    "NCCL_IGNORE_CPU_AFFINITY=1"
    "NCCL_DEBUG=${NCCL_DEBUG}"
    "TORCH_CUDA_ARCH_LIST=12.1a"
    "FLASHINFER_CUDA_ARCH_LIST=12.1a"
    "HF_HOME=/root/.cache/huggingface"
    "HF_HUB_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "TRITON_CACHE_DIR=/root/.triton"
    "HF_TOKEN=${HF_TOKEN:-}"
  )
  local v out=""
  for v in "${vars[@]}"; do out+=" -e ${v}"; done
  echo "${out# }"
}

launch() {
  header "Launch — TP=${TP_SIZE} across ${NNODES} nodes, image ${IMAGE}"

  local snap container_model
  snap="$(resolve_snapshot)" || { error "weights not resolved on head — run './start.sh download' first"; exit 1; }
  container_model="$(container_path "${snap}")"
  build_sglang_args "${container_model}"

  # NCCL ≥ 2.30.7 staged on both nodes (house GB10+CX7 cudagraph/TP pin).
  local head_nccl_dir="${NCCL_HOST_DIR:-${HOME}/nccl-2.30.7}"
  local worker_nccl_dir="${WORKER_NCCL_HOST_DIR:-$(remote_home)/nccl-2.30.7}"
  local head_preload=() worker_preload=()
  if [[ "${USE_HOST_NCCL}" == "1" ]]; then
    if [[ -f "${head_nccl_dir}/${NCCL_SO_NAME}" ]]; then
      head_preload=(-v "${head_nccl_dir}:/nccl:ro" -e LD_PRELOAD="/nccl/${NCCL_SO_NAME}")
      ok "head: LD_PRELOAD ${NCCL_SO_NAME}"
    else
      warn "head: ${head_nccl_dir}/${NCCL_SO_NAME} not found — using image NCCL (2.29.7)"
    fi
    if wrun "test -f '${worker_nccl_dir}/${NCCL_SO_NAME}'" 2>/dev/null; then
      worker_preload=(-v "${worker_nccl_dir}:/nccl:ro" -e LD_PRELOAD="/nccl/${NCCL_SO_NAME}")
      ok "worker: LD_PRELOAD ${NCCL_SO_NAME}"
    else
      warn "worker: ${worker_nccl_dir}/${NCCL_SO_NAME} not found — using image NCCL (2.29.7)"
    fi
  fi

  local pin=()
  [[ -n "${CPUSET}" ]] && pin=(--cpuset-cpus "${CPUSET}")
  mkdir -p "${TRITON_CACHE_DIR}" "${FLASHINFER_CACHE_DIR}"

  # Worker-side JIT caches (paths must exist on the WORKER, not here)
  local worker_cache_root
  worker_cache_root="$(dirname "$(worker_hf_home)")"
  wrun "mkdir -p '${worker_cache_root}/triton-qwen38-flash-next' '${worker_cache_root}/flashinfer-qwen38-flash-next'"

  # Log-follower cleanup (so this script's exit can't leave stray followers)
  cleanup_followers() {
    local p
    for p in "${HEAD_LOG_PID:-}" "${WORKER_LOG_PID:-}"; do
      [[ -n "${p}" ]] && kill "${p}" 2>/dev/null || true
    done
    return 0
  }
  trap cleanup_followers EXIT

  : >"${LOG_FILE}"; : >"${WORKER_LOG_FILE}"
  echo "[$(date -Is)] launching ${MODEL_ID} TP=${TP_SIZE} nnodes=${NNODES} image=${IMAGE}" >>"${LOG_FILE}"

  # Stale containers from a previous run
  if docker ps -a --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER}"; then
    warn "removing stale ${HEAD_CONTAINER}"
    docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1 || true
  fi
  if worker_docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${WORKER_CONTAINER}"; then
    warn "removing stale ${WORKER_CONTAINER}"
    worker_docker rm -f "${WORKER_CONTAINER}" >/dev/null 2>&1 || true
  fi

  # ---- Worker (node-rank 1) first, then head (rank 0 + API) — house order --
  info "starting worker (node-rank 1) on ${WORKER_SSH} …"
  worker_docker run -d \
    --name "${WORKER_CONTAINER}" \
    --network host --ipc host --gpus all \
    --shm-size 32g \
    --device /dev/infiniband --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${pin[@]}" \
    -v "$(worker_hf_home):/root/.cache/huggingface" \
    -v "${worker_cache_root}/triton-qwen38-flash-next:/root/.triton" \
    -v "${worker_cache_root}/flashinfer-qwen38-flash-next:/root/.cache/flashinfer" \
    "${worker_preload[@]}" \
    $(common_docker_env) \
    -e NCCL_SOCKET_IFNAME="${WORKER_CX7_IF}" \
    -e GLOO_SOCKET_IFNAME="${WORKER_CX7_IF}" \
    -e TP_SOCKET_IFNAME="${WORKER_CX7_IF}" \
    -e NCCL_IB_HCA="${WORKER_CX7_IB}" \
    "${IMAGE}" \
    python3 -m sglang.launch_server "${SGLANG_ARGS[@]}" \
    --nnodes "${NNODES}" --node-rank 1 --dist-init-addr "${HEAD_CX7_IP}:${DIST_PORT}"
  sleep 5
  worker_docker ps --format '{{.Names}}' | grep -qx "${WORKER_CONTAINER}" \
    || { error "worker container exited immediately"; wrun "docker logs --tail 80 ${WORKER_CONTAINER}" | tee -a "${WORKER_LOG_FILE}"; exit 1; }
  ok "worker started"
  wrun "docker logs -f ${WORKER_CONTAINER}" >>"${WORKER_LOG_FILE}" 2>&1 &
  WORKER_LOG_PID=$!

  info "starting head (node-rank 0, API on ${HOST_BIND}:${PORT}) …"
  docker run -d \
    --name "${HEAD_CONTAINER}" \
    --network host --ipc host --gpus all \
    --shm-size 32g \
    --device /dev/infiniband --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    "${pin[@]}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${TRITON_CACHE_DIR}:/root/.triton" \
    -v "${FLASHINFER_CACHE_DIR}:/root/.cache/flashinfer" \
    "${head_preload[@]}" \
    $(common_docker_env) \
    -e NCCL_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e GLOO_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e TP_SOCKET_IFNAME="${HEAD_CX7_IF}" \
    -e NCCL_IB_HCA="${HEAD_CX7_IB}" \
    "${IMAGE}" \
    python3 -m sglang.launch_server "${SGLANG_ARGS[@]}" \
    --nnodes "${NNODES}" --node-rank 0 --dist-init-addr "${HEAD_CX7_IP}:${DIST_PORT}"
  sleep 5
  docker ps --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER}" \
    || { error "head container exited immediately"; docker logs "${HEAD_CONTAINER}" 2>&1 | tee -a "${LOG_FILE}" | tail -80; exit 1; }
  ok "head started"
  docker inspect -f '{{.Id}}' "${HEAD_CONTAINER}" > "${PID_FILE}"
  docker logs -f "${HEAD_CONTAINER}" 2>&1 | tee -a "${LOG_FILE}" \
    | grep --line-buffered -v "Enabled fused SiLU+mul+FP4-quant" &
  HEAD_LOG_PID=$!
}

# =============================================================================
# Wait for readiness
# =============================================================================
wait_ready() {
  header "Waiting for SGLang API (first boot: load + JIT + graph capture)"
  local deadline=$(( $(date +%s) + WAIT_TIMEOUT_MIN * 60 )) start elapsed heartbeat
  start=$(date +%s); heartbeat=0
  while :; do
    if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
      echo ""
      ok "SGLang is ready at ${READY_URL}"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER}"; then
      echo ""
      error "head container died — last log lines:"
      tail -n 120 "${LOG_FILE}" || true
      exit 1
    fi
    if (( heartbeat % 4 == 0 )); then
      if ! worker_docker ps --format '{{.Names}}' | grep -qx "${WORKER_CONTAINER}"; then
        echo ""
        error "worker container died — last log lines:"
        tail -n 120 "${WORKER_LOG_FILE}" || true
        exit 1
      fi
    fi
    if (( $(date +%s) > deadline )); then
      error "timed out after ${WAIT_TIMEOUT_MIN} min — tail of head log:"
      tail -n 120 "${LOG_FILE}" || true
      exit 1
    fi
    if (( heartbeat % 12 == 0 )); then
      elapsed=$(( $(date +%s) - start ))
      echo "  still starting… ${elapsed}s elapsed (logs: ${LOG_FILE} / ${WORKER_LOG_FILE})"
    fi
    heartbeat=$(( heartbeat + 1 ))
    sleep 5
  done
}

# =============================================================================
# Subcommands
# =============================================================================
cmd_serve() {
  # Already running → report and exit cleanly (idempotent)
  if docker ps --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER}"; then
    ok "${HEAD_CONTAINER} already running"
    curl -fsS "${READY_URL}" >/dev/null 2>&1 && ok "API is ready: ${READY_URL}" || info "API not ready yet (still booting?)"
    info "logs: ./start.sh logs   ·   stop: ./start.sh stop"
    exit 0
  fi

  preflight serve
  ensure_images
  header "Weights"
  download_on_head
  sync_weights_to_worker

  launch
  wait_ready
  cleanup_followers
  trap - EXIT

  # LAN URLs: every non-docker IPv4 of the head (incl. tailscale, if present)
  local lan_ips; lan_ips="$(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]' | grep -v '^172\.17\.' | head -6)"
  echo ""
  echo -e "${BOLD}${GREEN}  Qwen3.8-Flash-Next-NVFP4 is up (2× DGX Spark, TP=2, NEXTN MTP)${NC}"
  echo -e "  ${BOLD}Model:${NC}      ${SERVED_MODEL_NAME}"
  echo -e "  ${BOLD}Draft:${NC}      in-checkpoint MTP (NEXTN ${SPEC_STEPS}/${SPEC_TOPK}/${SPEC_DRAFT}) — no separate download"
  echo -e "  ${BOLD}OpenAI API:${NC} http://<host>:${PORT}/v1   (bound ${HOST_BIND})"
  while read -r ip; do
    [[ -n "${ip}" ]] && echo -e "              http://${ip}:${PORT}/v1"
  done <<< "${lan_ips}"
  echo -e "  ${BOLD}Spec:${NC}       thinking always on (reasoning_parser=${REASONING_PARSER}); depth via reasoning_effort"
  echo -e "  ${BOLD}Context:${NC}    ${CONTEXT_LENGTH} tokens · mem-fraction ${MEM_FRACTION_STATIC} · max-running ${MAX_RUNNING_REQUESTS}"
  echo -e "  ${BOLD}Logs:${NC}       ${LOG_FILE} (head), ${WORKER_LOG_FILE} (worker)"
  echo ""
  echo -e "  ${BOLD}Quick test:${NC}"
  echo "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \\"
  echo "      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
  echo ""
  echo -e "  ${BOLD}Stop:${NC}  ./start.sh stop      ${BOLD}Status:${NC}  ./start.sh status"
  echo ""
}

cmd_stop() {
  header "Stopping"
  local did=0
  if docker ps -a --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER}"; then
    if docker rm -f "${HEAD_CONTAINER}" >/dev/null 2>&1; then ok "head stopped"; did=1; else warn "failed to remove head container"; fi
  fi
  if worker_docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${WORKER_CONTAINER}"; then
    if worker_docker rm -f "${WORKER_CONTAINER}" >/dev/null 2>&1; then ok "worker stopped"; did=1; else warn "failed to remove worker container"; fi
  fi
  for p in "${HEAD_LOG_PID:-}" "${WORKER_LOG_PID:-}"; do
    [[ -n "${p}" ]] && kill "${p}" 2>/dev/null || true
  done
  pgrep -f "docker logs -f ${HEAD_CONTAINER}"  >/dev/null && pkill -f "docker logs -f ${HEAD_CONTAINER}"  || true
  pgrep -f "docker logs -f ${WORKER_CONTAINER}" >/dev/null && pkill -f "docker logs -f ${WORKER_CONTAINER}" || true
  rm -f "${PID_FILE}"
  (( did )) || info "nothing to stop"
  return 0
}

cmd_status() {
  header "Status"
  echo "head (${HOSTNAME}):"
  docker ps -a --filter "name=${HEAD_CONTAINER}" --format '  {{.Names}}  {{.Status}}' 2>/dev/null || true
  echo "worker (${WORKER_SSH}):"
  worker_docker ps -a --filter "name=${WORKER_CONTAINER}" --format '  {{.Names}}  {{.Status}}' 2>/dev/null || true
  if curl -fsS --max-time 3 "${READY_URL}" >/dev/null 2>&1; then
    ok "API ready: ${READY_URL}"
    curl -fsS --max-time 3 "${READY_URL}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  served:", ", ".join(m["id"] for m in d["data"]))' 2>/dev/null || true
  else
    info "API not responding on ${READY_URL}"
  fi
  echo "GPU (head): $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"
  echo "GPU (worker): $(wrun "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1" || echo n/a)"
}

cmd_logs() {
  local which="${1:-head}"
  case "${which}" in
    head)   exec docker logs -f "${HEAD_CONTAINER}" ;;
    worker) exec wrun "docker logs -f ${WORKER_CONTAINER}" ;;
    *) echo "usage: ./start.sh logs [head|worker]"; exit 1 ;;
  esac
}

cmd_smoke() {
  header "Smoke test"
  curl -fsS --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Give me three words that rhyme with 'spark', then use one in a sentence.\"}], \"max_tokens\": 300}" \
    | python3 -c '
import json, sys
m = json.load(sys.stdin)["choices"][0]["message"]
r = getattr(m, "reasoning_content", None) or ""
print("reasoning:", (r[:200] + "…") if len(r) > 200 else r)
print("content:  ", m["content"])
' || { error "smoke test failed — check ./start.sh logs"; exit 1; }
}

cmd_doctor() { preflight doctor; }

case "${ACTION}" in
  serve)    cmd_serve ;;
  download) preflight download; ensure_images; download_on_head; sync_weights_to_worker; ok "download complete (--download-only)" ;;
  stop)     cmd_stop ;;
  status)   cmd_status ;;
  logs)     cmd_logs "${2:-head}" ;;
  smoke)    cmd_smoke ;;
  doctor)   cmd_doctor ;;
esac
