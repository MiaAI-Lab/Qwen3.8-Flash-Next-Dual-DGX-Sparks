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
#    · 262144-token native context, YaRN-scaled default 1048576 (1M) · always-thinking.
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
#  SM121 kernel work (DSpark patch = sglang#36806 + #36845, plus NVFP4 KV):
#    SM120 and SM121 need different QSA decode paths (sglang#36537):
#      * exact SM120 — FlashInfer TRT-LLM paged decode is numerically correct
#      * SM121 / GB10 — that same kernel silently emits token id 0 (`!`) at
#        long context (120k–210k) while still returning HTTP 200. sglang#36806
#        therefore gates TRT-LLM to SM100 + exact SM120 only.
#      * without TRT-LLM, QSA falls back to FA4 CuTe varlen, which does not
#        compile on GB10 (MLIR "weakly congruent" layout error). sglang#36845
#        replaces that fallback on SM121 with a packed one-query Triton kernel.
#    This script builds a derivative image (qwen38-flashnext-dspark:local)
#    that (1) forces `_resolve_trtllm_sparse_decode` to None on SM121 even if
#    a newer base image re-enables it, and (2) routes
#    `_resolve_flash_attn_varlen_func` to the #36845 Triton kernel. The
#    kernel reads cu_seqlens on-device so CUDA-graph replay stays valid.
#    DSpark extra vs upstream: fp8 K/V is allowed (upcast in-kernel) so
#    NVFP4_KV_CACHE=0 still works. A leftover long-thinking token-id-0 loop
#    (HTTP 200, then sticky `!` on later requests) is aborted after 16
#    consecutive token-0 samples; that completion is not inserted into radix
#    and the prefix cache is reset before the next prefill. KERNEL_PATCH=0
#    disables the patch (the stock image then dies at warmup with an
#    MLIRError, or worse, a silent token-0 loop if the image has the SM121
#    TRT-LLM route).
#
#    The same derivative image adds NVFP4 KV cache support for the QSA
#    layers (default NVFP4_KV_CACHE=1): the pool stores packed FP4 + per-block
#    FP8 scales (no FP8 dequant workspace — the QSA path never needed one),
#    and the QSA gather paths compact the packed rows with the stock Triton
#    kernel and dequantize via flashinfer's nvfp4_kv_dequantize (validated
#    CUDA-graph-safe on SM121). ~3.1x KV tokens at FP4 KV accuracy
#    (~9% relative K/V error; NIAH RELIABLE through 128k as a single huge prompt).
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
#    ./start.sh smoke           # one chat completion against :8888 (fails on `!` loops)
#    ./start.sh kv-eval         # NVFP4 KV passkey/NIAH reliability vs live API
#    ./start.sh doom-loop       # long thinking decode regression for token-id-0 `!`
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
#    MEM_FRACTION_STATIC=0.80   GB10 unified DRAM: PLE (~26 GB/rank) is INSIDE
#                               --mem-fraction-static (issue #8). 0.70
#                               double-counted it and starved KV/mamba.
#                               1M YaRN recipe uses 0.82 + chunk 1024 in .env.
#    CONTEXT_LENGTH=1048576     YaRN 1M default (factor 4.0 × native 262144).
#                               Set 262144 for native (no YaRN). Hard-capped at 1M.
#    CHUNKED_PREFILL_SIZE=1024  Keep ≤1024 when context > 262144 (indexer workspace).
#    MAX_PREFILL_TOKENS=        empty = 2048 under YaRN, else unset
#    MAX_RUNNING_REQUESTS=28    mamba ceiling at 0.80 + default mamba ratio
#                               (~141 slots / 5). 16 silently queues past c=16.
#    ALLOW_AUTO_TRUNCATE=0      1 = --allow-auto-truncate (silent trim). Off so
#                               over-length prompts 400 instead of shrinking.
#    SPEC_STEPS=3 SPEC_TOPK=1 SPEC_DRAFT=4     NEXTN chain (draft = steps + 1)
#    ENABLE_DECODE_GRAPHS=1     0 → --cuda-graph-backend-decode=disabled
#    CUDA_GRAPH_BS="1 2 3 4 5 6 7 8 10 12 14 16"  extended up to MAX_RUNNING
#    ATTENTION_BACKEND=flashinfer
#    NVFP4_KV_CACHE=1            1 = --kv-cache-dtype nvfp4 (the QSA FP4 KV
#                                path above; ~3.1x KV tokens, FP4 KV
#                                accuracy; default) · 0 = fp8_e4m3 KV
#                                (varlen + GQA prefill upcast fp8 before tl.dot)
#    KV_CACHE_DTYPE=            raw --kv-cache-dtype override when
#                                NVFP4_KV_CACHE=0 (e.g. bf16). Must be empty
#                                when NVFP4_KV_CACHE=1. Empty + NVFP4=0 → fp8_e4m3.
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
#    API_KEY=                   optional: serve with --api-key AND authenticate
#                               this script's own readiness/status/smoke curls
#                               (an --api-key inside EXTRA_ARGS wins for the
#                               server and is picked up here too, with a warning)
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

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"   # GB10: PLE is inside this budget (issue #8)
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}"   # YaRN 1M; 262144 = native, no YaRN
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"  # empty = 2048 when YaRN, else unset
YARN_NATIVE_CTX=262144
YARN_MAX_CTX=1048576
YARN_OVERRIDE_ARGS='{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144},"max_position_embeddings":1048576}}'
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-28}"
ALLOW_AUTO_TRUNCATE="${ALLOW_AUTO_TRUNCATE:-0}"      # 1 = --allow-auto-truncate
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"
ENABLE_DECODE_GRAPHS="${ENABLE_DECODE_GRAPHS:-1}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-"1 2 3 4 5 6 7 8 10 12 14 16"}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
NVFP4_KV_CACHE="${NVFP4_KV_CACHE:-1}"   # 1 = NVFP4 FP4 KV; 0 = fp8_e4m3
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"   # override when NVFP4_KV_CACHE=0 (e.g. bf16); empty + 0 → fp8_e4m3
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
API_KEY="${API_KEY:-}"
# Auth header for every local readiness/status/smoke curl in this script:
# with --api-key set, the server answers 401 to bare requests, which curl -f
# treats as failure -- without this, wait_ready times out on a healthy server.
# API_KEY is only the *requested* key: EXTRA_ARGS can carry its own --api-key
# and argparse is last-wins, so the header is re-derived from the key the
# server will really enforce by derive_effective_api_key() below.
EFFECTIVE_API_KEY="${API_KEY}"
AUTH_CURL=()   # populated by derive_effective_api_key (single owner of this rule)

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

# ---- Effective API key ------------------------------------------------------
# The server enforces the LAST --api-key on its command line. build_sglang_args
# appends API_KEY first and EXTRA_ARGS last, so an --api-key hidden in
# EXTRA_ARGS beats API_KEY server-side. Derive that winning key (SGLANG_ARGS
# when it has been built, EXTRA_ARGS otherwise) and point AUTH_CURL at it, so
# the script's own curls never 401 against a server it keyed itself.
derive_effective_api_key() {
  local -a scan=(); local from_extra=0 key="" a i n
  if [[ -n "${SGLANG_ARGS+x}" ]]; then
    scan=("${SGLANG_ARGS[@]}")
  else
    read -ra scan <<< "${EXTRA_ARGS}"
    from_extra=1
  fi
  n=${#scan[@]}
  for (( i = 0; i < n; i++ )); do
    a="${scan[i]}"
    if   [[ "${a}" == "--api-key" && $(( i + 1 )) -lt n ]]; then key="${scan[i+1]}"
    elif [[ "${a}" == --api-key=* ]];                     then key="${a#--api-key=}"
    fi
  done
  EFFECTIVE_API_KEY="${key:-${API_KEY}}"
  AUTH_CURL=()
  [[ -n "${EFFECTIVE_API_KEY}" ]] && AUTH_CURL=(-H "Authorization: Bearer ${EFFECTIVE_API_KEY}")
  # Warn once, from the EXTRA_ARGS pass (the SGLANG_ARGS pass cannot tell which
  # knob a key came from).
  if (( from_extra )) && [[ -n "${key}" ]]; then
    if [[ -z "${API_KEY}" ]]; then
      warn "EXTRA_ARGS carries --api-key while API_KEY is empty — serving keyed; this script's curls will use the EXTRA_ARGS key. Prefer API_KEY."
    else
      warn "API_KEY and EXTRA_ARGS both set --api-key — EXTRA_ARGS wins for the server (argparse last-wins), so that is the key being used."
    fi
  fi
  return 0
}

# ---- Remote helpers (worker = spark2 via ssh alias) -------------------------
wrun() { # remote shell snippet
  # -n (stdin from /dev/null) belongs HERE and not in SSH_OPTS: scp and rsync -e
  # share SSH_OPTS, and -n breaks rsync's transport. Without it, ssh inherits the
  # caller's stdin and silently consumes it -- running any script that calls wrun
  # from inside `ssh host bash -s <<EOF` swallows the rest of the heredoc, so the
  # worker step succeeds and every later line vanishes with no error.
  ssh -n "${SSH_OPTS[@]}" "${WORKER_SSH}" "$1"
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
  stop|status|logs|smoke|kv-eval|doom-loop|doctor|-h|--help) ;;
  *) echo "Unknown argument: ${ACTION}"; echo "Usage: $0 [serve|download|stop|status|logs|smoke|kv-eval|doom-loop|doctor]"; exit 1 ;;
esac
if [[ "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  # Print the whole header comment block — from line 2 until the first line
  # that is no longer a comment — so tunables documented at its end are not
  # truncated away (a hardcoded line range silently drops them as it grows).
  sed -n '2,${/^#/!{/^$/!q;};p;}' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

# Auth for the readiness/status/smoke curls, before any of them can run
# (status/smoke never reach build_sglang_args); launch() re-derives from the
# assembled SGLANG_ARGS.
derive_effective_api_key

# Decode-graph batch list must reach MAX_RUNNING_REQUESTS. The two defaults
# used to agree at 16; raising admission without extending the list leaves
# every decode step above 16 ungraphed (issue #8 follow-up).
sync_cuda_graph_bs() {
  local n="${MAX_RUNNING_REQUESTS}" top=0 x cand extra=""
  local -a bs
  read -ra bs <<< "${CUDA_GRAPH_BS}"
  for x in "${bs[@]}"; do
    [[ "${x}" =~ ^[0-9]+$ ]] || continue
    (( x > top )) && top=$x
  done
  if (( top >= n )); then
    return 0
  fi
  warn "CUDA_GRAPH_BS tops at ${top} < MAX_RUNNING_REQUESTS=${n} — extending so c>${top} stays graphed"
  for cand in 20 24 28 32 40 48; do
    if (( cand > top && cand < n )); then
      extra+=" ${cand}"
    fi
  done
  extra+=" ${n}"
  CUDA_GRAPH_BS="${CUDA_GRAPH_BS}${extra}"
  info "CUDA_GRAPH_BS=${CUDA_GRAPH_BS}"
}

# After the API is up: refuse a pool smaller than one advertised context
# request (that was silent truncation under --allow-auto-truncate).
boot_health_check() {
  header "KV pool vs advertised context"
  local pool=0 ctx=0
  pool="$(curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | python3 -c '
import re, sys
text = sys.stdin.read()
m = re.search(r"^sglang:max_total_num_tokens\{[^}]*\}\s+(\S+)", text, re.M)
print(int(float(m.group(1))) if m else 0)
' || true)"
  ctx="$(curl -fsS "${AUTH_CURL[@]}" --max-time 10 "http://127.0.0.1:${PORT}/get_server_info" 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(int(d.get("context_length") or 0))
' || true)"
  pool="${pool:-0}"; ctx="${ctx:-0}"
  local cap=""
  cap="$(grep -E "max_running_requests is capped" "${LOG_FILE}" 2>/dev/null | tail -1 || true)"
  if (( pool > 0 && ctx > 0 )); then
    info "KV pool ${pool} tokens · context_length ${ctx} · requested max-running ${MAX_RUNNING_REQUESTS}"
    if (( pool < ctx )); then
      error "KV pool (${pool}) is smaller than --context-length (${ctx}) — over-length prompts would truncate silently if --allow-auto-truncate is on (issue #8)"
      if [[ "${ALLOW_SHORT_KV_POOL:-0}" == "1" ]]; then
        warn "ALLOW_SHORT_KV_POOL=1 — continuing anyway"
      else
        error "refusing to advertise this context. Raise MEM_FRACTION_STATIC (0.80+), or ALLOW_SHORT_KV_POOL=1 to override"
        exit 1
      fi
    else
      ok "pool holds $(awk -v p="${pool}" -v c="${ctx}" 'BEGIN{printf "%.2f", p/c}')× one context-length request"
    fi
  else
    warn "could not read pool/context from the live API — skip size check"
  fi
  if [[ -n "${cap}" ]]; then
    warn "${cap##*] }"
  fi
}

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
    error "cannot reach ${WORKER_CX7_IP} over the CX7 fabric — TP rendezvous will hang, not fail fast"
    exit 1
  fi

  # MTU 9000, asserted end-to-end rather than read from one side. RoCE derives its
  # IB MTU from the netdev MTU, so a single end left at 1500 silently caps the whole
  # TP group -- and a default-size ping cannot see it.
  if ping -c 1 -W 2 -M do -s 8972 "${WORKER_CX7_IP}" >/dev/null 2>&1; then
    ok "fabric MTU 9000 verified end-to-end (8972B unfragmented)"
  else
    error "MTU 9000 not usable head→${WORKER_CX7_IP}: an 8972-byte unfragmented ping failed."
    error "  check both ends: ip link show ${HEAD_CX7_IF} / ${WORKER_CX7_IF} (expect mtu 9000)"
    exit 1
  fi
  local hmtu; hmtu="$(cat "/sys/class/net/${HEAD_CX7_IF}/mtu" 2>/dev/null || echo 0)"
  (( hmtu >= 9000 )) || { error "head ${HEAD_CX7_IF} MTU is ${hmtu}, expected 9000"; exit 1; }
  local wmtu; wmtu="$(wrun "cat /sys/class/net/${WORKER_CX7_IF}/mtu 2>/dev/null || echo 0")"
  (( wmtu >= 9000 )) || { error "worker ${WORKER_CX7_IF} MTU is ${wmtu}, expected 9000"; exit 1; }
  ok "netdev MTU head ${hmtu} / worker ${wmtu}"

  # The configured RoCE device must actually exist on each node: NCCL_IB_HCA naming
  # a device that is not present degrades silently instead of erroring.
  if ibv_devices 2>/dev/null | grep -qw "${HEAD_CX7_IB}"; then
    ok "head RoCE device ${HEAD_CX7_IB} present"
  else
    error "head RoCE device '${HEAD_CX7_IB}' not in ibv_devices — fix HEAD_CX7_IB"
    exit 1
  fi
  if wrun "ibv_devices 2>/dev/null | grep -qw '${WORKER_CX7_IB}'"; then
    ok "worker RoCE device ${WORKER_CX7_IB} present"
  else
    error "worker RoCE device '${WORKER_CX7_IB}' not in ibv_devices — fix WORKER_CX7_IB"
    exit 1
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

  # Decode-graph list must cover MAX_RUNNING_REQUESTS (issue #8 follow-up).
  sync_cuda_graph_bs

  # Config sanity
  case "${NVFP4_KV_CACHE}" in
    0|1) ;;
    *) error "NVFP4_KV_CACHE must be 0 or 1 (got '${NVFP4_KV_CACHE}')"; exit 1 ;;
  esac
  case "${ALLOW_AUTO_TRUNCATE}" in
    0|1) ;;
    *) error "ALLOW_AUTO_TRUNCATE must be 0 or 1 (got '${ALLOW_AUTO_TRUNCATE}')"; exit 1 ;;
  esac
  if [[ "${NVFP4_KV_CACHE}" == "1" && -n "${KV_CACHE_DTYPE}" ]]; then
    error "NVFP4_KV_CACHE=1 and KV_CACHE_DTYPE=${KV_CACHE_DTYPE} are both set — pick one"
    exit 1
  fi
  if (( CONTEXT_LENGTH > YARN_MAX_CTX )); then
    error "CONTEXT_LENGTH=${CONTEXT_LENGTH} exceeds YaRN 1M (${YARN_MAX_CTX})"
    exit 1
  fi
  if (( CONTEXT_LENGTH > YARN_NATIVE_CTX )); then
    if (( CHUNKED_PREFILL_SIZE > 1024 )); then
      warn "CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE} with ${CONTEXT_LENGTH} ctx froze GB10 at 300k history — clamping to 1024"
      CHUNKED_PREFILL_SIZE=1024
    fi
    ok "YaRN 1M context (${CONTEXT_LENGTH}; native ${YARN_NATIVE_CTX} × factor 4.0)"
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
  if [[ "${NVFP4_KV_CACHE}" == "1" ]]; then
    ok "NVFP4 KV cache enabled (kv-cache-dtype nvfp4)"
  else
    ok "KV cache dtype ${KV_CACHE_DTYPE:-fp8_e4m3} (NVFP4_KV_CACHE=0)"
  fi
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
    info "base image missing on worker — trying 'docker pull' there…"
    if worker_docker pull "${BASE_IMAGE}" >/dev/null 2>&1; then
      ok "base image pulled on worker"
    else
      warn "worker pull failed — syncing image head → worker via docker save/load"
      local tar="/tmp/qwen38-flash-next-image-$$.tar"
      docker save "${BASE_IMAGE}" -o "${tar}"
      scp "${SSH_OPTS[@]}" "${tar}" "${WORKER_SSH}:/tmp/qwen38-flash-next-image.tar"
      rm -f "${tar}"
      worker_docker load -i /tmp/qwen38-flash-next-image.tar
      wrun "rm -f /tmp/qwen38-flash-next-image.tar"
      worker_docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || { error "image sync to worker failed"; exit 1; }
      ok "base image loaded on worker"
    fi
  fi
}

# SM121 QSA fallback kernel (see header). Written as a docker build context
# under .patch/ and built identically on both nodes.
write_patch_context() {
  mkdir -p "${SCRIPT_DIR}/.patch"
  # Old name from before sglang#36845; the image context must not keep a stale copy.
  rm -f "${SCRIPT_DIR}/.patch/qsa_fa_fallback.py"
  cat > "${SCRIPT_DIR}/.patch/sm121_varlen.py" <<'QSA_EOF'
"""Packed varlen attention fallback for QSA decode on SM121.

Port of sgl-project/sglang#36845 (`qsa/sm121_varlen.py`). The QSA backend
compacts each request's selected KV rows into a packed buffer and issues one
query row per request. FlashAttention-4's CuTe varlen kernel does not compile
for this call shape on SM121, while FlashInfer's TRT-LLM decode kernel is not
numerically safe there (sglang#36806 / #36537: silent token-id-0 at long
context). This module implements only the narrow packed contract needed by QSA.

DSpark extra vs upstream: K/V may be fp8 (fp8_e4m3 / fp8_e5m2). The kernel
already upcasts loads to fp32, so ``NVFP4_KV_CACHE=0`` does not need a
host-side dequant (CUDA-graph safe).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_one_query_varlen_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    k_stride_t: tl.constexpr,
    k_stride_h: tl.constexpr,
    v_stride_t: tl.constexpr,
    v_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
):
    sequence_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    query_idx = tl.load(cu_seqlens_q_ptr + sequence_idx)
    kv_start = tl.load(cu_seqlens_k_ptr + sequence_idx)
    kv_end = tl.load(cu_seqlens_k_ptr + sequence_idx + 1)

    dim_offsets = tl.arange(0, PADDED_HEAD_DIM)
    dim_mask = dim_offsets < HEAD_DIM
    query = tl.load(
        q_ptr + query_idx * q_stride_t + query_head_idx * q_stride_h + dim_offsets,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    queries_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
    kv_head_idx = query_head_idx // queries_per_kv
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros([PADDED_HEAD_DIM], dtype=tl.float32)

    for block_start in range(kv_start, kv_end, BLOCK_KV):
        kv_offsets = block_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < kv_end
        key_offsets = kv_offsets * k_stride_t + kv_head_idx * k_stride_h
        keys = tl.load(
            k_ptr + key_offsets[:, None] + dim_offsets[None, :],
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(query[None, :] * keys, axis=1) * softmax_scale
        scores = tl.where(kv_mask, scores, -float("inf"))

        new_max = tl.maximum(running_max, tl.max(scores, axis=0))
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        accumulator *= old_scale

        value_offsets = kv_offsets * v_stride_t + kv_head_idx * v_stride_h
        values = tl.load(
            v_ptr + value_offsets[:, None] + dim_offsets[None, :],
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(probabilities[:, None] * values, axis=0)
        running_max = new_max

    # Empty selected-KV (or a 0 running_sum) must not divide through -inf
    # softmax state: that yields NaN attention, NaN logits, then token id 0
    # (`!`) until max_tokens and can poison later radix hits.
    valid = (kv_end > kv_start) & (running_sum > 0.0)
    output = accumulator / tl.where(valid, running_sum, 1.0)
    finite = output == output
    output = tl.where(finite & valid, output, 0.0)
    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + query_head_idx * out_stride_h
        + dim_offsets,
        output.to(out_ptr.dtype.element_ty),
        mask=dim_mask,
    )


def qsa_sm121_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 0,
    softmax_scale: float = 1.0,
    causal: bool = True,
    **_: object,
) -> torch.Tensor:
    """Run the one-query packed-varlen attention contract emitted by QSA."""

    del max_seqlen_k, causal
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise RuntimeError("SM121 QSA varlen attention requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"expected 3-D q/k/v, got {q.shape}/{k.shape}/{v.shape}")
    if max_seqlen_q != 1:
        raise ValueError(f"QSA requires max_seqlen_q=1, got {max_seqlen_q}")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"unsupported query dtype: {q.dtype}")
    _fp8 = {torch.float8_e4m3fn, torch.float8_e5m2}
    if k.dtype != v.dtype:
        raise TypeError(f"k/v dtypes must match, got {k.dtype}/{v.dtype}")
    if k.dtype != q.dtype and k.dtype not in _fp8:
        raise TypeError(
            f"k/v dtype {k.dtype} is not {q.dtype} or fp8"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v must be on the same CUDA device")
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        raise ValueError("cu_seqlens_q/k must be on the same CUDA device as q")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q/k must use torch.int32")

    total_queries, num_q_heads, head_dim = q.shape
    _, num_kv_heads, key_head_dim = k.shape
    if v.shape != k.shape:
        raise ValueError(f"k/v shapes must match, got {k.shape}/{v.shape}")
    if key_head_dim != head_dim:
        raise ValueError(
            f"q/k head dimensions must match, got {head_dim}/{key_head_dim}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"QSA GQA requires q heads divisible by kv heads, got "
            f"{num_q_heads}/{num_kv_heads}"
        )
    if cu_seqlens_q.numel() != total_queries + 1:
        raise ValueError("cu_seqlens_q must contain one entry per query plus the end")
    if cu_seqlens_k.numel() != total_queries + 1:
        raise ValueError("cu_seqlens_k must contain one entry per query plus the end")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)
    padded_head_dim = triton.next_power_of_2(max(head_dim, 16))
    _qsa_one_query_varlen_kernel[(total_queries, num_q_heads)](
        q,
        k,
        v,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        PADDED_HEAD_DIM=padded_head_dim,
        BLOCK_KV=64,
        q_stride_t=q.stride(0),
        q_stride_h=q.stride(1),
        k_stride_t=k.stride(0),
        k_stride_h=k.stride(1),
        v_stride_t=v.stride(0),
        v_stride_h=v.stride(1),
        out_stride_t=output.stride(0),
        out_stride_h=output.stride(1),
        num_warps=4,
    )
    return output
QSA_EOF

  cat > "${SCRIPT_DIR}/.patch/qsa_nvfp4_kv.py" <<'NVP4_EOF'
"""NVFP4 KV cache for the QSA (Qwen4-Exp sparse attention) path on DGX Spark.

Upstream SGLang's NVFP4 KV recipe assumes two consumers that do not exist on
the QSA path: FlashInfer prefill reading an FP8 *dequant workspace* covering
the whole pool, and TRT-LLM decode consuming native packed FP4.  The QSA
backend instead gathers a sparse subset of KV rows with Triton kernels and
wants dense BF16 rows.  On top of that, the full-size FP8 workspace would eat
most of the FP4 savings (fp4 data + scales + fp8 workspace is ~1.56 B/elem
against bf16's 2 B/elem; without the workspace it is 0.5625 B/elem — a 3.6x
cut, verified CUDA-graph safe on SM121 via flashinfer's nvfp4_kv_* kernels).

This module provides:

* ``QSANVFP4KVCacheMethod`` — an NVFP4 method whose attention-access rules
  declare PLAIN BF16 reads only (no DEQUANT_WORKSPACE, no NATIVE_FP4), so the
  pool allocates packed FP4 + per-block FP8 scales and nothing else.  Plain
  readers (``get_key_buffer``) dequantize on demand through
  ``dequantize_kv_tensor``.  Quantization, storage layout, per-layer global
  scales and slot moves are inherited unchanged from the upstream method.
* ``try_fp4_view`` / ``compact_and_dequant`` / ``gather_history_fp4`` — the
  gather-dequant helpers the patched ``QwenSparseAttnBackend`` calls on its
  decode/verify and chunked-prefill paths.  The stock Triton compaction
  kernel runs over the packed FP4 buffers and (a second time) over the
  per-block scale buffers — same leading dims, dim/16 — and the compacted
  rows are dequantized with flashinfer's ``nvfp4_kv_dequantize``.

Wiring (applied by the sibling ``apply_nvfp4_patches.py`` build step):

* ``get_kv_cache_quant_method("nvfp4")`` routes here (this image serves QSA
  models; non-QSA models should use the stock image for nvfp4 KV).
* ``_handle_kv4_compatibility`` allows nvfp4 KV for QSA hybrids whose
  ``--attention-backend`` flag only selects the GDN linear-attn kernels.
* The pool-configurator cell size skips the FP8 workspace share.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccess,
    KVCacheAttentionAccessKind,
    KVCacheAttentionPhase,
    KVCacheBackendMatcher,
    NVFP4KVCacheMethod,
)
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

_BF16 = torch.bfloat16
_U8 = torch.uint8
_FP8 = torch.float8_e4m3fn


def _plain_access(phase: KVCacheAttentionPhase) -> KVCacheAttentionAccess:
    return KVCacheAttentionAccess(
        phase,
        KVCacheAttentionAccessKind.PLAIN,
        KVCacheBackendMatcher(any_backend=True),
        storage_dtype=_U8,
        attention_kv_dtype=_BF16,
        scale_recipe="nvfp4",
    )


class QSANVFP4KVCacheMethod(NVFP4KVCacheMethod):
    """NVFP4 KV cache as consumed by the QSA backend: plain BF16 dequant reads.

    Differs from the upstream ``NVFP4KVCacheMethod`` only in its declared
    attention accesses: PLAIN for both phases and every backend.  As a
    consequence the pool allocates no FP8 dequant workspace
    (``needs_dequant_workspace()`` is False) and plain readers dequantize
    packed FP4 + scales on demand.
    """

    def __init__(self, num_layers: int, device: str):
        super().__init__(num_layers, device)
        # Pre-allocate so the out-of-range fallback never does torch.ones
        # during CUDA-graph capture.
        self._ones_scale = torch.ones(1, dtype=torch.float32, device=device)

    def attention_accesses(self) -> tuple[KVCacheAttentionAccess, ...]:
        return (
            _plain_access(KVCacheAttentionPhase.PREFILL),
            _plain_access(KVCacheAttentionPhase.DECODE),
        )

    def _layer_global_scale(
        self, scales_gpu: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        # The write path indexes k_scales_gpu by the GLOBAL layer id (with
        # load_scales_from_model resizing the vector to cover global ids);
        # guard against shorter vectors (all-ones scales) anyway.
        if 0 <= layer_id < scales_gpu.numel():
            return scales_gpu[layer_id : layer_id + 1]
        return self._ones_scale

    def dequantize_kv_tensor(
        self,
        fp4_tensor: torch.Tensor,
        scales: torch.Tensor,
        layer_id: int,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Dequantize one packed FP4 KV tensor (whole-pool view) for plain reads."""
        if scales.dtype != _FP8:
            scales = scales.view(_FP8)
        return NVFP4KVQuantizeUtil.dequantize(
            fp4_tensor.view(_U8),
            scales,
            self._layer_global_scale(self.k_scales_gpu, layer_id),
            dtype=dtype or _BF16,
        )


class QSAFP4KVView:
    """Packed FP4 + per-block-scale buffers of one QSA full-attention layer."""

    def __init__(self, k_fp4, v_fp4, k_sf, v_sf, k_gs, v_gs):
        self.k_fp4 = k_fp4  # uint8 [rows, head_num, head_dim // 2]
        self.v_fp4 = v_fp4
        self.k_sf = k_sf  # float8_e4m3 view [rows, head_num, head_dim // 16]
        self.v_sf = v_sf
        self.k_gs = k_gs  # 1-element fp32 global scale (on device)
        self.v_gs = v_gs
        self.head_num = k_fp4.shape[1]
        self.head_dim = k_fp4.shape[2] * 2
        self.device = k_fp4.device

    @property
    def k_sf_u8(self) -> torch.Tensor:
        return self.k_sf.view(_U8)

    @property
    def v_sf_u8(self) -> torch.Tensor:
        return self.v_sf.view(_U8)

    def dequant_rows(self, k_rows, k_sf_rows, v_rows, v_sf_rows):
        """Dequantize already-gathered packed rows -> (k_bf16, v_bf16)."""
        k = NVFP4KVQuantizeUtil.dequantize(
            k_rows, k_sf_rows.view(_FP8), self.k_gs
        )
        v = NVFP4KVQuantizeUtil.dequantize(
            v_rows, v_sf_rows.view(_FP8), self.v_gs
        )
        return k, v


def try_fp4_view(pool, layer_id: int) -> Optional[QSAFP4KVView]:
    """Return the layer's FP4 buffers, or None for an unquantized pool.

    On SM100 the stock native-FP4 consumers stay in charge, so the QSA
    gather-dequant path is SM120/SM121 (DGX Spark) only.
    """
    from sglang.srt.utils import is_sm100_supported

    if is_sm100_supported():
        return None
    full_pool = getattr(pool, "full_kv_pool", pool)
    quant_method = getattr(full_pool, "quant_method", None)
    if quant_method is None or getattr(quant_method, "name", None) != "nvfp4":
        return None
    k_fp4, v_fp4, k_sf, v_sf = pool.get_raw_kv_buffer(layer_id)
    k_gs = quant_method._layer_global_scale(quant_method.k_scales_gpu, layer_id)
    v_gs = quant_method._layer_global_scale(quant_method.v_scales_gpu, layer_id)
    return QSAFP4KVView(k_fp4, v_fp4, k_sf, v_sf, k_gs, v_gs)


def compact_and_dequant(
    backend,
    fp4_kv: QSAFP4KVView,
    scratch_capacity: int,
    req_indices,
    topk_indices,
    sequence_lens,
    cu_seqlens_k,
    batch: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse gather + dequant for the QSA decode/verify path.

    Runs the stock compaction kernel twice — once over the packed FP4 data,
    once over the per-block scale buffers — then dequantizes the compacted
    rows to BF16.  Kernel-only, so CUDA-graph capture/replay is safe (the
    dequant scratch rows past the packed prefix hold garbage that the varlen
    attention kernel never reads, exactly like the BF16 path).
    """
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_kv_extraction_compact_triton,
    )

    req_to_token = backend.req_to_token_pool.req_to_token
    n, d = fp4_kv.head_num, fp4_kv.head_dim
    # The scratch cache is keyed by (heads, dim, dtype, device), so the FP4
    # data (dim/2), scale (dim/16) and BF16 (dim) buffers never collide.
    pk_fp4, pv_fp4 = backend._get_fa2_scratch(
        scratch_capacity, n, d // 2, _U8, fp4_kv.device
    )
    pk_sf, pv_sf = backend._get_fa2_scratch(
        scratch_capacity, n, d // 16, _U8, fp4_kv.device
    )
    qwen_sparse_kv_extraction_compact_triton(
        fp4_kv.k_fp4,
        fp4_kv.v_fp4,
        req_to_token,
        req_indices,
        topk_indices,
        sequence_lens,
        cu_seqlens_k,
        pk_fp4,
        pv_fp4,
        batch,
        topk,
    )
    qwen_sparse_kv_extraction_compact_triton(
        fp4_kv.k_sf_u8,
        fp4_kv.v_sf_u8,
        req_to_token,
        req_indices,
        topk_indices,
        sequence_lens,
        cu_seqlens_k,
        pk_sf,
        pv_sf,
        batch,
        topk,
    )
    return fp4_kv.dequant_rows(pk_fp4, pk_sf, pv_fp4, pv_sf)


def gather_history_fp4(
    fp4_kv: QSAFP4KVView, req_to_token, req_indices, sequence_lens
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked-prefill history gather: index_select + dequant per request.

    Mirrors the BF16 path's per-request ``index_select`` + ``cat`` (the
    validated chunk-prefill kernel consumes tightly packed full-context
    K/V); the only difference is that the gathered packed rows are
    dequantized to BF16 on the way out.
    """
    k_parts = []
    v_parts = []
    k_fp4, v_fp4 = fp4_kv.k_fp4, fp4_kv.v_fp4
    k_sf_u8, v_sf_u8 = fp4_kv.k_sf_u8, fp4_kv.v_sf_u8
    for i, seq_len in enumerate(sequence_lens):
        slots = req_to_token[req_indices[i], :seq_len].long()
        k_parts.append(
            NVFP4KVQuantizeUtil.dequantize(
                k_fp4[slots], k_sf_u8[slots].view(_FP8), fp4_kv.k_gs
            )
        )
        v_parts.append(
            NVFP4KVQuantizeUtil.dequantize(
                v_fp4[slots], v_sf_u8[slots].view(_FP8), fp4_kv.v_gs
            )
        )
    return torch.cat(k_parts), torch.cat(v_parts)
NVP4_EOF

  cat > "${SCRIPT_DIR}/.patch/apply_nvfp4_patches.py" <<'APPLY_EOF'
#!/usr/bin/env python3
"""Apply the DSpark NVFP4-KV source patches inside the qwen38flashnext image.

All patches are QSA-scoped and inert unless ``--kv-cache-dtype nvfp4`` is set:

1. ``qwen_sparse_attn_backend.py`` — the QSA gather paths (decode/verify and
   chunked prefill) read the packed FP4 + per-block-scale buffers and
   dequantize the gathered rows, instead of pulling whole BF16 pool views.
2. ``fp4_kv_cache_quant_method.py`` — route ``nvfp4`` to the QSA plain-dequant
   method (no FP8 dequant workspace, no native-FP4 decode path).
3. ``server_args.py`` — allow nvfp4 KV for QSA hybrid models whose
   ``--attention-backend`` only selects the GDN linear-attn kernels.
4. ``pool_configurator.py`` — don't reserve the FP8 workspace share of the
   FP4 cell size for QSA models (the QSA method allocates none).
5. ``memory_pool.py`` — NVFP4 write path: ignore hybrid pool's Python
   ``k_scale=1.0`` default and use on-device ``k_scales_gpu`` (host→CUDA
   ``torch.tensor`` is illegal during decode CUDA-graph capture).
6. ``sparse_attn.py`` — SM121 Triton cannot ``tl.dot`` fp8e4nv. The
   chunk-prefill / prefill GQA kernels upcast K/V (and Q) to fp32 before
   the dots so ``--kv-cache-dtype fp8_e4m3`` survives extend, not only
   the paged-varlen decode fallback.
"""

import pathlib

SRT = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt")

MARKER = "qsa_nvfp4_kv"


def patch(path, replacements):
    s = path.read_text()
    if MARKER in s:
        print(f"{path.name}: already patched")
        return
    for anchor, replacement in replacements:
        count = s.count(anchor)
        assert count == 1, f"{path.name}: anchor matched {count} times (want 1):\n{anchor}"
        s = s.replace(anchor, replacement, 1)
    path.write_text(s)
    print(f"{path.name}: patched")


# ---------------------------------------------------------------------------
# 1. QSA attention backend: FP4-aware gather paths
# ---------------------------------------------------------------------------
BACKEND = SRT / "layers" / "attention" / "qwen_sparse_attn_backend.py"

IMPORT_ANCHOR = """from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
"""
IMPORT_REPLACEMENT = IMPORT_ANCHOR + (
    "from sglang.srt.layers.attention import qsa_nvfp4_kv  # dspark: NVFP4 KV cache\n"
)

PAGED_HEAD_ANCHOR = """        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)
"""
PAGED_HEAD_REPLACEMENT = """        pool = self.token_to_kv_pool
        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)
        if fp4_kv is None:
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
        else:
            k_buffer = v_buffer = None
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            if fp4_kv is not None:
                # Whole-pool plain dequant (slow; CPU fallback only).
                k_buffer = pool.get_key_buffer(layer.layer_id)
                v_buffer = pool.get_value_buffer(layer.layer_id)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)
"""

EXTRACTION_ANCHOR = """        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_seqlens_k,
            packed_k,
            packed_v,
            batch,
            topk,
        )
"""
EXTRACTION_REPLACEMENT = """        if fp4_kv is not None:
            # dspark NVFP4: compact the packed FP4 rows and the per-block
            # scale rows, then dequantize the gathered rows to BF16.
            packed_k, packed_v = qsa_nvfp4_kv.compact_and_dequant(
                self,
                fp4_kv,
                scratch_capacity,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_seqlens_k,
                batch,
                topk,
            )
        else:
            packed_k, packed_v = self._get_fa2_scratch(
                scratch_capacity,
                k_buffer.shape[1],
                k_buffer.shape[2],
                k_buffer.dtype,
                k_buffer.device,
            )
            qwen_sparse_kv_extraction_compact_triton(
                k_buffer,
                v_buffer,
                self.req_to_token_pool.req_to_token,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_seqlens_k,
                packed_k,
                packed_v,
                batch,
                topk,
            )
"""

EXTEND_CHUNK_ANCHOR = """        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        k_parts = [
            k_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        v_parts = [
            v_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
"""
EXTEND_CHUNK_REPLACEMENT = """        pool = self.token_to_kv_pool
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)
        if fp4_kv is not None:
            # dspark NVFP4: gather each request's history from the packed
            # FP4 + scale buffers and dequantize to BF16.
            k_cat, v_cat = qsa_nvfp4_kv.gather_history_fp4(
                fp4_kv, req_to_token, req_indices, sequence_lens
            )
        else:
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
            k_parts = [
                k_buffer.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            v_parts = [
                v_buffer.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            k_cat = torch.cat(k_parts)
            v_cat = torch.cat(v_parts)
"""

EXTEND_CK_ANCHOR = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            torch.cat(k_parts),
            torch.cat(v_parts),
"""
EXTEND_CK_REPLACEMENT = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            k_cat,
            v_cat,
"""

# ---------------------------------------------------------------------------
# 2. NVFP4 recipe routing: QSA plain-dequant method
# ---------------------------------------------------------------------------
FP4_METHOD = SRT / "layers" / "quantization" / "fp4_kv_cache_quant_method.py"

FP4_METHOD_ANCHOR = """    return KV_CACHE_QUANT_REGISTRY[name](**kwargs)
"""
FP4_METHOD_REPLACEMENT = """    if name == "nvfp4":
        # dspark (DGX Spark / SM121): QSA models consume the NVFP4 pool via
        # plain BF16 dequant reads — no FP8 dequant workspace, no
        # native-FP4 decode. See attention/qsa_nvfp4_kv.py.
        from sglang.srt.layers.attention.qsa_nvfp4_kv import QSANVFP4KVCacheMethod

        return QSANVFP4KVCacheMethod(**kwargs)
    return KV_CACHE_QUANT_REGISTRY[name](**kwargs)
"""

# ---------------------------------------------------------------------------
# 3. server_args: allow nvfp4 KV for QSA hybrid models
# ---------------------------------------------------------------------------
SERVER_ARGS = SRT / "server_args.py"

SERVER_ARGS_ANCHOR = """        if is_cuda():
            if self.kv_cache_dtype == "nvfp4" and not (
                is_sm100_supported() or is_sm120_supported()
            ):
                raise RuntimeError(
                    "--kv-cache-dtype=nvfp4 requires Blackwell SM100 or SM120. "
                    "Use --kv-cache-dtype=fp4_mx_block16 for the block-size-16 FP4 recipe."
                )
"""
SERVER_ARGS_REPLACEMENT = SERVER_ARGS_ANCHOR + """            # dspark (DGX Spark / SM121, see qsa_nvfp4_kv): on QSA hybrid
            # models the full-attention layers read the FP4 pool through the
            # QSA backend's Triton plain-dequant path; --attention-backend
            # only selects the GDN linear-attention kernels, so the MHA
            # allow-list below does not apply.
            if self.kv_cache_dtype == "nvfp4":
                try:
                    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

                    if is_qwen_qsa(self.get_model_config().hf_config):
                        return
                except Exception:
                    pass
"""

# ---------------------------------------------------------------------------
# 4. pool_configurator: no FP8 workspace share for QSA FP4 cell size
# ---------------------------------------------------------------------------
POOL_CFG = SRT / "model_executor" / "pool_configurator.py"

POOL_CFG_ANCHOR = """                # FP4 prefill uses one shared FP8 dequant workspace across layers.
                cell_size += n * k * 2 * kv_size
"""
POOL_CFG_REPLACEMENT = """                # FP4 prefill uses one shared FP8 dequant workspace across
                # layers — except on the QSA path (dspark qsa_nvfp4_kv),
                # whose method allocates no FP8 workspace.
                _is_qsa_kv4 = False
                try:
                    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

                    _is_qsa_kv4 = is_qwen_qsa(model_config.hf_config)
                except Exception:
                    pass
                if not _is_qsa_kv4:
                    cell_size += n * k * 2 * kv_size
"""

# ---------------------------------------------------------------------------
# 5. memory_pool: NVFP4 store must use on-device global scales
# ---------------------------------------------------------------------------
POOL = SRT / "mem_cache" / "memory_pool.py"

QUANT_SCALES_ANCHOR = """    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        if k_scale is None and hasattr(self.quant_method, "k_scales_gpu"):
            k_scale = self.quant_method.k_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
            v_scale = self.quant_method.v_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
        return k_scale, v_scale
"""
QUANT_SCALES_REPLACEMENT = """    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        # dspark qsa_nvfp4_kv: HybridTokenToKVPool.set_kv_buffer defaults
        # k_scale=v_scale=1.0 (Python floats). That skips the on-device
        # per-layer NVFP4 global scales, and NVFP4KVQuantizeUtil.quantize
        # then does torch.tensor(..., device=cuda) — illegal during CUDA
        # graph capture. Host scalars are treated as unset on nvfp4 only.
        use_gpu = hasattr(self.quant_method, "k_scales_gpu")
        nvfp4 = getattr(self.quant_method, "name", None) == "nvfp4"
        host_scalar = nvfp4 and not (
            torch.is_tensor(k_scale) and k_scale.is_cuda
        )
        if use_gpu and (k_scale is None or host_scalar):
            k_scale = self.quant_method.k_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
            v_scale = self.quant_method.v_scales_gpu[
                global_layer_id : global_layer_id + 1
            ]
        return k_scale, v_scale
"""

# ---------------------------------------------------------------------------
# 6. sparse_attn.py: Triton GQA cannot tl.dot fp8e4nv on SM121
# ---------------------------------------------------------------------------
SPARSE_ATTN = SRT / "layers" / "attention" / "qsa" / "sparse_attn.py"
FP8_DOT_MARKER = "dspark: SM121 Triton cannot tl.dot fp8e4nv"

GQA_DOT_ANCHOR = """        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
"""
GQA_DOT_REPLACEMENT = """        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
        # dspark: SM121 Triton cannot tl.dot fp8e4nv (fp8 KV cache).
        keys = keys.to(tl.float32)
        values = values.to(tl.float32)
        q_dot = q_values.to(tl.float32)
        scores = tl.where(valid[None, :], tl.dot(q_dot, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(probabilities, values, accumulator * alpha[:, None])
"""


def patch_count(path, anchor, replacement, expected, marker):
    s = path.read_text()
    if marker in s:
        print(f"{path.name}: already patched")
        return
    count = s.count(anchor)
    assert count == expected, (
        f"{path.name}: anchor matched {count} times (want {expected}):\n{anchor}"
    )
    path.write_text(s.replace(anchor, replacement))
    print(f"{path.name}: patched ({expected} sites)")


def main() -> None:
    patch(
        BACKEND,
        [
            (IMPORT_ANCHOR, IMPORT_REPLACEMENT),
            (PAGED_HEAD_ANCHOR, PAGED_HEAD_REPLACEMENT),
            (EXTRACTION_ANCHOR, EXTRACTION_REPLACEMENT),
            (EXTEND_CHUNK_ANCHOR, EXTEND_CHUNK_REPLACEMENT),
            (EXTEND_CK_ANCHOR, EXTEND_CK_REPLACEMENT),
        ],
    )
    patch(FP4_METHOD, [(FP4_METHOD_ANCHOR, FP4_METHOD_REPLACEMENT)])
    patch(SERVER_ARGS, [(SERVER_ARGS_ANCHOR, SERVER_ARGS_REPLACEMENT)])
    patch(POOL_CFG, [(POOL_CFG_ANCHOR, POOL_CFG_REPLACEMENT)])
    patch(POOL, [(QUANT_SCALES_ANCHOR, QUANT_SCALES_REPLACEMENT)])
    patch_count(
        SPARSE_ATTN,
        GQA_DOT_ANCHOR,
        GQA_DOT_REPLACEMENT,
        expected=2,
        marker=FP8_DOT_MARKER,
    )
    patch_token0_guard()
    print("NVFP4 KV patches applied")


TOKEN0_MARKER = "dspark_token0_guard"
TOKEN0_RUN = 16  # matches logs/qwen38_doom_loop_repro.py


def patch_token0_guard() -> None:
    """Abort a token-id-0 (`!`) decode loop and drop the poisoned prefix cache.

    sglang#36806/#36845 closed the SM121 TRT-LLM path. A later long thinking
    decode can still emit token 0, finish HTTP 200, and poison the next
    request via radix reuse. See logs/qwen38-doom-loop-bug-report.md.
    """
    schedule_batch = SRT / "managers" / "schedule_batch.py"
    processor = (
        SRT / "managers" / "scheduler_components" / "batch_result_processor.py"
    )
    scheduler = SRT / "managers" / "scheduler.py"

    pad_anchor = '''def _compute_pad_value(hash: int) -> int:
    """Compute pad value from hash."""
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))
'''
    pad_repl = pad_anchor + '''
# dspark_token0_guard: set when a request hits a repeated token-id-0 run.
DSPARK_TOKEN0_FLUSH_NEEDED = False


def dspark_note_token0_loop() -> None:
    global DSPARK_TOKEN0_FLUSH_NEEDED
    DSPARK_TOKEN0_FLUSH_NEEDED = True


def dspark_consume_token0_flush() -> bool:
    global DSPARK_TOKEN0_FLUSH_NEEDED
    needed = DSPARK_TOKEN0_FLUSH_NEEDED
    DSPARK_TOKEN0_FLUSH_NEEDED = False
    return needed

'''
    vocab_anchor = '''    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
'''
    vocab_repl = f'''    def _check_token0_loop_finish(self) -> bool:
        """Stop a decoded token-id-0 (`!`) run before it fills max_tokens."""
        ids = self.output_ids
        if len(ids) < {TOKEN0_RUN}:
            return False
        if any(token != 0 for token in ids[-{TOKEN0_RUN}:]):
            return False
        self.finished_reason = FINISH_ABORT(
            "token-id-0 loop (decoded '!'); aborting to avoid poisoning later requests",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "InternalServerError",
        )
        self.finished_len = len(ids) - {TOKEN0_RUN} + 1
        self.token0_loop = True
        dspark_note_token0_loop()
        logger.error(
            "dspark: token-id-0 loop after %s output tokens rid=%s",
            len(ids),
            self.rid,
        )
        return True

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
'''
    finish_anchor = '''        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        # Sanitize out-of-range / NaN token ids before any decode.
        if self._check_vocab_boundary_finish(new_accepted_tokens):
            self._cap_finished_len_at_max_new_tokens()
            return
'''
    finish_repl = '''        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        # Sanitize out-of-range / NaN token ids before any decode.
        if self._check_vocab_boundary_finish(new_accepted_tokens):
            self._cap_finished_len_at_max_new_tokens()
            return

        # dspark_token0_guard: abort a repeated token-id-0 (`!`) run.
        if self._check_token0_loop_finish():
            return
'''
    insert_anchor = '''                is_insert = (
                    req.mamba_lazy_is_insert
                    if mamba_extra_buffer_lazy_enabled()
                    else True
                )
                release_kv_cache(req, self.tree_cache, is_insert=is_insert)
'''
    insert_repl = '''                is_insert = (
                    req.mamba_lazy_is_insert
                    if mamba_extra_buffer_lazy_enabled()
                    else True
                )
                if getattr(req, "token0_loop", False):  # dspark_token0_guard
                    is_insert = False
                release_kv_cache(req, self.tree_cache, is_insert=is_insert)
'''
    next_anchor = '''    def get_next_batch_to_run(
        self, running_batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()
'''
    next_repl = '''    def get_next_batch_to_run(
        self, running_batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()
        # dspark_token0_guard: drop the radix tree before the next prefill so
        # a later request cannot reuse prefix KV corrupted by a token-id-0 run.
        if running_batch.is_empty():
            from sglang.srt.managers.schedule_batch import dspark_consume_token0_flush

            if dspark_consume_token0_flush():
                logger.warning(
                    "dspark: token-id-0 loop; resetting prefix cache before the next prefill"
                )
                self.tree_cache.reset()
'''

    def _one(path, replacements):
        s = path.read_text()
        if TOKEN0_MARKER in s:
            print(f"{path.name}: token0 guard already patched")
            return
        for anchor, replacement in replacements:
            count = s.count(anchor)
            assert count == 1, (
                f"{path.name}: token0 anchor matched {count} times (want 1):\\n{anchor}"
            )
            s = s.replace(anchor, replacement, 1)
        path.write_text(s)
        print(f"{path.name}: token0 guard patched")

    _one(schedule_batch, [(pad_anchor, pad_repl), (vocab_anchor, vocab_repl), (finish_anchor, finish_repl)])
    _one(processor, [(insert_anchor, insert_repl)])
    _one(scheduler, [(next_anchor, next_repl)])


if __name__ == "__main__":
    main()
APPLY_EOF

  cat > "${SCRIPT_DIR}/.patch/Dockerfile" <<DKR_EOF
# DSpark kernel-work image for Qwen3.8-Flash-Next-NVFP4 on DGX Spark (SM121/GB10).
# sglang#36806 (exclude SM121 from TRT-LLM sparse decode) + #36845 (Triton
# packed-varlen fallback) + NVFP4 KV for the QSA path + token-id-0 abort
# guard. See start.sh header.
FROM ${BASE_IMAGE}

COPY sm121_varlen.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sm121_varlen.py
COPY qsa_nvfp4_kv.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_nvfp4_kv.py
COPY apply_nvfp4_patches.py /tmp/apply_nvfp4_patches.py
RUN python3 - <<'PYEOF'
p = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
s = open(p).read()

# sglang#36845: SM121 packed-varlen Triton fallback (not FA4 CuTe).
if "qsa.sm121_varlen" not in s:
    anchor = "    try:\\n        from flash_attn import flash_attn_varlen_func"
    assert anchor in s, "flash_attn varlen anchor not found — upstream image layout changed"
    insert = (
        "    from sglang.srt.utils import is_sm121\\n"
        "\\n"
        "    if is_sm121():\\n"
        "        from sglang.srt.layers.attention.qsa.sm121_varlen import (\\n"
        "            qsa_sm121_varlen_attention,\\n"
        "        )\\n"
        "\\n"
        "        return qsa_sm121_varlen_attention\\n"
    )
    s = s.replace(anchor, insert + anchor, 1)

# sglang#36806: never take FlashInfer TRT-LLM sparse decode on SM121.
# A newer base image may re-enable it (token-id-0 at long context).
marker = "dspark: SM121 must not use TRT-LLM sparse decode"
if marker not in s:
    fn = "def _resolve_trtllm_sparse_decode():"
    i = s.find(fn)
    assert i >= 0, "trtllm resolver not found — upstream image layout changed"
    ds = s.find('"""', i)
    assert ds > 0, "trtllm resolver docstring not found"
    ds_end = s.find('"""', ds + 3)
    assert ds_end > 0, "trtllm resolver docstring unterminated"
    ds_end += 3
    s = s[:ds_end] + (
        "\\n    from sglang.srt.utils import is_sm121\\n"
        "\\n"
        "    # dspark: SM121 must not use TRT-LLM sparse decode\\n"
        "    # (sglang#36806 / #36845). That path silently emits token id 0\\n"
        "    # at long context on GB10.\\n"
        "    if is_sm121():\\n"
        "        return None\\n"
    ) + s[ds_end:]

open(p, "w").write(s)
print("qwen_sparse_attn_backend.py patched for SM121 (sglang#36806 + #36845)")
PYEOF

# DSpark: build GDN chunk indices once from host-side extend lengths.
# Avoid prepare_chunk_indices(...).tolist() CUDA -> CPU sync in every GDN layer.
RUN python3 - <<'PYEOF'
from pathlib import Path

ROOT = Path("/sgl-workspace/sglang/python/sglang")
MARK = "dspark_gdn_chunk_indices_cpu"

def patch(path, replacements):
    s = path.read_text()

    if MARK in s:
        print(f"{path.name}: GDN D2H patch already present")
        return

    for old, new in replacements:
        count = s.count(old)
        assert count == 1, (
            f"{path}: anchor matched {count} times, expected 1:\n{old}"
        )
        s = s.replace(old, new, 1)

    path.write_text(s)
    print(f"{path.name}: GDN D2H patch applied")


# 1. Precompute chunk indices once per forward from the CPU lengths.
p = ROOT / "srt/layers/attention/linear/gdn_backend.py"

patch(p, [
(
'''    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
''',
'''    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)

        # dspark_gdn_chunk_indices_cpu:
        # Build once on CPU instead of calling .tolist() on CUDA in every GDN layer.
        self.forward_metadata.dspark_gdn_chunk_indices = None
        if forward_batch.extend_seq_lens_cpu is not None:
            pairs = [
                (seq_idx, chunk_idx)
                for seq_idx, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                for chunk_idx in range(
                    (int(seq_len) + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE
                )
            ]
            self.forward_metadata.dspark_gdn_chunk_indices = (
                torch.tensor(pairs, dtype=torch.int32, device=self.device)
                if pairs
                else torch.empty((0, 2), dtype=torch.int32, device=self.device)
            )
'''
),
(
'''                query_start_loc=query_start_loc,
                state_checkpoint_cu_starts=(
''',
'''                query_start_loc=query_start_loc,
                chunk_indices=getattr(
                    forward_metadata, "dspark_gdn_chunk_indices", None
                ),
                state_checkpoint_cu_starts=(
'''
),
])


# 2. Pass the precomputed indices through the Triton backend.
p = ROOT / "srt/layers/attention/linear/kernels/gdn_triton.py"

patch(p, [
(
'''            cu_seqlens=query_start_loc,
            head_first=False,
''',
'''            cu_seqlens=query_start_loc,
            chunk_indices=kwargs.get("chunk_indices"),  # dspark_gdn_chunk_indices_cpu
            head_first=False,
'''
),
])


# 3. Use them instead of recomputing via CUDA -> CPU .tolist().
p = ROOT / "kernels/ops/attention/fla/chunk.py"

patch(p, [
(
'''        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
''',
'''        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_indices: Optional[torch.LongTensor] = None,  # dspark_gdn_chunk_indices_cpu
        use_qk_l2norm_in_kernel: bool = False,
'''
),
(
'''        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
            if cu_seqlens is not None
            else None
        )
''',
'''        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
'''
),
(
'''    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
''',
'''    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
    head_first: bool = False,
'''
),
(
'''        cu_seqlens,
        use_qk_l2norm_in_kernel,
''',
'''        cu_seqlens,
        chunk_indices,
        use_qk_l2norm_in_kernel,
'''
),
])

print("DSpark GDN CPU chunk-index patch complete")
PYEOF

RUN python3 /tmp/apply_nvfp4_patches.py && rm -f /tmp/apply_nvfp4_patches.py
DKR_EOF
}

ensure_patched_image() {
  header "SM121 kernel patch → ${PATCHED_IMAGE}"
  write_patch_context
  local stamp
  stamp="$(cat "${SCRIPT_DIR}/.patch/sm121_varlen.py" "${SCRIPT_DIR}/.patch/qsa_nvfp4_kv.py" "${SCRIPT_DIR}/.patch/apply_nvfp4_patches.py" "${SCRIPT_DIR}/.patch/Dockerfile" | sha256sum | cut -d' ' -f1)"

  if docker image inspect "${PATCHED_IMAGE}" >/dev/null 2>&1 \
     && [[ "$(cat "${SCRIPT_DIR}/.patch/.stamp" 2>/dev/null)" == "${stamp}" ]]; then
    ok "patched image present on head (context unchanged)"
  else
    docker build -t "${PATCHED_IMAGE}" "${SCRIPT_DIR}/.patch" || { error "failed to build ${PATCHED_IMAGE} on head"; exit 1; }
    echo "${stamp}" > "${SCRIPT_DIR}/.patch/.stamp"
    ok "patched image built on head"
  fi

  # .stamp is excluded: it is only written on the worker AFTER a successful
  # worker-side build. Syncing it here would make the freshness check below
  # compare the just-synced stamp against itself and keep a stale worker
  # image when the head rebuilds after a context change (head/worker drift).
  rsync -a -e "ssh ${SSH_OPTS[*]}" --exclude '.stamp' --exclude '__pycache__' \
    "${SCRIPT_DIR}/.patch/" "${WORKER_SSH}:/tmp/qwen38-dspark-patch/"
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
    warn "KERNEL_PATCH=0 — the stock image cannot serve Qwen4Exp on SM121 (FA4-cute MLIR failure, or silent token-0 if TRT-LLM sparse decode is enabled)"
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

# The per-file byte check, kept as a program string so it can run on either
# node: python3 on the head, and shipped to the worker by verify_snapshot_worker.
VERIFY_SNAPSHOT_PY="$(cat <<'PY'
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
)"

verify_snapshot() { # <snapshot_dir> <label> — byte-size check against the HF API
  local dir="$1" label="$2"
  python3 - "${MODEL_ID}" "${dir}" "${label}" <<<"${VERIFY_SNAPSHOT_PY}" || return 1
}

verify_snapshot_worker() { # <worker snapshot_dir> <label> — the same check, on the worker
  local dir="$1" label="$2" b64
  if ! wrun "command -v python3 >/dev/null 2>&1"; then
    warn "${label}: no python3 on the worker — skipping the per-file check (the byte total is still compared)"
    return 0
  fi
  # Ship the program as part of the command, never on stdin: wrun's ssh has to
  # leave stdin alone or it eats a caller's heredoc.
  b64="$(printf '%s' "${VERIFY_SNAPSHOT_PY}" | base64 | tr -d '\n')"
  wrun "printf %s '${b64}' | base64 -d | python3 - '${MODEL_ID}' '${dir}' '${label}'" || return 1
}

# Total logical bytes under a snapshots/ tree. -L dereferences, because a
# snapshot is a directory of symlinks into ../../blobs/ and the link's own size
# is meaningless. %.0f not %d: mawk overflows a 135 GB total with %d.
snapshot_bytes() { # <dir> → bytes
  find -L "$1" -type f -printf '%s\n' 2>/dev/null \
    | awk '{t+=$1} END{printf "%.0f\n", t+0}'
}

worker_snapshot_bytes() { # <worker dir> → bytes
  wrun "find -L '$1' -type f -printf '%s\n' 2>/dev/null | awk '{t+=\$1} END{printf \"%.0f\n\", t+0}'" \
    || echo 0
}

worker_snapshot_dir() { # <worker repo dir> → the snapshot holding config.json (or fail)
  local wrepo="$1" d=""
  if [[ -n "${HF_REVISION}" ]] \
     && wrun "test -f '${wrepo}/snapshots/${HF_REVISION}/config.json'" 2>/dev/null; then
    echo "${wrepo}/snapshots/${HF_REVISION}"; return 0
  fi
  d="$(wrun "find '${wrepo}/snapshots' -maxdepth 2 -name config.json -printf '%h\n' 2>/dev/null | head -1" || true)"
  [[ -n "${d}" ]] || return 1
  echo "${d}"
}

# Byte-level acceptance for the worker copy. A truncated shard keeps the file
# count intact, so the count check alone accepts it and you find out at weight
# load — or as garbage output.
verify_worker_weights() { # <worker repo dir> [head_bytes]
  local wrepo="$1" head_bytes="${2:-}" wsnap wbytes
  wsnap="$(worker_snapshot_dir "${wrepo}")" \
    || { error "worker snapshot has no config.json under ${wrepo}/snapshots"; return 1; }
  wbytes="$(worker_snapshot_bytes "${wrepo}/snapshots")"
  if [[ -n "${head_bytes}" ]]; then
    if [[ "${wbytes}" != "${head_bytes}" ]]; then
      error "worker bytes ${wbytes} != head ${head_bytes} — truncated or partial shard; rerun to resume"
      return 1
    fi
    ok "worker byte total matches head (${wbytes} bytes)"
  else
    info "worker byte total ${wbytes}"
  fi
  verify_snapshot_worker "${wsnap}" "worker cache" || return 1
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
      verify_worker_weights "${wrepo}" || {
        error "worker weights failed verification — delete ${wrepo} on the worker and rerun"
        exit 1
      }
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
    verify_worker_weights "${wrepo}" || {
      error "worker download incomplete — rerun (snapshot_download resumes)"
      exit 1
    }
    ok "weights downloaded on worker"
    return 0
  fi

  # rsync head → worker over the CX7 link (single internet download; resumable)
  # Count under snapshots/ only — .locks lives in the repo dir on the head but
  # is excluded from the rsync, so counting the whole repo would never match.
  local hcount wcount hbytes wbytes
  hcount="$(find "$(snapshot_root)/snapshots" \( -type f -o -type l \) 2>/dev/null | wc -l)"
  wcount="$(wrun "find '${wrepo}/snapshots' \( -type f -o -type l \) 2>/dev/null | wc -l" || echo 0)"
  hbytes="$(snapshot_bytes "$(snapshot_root)/snapshots")"
  wbytes="$(worker_snapshot_bytes "${wrepo}/snapshots")"
  info "head ${hcount} files / ${hbytes} bytes · worker ${wcount} files / ${wbytes} bytes"
  if [[ "${hcount}" == "${wcount}" && "${hbytes}" == "${wbytes}" ]] \
     && wrun "test -f '${wrepo}/snapshots'/*/config.json" 2>/dev/null; then
    verify_worker_weights "${wrepo}" "${hbytes}" || {
      error "worker weights failed verification — delete ${wrepo} on the worker and rerun"
      exit 1
    }
    ok "weights already synced to worker (${wcount} files / ${wbytes} bytes)"
    return 0
  fi
  # A count match with a byte mismatch is the truncated-shard case: fall through
  # and let rsync repair it rather than failing here (it resumes).
  if [[ "${hcount}" == "${wcount}" && "${hbytes}" != "${wbytes}" ]]; then
    warn "worker file count matches but bytes differ (${wbytes} != ${hbytes}) — re-syncing"
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
  verify_worker_weights "${wrepo}" "${hbytes}" || {
    error "worker weights failed verification — rerun to resume"
    exit 1
  }
  ok "weights synced to worker (${wcount} files / ${hbytes} bytes)"
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
    --enable-metrics
    --enable-cache-report
    # SM121 house rule (27B Spark cell / Inkling champion): prefill CUDA
    # graphs are unreliable on GB10.
    --disable-prefill-cuda-graph
    --host "${HOST_BIND}"
    --port "${PORT}"
  )
  if [[ "${NVFP4_KV_CACHE}" == "1" ]]; then
    SGLANG_ARGS+=(--kv-cache-dtype nvfp4)
  elif [[ -n "${KV_CACHE_DTYPE}" ]]; then
    SGLANG_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
  else
    SGLANG_ARGS+=(--kv-cache-dtype fp8_e4m3)
  fi
  if (( CONTEXT_LENGTH > YARN_NATIVE_CTX )); then
    SGLANG_ARGS+=(--json-model-override-args "${YARN_OVERRIDE_ARGS}")
    SGLANG_ARGS+=(--max-prefill-tokens "${MAX_PREFILL_TOKENS:-2048}")
  elif [[ -n "${MAX_PREFILL_TOKENS}" ]]; then
    SGLANG_ARGS+=(--max-prefill-tokens "${MAX_PREFILL_TOKENS}")
  fi
  [[ "${ALLOW_AUTO_TRUNCATE}" == "1" ]]    && SGLANG_ARGS+=(--allow-auto-truncate)
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
  [[ -n "${API_KEY}" ]] && SGLANG_ARGS+=(--api-key "${API_KEY}")
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
    "SGLANG_SANITIZE_NAN_LOGITS=1"
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
  # Re-derive from the assembled command line: whatever --api-key ends up last
  # there is what wait_ready must authenticate with.
  derive_effective_api_key

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
  # The sglang startup banner captured here contains server_args, which embeds
  # the configured --api-key verbatim. Create the logs owner-only so the key
  # set is not left world-readable between runs.
  chmod 600 "${LOG_FILE}" "${WORKER_LOG_FILE}"
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
    -e SGLANG_HOST_IP="${WORKER_CX7_IP}" \
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
    -e SGLANG_HOST_IP="${HEAD_CX7_IP}" \
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
    if curl -fsS "${AUTH_CURL[@]}" "${READY_URL}" >/dev/null 2>&1; then
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
  # Already running → report and exit cleanly (idempotent). Both ranks must be
  # up: a TP=2 group is not "running" because rank 0 is. If the head is up and
  # the worker is gone, rank 0 sits in NCCL rendezvous forever (or against a
  # stale rank 1), and exiting 0 here reports success for a pair that can never
  # serve. Recovery is stop-then-start, which the operator must be told to do.
  if docker ps --format '{{.Names}}' | grep -qx "${HEAD_CONTAINER}"; then
    if worker_docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${WORKER_CONTAINER}"; then
      ok "${HEAD_CONTAINER} already running"
      curl -fsS "${AUTH_CURL[@]}" "${READY_URL}" >/dev/null 2>&1 && ok "API is ready: ${READY_URL}" || info "API not ready yet (still booting?)"
      info "logs: ./start.sh logs   ·   stop: ./start.sh stop"
      exit 0
    fi
    error "half-assembled pair: ${HEAD_CONTAINER} is running on the head but"
    error "${WORKER_CONTAINER} is NOT running on ${WORKER_SSH}."
    error "Rank 0 cannot form a TP group alone -- it will wait in rendezvous."
    error "Recover with:  ./start.sh stop  &&  ./start.sh"
    exit 1
  fi

  preflight serve
  ensure_images
  header "Weights"
  download_on_head
  sync_weights_to_worker

  launch
  wait_ready
  boot_health_check
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
  echo -e "  ${BOLD}Context:${NC}    ${CONTEXT_LENGTH} tokens · mem-fraction ${MEM_FRACTION_STATIC} · max-running ${MAX_RUNNING_REQUESTS} (engine may cap via mamba) · graphs ${CUDA_GRAPH_BS}"
  echo -e "  ${BOLD}Logs:${NC}       ${LOG_FILE} (head), ${WORKER_LOG_FILE} (worker)"
  echo ""
  echo -e "  ${BOLD}Quick test:${NC}"
  echo "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \\"
  # Keyed boot → the example needs the header or it 401s. Printed as the
  # literal placeholder, never the secret itself.
  if [[ -n "${EFFECTIVE_API_KEY}" ]]; then
    local _qt_ph='$API_KEY'; [[ -z "${API_KEY}" ]] && _qt_ph='<your-api-key>'
    # shellcheck disable=SC2016,SC1003  # literal $API_KEY placeholder is intentional
    echo "      -H \"Authorization: Bearer ${_qt_ph}\" \\"
  fi
  echo "      -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
  if [[ -n "${EFFECTIVE_API_KEY}" ]]; then
    echo "    (served with --api-key: every /v1 request needs that Authorization header)"
  fi
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
  if curl -fsS --max-time 3 "${AUTH_CURL[@]}" "${READY_URL}" >/dev/null 2>&1; then
    ok "API ready: ${READY_URL}"
    curl -fsS --max-time 3 "${AUTH_CURL[@]}" "${READY_URL}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  served:", ", ".join(m["id"] for m in d["data"]))' 2>/dev/null || true
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
  curl -fsS --max-time 300 "${AUTH_CURL[@]}" "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Give me three words that rhyme with 'spark', then use one in a sentence.\"}], \"max_tokens\": 300}" \
    | python3 -c '
import json, re, sys
data = json.load(sys.stdin)
choice = data["choices"][0]
m = choice["message"]
r = m.get("reasoning_content") or ""
c = m.get("content") or ""
text = r + c
print("reasoning:", (r[:200] + "…") if len(r) > 200 else r)
print("content:  ", c)
if choice.get("finish_reason") == "abort":
    print("SMOKE FAIL: finish_reason=abort", file=sys.stderr)
    sys.exit(1)
if re.search(r"!{16,}", text):
    print("SMOKE FAIL: token-0 exclamation loop", file=sys.stderr)
    sys.exit(1)
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
  kv-eval)  shift; exec python3 "${SCRIPT_DIR}/evals/nvfp4_kv_eval.py" --base-url "http://127.0.0.1:${PORT}" --model "${SERVED_MODEL_NAME}" --api-key "${EFFECTIVE_API_KEY}" "$@" ;;
  doom-loop) shift; API_KEY="${EFFECTIVE_API_KEY}" exec python3 "${SCRIPT_DIR}/evals/doom_loop_repro.py" --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED_MODEL_NAME}" "$@" ;;
  doctor)   cmd_doctor ;;
esac
