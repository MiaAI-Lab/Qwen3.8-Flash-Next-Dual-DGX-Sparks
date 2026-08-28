<h1 align="center">Qwen3.8-Flash-Next-NVFP4 · 2× DGX Spark · SGLang TP2</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

<p align="center">
  <b>Serving a 176B-parameter NVFP4 MoE across two DGX Sparks</b> — including the SM121 kernel work that makes the Qwen4Exp sparse-attention path boot at all, and the GB10 unified-memory math that keeps two 128 GB machines from wedging.
</p>

---

## What is this?

A single-file, production recipe (`start.sh`) that serves
**[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)**
(≈176B params, ~135 GB, NVFP4-quantized MoE) with **SGLang** on **two NVIDIA DGX
Spark (GB10 / SM121)** nodes in tensor parallel over a direct ConnectX-7
200 Gb RoCEv2 link:

- **One command** downloads, verifies, and rsyncs the weights, builds the
  kernel-patched container image on *both* nodes, boots the 2-node cluster,
  and waits for readiness
- **OpenAI-compatible API** on `0.0.0.0:8888` — completions, chat, reasoning
  (`<think>`) and tool-call parsing, 1M-token context (YaRN)
- **NEXTN speculative decoding** (`3/1/4`) with CUDA-graph decode on both nodes
- Self-healing and idempotent: rerun after a failure and it resumes the
  download, reuses caches, and replaces stale containers

Measured on this cluster (2× GB10, TP=2): **64 tok/s single-stream decode,
117 tok/s aggregate at ×2 concurrency** with NEXTN speculative decoding;
vision input working (`text` + `image`).

## Why it needed kernel work

SGLang's message about this model on DGX Spark was blunt: *"We tried two DGX
Spark but it will need some more kernel work."* This repo contains that work.

The Qwen4Exp architecture routes attention through a **Qwen Sparse Attention
(QSA)** backend. SM120 and SM121 need different decode paths
([sglang#36537](https://github.com/sgl-project/sglang/issues/36537),
[#36806](https://github.com/sgl-project/sglang/pull/36806),
[#36845](https://github.com/sgl-project/sglang/pull/36845)):

- Exact SM120 is numerically correct with FlashInfer's TRT-LLM paged decode.
- SM121 silently corrupts long-context decode on that path: 120k / 190k /
  210k prompts return 32/32 token id 0 (`!`) while the server stays HTTP 200.
- Excluding SM121 from TRT-LLM leaves packed FA4 CuTe varlen, which **fails to
  compile on GB10** with an MLIR layout-congruence error:

```
error: layout #expected and #got are not considered equivalent
in the layout composition because their non-involved dimension...
```

The fix in `.patch/` (embedded in `start.sh`, built automatically into a
derivative Docker image):

- `sm121_varlen.py` — sglang#36845's Triton **packed one-query varlen kernel**
  for the exact QSA call contract:
  - one query row per varlen sequence (every QSA call shape, prefill included),
  - GQA, any head dim ≤ 256, online softmax,
  - **`cu_seqlens` read on-device** so CUDA-graph replay stays valid when the
    backend rewrites the sequence table
  - DSpark extra: fp8 K/V is allowed (upcast in-kernel) so `NVFP4_KV_CACHE=0`
    still works
- A Docker build step that (1) forces `_resolve_trtllm_sparse_decode` to
  `None` on SM121 (sglang#36806 — even if a newer base image re-enables it)
  and (2) returns the Triton fallback from `_resolve_flash_attn_varlen_func`
  when `is_sm121()` is true. SM100/SM120 keep their native paths.

The result: `qwen38-flashnext-dspark:local`, built on both nodes by `start.sh`,
boots, serves, and captures decode CUDA graphs across both machines.

### NVFP4 KV cache (`NVFP4_KV_CACHE=1`)

The same derivative image adds **NVFP4 KV cache** for the QSA layers — another
path upstream SGLang never wired for this architecture. Upstream's NVFP4 recipe
assumes FlashInfer prefill reading an FP8 *dequant workspace* covering the whole
pool plus TRT-LLM decode consuming native packed FP4 — neither consumer exists
on the QSA path (and the FP8 workspace alone would eat most of the FP4 savings:
fp4 + scales + fp8 workspace ≈ 1.56 B/elem vs bf16's 2; without it, 0.5625).

The patch (`qsa_nvfp4_kv.py` + `apply_nvfp4_patches.py`, applied at image
build):

- an `NVFP4KVCacheMethod` variant declaring **plain BF16 dequant reads** for
  every backend/phase — the pool allocates packed FP4 + per-block FP8 scales,
  no FP8 workspace
- the QSA decode/verify path runs the stock Triton compaction kernel over the
  packed FP4 buffers and (a second pass) the scale buffers, then dequantizes
  the gathered rows with flashinfer's `nvfp4_kv_dequantize`
- the chunked-prefill history gather dequantizes per-request on the way out
- `--kv-cache-dtype nvfp4` is allowed for QSA hybrids (upstream's MHA
  allow-list doesn't apply) and the pool-sizing math skips the FP8 workspace
  share

**On by default** (`NVFP4_KV_CACHE=1`). Opt out with `NVFP4_KV_CACHE=0` in
`.env` (or inline: `NVFP4_KV_CACHE=0 ./start.sh serve`) for **fp8_e4m3** KV
— not bf16. SM121 Triton cannot `tl.dot` fp8, so two QSA paths upcast K/V
loads to fp32 in-kernel (q stays bf16): the paged-varlen decode fallback
**and** the GQA prefill / chunk-prefill kernels. A first FP8 boot captured
decode graphs then died on the first extend-with-prefix (`Unsupported rhs
dtype fp8e4nv` in `_sparse_gqa_chunk_prefill`). Set `KV_CACHE_DTYPE=bf16`
only if you actually want bf16. `.env` is first-assignment wins, so a leftover
`NVFP4_KV_CACHE=0` above a later `=1` keeps fp8.

Measured KV pools on this cluster (1M YaRN, `MEM_FRACTION_STATIC=0.82`,
both QSA + indexer pools):

| KV | Flag | Pool | Memory | vs 1M context |
|---|---|---:|---:|---:|
| **NVFP4** | `NVFP4_KV_CACHE=1` | **2,851,328** tokens | 11.22 GB | **2.72×** |
| **FP8 E4M3** | `NVFP4_KV_CACHE=0` | **1,751,552** tokens | 11.28 GB | **1.67×** |
| bf16 (earlier boot) | `KV_CACHE_DTYPE=bf16` | 925,504 tokens | ~11.3 GB | 0.88× (under 1M) |

NVFP4 earlier boots landed **2.90–2.91M** at the same recipe (page rounding).
All kernels are CUDA-graph-safe on SM121 (verified bit-exact replay;
decode-graph capture needs on-device `k_scales_gpu` — a Python `k_scale=1.0`
default illegal-copied host→CUDA during capture and was patched). FP4 KV
accuracy is ~9 % relative K/V error on the tensors; retrieval below.

Measured retrieval (`NVFP4_KV_CACHE=1`, thinking off, temp 0). Verdict
**RELIABLE through 128k as a single huge prompt** (0/50/100% passkey). 64k
and 128k runs were 3/3 at every position:

| Case | Prompt tokens | Position | Result |
|---|---:|---|---|
| control / no haystack | 47 | — | PASS |
| decode / exact copy | 30 | — | PASS |
| NIAH 1,024 | 1,102 | 50% | PASS |
| NIAH 4,096 | 4,134 / 4,136 / 4,136 | 0% / 50% / 100% | PASS ×3 |
| NIAH 16,384 | 16,296 / 16,294 / 16,293 | 0% / 50% / 100% | PASS ×3 |
| NIAH 32,768 | ~32,587 | 0% / 50% / 100% | PASS ×3 |
| NIAH 65,536 | ~65,020 | 0% / 50% / 100% | PASS ×3 |
| NIAH 131,072 (single huge prompt) | ~130,040 | 0% / 50% / 100% | PASS ×3 |
| multi-fact binding | 8,229 | 15% | PASS |
| radix follow-up | 4k / 128k prefix | 50% | PASS |

At **256k** same-turn 0%/50% previously hit the token-0 `!!!!!!` loop
([sglang#36537](https://github.com/sgl-project/sglang/issues/36537)) — that
was the SM121 TRT-LLM sparse-decode path. This recipe now uses sglang#36845's
Triton fallback and keeps TRT-LLM off SM121. Upstream validated exact NIAH at
120k / 190k / 210k on one GB10 with zero token id 0; re-run
`./start.sh kv-eval --suite long` after rebuilding the image before treating
256k as trusted. Recency (100%) and a two-turn follow-up still retrieved the
passkey even on the old path — so the KV still held it. Treat **128k
same-turn** as the last fully measured retrieval depth on this cluster.

Re-run:

```bash
./start.sh kv-eval --require-nvfp4                  # ≤16k, a few minutes
./start.sh kv-eval --suite full --require-nvfp4 --json kv-eval.json   # up to 64k
./start.sh kv-eval --suite long --require-nvfp4 --json kv-eval.json   # 128k
```

`evals/nvfp4_kv_eval.py` plants a unique passkey at 0/50/100% of a synthetic haystack
and asks for it back (plus a 2-turn radix follow-up). `--suite long` is 128k.

## The GB10 memory cliff (crash post-mortem)

Two of our early boot attempts **hard-froze both machines** — no kernel panic,
no Xid, no pstore; just two 128 GB bricks that needed a power cycle. Root
causes, in case you're porting other big MoEs to DGX Spark:

1. **Never `--load-format dummy` this model on GB10.** The dummy-weights
   initializer upcasts fp8 parameters to fp16 *as a temporary copy*: the
   51.2 GB fp8 PLE n-gram table briefly demands >150 GB on a 121 GB unified
   memory part → wedge. Real safetensors loading streams weights and never
   does this.
2. **The PLE embedding table must stay CPU-offloaded.** SGLang's auto-rule
   (`ple_offload_embedding = CUDA && bf16`) is correct — overriding it with
   `--no-ple-offload-embedding` puts ~26 GB/rank of pinned host memory on the
   GPU side and overcommits the box.
3. **Budget against *unified* DRAM.** On GB10, pinned host memory, CUDA
   allocations, and the OS share one 121.7 GB pool:
   `0.82 × 122 GB (GPU budget) + ~6 GB (OS/docker) ≈ 106 GB`.
   The pinned PLE table is already inside the CUDA accounting on GB10 (same
   physical DRAM), so it doesn't need a separate line — but the QSA indexer's
   prefill workspace does: it materializes a `fp32 [chunk × history]` logits
   matrix per sparse layer per chunk, which at chunk 4096 and 300k history
   demands ~8-10 GB transient and wedged the box at `MEM_FRACTION_STATIC=0.85`.
   The shipped recipe uses `0.82` + `CHUNKED_PREFILL_SIZE=1024` (quarter the
   transient) + `MAMBA_FULL_MEMORY_RATIO=0.3` (reclaim mamba's over-provisioned
   47% share of the post-weights budget for KV tokens).
4. **Skip `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — unproven on
   GB10 and present in exactly zero of our known-good Spark recipes.

Also avoid `expandable_segments`-style allocator tinkering unless you can
survive a hard reset of every node in the job.

## Quick start

Prereqs (head node, a.k.a. spark1):

- Docker + the `hf` CLI (`pip install hf`), an HF token in `~/.bashrc` if the
  repo needs auth
- Passwordless `ssh` to the worker node
  - Default assumes a `~/.ssh/config` alias `spark2` (override with
    `WORKER_HOST`/`WORKER_SSH` in `.env` — see `.env.example`)
  - The worker login user defaults to the **same user** you run `start.sh`
    as on the head; set `WORKER_USER` in `.env` only if the worker uses a
    different account
- Both nodes: the ConnectX-7 ports cabled **directly** (no switch) and
  `rocep1s0f1/enp1s0f1np1` ↔ `rocep1s0f0/enp1s0f0np0` up

```bash
git clone <this-repo> && cd <this-repo>
cp .env.example .env       # then edit HEAD_CX7_IP / WORKER_CX7_IP / WORKER_HOST
./start.sh doctor      # fabric / GPU / RAM / disk / ssh preflight
./start.sh download    # ~135 GB → head, then rsync → worker (resumable)
./start.sh serve       # builds patched image (both nodes), boots TP2, waits
```

Or just `./start.sh` — it runs doctor → download → serve in order.

### Using it

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
        "model": "Qwen3.8-Flash-Next-NVFP4",
        "messages": [{"role": "user", "content": "Explain RoCEv2 in one paragraph."}]
      }'
```

The `Authorization` header is only needed when you set `API_KEY` (see [Key
tunables](#key-tunables)); on an open server you can drop that line. Keyed
servers answer bare requests with `401`.

Thinking is on by default (`reasoning_content` streams separately). To turn it
off per-request:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

If an agent session starts streaming `!!!!!!` until `max_tokens` *after* a
rebuild of the patched image, see
[Known quirks](#known-quirks-read-before-filing-a-bug). The SM121 decode
path that caused that ([sglang#36537](https://github.com/sgl-project/sglang/issues/36537))
is the kernel work in this recipe. This recipe keeps thinking on.

Images work through the standard `image_url` content part.

Wiring it into a local agent harness (e.g. pi's `~/.pi/agent/models.json`):

```json
"qwen38-flashnext-nvfp4-dgx": {
  "baseUrl": "http://127.0.0.1:8888/v1",
  "apiKey": "dummy",
  "api": "openai-completions",
  "authHeader": false,
  "auth": "none",
  "models": [{
    "id": "Qwen3.8-Flash-Next-NVFP4",
    "name": "Qwen3.8 Flash Next 176B NVFP4 · SGLang NEXTN · 262k (2×Spark TP2)",
    "reasoning": true,
    "input": ["text", "image"],
    "contextWindow": 1048576,
    "maxTokens": 32768,
    "compat": {
      "supportsDeveloperRole": false,
      "supportsReasoningEffort": false,
      "maxTokensField": "max_tokens",
      "thinkingFormat": "chat-template",
      "chatTemplateKwargs": { "enable_thinking": { "$var": "thinking.enabled" } }
    }
  }]
}
```

That block assumes an open server; with `API_KEY` set, put the key in
`"apiKey"` and flip `"authHeader": true` / `"auth": "bearer"`.

## Commands

| Command | What it does |
|---|---|
| `./start.sh serve` | Preflight → ensure images/weights → launch TP2 → wait for ready |
| `./start.sh download` | Download + verify + rsync weights only (`--download-only` equivalent) |
| `./start.sh stop` | Remove both containers, kill log followers |
| `./start.sh status` | Container status on both nodes |
| `./start.sh logs [N]` | Tail the last N lines of head/worker logs |
| `./start.sh smoke` | A quick greedy completion against the running server |
| `./start.sh doctor` | Full preflight (fabric, GPUs, RAM, disk, ports, recipe constraints) |

## Topology

```
                clients (LAN / Tailscale)
                       │  OpenAI API 0.0.0.0:8888
                       ▼
        ┌──────────────────────────┐
        │ spark1 — head, rank 0    │
        │ GB10 · SM121 · 128 GB    │
        │ SGLang server + router   │
        └────────────┬─────────────┘
                     │  CX7 200 Gb RoCEv2, direct cable
                     │  10.0.22.1 ↔ 10.0.22.2, GID 3
                     ▼
        ┌──────────────────────────┐
        │ spark2 — worker, rank 1  │
        │ GB10 · SM121 · 128 GB    │
        └──────────────────────────┘
```

NCCL is pinned per-node to the RoCE device (`NCCL_NET=IB`,
`NCCL_IB_DISABLE=0`, RoCE v2, NVLS/CUMEM off) with the option to
`LD_PRELOAD` a staged host NCCL 2.30.7 (`USE_HOST_NCCL=1`, the default) for
multi-node-TP stability on GB10.

## Memory layout (per node, TP=2)

| Component | Where | Size | nvidia-smi? |
|---|---|---|---|
| NVFP4 expert + dense/MTP/vision weights | GPU device | ~62.5 GB | ✅ `sglang::scheduler_TP0 63971MiB` |
| PLE n-gram table (fp8, cudaHostAlloc) | **pinned host** | ~11 GB | ❌ host-side, invisible |
| KV cache (both pools) | GPU device | **NVFP4: 11.22 GB → 2,851,328 tokens**; **FP8: 11.28 GB → 1,751,552 tokens**; bf16: ~11.3 GB → 925,504 | ✅ |
| Mamba/GDN state cache | GPU device | ~3.6 GB (73 slots) | ✅ |
| CUDA graphs + NCCL/cuBLAS workspaces | GPU device | ~8 GB | ✅ |
| **Total CUDA-visible** | | **~95.6 GB** | |
| **Free (avail_gpu_mem)** | | **~17.5 GB** | |

nvidia-smi per-process only shows CUDA device allocations — the pinned PLE
(`cudaHostAlloc`) lives in host RAM and is invisible to nvidia-smi, but GB10's
unified memory means it *does* consume from the same 121.7 GB physical DRAM.
The number that matters is `available_gpu_mem` in the SGLang log (or `free -h`'s
"available" column), which accounts for everything.

## Key tunables

Set via shell env or a `.env` next to `start.sh` (shell env wins; see
`.env.example` for a ready-to-edit template). Full list in the script header;
the ones you'll actually touch:

| Variable | Default | Notes |
|---|---|---|
| `HEAD_CX7_IP` | `10.0.22.1` | head's CX7 RoCE IP (rendezvous + NCCL rail) |
| `WORKER_CX7_IP` | `10.0.22.2` | worker's CX7 RoCE IP |
| `WORKER_HOST` | `spark2` | worker hostname/IP or `~/.ssh/config` alias for ssh |
| `WORKER_USER` | *(empty)* | empty = reuse the head login user (most setups); set only if the worker uses a different account |
| `WORKER_SSH` | *(derived)* | full `user@host` override if you need it |
| `PORT` | `8888` | API port, bound on all interfaces |
| `API_KEY` | *(empty)* | Empty = open (LAN-trusted) server. Set = serve with `--api-key` and send `Authorization: Bearer <key>` on the script's own readiness/status/smoke curls |
| `MEM_FRACTION_STATIC` | `0.80` | Script default. PLE is *inside* this budget on GB10 ([issue #8](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/8) — `0.70` double-counted it). This cluster's 1M YaRN `.env` uses `0.82` + chunk 1024 |
| `PLE_OFFLOAD` | *(auto)* | Empty = auto-rule (recommended on GB10); `1`/`0` to force |
| `NVFP4_KV_CACHE` | `1` | `1` = NVFP4 KV (**2,851,328 tokens** / 11.22 GB); `0` = **fp8_e4m3** KV (**1,751,552 tokens** / 11.28 GB) |
| `KV_CACHE_DTYPE` | *(empty)* | raw `--kv-cache-dtype` when `NVFP4_KV_CACHE=0` (empty → `fp8_e4m3`; `bf16` to force bf16). Must be empty when `NVFP4_KV_CACHE=1` |
| `CONTEXT_LENGTH` | `1048576` | YaRN 1M default (factor 4.0 × native 262144). Set `262144` for native (no YaRN). Pool is **2.85M (NVFP4)** or **1.75M (FP8)** — both > 1M |
| `MAX_RUNNING_REQUESTS` | `28` | Script default = mamba ceiling at 0.80 + default mamba ratio. `CUDA_GRAPH_BS` is extended to match. YaRN `.env` at mamba ratio 0.3 still caps ~14 |
| `CHUNKED_PREFILL_SIZE` | `1024` | Keep ≤1024 for 1M ctx — the QSA indexer logits buffer is `[chunk × history]` fp32 |
| `MAMBA_FULL_MEMORY_RATIO` | `0.3` | Default 0.9 over-provisions mamba (47% of budget); 0.3 is enough for 14 requests |
| `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT` | `3` / `1` / `4` | NEXTN spec-decode chain |
| `KERNEL_PATCH` | `1` | Build+use the SM121 QSA fallback image |
| `IMAGE` | *(auto)* | Use a specific image verbatim, skip patching |
| `DOWNLOAD_MODE` | `rsync` | `direct` = worker pulls from HF itself |
| `CPUSET` | `5-9,15-19` | GB10 big cores; empty disables pinning |
| `EXTRA_ARGS` | *(empty)* | Appended last — argparse last-wins overrides anything |

## Known quirks (read before filing a bug)

- **Agent `!!!!!!` loop (thinking + tools)** —
  [sgl-project/sglang#36537](https://github.com/sgl-project/sglang/issues/36537).
  Two different SM12x bugs produced token ID 0 (`!` after decoding) while the
  server stayed HTTP 200: FA4 CuTe failing to compile (SM120, or SM121 without
  a fallback) and FlashInfer TRT-LLM sparse decode silently corrupting long
  context on SM121. This recipe ports [sglang#36806](https://github.com/sgl-project/sglang/pull/36806)
  (never TRT-LLM on SM121) and [sglang#36845](https://github.com/sgl-project/sglang/pull/36845)
  (Triton packed-varlen fallback). Upstream's SM121 validation then passed
  structured `get_weather({"city": "Paris"})` plus exact NIAH at 120k/190k/210k
  with zero token id 0. If a client still streams `!!!!!!` until `max_tokens`
  after you rebuild the patched image, file it — the remaining workaround is
  still to turn thinking off for those sessions:

  Per request (preferred — keeps thinking for everything else):

  ```json
  "chat_template_kwargs": {"enable_thinking": false}
  ```

  Or server-wide, then `./start.sh stop && ./start.sh serve`:

  ```bash
  EXTRA_ARGS='--tool-call-parser qwen3_coder --default-chat-template-kwargs {"enable_thinking":false}' ./start.sh serve
  ```

  (`EXTRA_ARGS` is appended last; if `.env` already sets `EXTRA_ARGS`, merge
  into that line — no spaces inside the JSON.) Do **not** opt thinking back on
  in the same request that sends `tools` if you use this workaround. Without
  the parser, thinking works but tool calls leak as `<tool_call>` XML in
  `content` instead of `message.tool_calls`. Cap agent temperature at ≤ 0.7
  if you use the workaround; a residual loop has been seen at temp 1.0.

- **TileLang data-race warning at JIT time** —
  `Logits(bx, position) is written by multiple threads in loop (token,)` from
  the QSA indexer kernel (`qsa/mqa.py`). The stores are
  `position = group*GROUP + token` — disjoint by construction; the checker
  can't prove injectivity because `group` is a runtime value. False positive;
  silence with `PassKey.TL_DISABLE_DATA_RACE_CHECK` if it bothers you.
- **Greedy decoding is not bit-reproducible run-to-run.** Near-tie tokens
  occasionally flip ("2+2=" → "4" usually, "5" sometimes). No CUDA/CUTLASS
  errors; outputs stay coherent. Most plausible source is run-to-run FP
  reduction variance in the NVFP4 MoE GEMMs on SM121. If you need strict
  determinism, restart with `EXTRA_ARGS="--moe-runner-backend triton"`
  (measurably slower, deterministic).
- **Boot takes ~10 minutes** (135 GB weight load + JIT + CUDA-graph capture).
  The script waits up to `WAIT_TIMEOUT_MIN=90`.
- **The QSA indexer prefill workspace scales with `chunk × history`.** The
  TileLang MQA kernel allocates a `fp32 [chunk_size, history_length]` logits
  matrix per sparse layer per chunk, plus gather/topk copies that roughly
  double it. At `CHUNKED_PREFILL_SIZE=4096` and 300k history that's ~8-10 GB
  transient — enough to wedge the box at `MEM_FRACTION_STATIC=0.85` (only
  ~12 GB free). The shipped recipe uses chunk 1024 (quarter the transient) and
  fraction 0.82 (~17 GB free). Don't raise chunk above 1024 without profiling
  the peak transient against your free budget.

## Repository layout

```
start.sh          # everything: download, sync, image build, launch, ops
evals/            # live-API evals (not required to serve)
  nvfp4_kv_eval.py  # passkey/NIAH eval for NVFP4 KV reliability
.env.example      # copy to .env and edit (cluster IPs, worker ssh, recipe)
.env              # your local config (gitignored — create from .env.example)
.patch/           # (generated) SM121 kernel-patch Docker build context
                  #   sm121_varlen.py        — sglang#36845 Triton packed-varlen fallback
                  #   qsa_nvfp4_kv.py        — NVFP4 KV cache for the QSA path
                  #   apply_nvfp4_patches.py — source patches applied at build
.serve.log        # launcher output
.sglang.log       # head container log
.sglang-worker.log# worker container log
```

## Performance (decode, structural)

Benchmarked with `sglang bench_serving` on the live 2-node cluster:

| Streams | TTFT | Aggregate tok/s | Per-stream tok/s |
|---|---|---|---|
| ×1 | 117 ms | 64.4 | 64.4 |
| ×2 | 169 ms | 116.8 | 60.3 |
| ×4 | 517 ms | 114.1 | 33.2 |

NEXTN speculative decoding (`3/1/4`) is the primary throughput driver — each
decode step verifies 4 draft tokens in a single forward pass. Aggregate
throughput peaks at ×2 streams (117 tok/s) and plateaus beyond that as the
QSA indexer prefill workspace and mamba cache contend for the same ~17 GB of
free GPU memory.

## Credits

- [SGLang](https://github.com/sgl-project/sglang) — serving stack and the
  `qwen38flashnext` image; the DGX Spark cookbook and the 27B two-node recipes
  were the blueprint (and the proof that `flashinfer_cutlass` NVFP4 works on
  SM121)
- [RadixArk](https://huggingface.co/RadixArk) for the Qwen3.8-Flash-Next NVFP4
  checkpoint
- The DGX Spark community threads on multi-node NCCL/RoCEv2 — the per-node NIC
  pinning recipe comes from that collective debugging

---

<p align="center">
  <sub>Built on two machines that froze, rebooted, and froze again until they didn't.</sub>
</p>
