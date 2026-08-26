<h1 align="center">Qwen3.8-Flash-Next-NVFP4 · 2× DGX Spark · SGLang TP2</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
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
  (`<think>`) and tool-call parsing, 900k-token context
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
(QSA)** backend. Its flash-attn resolver prefers classic FA2, and otherwise
falls back to flash-attn-4's **CuTe DSL** interface — which **fails to compile
on SM121** with an MLIR layout-congruence error:

```
error: layout #expected and #got are not considered equivalent
in the layout composition because their non-involved dimension...
```

The fix in `.patch/` (embedded in `start.sh`, built automatically into a
derivative Docker image):

- `qsa_fa_fallback.py` — a Triton **FlashDecoding-style varlen kernel**
  specialized for the exact QSA call contract:
  - one query row per varlen sequence (every QSA call shape, prefill included),
  - GQA, any head dim ≤ 256, online softmax,
  - **`cu_seqlens` read on-device** so CUDA-graph replay stays valid when the
    backend rewrites the sequence table,
  - host-sync guards disabled during graph capture
- A Docker build step that patches `qwen_sparse_attn_backend.py` to return the
  Triton fallback whenever `is_sm100_supported()` is false — i.e. everywhere
  except B100/B200, so the stock path is untouched on datacenter GPUs

The result: `qwen38-flashnext-dspark:local`, built on both nodes by `start.sh`,
boots, serves, and captures decode CUDA graphs across both machines.

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
  -d '{
        "model": "Qwen3.8-Flash-Next-NVFP4",
        "messages": [{"role": "user", "content": "Explain RoCEv2 in one paragraph."}]
      }'
```

Thinking is on by default (`reasoning_content` streams separately). To turn it
off per-request:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

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
    "contextWindow": 900000,
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
| KV cache (bf16, both pools) | GPU device | 9.0 GB → **956,800 tokens** | ✅ |
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
| `MEM_FRACTION_STATIC` | `0.82` | Budget for unified DRAM; 0.82 leaves ~17 GB free for prefill transients |
| `PLE_OFFLOAD` | *(auto)* | Empty = auto-rule (recommended on GB10); `1`/`0` to force |
| `CONTEXT_LENGTH` | `900000` | YaRN-scaled (factor 4.0, native 262144); pool is 956,800 tokens |
| `MAX_RUNNING_REQUESTS` | `16` | Concurrency (capped at 14 by mamba slots at ratio 0.3) |
| `CHUNKED_PREFILL_SIZE` | `1024` | Keep ≤1024 for 900k ctx — the QSA indexer logits buffer is `[chunk × history]` fp32 |
| `MAMBA_FULL_MEMORY_RATIO` | `0.3` | Default 0.9 over-provisions mamba (47% of budget); 0.3 is enough for 14 requests |
| `SPEC_STEPS` / `SPEC_TOPK` / `SPEC_DRAFT` | `3` / `1` / `4` | NEXTN spec-decode chain |
| `KERNEL_PATCH` | `1` | Build+use the SM121 QSA fallback image |
| `IMAGE` | *(auto)* | Use a specific image verbatim, skip patching |
| `DOWNLOAD_MODE` | `rsync` | `direct` = worker pulls from HF itself |
| `CPUSET` | `5-9,15-19` | GB10 big cores; empty disables pinning |
| `EXTRA_ARGS` | *(empty)* | Appended last — argparse last-wins overrides anything |

## Known quirks (read before filing a bug)

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
.env.example      # copy to .env and edit (cluster IPs, worker ssh, recipe)
.env              # your local config (gitignored — create from .env.example)
.patch/           # (generated) SM121 kernel-patch Docker build context
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
