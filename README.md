<h1 align="center">Qwen3.8-Flash-Next-NVFP4-vLLM-Dual-DGX-Spark</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Multi-node inference for [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) across 2 DGX Sparks using vLLM with TP2+EP+MTP3.

Based on [getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark).

All numbers in **KV cache budget** and **Default runtime** below were read from the running
server (`docker logs vllm-fn`, `docker inspect vllm-fn`) — they are measurements, not estimates.
That container runs `GPU_MEMORY_UTILIZATION=0.835`, i.e. the value `.env` / `.env.sample` ship today.

## Prerequisites

- 2 DGX Spark nodes (GB10, 128 GB unified memory, sm_121) connected via ConnectX RoCE/IB
- Passwordless SSH between nodes
- Docker on both nodes
- ~126 GiB free on each node for the checkpoint

## Quick Start

```bash
# 1. Edit .env (set IPs, interface, IB HCA, etc.)
cp .env.sample .env
vim .env

# 2. Download weights, rsync to worker, apply patch, launch
./start.sh

# 3. Confirm the KV cache pool that vLLM actually allocated (~11 min after launch)
docker logs vllm-fn 2>&1 | grep -E "Available KV cache memory|GPU KV cache size"
```

## Flags

| Flag | Description |
|------|-------------|
| `--no-download` | Skip HF download (weights already in local cache) |
| `--no-launch` | Download + rsync only, don't start the server |
| `--launch` | Apply patch + launch only (weights already on both nodes) |

## What Happens

1. **Download** — pulls `RadixArk/Qwen3.8-Flash-Next-NVFP4` to local HF cache
2. **Rsync** — copies weights to the worker node
3. **Image sync** — ensures `vllm/vllm-openai:qwen38-flash-next` is on both nodes
4. **PLE patch** — extracts `ple_layer.py` from the image and patches it into `files/ple_layer_patched.py` (no image rebuild; bind-mounted at runtime)
5. **Drop caches** — `sync && echo 3 > /proc/sys/vm/drop_caches` (mandatory for unified memory on GB10)
6. **Launch** — worker (rank 1) starts first, then head (rank 0) serves on `:8888`

Both containers are named **`vllm-fn`** (head and worker); `./stop.sh` removes both.

## .env Reference

`sample` = value in `.env.sample` · `live` = value in the `.env` on this box, which produced
every measurement in this README.

| Variable | sample | live | Description |
|----------|--------|------|-------------|
| `HEAD_IP` | `10.0.0.1` | `10.0.0.1` | Head node inter-node IP |
| `WORKER_IP` | `10.0.0.2` | `10.0.0.2` | Worker node inter-node IP |
| `WORKER_USER` | *(empty)* | `zurih` | SSH user on worker (blank = same user) |
| `IFACE` | `enp1s0f0np0` | `enp1s0f1np1` | Head-side inter-node interface |
| `WORKER_IFACE` | *(unset → `IFACE`)* | `enp1s0f0np0` | Worker-side interface (nodes are cross-wired) |
| `IB_HCA` | `=rocep1s0f0` | `=rocep1s0f1` | NCCL IB HCA (leading `=` = exact match, one device) |
| `WORKER_IB_HCA` | *(unset → `IB_HCA`)* | `=rocep1s0f0` | Worker-side HCA (cross-wired link) |
| `IB_GID_INDEX` | `3` | `3` | IB GID index (3 for ConnectX + RoCE) |
| `MODEL_ID` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | same | HuggingFace model |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | same | Name in `/v1/models` |
| `MAX_MODEL_LEN` | `1000000` | `1000000` | Context length (262144 = native, no YaRN) |
| `YARN_ENABLE` | `true` | `true` | Extend context via YaRN rope scaling — **auto force-disabled when `MAX_MODEL_LEN` ≤ 262144** |
| `YARN_FACTOR` | `4.0` | `4.0` | 262144 × 4.0 ≈ 1M |
| `GPU_MEMORY_UTILIZATION` | `0.835` | `0.835` | Fraction of the 121.69 GiB budgeted by vLLM (see [KV cache budget](#kv-cache-budget)) |
| `MAX_NUM_SEQS` | `8` | `8` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | `8192` | Prefill chunk / cudagraph ceiling |
| `PORT` | `8888` | `8888` | API server port (`--network host`) |
| **`KV_CACHE_DTYPE`** | `auto` | **`auto`** | **`auto` = inherits model dtype = bfloat16 → the served cache is bf16** |
| `TENSOR_PARALLEL_SIZE` | `2` | `2` | 1 GB10 per node × 2 nodes |
| `ENABLE_EXPERT_PARALLEL` | `true` | `true` | EP for the NVFP4 experts (required) |
| `MTP_NUM_SPECULATIVE_TOKENS` | `3` | `3` | MTP draft tokens (`0` = disable) |
| `PLE_OFFLOAD` | `false` | `false` | `true` → CPU-RAM offload of the 51 GB PLE table (**see gotcha**) |
| `IMAGE` | `vllm/vllm-openai:qwen38-flash-next` | same | Day-0 image |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | `1` | Required when `MAX_MODEL_LEN` > 262144 |
| `MASTER_PORT` | `50000` | `50000` | Distributed coordination port |
| `EXTRA_VLLM_ARGS` / `EXTRA_DOCKER_ARGS` / `HF_TOKEN` | unset | unset | Escape hatches (`EXTRA_VLLM_ARGS` is appended last) |

> **KV cache dtype:** the default is **`auto`, which resolves to bfloat16** (the engine logs
> `kv_cache_dtype=auto` with `dtype=torch.bfloat16`). `.env`, `.env.sample` and the `start.sh`
> fallback (`${KV_CACHE_DTYPE:-auto}`) all agree on this. **This checkpoint only supports bf16 KV
> cache as of now** — do not set `KV_CACHE_DTYPE=fp8`. See
> [KV cache budget](#kv-cache-budget) for the measured bf16 numbers.

## KV cache budget

Measured on the running container at the shipped defaults (`KV_CACHE_DTYPE=auto` → bf16,
`GPU_MEMORY_UTILIZATION=0.835`, `MAX_MODEL_LEN=1000000`, MTP3, TP2):

```
[gpu_worker.py:693]   Available KV cache memory: 35.11 GiB
[kv_cache_utils.py]   GPU KV cache size: 2,481,424 tokens,
                      Maximum concurrency for 1,000,000 tokens per request: 2.48x
```

> **How to read those two lines.** They are the same number in two units: **35.11 GiB** is the
> byte budget, **2,481,424 tokens** is what fits in it (~15 KB/token). The token figure is the
> capacity of the whole block pool, shared by every concurrent request — it is *not* per-request,
> and it is *not* summed over TP (TP2 shards each token across both GPUs, so ~70 GiB of VRAM is
> committed while the usable token space is 2.48M once). The third figure is the practical one:
> **2.48×** = 2.48 resident requests at full 1M context, or 8 concurrent sequences only while
> each stays under ~310K tokens.

**Per-GPU memory accounting (GB10, per node — from `gpu_worker.py:919`):**

| Line | GiB |
|------|-----|
| Total memory visible to CUDA | 121.69 |
| Free at startup | 109.41 |
| Budgeted at GMU 0.835 | 101.61 |
| Weights + non-torch | 64.45 |
| Peak activation | 2.04 |
| CUDA graphs | 0.39 |
| **KV cache** | **35.11** |

Checkpoint is 125.91 GiB on disk (NVFP4 experts ≈ 68 GB, FP8 PLE n-gram table ≈ 51 GB,
bf16 attention/dense/vision ≈ 16 GB). With EP + TP the per-GPU weight footprint lands at
64.45 GiB — roughly experts 34 GB + PLE shard ~25 GB + bf16 dense ~10 GB. Weights, not KV,
are what eats this box.

**Why the cache is so cheap:** of the 48 layers only every 4th is `full_attention`
(`full_attention_interval: 4`) → **12 KV-bearing layers + 1 MTP draft layer = 13**. The other
36 are Gated-DeltaNet linear attention: constant-size recurrent state per *request*, not per
token. With `num_key_value_heads: 2`, `head_dim: 256`:

```
13 layers × 2 (K,V) × 2 kv_heads × 256 × 2 B (bf16) = 26,624 B/token  (= 13,312 B/GPU at TP2)
```

vLLM reports 35.11 GiB / 2,481,424 tokens = **15,192 B/token/GPU**, ~14% above the pure
attention figure — that delta is the 36 GDN layers' state plus the sparse-attention (QSA)
index state, rounded into cache blocks. Rounding is forced: vLLM sets the **attention block
size to 1600 tokens** so the attention page size matches (and never falls below) the Mamba/GDN
page size, padding the mamba page by 0.25%.

| Config | KV pool | Cache tokens | 1M-ctx concurrency | 262K-ctx concurrency |
|--------|---------|--------------|--------------------|----------------------|
| **bf16 KV, GMU 0.835 — measured (shipped default, running now)** | **35.11 GiB** | **2,481,424** | **2.48×** | **9.5×** |
| bf16 KV, GMU 0.85 — measured (previous container) | 36.35 GiB | 2,572,755 | 2.57× | 9.8× |
| bf16 KV, `--kv-cache-memory=45502283776` (42.38 GiB, log-suggested) | 42.38 GiB | ≈ 3.00M | ≈ 3.0× | ≈ 11.4× |

There **is** headroom at 0.835 — the startup log says so explicitly:

```
Replace gpu_memory_utilization config with `--kv-cache-memory=37127157105` (34.58 GiB) to fit
into requested memory, or `--kv-cache-memory=45502283776` (42.38 GiB) to fully utilize gpu memory.
Current kv cache memory in use is 35.11 GiB.
```

So ~7.3 GiB (≈ +0.5× of 1M-context concurrency) is sitting unused; the 34.58 GiB "strict fit"
figure is just vLLM recomputing the 0.835 budget from its own profiling deltas — the pool ended
up ~0.5 GiB larger than that.

> **The default `.env` is oversubscribed.** `MAX_MODEL_LEN=1000000` × `MAX_NUM_SEQS=8` asks for
> 8M cache tokens; the pool holds 2.48M. That is fine — vLLM preempts/evicts rather than
> OOM — but only ~2 concurrent 1M requests stay resident, and the 3rd+ gets preempted and
> re-prefilled. Pick one of:
>
> | Goal | Settings | Result |
> |------|----------|--------|
> | Deep-context, sequential | `MAX_MODEL_LEN=1000000` `MAX_NUM_SEQS=2` | matches bf16 pool, no thrash |
> | 1M + more concurrency | `MAX_MODEL_LEN=1000000` `EXTRA_VLLM_ARGS="--kv-cache-memory=45502283776"` `MAX_NUM_SEQS=3` | ≈ 3 resident (42.38 GiB → ≈ 3.0M tokens) |
> | Throughput / agentic | `MAX_MODEL_LEN=262144` `YARN_ENABLE=false` `MAX_NUM_SEQS=8` | 8 resident, ~15% headroom (8 of 9.5 possible), no YaRN accuracy risk |
> | Middle ground | `MAX_MODEL_LEN=512000` `MAX_NUM_SEQS=4` | ≈ 4 resident at full 512K (2.48M bf16 pool) |

**Live occupancy / re-check after any change:**

```bash
docker logs vllm-fn 2>&1 | grep -E "Available KV cache|GPU KV cache size|Free memory on device"
curl -s localhost:8888/metrics | grep -i "kv_cache_usage\|preemption"
```

vLLM also prints its own arithmetic on startup — the `Free memory on device` line suggests exact
`--kv-cache-memory=<bytes>` values (quoted above). `start.sh` always emits
`--gpu-memory-utilization` (and validates it as required), so prefer `GPU_MEMORY_UTILIZATION` for
headroom control; `EXTRA_VLLM_ARGS` is appended last if you want `--kv-cache-memory`. Rough rule
of thumb for extrapolating: **1 GiB of KV pool ≈ 71–74K tokens ≈ +0.07× at 1M ctx** — from the
measured 15,192 B/token, and from the 0.835 → 0.85 delta (+1.24 GiB → +91,331 tokens). Block/page
rounding in the hybrid allocator means it is ±5%, not exact.

## Default runtime

`docker inspect vllm-fn` on the head:

```bash
vllm serve RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 2 --nnodes 2 --node-rank 0 \
  --master-addr 10.0.0.1 --master-port 50000 \
  --distributed-executor-backend mp \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --gpu-memory-utilization 0.835 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --max-model-len 1000000 --kv-cache-dtype auto \
  --load-format safetensors --safetensors-load-strategy lazy \
  --enable-chunked-prefill --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,
                    "original_max_position_embeddings":262144}}' \
  --host 0.0.0.0 --port 8888
```

As measured from the running container (`--gpu-memory-utilization 0.835`, the shipped default —
see [KV cache budget](#kv-cache-budget) for what that buys in tokens).

Container: `vllm/vllm-openai:qwen38-flash-next` (`sha256:d464f3b4…`, ~20.6 GB, NVIDIA dev build),
`--gpus all --network host --ipc host`, `--cap-add SYS_NICE`, `memlock=-1`,
`--device /dev/infiniband`, `--restart` unset (die-on-crash).

Env injected by `start.sh`: `PLE_QUANT_OVERRIDE=fp8`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `NCCL_IB_HCA`/`NCCL_IB_GID_INDEX`/`NCCL_IB_DISABLE=0`,
`{GLOO,NCCL,TP}_SOCKET_IFNAME=$IFACE`, `NCCL_DEBUG=WARN`.
Mounts: patched `ple_layer.py` → `/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro`,
`$HF_CACHE_DIR` → `/root/.cache/huggingface`, `$HOME/.cache/vllm` → `/root/.cache/vllm`
(container runs as root — the mount target must be `/root`, or offline HF lookups fail).

**Versions / backends actually selected:**

| Component | Selected |
|-----------|----------|
| vLLM | `v0.1.dev20073+g8e685d198`, V1 engine, V2 model runner |
| NCCL | 2.30.7, `PYNCCL` all-reduce only (no MNNVL multicast / symm-mem on sm_121) |
| Full attention | FlashAttention **v2** |
| Linear attention (36 GDN layers) | Triton/FLA GDN prefill, CUDA GDN decode (`head_k_dim=128`) |
| Sparse attention state | `QWEN38_FLASH_NEXT_EXP_QSA_STATE` |
| NVFP4 MoE | FlashInfer **CUTLASS**; shared expert → FlashInferExperts |
| Sampling | FlashInfer top-k/top-p; generation_config defaults `temp 1.0, top_p 0.95, top_k 20` |
| Prefix caching | on (shared across requests) |
| KV block size | attention page = **1600 tokens** (raised so it matches/pads the GDN page) |
| CUDA graphs | `FULL_DECODE_ONLY`, sizes 1–64, 0.39 GiB |

**Cold start ≈ 10m55s** (21:47:41 → API up 21:58:36 UTC): NCCL setup ~40 s, weight load 458 s
(target 392 s + MTP drafter 67 s, lazy safetensors), engine init 92.4 s (profile + KV alloc +
warmup), graph capture ~7 s. First request after launch is slow while FlashInfer autotunes.

**Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`,
`/detokenize`, `/metrics`, `/health`, `/version`, `/docs` on `http://$HEAD_IP:8888`.

## The PLE Patch

The NVFP4 checkpoint stores the 51B-param N-gram/PLE embedding table as FP8 shards + one global
`weight_scale`, but declares `*.ple.*` excluded in the ModelOpt-NVFP4 quant config. vLLM's PLE
resolver only enables FP8-PLE when the *whole* checkpoint is FP8-serialized, so it builds a BF16
embedding (≈102 GB) and crashes.

The fix: a resolver shim installed in `ple_layer.py` — when `PLE_QUANT_OVERRIDE=fp8` is set, the
quant-method lookup for the PLE embedding short-circuits to the image's own
`Qwen3_8FlashNextPLEFp8EmbeddingMethod` (which handles exactly this "FP8 shards + one global
weight_scale" layout), bypassing the FP8-checkpoint and `ignored_layers` checks. Applied at
runtime via bind-mount; no image rebuild needed. `start.sh` extracts and patches automatically;
delete `files/ple_layer_patched.py` to force re-extraction after an image update.

## Multimodal

The model supports **Text, Image, and Video** input. vLLM auto-detects multimodal capabilities
from the config — no extra flags needed. Caveat: the MTP drafter logs
`Draft model … does not support external multimodal embeddings`, so image/video prompts draft
from text-only inputs (lower acceptance rate on those requests).

```bash
curl http://localhost:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":[{"type":"text","text":"What is in this image?"},{"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}]}]}'
```

> **Verified live on this deployment** — fine OCR (48 px text inside a 3840×2160 frame,
> ≈8.2k vision tokens), color/shape/scene QA, and correct *refusal* to read text that isn't
> in the image. Two caveats: **(1)** this is a reasoning model and thinking consumes
> `max_tokens` first — if `content` is empty with `finish_reason: "length"`, just raise
> `max_tokens` (2–4k covers most image QA); `chat_template_kwargs.thinking_budget` is
> **not** honored by this build. **(2)** media must be a URL or base64 data-URL (no local
> paths — `--allowed-local-media-path` is not set), default limit is 1 image + 1 video per
> prompt, and the vision encoder budget is 16,384 tokens (larger inputs are auto-resized /
> sparse-sampled for video).

## YaRN (1M context)

```bash
# .env
MAX_MODEL_LEN=1000000
YARN_ENABLE=true
YARN_FACTOR=4.0
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

`max_position_embeddings` is 262144, so vLLM logs a `VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used
with extreme caution` warning — YaRN is what makes the 1M positions valid. Quality at 1M with
`factor 4.0` is beyond the checkpoint's trained range: benchmark before trusting long-context
answers, and fall back to `MAX_MODEL_LEN=262144` / `YARN_ENABLE=false` for comparison runs.

> **Rule:** YaRN exists to *extend* context beyond native. At `MAX_MODEL_LEN` ≤ 262144 it has no
> benefit and costs accuracy, so `start.sh` now force-disables it automatically (prints a NOTE) —
> no need to remember `YARN_ENABLE=false` when testing at native context.

## Performance (batch-1, greedy, default runtime above)

Captured on the previous container (GMU 0.85, same bf16 KV): batch-1 decode is memory-bandwidth
bound and does not care about pool size, so these carry over — but they predate the 0.835 restart.

| Content | Tokens | TTFT | Decode |
|---------|--------|------|--------|
| code | 545 | 0.26s | **55.8 tok/s** |
| reasoning | 575 | 0.28s | 56.0 tok/s |
| C# | 1200 | 0.27s | 44.4 tok/s |
| prose | 718 | 0.31s | 36.4 tok/s |

Re-measure after changing `KV_CACHE_DTYPE`, `MAX_MODEL_LEN`, or `MTP_NUM_SPECULATIVE_TOKENS` —
decode speed is dominated by KV bandwidth, so cache dtype moves it.

## Gotchas

- **`PLE_OFFLOAD=true` needs ~51 GB of free CPU RAM.** The launch log reported
  `Available RAM: 44.92 GiB` at target-weight load and `41.71 GiB` before the MTP drafter
  loads on this box — offloading will OOM or thrash swap. Keep it `false`
  here; the FP8 PLE shard fits comfortably on the GPU.
- **Drop page caches before every launch** (`start.sh` does this). Skipping it is the usual
  cause of a `CUDA out of memory` that "worked yesterday" on unified memory.
- **Never point `IB_HCA` at an HCA cabled to another cluster** — NCCL will hang mid-NCCL-init
  with no useful error. One exact-match device per node (leading `=`).
- **Cross-wired nodes**: head uses `f1`, worker `f0` — hence the separate `WORKER_IFACE` /
  `WORKER_IB_HCA` overrides. If you re-cable, set both.
- **fp8 KV is not supported by this checkpoint** — the weights only support bf16 KV cache as of
  now. Keep `KV_CACHE_DTYPE=auto`; do not pass `--kv-cache-dtype fp8`.
- **MTP >1 token** logs `running multiple times of forward on same MTP layer, which may result
  in lower acceptance rate`, and the QSA backend can't fuse multi-step draft decode (it rebuilds
  attention metadata per draft step). `MTP_NUM_SPECULATIVE_TOKENS=1` is the safe comparison point.
- `./stop.sh` force-removes `vllm-fn` on both nodes; nothing persists between launches
  (FlashInfer autotune cache in `~/.cache/vllm` is the only thing that carries over).

## Scripts

| Script | Purpose |
|--------|---------|
| `start.sh` | download → rsync → verify → image sync → PLE patch → launch rank 1 then rank 0 |
| `stop.sh` | `docker rm -f vllm-fn` on worker, then head |
| `check-weights.sh` | verify the checkpoint exists (and its size) on both nodes |
