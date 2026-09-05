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
- ~126 GiB free **on each node** for the checkpoint. By default both nodes keep their own copy in `~/.cache/huggingface`; `start.sh` rsyncs the worker copy from the head once. Enable [NFS weight sharing](#nfs-weight-sharing-optional) to skip the worker copy entirely.

## Quick Start

```bash
# 1. Edit .env (set IPs, interface, IB HCA, etc.)
cp .env.sample .env
vim .env

# 2. Download weights onto the head
./download.sh
#    ./download.sh --fp8    # official FP8 instead

# 3. Sync to worker, apply patches, launch (NVFP4)
./start.sh --no-download
#    or ./start.sh       if you want start.sh to download as well
#    or ./start.sh --nfs to share the head cache over NFS instead of rsyncing

#    Or serve official FP8 instead: https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8
#    Same cluster .env; native 262K, no YaRN.
#    API name: qwen3.8-flash-next-fp8
./start-fp8.sh --no-download

# 4. Confirm the KV cache pool that vLLM actually allocated (~11 min after launch)
docker logs vllm-fn 2>&1 | grep -E "Available KV cache memory|GPU KV cache size"
```

`./start.sh` and `./start-fp8.sh` both launch containers named **`vllm-fn`** on the same port — stop first with `./stop.sh` before switching checkpoints.

## Flags

| Flag | Description |
|------|-------------|
| `--no-download` | Skip HF download (weights already in the head cache) |
| `--no-launch` | Download + distribute weights only, don't start the server |
| `--launch` | Skip download + sync; apply patches and launch (weights already on both nodes) |
| `--nfs` | Distribute weights over NFS instead of rsync (see [below](#nfs-weight-sharing-optional)) |
| `--no-nfs` | Force rsync distribution even if `NFS_SHARE=true` in `.env` |

## What Happens

1. **Download** — `./download.sh` pulls `RadixArk/Qwen3.8-Flash-Next-NVFP4` to the **head** HF cache (`./download.sh --fp8` for official FP8).
2. **Distribute** — by default `rsync` copies the checkpoint into the worker's own `~/.cache/huggingface/hub` over the ConnectX link, skipped when the worker already has it. With `NFS_SHARE=true` / `--nfs` this is replaced by the [NFS share](#nfs-weight-sharing-optional).
3. **Image sync** — ensures `vllm/vllm-openai:qwen38-flash-next` is on both nodes
4. **PLE patch** — extracts `ple_layer.py` from the image and patches it into `files/ple_layer_patched.py` (no image rebuild; bind-mounted at runtime)
5. **MXFP8 patch** — extracts `modelopt.py` and patches it into `files/modelopt_patched.py`, routing the MXFP8 shapes FlashInfer's `mm_mxfp8` cannot run to the BF16 emulation kernel (also bind-mounted)
6. **Preflight + overlays** — refuses to launch if another process holds a GPU on either node
   (`REQUIRE_IDLE_GPU=false` to override); prepares the `FP8_DENSE` / `QSA_PROFILE` bind-mounts
7. **Launch** — worker (rank 1) starts first, then head (rank 0) serves on `:8888`. Both launch
   scripts render the same `VLLM_ARGS` array (`EXTRA_VLLM_ARGS` really is appended last now;
   `ENABLE_EXPERT_PARALLEL=false` and `MTP_NUM_SPECULATIVE_TOKENS=0` are honored). The head's
   rendered script is kept as `.last_head_launch.sh` for inspection.

> **Not done for you: dropping page caches.** `start.sh` needs no root and does
> **not** drop page caches. Do it yourself on **both** nodes before a launch —
> it matters on GB10 unified memory (see [Gotchas](#gotchas)):
>
> ```bash
> sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
> ```

Both containers are named **`vllm-fn`** (head and worker); `./stop.sh` removes both.

## NFS weight sharing (optional)

**Off by default.** By default each node keeps its own copy of the checkpoint in
`~/.cache/huggingface`, and `start.sh` rsyncs the worker's copy from the head once (subsequent
launches detect it and skip the transfer). That costs ~126 GiB on the worker and one long copy the
first time, but the worker is then self-sufficient.

Turn NFS sharing on to skip the worker copy entirely — the head exports its HF cache and the worker
mounts it read-only:

```bash
NFS_SHARE=true ./start.sh --launch     # or set NFS_SHARE=true in .env
./start.sh --nfs                       # same, per-run flag
./start.sh --no-nfs                    # force rsync even with NFS_SHARE=true in .env
```

A privileged `vllm-fn-nfs` container on the head exports `$HF_CACHE_DIR` over NFSv4 on the ConnectX
address (`NFS_SERVER_IP`, auto-detected from `IFACE` — do **not** use the `10.0.0.1` loopback
alias). The worker Docker volume `vllm-fn-hf` mounts it read-only at `/root/.cache/huggingface`.

| | rsync (default) | `NFS_SHARE=true` |
|---|---|---|
| Worker disk | ~126 GiB | none |
| First launch | one full copy, then free | no copy |
| Every cold start | reads local disk | streams ~126 GiB over ConnectX |
| Head must stay up | only to launch | **for the whole serving run** |
| Failure mode | worker cache goes stale silently | share dies → worker loses its weights |

Prefer NFS when worker disk is tight or you re-pull checkpoints often; prefer the default when you
want the two nodes decoupled. Switching checkpoints is where NFS earns its keep — no re-sync.

Notes when it is on:

- `./stop.sh` leaves the share up so the next `--launch` does not rebuild it; `./stop.sh --nfs`
  tears it down and removes the worker volume. Kernel NFS in Docker can ignore SIGKILL when
  rpcbind is in D-state, so that path has a 15 s timeout and reports if the container survives.
- **Do not stop `vllm-fn-nfs` while vLLM is loading or running** — the worker reads shards from it.
- `./check-weights.sh` follows `NFS_SHARE`: it verifies the worker's local copy by default, or the
  NFS volume when sharing is on.

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
| `MODEL_ID` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | same | HuggingFace model (the snapshot in the head cache; `local-inference-lab/…` was the old name of the same weights and no longer resolves offline) |
| `FP8_DENSE` | `false` | `false` | `true` → serve the hybrid NVFP4+FP8-dense checkpoint with the `files/overlay` loader patches (see below) |
| `QSA_PROFILE` | `stock` | `stock` | `gb10` or a JSON from `files/qsa_gb10/bench_qsa_kernels.py` |
| `REQUIRE_IDLE_GPU` | `true` | `true` | abort if `nvidia-smi` shows a compute process on either node |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | same | Name in `/v1/models` |
| `MAX_MODEL_LEN` | `1000000` | `1000000` | Context length (262144 = native, no YaRN) |
| `YARN_ENABLE` | `true` | `true` | Extend context via YaRN rope scaling — **auto force-disabled when `MAX_MODEL_LEN` ≤ 262144** |
| `YARN_FACTOR` | `4.0` | `4.0` | 262144 × 4.0 ≈ 1M |
| `GPU_MEMORY_UTILIZATION` | `0.835` | `0.835` | Fraction of the 121.69 GiB budgeted by vLLM (see [KV cache budget](#kv-cache-budget)) |
| `MAX_NUM_SEQS` | `8` | `8` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | `8192` | Prefill chunk / cudagraph ceiling |
| `PORT` | `8888` | `8888` | API server port (`--network host`) |
| **`KV_CACHE_DTYPE`** | `fp8` | **`fp8`** | **Default. 3.65M cache tokens (13.93× at 262K) via `files/patch_qsa_fp8_kv.py`, applied automatically. `auto` = bf16 = 2.13M (8.13×)** |
| `TENSOR_PARALLEL_SIZE` | `2` | `2` | 1 GB10 per node × 2 nodes |
| `ENABLE_EXPERT_PARALLEL` | `true` | `true` | EP for the NVFP4 experts (required) |
| `MTP_NUM_SPECULATIVE_TOKENS` | `3` | `3` | MTP draft tokens (`0` = disable) |
| `PLE_OFFLOAD` | `false` | `false` | `true` → CPU-RAM offload of the 51 GB PLE table (**see gotcha**) |
| `IMAGE` | `vllm/vllm-openai:qwen38-flash-next` | same | Day-0 image |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | `1` | Required when `MAX_MODEL_LEN` > 262144 |
| `MASTER_PORT` | `50000` | `50000` | Distributed coordination port |
| `NFS_SHARE` | `false` | `false` | `true` → share the head HF cache over NFS instead of rsyncing a worker copy ([details](#nfs-weight-sharing-optional)) |
| `NFS_SERVER_IP` | *(unset → `IFACE` IPv4)* | *(unset → `10.0.22.1`)* | Head ConnectX address that exports the HF cache. Only used when `NFS_SHARE=true`. Do **not** use the `10.0.0.1` loopback alias |
| `EXTRA_VLLM_ARGS` / `EXTRA_DOCKER_ARGS` / `HF_TOKEN` | unset | unset | Escape hatches (`EXTRA_VLLM_ARGS` is appended last) |

> **KV cache dtype:** the default is now **`fp8`**, which needs `files/patch_qsa_fp8_kv.py` —
> the stock QSA kernels declare `supported_kv_cache_dtypes = ["auto", "bfloat16"]` and raise
> `Qwen3.8-Flash-Next QSA requires a BF16 main KV cache` otherwise. `start.sh` applies the patch
> automatically whenever `KV_CACHE_DTYPE` starts with `fp8`. Set `KV_CACHE_DTYPE=auto` for bf16.
> See [KV cache budget](#kv-cache-budget) for both measured sets.

## KV cache budget

Measured on the running container at the shipped defaults (`KV_CACHE_DTYPE=fp8`,
`GPU_MEMORY_UTILIZATION=0.835`, `MAX_MODEL_LEN=262144`, MTP3, TP2, nvidia NVFP4 checkpoint):

```
[gpu_worker.py:693]   Available KV cache memory: 32.02 GiB
[kv_cache_utils.py]   GPU KV cache size: 3,652,200 tokens,
                      Maximum concurrency for 262,144 tokens per request: 13.93x
```

**FP8 vs bf16 on this kit**, same GMU, same checkpoint, KV dtype the only change:

| | `auto` (bf16) | **`fp8`** (default) |
|---|---|---|
| KV pool | 31.7 GiB | 32.02 GiB |
| Cache size | 2,131,159 tok | **3,652,200 tok** |
| Concurrency at 262,144 | 8.13× | **13.93×** |
| Tokens per GiB | 67,229 | **114,060** (1.70×) |
| Reasoning + needle suite | 12/12 | 12/12 (no regressions) |

1.70×, not 2×, because the QSA indexer and compressor stay BF16 — only the main K/V halve.
A 1M-token context now fits 3.5× over. Weights (68.52 GiB/GPU), not KV, are the binding
constraint on this box.

> **Credit.** The FP8-KV kernel work in `files/patch_qsa_fp8_kv.py` is vendored from
> [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) (AGPL-3.0-or-later),
> which credits the underlying approach to
> [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX) (Apache-2.0).
> Applied here unchanged — every anchor matched this image's QSA sources.

> **How to read those two lines.** They are the same number in two units: **32.02 GiB** is the
> byte budget, **3,652,200 tokens** is what fits in it (~9.2 KB/token at fp8). The token figure is
> the capacity of the whole block pool, shared by every concurrent request — it is *not*
> per-request, and it is *not* summed over TP (TP2 shards each token across both GPUs, so ~64 GiB
> of VRAM is committed while the usable token space is 3.65M once). The third figure is the
> practical one: **13.93×** = 13.93 resident requests at the full 262,144 context — more than
> `MAX_NUM_SEQS=8` can even admit, so at these defaults the scheduler, not the cache, is the
> limit. At a 1M-token context (YaRN) the same pool is 3.5×.

**Per-GPU memory accounting (GB10, per node — from `gpu_worker.py:919`):**

| Line | GiB |
|------|-----|
| Total memory visible to CUDA | 121.69 |
| Free at startup | 108.79 |
| Budgeted at GMU 0.835 | 101.61 |
| Weights (`model_runner.py:407`) | 64.46 |
| Weights + non-torch | 68.52 |
| Peak activation | 1.07 |
| CUDA graphs | 0.54 |
| **KV cache** | **32.02** |

vLLM reports headroom for `--kv-cache-memory=41360080896` (38.52 GiB) if you want to push
`GMU` higher.

The nvidia checkpoint is 124 GiB on disk / 11 shards (NVFP4 experts, FP8 PLE n-gram table ≈ 51 GB,
bf16 attention/dense/vision). With EP + TP the per-GPU weight footprint lands at 64.46 GiB —
roughly experts 34 GB + PLE shard ~25 GB + bf16 dense ~10 GB. Weights, not KV, are what eats
this box, and that is more true at fp8 than it was at bf16.

**Why the cache is so cheap:** of the 48 layers only every 4th is `full_attention`
(`full_attention_interval: 4`) → **12 KV-bearing layers + 1 MTP draft layer = 13**. The other
36 are Gated-DeltaNet linear attention: constant-size recurrent state per *request*, not per
token. With `num_key_value_heads: 2`, `head_dim: 256`:

```
13 layers × 2 (K,V) × 2 kv_heads × 256 × 2 B (bf16) = 26,624 B/token  (= 13,312 B/GPU at TP2)
13 layers × 2 (K,V) × 2 kv_heads × 256 × 1 B (fp8)  = 13,312 B/token  (=  6,656 B/GPU at TP2)
```

At fp8 vLLM reports 32.02 GiB / 3,652,200 tokens = **9,418 B/token/GPU**, ~41% above the pure
attention figure (bf16 was 15,192 B/token/GPU, ~14% above). The overhead grows in relative
terms precisely because the attention half halved while the rest did not: the 36 GDN layers'
state and the sparse-attention (QSA) index state stay BF16, which is why the pool expands
1.70× rather than 2×. Rounding is forced: vLLM sets the **attention block size to 1600 tokens**
so the attention page size matches (and never falls below) the Mamba/GDN page size.

| Config | KV pool | Cache tokens | 1M-ctx concurrency | 262K-ctx concurrency |
|--------|---------|--------------|--------------------|----------------------|
| **fp8 KV, GMU 0.835 — measured (shipped default, running now)** | **32.02 GiB** | **3,652,200** | **3.65×** | **13.93×** |
| bf16 KV, GMU 0.835 — measured (same container, dtype only change) | 31.70 GiB | 2,131,159 | 2.13× | 8.13× |
| fp8 KV, `--kv-cache-memory=41360080896` (38.52 GiB, log-suggested) | 38.52 GiB | ≈ 4.39M | ≈ 4.4× | ≈ 16.8× |
| bf16 KV, GMU 0.85 — measured (older container, older checkpoint) | 36.35 GiB | 2,572,755 | 2.57× | 9.8× |

There **is** headroom at 0.835 — the startup log says so explicitly:

```
Replace gpu_memory_utilization config with `--kv-cache-memory=33651431445` (31.34 GiB) to fit
into requested memory, or `--kv-cache-memory=41360080896` (38.52 GiB) to fully utilize gpu memory.
Current kv cache memory in use is 32.02 GiB.
```

So ~6.5 GiB (≈ +740k fp8 cache tokens) is sitting unused; the 31.34 GiB "strict fit" figure is
just vLLM recomputing the 0.835 budget from its own profiling deltas — the pool ended up
~0.7 GiB larger than that. At 13.93× against `MAX_NUM_SEQS=8` there is little reason to chase
it: the scheduler admits fewer requests than the cache can already hold.

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
patched `modelopt.py` → `…/vllm/model_executor/layers/quantization/modelopt.py:ro`,
`$HOME/.cache/vllm` → `/root/.cache/vllm` on each node. HuggingFace cache: head bind-mounts
`$HF_CACHE_DIR`; the worker bind-mounts its own `~/.cache/huggingface` by default, or the NFS
volume `vllm-fn-hf` read-only when `NFS_SHARE=true`.
Container runs as root — the mount target must be `/root`, or offline HF lookups fail.

**Versions / backends actually selected:**

| Component | Selected |
|-----------|----------|
| vLLM | `v0.1.dev20073+g8e685d198`, V1 engine, V2 model runner |
| NCCL | 2.30.7, `PYNCCL` all-reduce only (no MNNVL multicast / symm-mem on sm_121) |
| Full attention (12 QSA layers) | Model-specific **Triton QSA kernels** (`_qsa_sparse_paged_gqa_splitk_kernel`, `persistent_topk`, `_qsa_mqa_paged_kernel`); FlashAttention v2 is only the nominal base class and never executes for text decode |
| Linear attention (36 GDN layers) | Triton/FLA GDN prefill, CUDA GDN decode (`head_k_dim=128`) |
| Sparse attention state | `QWEN38_FLASH_NEXT_EXP_QSA_STATE` |
| NVFP4 MoE | FlashInfer **CUTLASS**; shared expert → FlashInferExperts |
| MXFP8 linear | `FlashInferCutlassMxfp8LinearKernel`, except `linear_attn.in_proj_a/b` (`[48, 2560]`) and all `visual.*` → `EmulationMxfp8LinearKernel` (BF16, dequantized at load time) — see [The MXFP8 Kernel-Fallback Patch](#the-mxfp8-kernel-fallback-patch) |
| Sampling | FlashInfer top-k/top-p; generation_config defaults `temp 1.0, top_p 0.95, top_k 20` |
| Prefix caching | on (shared across requests) |
| KV block size | attention page = **1600 tokens** (raised so it matches/pads the GDN page) |
| CUDA graphs | `FULL_DECODE_ONLY`, sizes 1–64, 0.39 GiB |

**Cold start ≈ 10m55s** (21:47:41 → API up 21:58:36 UTC): NCCL setup ~40 s, weight load 458 s
(target 392 s + MTP drafter 67 s, lazy safetensors), engine init 92.4 s (profile + KV alloc +
warmup), graph capture ~7 s. First request after launch is slow while FlashInfer autotunes.

**Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/tokenize`,
`/detokenize`, `/metrics`, `/health`, `/version`, `/docs` on `http://$HEAD_IP:8888`.

## FP8-dense hybrid checkpoint (`FP8_DENSE=true`)

The routed experts are the only NVFP4 tensors in `RadixArk/Qwen3.8-Flash-Next-NVFP4`; every dense
projection (GDN in/out, attention q/k/v/o, both HyperConnections per layer, shared experts, lm_head —
≈9.1 GiB bf16) is streamed from LPDDR5x on every decode step and is ~6.9 of the ≈9.2 GiB a step
reads per GPU. (Qwen's official `Qwen3.8-Flash-Next-FP8` does **not** help here: it keeps exactly
the same dense tensors in bf16 and only makes the experts FP8, i.e. *more* bytes per step.)

`files/fp8dense/` builds a hybrid checkpoint on the head, CPU-only, in ~2 minutes and ~1 GB RAM:

```bash
./files/fp8dense/build.sh        # -> ~/.cache/huggingface/hub/models--MiaAI-Lab--Qwen3.8-Flash-Next-NVFP4-FP8dense
# then in .env:
FP8_DENSE=true
./start.sh --launch
```

- Experts / PLE shards are hard-linked (no extra disk); only the 4 bf16 shards are rewritten (11.8 GB).
- 591 dense linears become FP8 E4M3 with **per-output-channel** weight scales and dynamic per-token
  activation quantization at runtime (no calibration set needed, no block-size constraints, so the
  rank-320 HyperConnection projections quantize too). Relative RMSE of the dequantized weights is
  ≈2.6 % per tensor (the E4M3 rounding floor); GSM8K/AIME re-evaluation is still owed.
- Kept bf16 on purpose: embeddings, router gates, shared-expert gate, QSA indexer projection, GDN
  `in_proj_a/b`, conv1d, norms, PLE projections, the MTP layer, vision tower.
- The checkpoint is a ModelOpt `MIXED_PRECISION` config (`quantized_layers` per tensor). vLLM's
  loader needs four bind-mounted patch files (`files/overlay/*.py`, generated from the image's own
  sources by `apply_patches.py`, diffs alongside): the mixed config learns to dispatch
  `FP8_PER_CHANNEL_PER_TOKEN`, and the HyperConnection / lm_head modules stop hard-coding
  `quant_config=None`. `start.sh` mounts them on both nodes when `FP8_DENSE=true`.
- Expected effect (analytical, §3.1 of the report): ≈9.2 → ≈5.7 GiB per decode step, i.e. roughly
  1.5× batch-1 tok/s at equal MTP acceptance. **Not yet measured on GPU** (both Sparks were busy with
  another deployment when this was built) — run the §6 benchmark from the report before trusting it,
  and watch the first launch for `Selected CutlassFP8ScaledMMLinearKernel` in the log.

## QSA launch profiles (`QSA_PROFILE`)

The 12 full-attention layers run the model's own Triton sparse-attention kernels, whose tile tables
were tuned on GB300 (160 SMs). `files/qsa_gb10/` adds a bind-mounted `ops/qsa.py` that reads the
launch profile from `VLLM_QSA_PROFILE` / `VLLM_QSA_PROFILE_JSON`, a conservative 48-SM table
(`QSA_PROFILE=gb10`), and `bench_qsa_kernels.py`, which sweeps block/split/warp configs on the
real kernels with this deployment's decode shapes and writes the best table as JSON:

```bash
docker run --rm --gpus all --ipc host \
  -v $PWD/files/qsa_gb10/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py:ro \
  -v $PWD/files/qsa_gb10:/work -v $HOME/.cache/vllm:/root/.cache/vllm \
  --entrypoint python3 vllm/vllm-openai:qwen38-flash-next /work/bench_qsa_kernels.py --tp 2 --out /root/.cache/vllm/qsa_gb10.json
# .env: QSA_PROFILE=$HOME/.cache/vllm/qsa_gb10.json
```

`stock` (default) leaves the image untouched. The `gb10` table is a starting point, not a result.

**`reasoning_effort`:** the chat template accepts **`low`, `medium`, `xhigh`** (default
`xhigh`) and rejects anything else at template-render time with HTTP 400:

```
Unexpected reasoning effort max. Supported types are xhigh (default), medium, and low.
```

`max` and `high` are *not* accepted — worth knowing if you point a client at this lane
that also talks to models where `high`/`max` are valid, or share one client config across
models. Two related caveats: thinking tokens are billed against the request's `max_tokens`
(a small `max_tokens` returns empty `content` with `finish_reason: "length"`), and
`chat_template_kwargs.thinking_budget` is **not** honored by this build.

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

## The MXFP8 Kernel-Fallback Patch

On GB10 the MXFP8 linear layers run through `FlashInferCutlassMxfp8LinearKernel`, whose
`mm_mxfp8` only accepts weights with **N ≥ 128, N % 32 == 0, K ≥ 128, K % 32 == 0**
(measured on sm_121 against `vllm/vllm-openai:qwen38-flash-next`; the limits do not depend on
the token count). Two shapes in this checkpoint miss that:

| Layer | `[N, K]` | Why it fails |
|-------|----------|--------------|
| `language_model.layers.*.linear_attn.in_proj_a` / `in_proj_b` (×72) | `[48, 2560]` | `N < 128` → `AssertionError: mm_mxfp8 requires N >= 128, got N=48` |
| `visual.blocks.*.mlp.linear_fc1` (×27) | `[4304, 1152]` | `4304 % 32 == 16` → `ValueError: Problem size is not supported for mm_mxfp8` |

The first one is fatal at engine start — it fires on the very first forward pass, during the
`determine_available_memory()` profile run, ~7 minutes into the launch and after the weights are
already resident.

The fix: `files/patch_modelopt_mxfp8.py` rewrites `ModelOptMxFp8LinearMethod.create_weights` to
check the *post-TP-split* `(N, K)` it is about to hand the kernel and swap in
`EmulationMxfp8LinearKernel` when the native GEMM cannot take them. Emulation dequantizes MXFP8 →
BF16 once at load time, so those layers then run as plain BF16 linears (≈17 MB extra across all
72 `in_proj_a/b`). Each layer gets its own quant-method instance, so the downgrade is per-layer —
everything else keeps the native kernel. The fallback logs one line per distinct shape:

```
WARNING [modelopt.py] MXFP8 layer [N=48, K=2560] is not supported by
FlashInferCutlassMxfp8LinearKernel (needs N,K >= 128 and divisible by 32);
falling back to BF16 emulation for this shape.
```

On top of the shape check, **all** `visual.*` MXFP8 layers are routed to emulation by prefix.
That is the verified multimodal configuration; emulating only `visual.*` (rather than the whole
model) is also what keeps the load-time BF16 dequant from OOMing the GPU.

Applied at runtime via bind-mount on both nodes; no image rebuild. Delete
`files/modelopt_patched.py.orig` to force re-extraction after an image update.

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

## Checkpoint: nvidia/Qwen3.8-Flash-Next-NVFP4

133 GB / 11 shards: BF16 dense (attention, GDN, hyper-connections, shared experts,
vision, lm_head) + NVFP4 routed experts + a 51 GB FP8 per-tensor PLE table.
Measured on this 2-Spark kit at `MAX_MODEL_LEN=262144`, `GMU=0.835`:

| | value |
|---|---|
| Weights resident | 62.72 GiB / node (66.14 GiB incl. non-torch) |
| KV cache (fp8, default) | 32.02 GiB — **3,652,200 tokens**, 13.93× at 262K |
| Peak activation / CUDA graphs | 2.15 GiB / 0.4 GiB |
| Load time | ~6:15 weights, 122 s engine init |
| Decode (batch-1, greedy) | ~24.5 tok/s |

vLLM reports headroom for `--kv-cache-memory=44208844800` (41.17 GiB) if you want
to push `GMU` up. At 2.66M cache tokens a 1M-token context still fits ~2.6×.

Two checkpoint-specific fixes are applied automatically by `start.sh`; both exist
because this checkpoint records metadata differently from `local-inference-lab/…`:

1. **PLE table dtype** (`files/detect_ple_dtype.py`) — the FP8 PLE table is declared
   only in `quantization_config.config_groups`, not as `text_config.ple_embedding_dtype`
   which the patched `ple_layer.py` dispatches on. Recovered and re-injected via
   `--hf-overrides`. Without it the 51 GB FP8 PLE table fails to load.
2. **MTP layer-index alias** (`files/patch_checkpoint_config.py`) — vLLM builds the MTP
   draft layer at the absolute index `mtp.layers.48`, and matches quantization metadata
   by exact string. This checkpoint records only `mtp.layers.0`. The alias is added in
   **both** `config.json` and the legacy `hf_quant_config.json` (vLLM reads the legacy
   file on the draft-model path) and bind-mounted in; the HF cache is never modified.

3. **FP8_BLOCK_SCALES routed experts** (`files/patch_modelopt_fp8_block_moe.py`) — this
   checkpoint's MTP routed experts are 128x128 block-scaled FP8, but
   `ModelOptMixedPrecisionConfig.get_quant_method` builds `RoutedExperts` only for
   `FP8` / `NVFP4` / `W4A16_NVFP4` / `MXFP8`. Anything else returns `None`, giving a
   silently *unquantized* MoE that dies ~7 min into the load with
   `Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`.
   The patch adds the missing branch, routing to vLLM's own `Fp8MoEMethod` with
   `weight_block_size` read from the checkpoint's `group_size` (it refuses to guess —
   a wrong block shape applies misaligned scales silently rather than failing).

> **This gap is not specific to this image.** Upstream vLLM has no `FP8_BLOCK_SCALES`
> branch either — not at commit `d4d703ca` (which the model card recommends, and which
> only fixes FP8 PLE loading, #54882), nor on current `main`. Upgrading vLLM does not
> enable MTP here. `start.sh` preflights the MTP expert algo and fails in seconds if it
> ever sees one the dispatch cannot build.

### MTP measured on this kit (TP2+EP, MTP=3, 262144 ctx, GMU 0.835)

| | MTP off | MTP=3 |
|---|---|---|
| Decode (batch-1, greedy) | 24.5 tok/s | **52.1 tok/s (2.13x)** |
| Draft acceptance | — | **72.8%** (823/1131) |
| Acceptance by position | — | 89% / 74.5% / 60% |
| Weights resident | 62.72 GiB | 64.3 GiB |
| KV cache (bf16) | 2,663,445 tok (10.16x) | 2,131,159 tok (8.13x) |
| KV cache (fp8, default) | — | **3,652,200 tok (13.93x)** |

The decaying per-position acceptance curve is the check that matters: a wrong block
shape would show up as near-random acceptance, not as a crash.

Concurrency and prefill numbers for this configuration are in
[Performance](#performance-nvidiaqwen38-flash-next-nvfp4-tp2ep-mtp3-262144-ctx-gmu-0835).

## Reduced-vocabulary MTP drafting

**Off by default** — set `MTP_DRAFT_VOCAB` to a token-id list to enable it.

This checkpoint's vocabulary is **248,320 tokens**, and the MTP drafter carries its *own* BF16
`ParallelLMHead` over all of it (`tie_word_embeddings` is false): 248320 × 2560 × 2 B = **1.18 GiB,
0.59 GiB per GPU at TP=2**, read once per draft step. At MTP=3 that is three of the four lm_head
reads in an engine step — roughly a quarter of the ~9.2 GiB a single-stream step moves per GPU.
Slicing that head down for drafting is the largest single bandwidth lever in the decode step.

**It cannot change what the server emits.** Draft sampling is greedy and every draft is verified
against the target model, so a token outside the subset is simply rejected, exactly like any other
bad draft. The cost is acceptance, not correctness.

```bash
python3 bench/gen_draft_corpus.py --out corpus.jsonl          # the model's own output
python3 files/build_draft_vocab.py corpus.jsonl \
        --out draft_vocab.txt --size 65536 --fill-to-size --balance-shards 2
MTP_DRAFT_VOCAB=$PWD/draft_vocab.txt ./start.sh --launch
```

`start.sh` then patches `nvidia/mtp.py` (step 4e) and sets `use_local_argmax_reduction`, which is
the only path that reaches the reduced head — `compute_logits` keeps the full vocabulary.

**Balance the vocabulary across TP ranks.** A vocab-parallel lm_head splits by *id range*, and a
decode step waits for the slowest rank. Filling with the lowest-numbered ids put 65,392 of 65,536
on rank 0, so only one rank saved anything; `--balance-shards 2` splits them 32,768/32,768:

| draft vocab | draft head per rank | decode: prose / code / entropy / copy | acceptance |
|---|---|---|---|
| full 248,320 (default) | 0.59 GiB | 40.0 / 43.2 / 40.1 / **59.1** | **56.5%** |
| 65,536, lowest-id fill | 0.31 GiB | 40.3 / 44.1 / 42.3 / 65.0 | 53.2% |
| 65,536, `--balance-shards 2` | **0.16 GiB** | **42.5 / 44.5 / 45.4** / 56.5 | 47.8% |

*(bf16 KV, 400 decoded tokens at 1k context, greedy, `bench/decodebench.py`; acceptance as a delta
via `bench/mtp_accept.py`.)*

**Read this honestly: it is a real trade, not a free win.** Balancing helps prose, code and
entropy, but acceptance falls 56.5% → 47.8% and the `copy` task — the one that leans hardest on
the drafter predicting quoted text — ends up *below* the full-vocabulary baseline. Bandwidth and
acceptance pull in opposite directions.

**The vocabulary above is mostly padding, and that is the likely culprit.** The corpus behind it
was 85 documents / 49,141 token occurrences, which yielded only **8,305 distinct ids** — coverage
saturates by 16,384, so `--fill-to-size` padded the rest with the lowest-numbered unseen ids on the
theory that byte-level BPE is built in merge order. That is a *proxy, not a measurement*. Two
things worth trying before judging the lever: a much larger corpus so 65k can be ranked honestly,
and the corpus-only ~8.3k vocabulary, which would cut the draft head to ~0.02 GiB per rank.
`build_draft_vocab.py --report-only` prints coverage at several sizes; tune on coverage, not size.

`files/test_draft_vocab.py` checks the shard slicing and the cross-rank reduction against a
full-vocabulary argmax restricted to the draft set — run it after touching the patch.

> **Credit.** The idea and `files/build_draft_vocab.py` come from
> [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) (AGPL-3.0-or-later),
> which implements it for TP=1. `files/patch_mtp_draft_vocab.py` is a TP-aware rewrite:
> upstream refuses to engage at `tp_size != 1`, since its reduced head is a plain matmul rather
> than a vocab-parallel one. FR-Spec is the general technique.

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

> **Fixed Sep 2026 — YaRN was previously a silent no-op.** `start.sh` emitted
> `--hf-overrides '{"rope_parameters":{...}}'` at the *top level*. vLLM's
> `ModelConfig._apply_dict_overrides` only recurses into keys that are themselves nested configs;
> for `qwen4_exp` the parent config exposes `rope_parameters` as a plain dict, so the override was
> `setattr`'d onto the parent and never reached `text_config`, which the model actually reads.
> Verified in the image: after the top-level override `text_config.rope_parameters` still read
> `rope_type: "default"`. Every earlier "1M context" run was therefore serving 1M positions on
> *unscaled* rope. The override is now nested under `text_config` (together with
> `ple_embedding_dtype`), so `YARN_ENABLE=true` changes rope for the first time — treat 1M as
> **unvalidated** on this kit and benchmark it before trusting long-context answers.

## Performance (nvidia/Qwen3.8-Flash-Next-NVFP4, TP2+EP, MTP=3, 262144 ctx, GMU 0.835)

Measured with [**sparkDash**](https://github.com/MiaAI-Lab/sparkDash) against the running server.
Configuration at capture: **fp8 KV** (the shipped default) **plus the 65,536-id balanced MTP draft
vocabulary** — the draft vocab is *not* a default, it was passed via `MTP_DRAFT_VOCAB` (see
[Reduced-vocabulary MTP drafting](#reduced-vocabulary-mtp-drafting)). Drop that and expect the
decode column to move; the prefill column should not.

### Decode — prose, concurrency sweep

| Concurrency | Aggregate | Per stream | TTFT |
|---|---|---|---|
| ×1 | 54.4 tok/s | **54.4 tok/s** | 160 ms |
| ×2 | 86.5 tok/s | 45.1 tok/s | 426 ms |
| ×4 | 128.1 tok/s | 34.2 tok/s | 471 ms |
| ×6 | 169.7 tok/s | 29.7 tok/s | 490 ms |
| ×8 | **207.0 tok/s** | 26.7 tok/s | 432 ms |

Aggregate scales 3.8× from ×1 to ×8 while per-stream falls to 49%. Batch-1 decode is bound by
**weight** bandwidth, so extra streams ride the same weight fetch nearly for free at first, and the
per-stream cost only starts to bite once the batch itself fills the step. (Speculative decoding
normally contributes to that decay — drafts compete with real tokens for the step budget — but
acceptance was not measured per concurrency level here, so treat the split as unattributed.)

### Prefill

| Prompt | Tokens | Throughput | TTFT |
|---|---|---|---|
| 8k | 8,230 | 2874.9 tok/s | 2.86 s |
| 16k | 16,423 | 2962.1 tok/s | 5.54 s |
| 32k | 32,806 | **2962.2 tok/s** | 11.08 s |
| 64k | 65,575 | 2893.4 tok/s | 22.66 s |
| 128k | 131,109 | 2727.1 tok/s | 48.08 s |

Prefill is flat at ~2.96k tok/s from 16k to 64k and sheds 8% by 128k — QSA keeps attention from
dominating at long context, so TTFT stays close to linear in prompt length.

### Against the previous bf16-KV, full-vocabulary run

Same kit, same checkpoint, same GMU. Both changes (fp8 KV, reduced draft vocab) are folded in, so
the deltas are not individually attributable — the per-lever measurements are in
[FP8 KV](#kv-cache-budget) and [draft vocab](#reduced-vocabulary-mtp-drafting).

| | before | now | Δ |
|---|---|---|---|
| decode ×1 | 52.1 tok/s | **54.4** | +4.4% |
| decode ×2 | 83.3 | **86.5** | +3.8% |
| decode ×4 | **136.5** | 128.1 | −6.2% |
| decode ×6 | **170.7** | 169.7 | −0.6% |
| decode ×8 | **210.1** | 207.0 | −1.5% |
| TTFT ×1 | 236 ms | **160 ms** | −32% |
| prefill 32k | 2933.3 tok/s | **2962.2** | +1.0% |
| prefill 128k | 2697.1 tok/s | **2727.1** | +1.1% |
| KV cache | 2,131,159 tok | **3,652,200** | +71% |

Low concurrency and TTFT improved; mid-range concurrency (×4) regressed. Prefill is essentially
unchanged, which is the expected shape — these levers act on the decode step and the cache, not
on prefill compute.

Re-measure after changing `MAX_MODEL_LEN`, `MTP_NUM_SPECULATIVE_TOKENS`, `KV_CACHE_DTYPE`,
`MTP_DRAFT_VOCAB`, `FP8_DENSE` or `QSA_PROFILE`. Each decode step streams ≈9 GiB per GPU (lm_head
×4 because of the three MTP draft passes, routed experts, GDN projections, replicated bf16
HyperConnection); a reduced draft vocabulary is a direct cut into three of those four lm_head
reads. See `docs/CLAUDE/fable5-1-report.md` §2 for the byte model and `FP8_DENSE` below for
another lever.

<details>
<summary>Previous checkpoint, batch-1 content mix (superseded)</summary>

Captured on an earlier container (GMU 0.85, MTP off, different weights) — kept only as a content-mix
reference; the numbers above supersede these.

| Content | Tokens | TTFT | Decode |
|---------|--------|------|--------|
| code | 545 | 0.26s | 55.8 tok/s |
| reasoning | 575 | 0.28s | 56.0 tok/s |
| C# | 1200 | 0.27s | 44.4 tok/s |
| prose | 718 | 0.31s | 36.4 tok/s |

</details>

## Gotchas

- **`PLE_OFFLOAD=true` needs ~51 GB of free CPU RAM.** The launch log reported
  `Available RAM: 44.92 GiB` at target-weight load and `41.71 GiB` before the MTP drafter
  loads on this box — offloading will OOM or thrash swap. Keep it `false`
  here; the FP8 PLE shard fits comfortably on the GPU.
- **Drop page caches before every launch** if you hit a `CUDA out of memory` that
  "worked yesterday" on unified memory: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`
  on **both** nodes. `start.sh` does **not** do this for you (it needs no root).
- **Never point `IB_HCA` at an HCA cabled to another cluster** — NCCL will hang mid-NCCL-init
  with no useful error. One exact-match device per node (leading `=`).
- **Cross-wired nodes**: head uses `f1`, worker `f0` — hence the separate `WORKER_IFACE` /
  `WORKER_IB_HCA` overrides. If you re-cable, set both.
- **fp8 KV needs the QSA patch, which `start.sh` applies for you.** The stock kernels declare
  `supported_kv_cache_dtypes = ["auto", "bfloat16"]`; passing `--kv-cache-dtype fp8` to an
  unpatched image raises `Qwen3.8-Flash-Next QSA requires a BF16 main KV cache`. Do not hand-roll
  the flag — set `KV_CACHE_DTYPE` in `.env` so the patch is applied with it.
- **`.env` beats the environment.** `start.sh` sources `.env` after reading the environment, so
  `KV_CACHE_DTYPE=fp8 ./start.sh` is silently ignored for any key that `.env` already defines.
  Edit `.env`, or check the `KV dtype:` line in the launch summary.
- **MTP >1 token** logs `running multiple times of forward on same MTP layer, which may result
  in lower acceptance rate`, and the QSA backend can't fuse multi-step draft decode (it rebuilds
  attention metadata per draft step). `MTP_NUM_SPECULATIVE_TOKENS=1` is the safe comparison point.
- `./stop.sh` force-removes `vllm-fn` on both nodes. If the NFS share is in use it stays up
  so the next `--launch` does not rebuild it; `./stop.sh --nfs` tears it down too.
  FlashInfer autotune cache in `~/.cache/vllm` is the only per-node state that carries over.
- **`NFS_SHARE=true` only:** do not stop `vllm-fn-nfs` while vLLM is loading or running — the
  worker reads shards from it. Cold start streams ~126 GiB over CX7 (lazy safetensors); once
  weights are in GPU memory the share is idle.

## Credits

| | |
|---|---|
| Base deployment | [getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark) |
| FP8 KV cache kernels, draft-vocabulary builder | [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) (AGPL-3.0-or-later) — the single-Spark TP=1 recipe |
| FP8-KV approach (via the above) | [lancelind/qwen3.8-Flash-DGX](https://github.com/lancelind/qwen3.8-Flash-DGX) (Apache-2.0) |
| Concurrency / prefill benchmarks | [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) |
| PLE quant dispatch | ported from vLLM PR #53899 (`qwen4_exp`) |
| Model | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |

vLLM remains Apache-2.0; the container image and checkpoint are governed by their upstream terms.
Files under `files/` that carry an `SPDX-License-Identifier` header keep the license of their
origin — see each file.

## Scripts

| Script | Purpose |
|--------|---------|
| `download.sh` | fetch weights onto the **head** (`--fp8` for official FP8); `start.sh` handles the worker |
| `start.sh` | optional download on head → distribute to worker (rsync, or NFS with `--nfs`) → verify → image sync → PLE + MXFP8 patches → launch rank 1 then rank 0 |
| `stop.sh` | `docker rm -f vllm-fn` on worker, then head (`--nfs` also stops the share) |
| `check-weights.sh` | verify the checkpoint on the head and on the worker (local copy, or over NFS when `NFS_SHARE=true`) |
