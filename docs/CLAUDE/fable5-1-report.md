# Qwen3.8-Flash-Next-NVFP4 on 2× DGX Spark (vLLM TP2+EP+MTP3): performance and stability review

Author: Claude Fable 5.1 · Date: 2026-09-02 · Repo: `Qwen3.8-Flash-vLLM` (branch `main`, dirty tree)

## 0. Scope and method

- **What was reviewed.** The launch scripts (`start.sh`, `start-fp8.sh`, `stop.sh`, `files/nfs-share.sh`), the
  bind-mounted PLE patch, the `.env` in use, and the full model/kernel source inside the day-0 image
  `vllm/vllm-openai:qwen38-flash-next` (`sha256:d464f3b4…`, vLLM `0.1.dev20073+g8e685d198`, FlashInfer 0.6.17,
  torch 2.13+cu130, Triton 3.7.1, NCCL 2.30.7, deep_ep 2.0.0). Sources were extracted with `docker cp`; every
  claim below about "what runs" is from that code, not from the README.
- **Checkpoint.** The safetensors headers of the checkpoint actually on disk
  (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, rev `7b71922…`, 125.87 GiB) were parsed to get per-tensor dtypes and
  sizes. That is what the per-step bandwidth model in §2 is built on.
- **What was NOT done.** No live benchmark. `vllm-fn` is not running on either node, and the **worker (spark2)
  is currently running a different vLLM container** (`deepseek-v4-flash-vllm-dspark-1`, anemll dspark image),
  so launching this stack now would contend for the GPU. All performance numbers are therefore either the
  README's measured values (55.8 tok/s batch-1 greedy) or analytical estimates clearly marked as such.
  §6 gives the exact commands to turn each hypothesis into a measurement.

Hardware facts confirmed on the head (spark1) and worker (spark2): GB10, sm_121, 121 GiB unified LPDDR5x
(~273 GB/s theoretical), 20 ARM cores, driver 580.159.03 / CUDA 13.0, kernel 6.17.0-1026-nvidia, CPU governor
`performance`, THP `madvise`. Two ConnectX-7 chips per node, four 200 GbE RoCE v2 ports each, all `ACTIVE`.

---

## 1. What is actually executing (per component)

| Component | Kernel / path actually selected on GB10 | Where |
|---|---|---|
| Full attention (12 layers) | **Not FlashAttention.** The QSA owner runs its own Triton kernels: `_qsa_mqa_paged_kernel` (index scoring), `torch.ops._C.persistent_topk` (cooperative variant is excluded on sm12x), `_expand_qsa_indices_kernel`, `_qsa_sparse_paged_gqa_splitk_kernel` + `_qsa_merge_splitk_kernel` (attention). `FlashAttentionImpl` is only the base class; "FA2" in the README is nominal. | `vllm/models/qwen3_8_flash_next/nvidia/{qsa.py,indexer_qsa.py,ops/qsa.py}` |
| QSA pre-indexer | Fused Triton `_qsa_pre_indexer_kernel` (norm + RoPE + raw/compressed key writes), `num_warps=1` | `nvidia/ops/qsa_pre_indexer.py` |
| GDN linear attention (36 layers), decode with MTP | vLLM CUDA `fused_gdn_decode_post_conv_mtp` (K=V=128, fp32 state, ≤8 spec tokens) | `mamba/gdn/qwen_gdn_linear_attn.py:1705-1835` |
| GDN decode without MTP | Triton FLA `fused_recurrent_gated_delta_rule_packed_decode` | same, `_forward_core_decode_non_spec` |
| GDN prefill | **Triton FLA `chunk_gated_delta_rule`.** FlashInfer/CuteDSL GDN prefill is gated to SM90 / SM10x in `_resolve_gdn_prefill_backend`; sm_121 falls to Triton even though FlashInfer 0.6.17 ships an SM120 CP delta-rule DSL kernel (`gdn_prefill.py` accepts `arch_major == 12`). | `qwen_gdn_linear_attn.py:112-160` |
| MoE (48 layers, 512 experts, top-10, NVFP4 W4A4) | FlashInfer **CUTLASS SM120** grouped GEMM (`gen_cutlass_fused_moe_sm120_module`). Backend order tried: TRTLLM (SM10x only) → CuteDSL (SM10x only) → **CUTLASS ✓**. The SM12x-native `FLASHINFER_B12X` CuteDSL fused MoE exists but is excluded from auto-select and requires EP off. | `fused_moe/oracle/nvfp4.py`, `experts/flashinfer_cutlass_moe.py`, `experts/flashinfer_b12x_moe.py` |
| MoE communication | `--all2all-backend allgather_reducescatter` is **never invoked**: `use_all2all_kernels` is false when DP=1 and PCP=1 (`fused_moe/config.py:1056`). Tokens are TP-replicated; each rank computes its 256 local experts and the result goes through the normal TP all-reduce. So EP here only changes *which* expert weights each GPU holds, not the collective. | `fused_moe/config.py`, `all2all.py:44-155` |
| Dense bf16 GEMMs (GDN/attention projections, HC, router, shared expert, lm_head) | **cuBLAS.** The model ships a CuteDSL "low-latency skinny GEMM" plan table for decode shapes but it is hard-gated to SM103 (`_is_sm103()`), and its plans are keyed on TP=4 shapes. | `nvidia/low_latency_gemm.py` |
| HyperConnection (2 per layer + final mixer) | Triton fused ops (`hc_combine_norm`, `hc_gate_mix`, `grouped_gemma_rmsnorm`) + **replicated** bf16 linears (`disable_tp=True`, `quant_config=None` hard-coded) | `nvidia/hyperconnection.py`, `ops/hc.py` |
| PLE n-gram embedding (layer index 1) | FP8 table (47.7 GiB total, vocab-sharded across TP → ~24 GiB per GPU), gathered by `index_select`; dilated short-conv via `PleShortConvAttentionBackend` | `files/ple_layer_patched.py` |
| Cross-node collectives | **PyNCCL only** over RoCE v2 (custom all-reduce needs P2P/same node; NCCL symm-mem needs world ≥ 4; torch symm-mem needs same node). | `cuda_communicator.py`, `all_reduce_utils.py` |
| Speculative decoding | `Qwen3_8FlashNextMTPProposer(EagleProposer)`: 1 draft forward on the padded batch + 2 more single-token forwards per step (3 total for K=3), each ending in a **full 248 320-vocab lm_head GEMM**. `index_share_for_mtp_iteration` is **off** (not in the HF config, defaults False), so the QSA indexer + top-k is re-run in every draft step. | `spec_decode/llm_base_proposer.py:514-760`, `config/speculative.py:1180` |
| CUDA graphs | `FULL_DECODE_ONLY`, compile mode 0 (no inductor). Capture sizes 1…64 tokens (= `max_num_seqs 8 × (1+3) × 2`). Drafter passes also dispatch through the cudagraph dispatcher. | `config/vllm.py:1916-1960` |
| Scheduler | Async scheduling auto-enables for `mtp` + `mp` executor; mamba cache mode is forced to `align` by the model (`"all"` raises). Prefix caching on, KV block = 1600 tokens (padded to the GDN page). | `models/config.py:604`, `nvidia/model.py` |

Two facts that matter for everything below:

1. **Full attention is only 12 of 48 layers and is sparse (QSA, budget 2048 tokens).** At short context the
   attention kernels are cheap; the decode step is dominated by *weight* traffic, not KV traffic. The README's
   line "decode speed is dominated by KV bandwidth" is not right for this model at batch 1.
2. **Everything that is not a routed expert is bf16 and much of it is replicated across the two GPUs.** See §2.

---

## 2. Where a decode step's time goes (bandwidth model, batch 1, MTP3)

Checkpoint composition (parsed from safetensors headers, whole model):

| Category | Size | dtype |
|---|---|---|
| Routed experts (48 layers × 512) | 63.28 GiB | NVFP4 packed (56.25) + FP8 block scales (7.03) |
| PLE n-gram table | 47.75 GiB | FP8 (vocab-sharded, ~24 GiB per GPU) |
| MTP layer routed experts | **4.69 GiB** | **bf16 (unquantized)** |
| GDN projections (36 layers) | 3.89 GiB | bf16 |
| HyperConnection (97 modules) | 1.19 GiB | bf16, replicated |
| lm_head | 1.18 GiB | bf16 |
| embed_tokens | 1.18 GiB | bf16 (gather only) |
| Full-attention q/k/v/o (12 layers) | 1.11 GiB | bf16 |
| Shared expert (48) | 0.44 GiB | bf16 |
| MoE router gate (48) | 0.12 GiB | bf16, replicated |
| QSA indexer proj | 0.04 GiB | bf16, replicated |
| Vision tower | 0.84 GiB | bf16 (not touched in text decode) |

Bytes each GPU must stream from LPDDR5x per decode step (TP2/EP2, one request, verify pass of 4 tokens plus
three 1-token draft passes). Sharded tensors are halved; replicated ones are not.

| Read per step per GPU | GiB | Share | Note |
|---|---|---|---|
| lm_head: 1 verify + 3 draft passes | **2.36** | 26% | 0.59 GiB × 4. The drafter samples from full logits each step. |
| Routed experts (verify, ~32 unique experts/layer × 48 × 2.77 MB, EP-split) | ~2.1 | 23% | 1.7–2.7 depending on routing overlap |
| GDN projections (TP-sharded) | 1.95 | 21% | |
| HyperConnection (replicated, bf16) | 1.19 | 13% | not sharded, not quantized |
| Full-attention projections + indexer | 0.60 | 7% | |
| MTP dense ×3 + MTP bf16 experts ×3 | ~0.5 | 5% | |
| Router + shared expert + PLE proj | 0.40 | 4% | |
| GDN recurrent state (36 × 1.5 MiB fp32, R+W) | 0.1 | 1% | |
| **Total** | **≈ 9.2 GiB (9.9 GB)** | | |

At 273 GB/s theoretical that is **36 ms/step**; at a realistic 235–245 GB/s achieved it is **40–42 ms/step**
(24–25 steps/s). With MTP3 accepting ~2.2 tokens/step that is **53–56 tok/s**, which is exactly the README's
measured 55.8 tok/s. Batch-1 decode on this box is therefore **at the memory-bandwidth roofline already**;
there is very little "kernel inefficiency" left to recover in the verify pass. The levers that remain are:

1. **Fewer bytes per step** (quantize the dense bf16 paths; stop re-reading lm_head three extra times).
2. **More accepted tokens per step** (MTP acceptance).
3. **More sequences per step** (concurrency amortizes all weight reads).
4. **Shave the non-bandwidth floor**: ~106 cross-node NCCL collectives per step (36 GDN + 12 attention + 48 MoE
   + logits gather, plus 3 per draft pass), each 5–20 KB. At 20–30 µs each on RoCE that is **2–3 ms/step
   (~6–8%)** and it does not shrink with batch size.

---

## 3. Ranked recommendations for decode tok/s

Effort: S = flag/env change, M = patch file bind-mounted like the PLE patch, L = new checkpoint or kernel work.
Gain estimates are analytical from §2 and must be confirmed with §6.

### 3.1 [L, gain ≈ +40–60% at batch 1] Quantize the dense bf16 paths to FP8

The dense bf16 weights (GDN, attention, HC, lm_head, router, shared expert) are ~6.9 GiB of the 9.2 GiB read
per step. At FP8 they are ~3.45 GiB, dropping the step to ≈ 5.7 GiB → ~25 ms → **~85–90 tok/s** at the same
acceptance length. Options, in order of practicality:

- **Correction (2026-09-02, after parsing the official FP8 checkpoint's headers):** Qwen's
  `Qwen3.8-Flash-Next-FP8` does **not** quantize the dense layers. Its `modules_to_not_convert` lists every
  `linear_attn.*`, `hyper_connection*`, `lm_head`, and the tensor scan confirms GDN 3.89 GiB / attention 1.11 /
  HC 1.19 / lm_head 1.18 are all bf16 there; only the routed experts (112.5 GiB) and the MTP experts are FP8.
  So the official FP8 checkpoint reads *more* bytes per step than NVFP4 (FP8 experts) with no dense saving,
  and there is no vendor-validated FP8 dense to borrow. The dense layers had to be quantized here.
- **Done: hybrid checkpoint `MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense` + loader overlay** — see §8.
  Per-output-channel FP8 E4M3 with dynamic per-token activation scales was chosen over block-128 because
  (a) it needs no calibration data, (b) HyperConnection's rank-320 projections and the TP-sharded shared
  expert (320 rows) are not 128-divisible, and (c) vLLM's `ParallelLMHead` loader shards block scales
  incorrectly (it narrows by vocab rows, not scale blocks) while per-channel scales shard like the weight.
- **lm_head is included** (it is 26% of the step because the drafter re-reads it three times). If MTP
  acceptance drops measurably, rebuild with lm_head left bf16 (remove `lm_head.weight` from
  `QUANT_PATTERNS` in the converter) and compare.

### 3.2 [S, gain 5–15%] Tune MTP: draft length and index sharing

- The drafter costs ~2.4 GiB/step (26%) mostly through **three lm_head reads**. Test
  `MTP_NUM_SPECULATIVE_TOKENS=2` and `4` against `3`: K=2 saves ~0.7 GiB/step (~8% step time) and loses the
  third position's acceptance. vLLM already warns that re-running the *same* MTP layer for K>1 lowers acceptance
  (`config/speculative.py:1013`); the per-position metric `vllm:spec_decode_num_accepted_tokens_per_pos` tells
  you exactly where the cut-off is (if position 3 accepts < ~35%, K=2 is faster).
- Enable **`index_share_for_mtp_iteration`** via `EXTRA_VLLM_ARGS` /
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"index_share_for_mtp_iteration":true}'`.
  Draft steps 2–3 then skip the QSA scoring + `persistent_topk` + expand kernels (per full-attention layer, per
  step) and reuse step-1 indices. Kernel-time saving only (not bandwidth), so a few % at most; check accuracy on
  long contexts (the indices are one position stale).
- Note the benchmark numbers are **greedy**. The served `generation_config` is `temperature 1.0, top_p 0.95,
  top_k 20`; rejection sampling at T=1.0 accepts fewer draft tokens, so real chat throughput will be lower than
  55 tok/s. Consider `--override-generation-config '{"temperature":0.7}'` if the product tolerates it, and
  benchmark at the temperature you actually serve.

### 3.3 [S, gain up to ~2× aggregate] Run concurrency, not just batch 1

Weight bytes per step are nearly flat in batch size (only unique-expert count grows), so 2–4 concurrent
streams should give close to 2–3.5× aggregate tok/s until the per-step time starts to grow. The SGLang recipe on
this same cluster measured 64 → 117 tok/s aggregate at ×2. The current config supports it (cudagraphs to 64
tokens = 16 sequences × 4). Sweep `MAX_NUM_SEQS ∈ {1,2,4,8}` with the §6 harness and pick the knee. If the
target is agentic/short context, set `MAX_MODEL_LEN=262144` (see 3.4) so 8 sequences are actually resident.

### 3.4 [S, gain a few %; also memory] Do not pay for 1M context when serving short prompts

`MAX_MODEL_LEN` sizes several per-step structures regardless of actual sequence length:

- The QSA index scorer allocates a `[rows, page_table_width × page_size]` fp32 logits buffer per QSA layer per
  forward: at 1M tokens and compress ratio 4 that is 250 K columns → **4 MB per verify pass per layer** (13 layers,
  ×4 forwards per step) and `persistent_topk` is launched over `columns` = 250 K. Programs past `visible` early
  exit, but the grid is still `cdiv(250K, 64)` = 3.9 K programs per row per layer.
- Block tables are 625 entries per request; `_mtp_hidden_buffer`, `topk_indices_buffer` etc. scale with
  `max_num_batched_tokens`, not context, so they are unaffected.

For the "throughput / agentic" profile in the README, `MAX_MODEL_LEN=262144` + `YARN_ENABLE=false` is the
right default and should be measured against 1M at identical prompts; expect a small but free win.

### 3.5 [S, gain 3–8%] NCCL and fabric settings

The ~106 small cross-node collectives per step are latency-bound. Things to test (env is passed via
`EXTRA_DOCKER_ARGS="-e X=Y -e ..."`, both nodes):

| Setting | Why |
|---|---|
| `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_NET=IB`, `NCCL_NET_PLUGIN=none` | Pin RoCE v2 and the IB transport explicitly (the SGLang recipe on this cluster used these; the vLLM launch does not). |
| `NCCL_PROTO=LL` or `LL128`, `NCCL_ALGO=Ring` | For 5–20 KB messages the low-latency protocol is what matters; force it and measure with `nccl-tests` first. |
| `NCCL_IB_QPS_PER_CONNECTION=2..4`, `NCCL_IB_SPLIT_DATA_ON_QPS=0` | Helps bandwidth-bound prefill all-reduces, neutral for decode. |
| `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_IGNORE_CPU_AFFINITY=1` | Used by both other GB10 recipes on this box (dspark image and the SGLang recipe); harmless, avoids cuMem/NVLS probing on an iGPU. |
| **Second CX7 link** | `enP2p1s0f1np1` (10.0.122.1) ↔ worker `enP2p1s0f0np0` (10.0.122.2) is up at 200 GbE, MTU 9000, and pings. `NCCL_IB_HCA="=rocep1s0f1,roceP2p1s0f1"` on head, `"=rocep1s0f0,roceP2p1s0f0"` on worker, plus `NCCL_CROSS_NIC=1` lets NCCL stripe rings over both NICs. Doubles cross-node bandwidth (prefill, §5) and gives no decode-latency change. Never list a port cabled elsewhere (README gotcha still applies). |
| GPUDirect RDMA | No `nvidia_peermem` module is loaded; `DmaRemapPeerMmio=1`. Whether NCCL is staging through host buffers on GB10's unified memory is unknown from here. `nccl-tests` with `NCCL_DEBUG=INFO` (look for `GDRDMA`/`dmabuf` in the transport line) answers it in one run; if it is not using dmabuf, try `NCCL_DMABUF_ENABLE=1` and `NCCL_NET_GDR_LEVEL=SYS`. |

Control-plane traffic (`VLLM_HOST_IP=10.0.0.x`, the scheduler→worker ZMQ broadcast each step) already routes
over the CX7 link (`ip route get 10.0.0.2 → via 10.0.22.2 dev enp1s0f1np1`, ~0.15–0.4 ms RTT). Good; keep it
that way.

### 3.6 [M] Kernel-level modifications worth trying (bind-mount patches, same mechanism as the PLE patch)

1. **Enable the CuteDSL skinny GEMM on sm_121** (`nvidia/low_latency_gemm.py`): change `_is_sm103()` to also
   accept `(12, 1)`, and add TP=2 plan entries. TP=2 shapes on this deployment: GDN qkvz `(8192, 2560)`, GDN
   out `(2560, 3072)`, GDN ba `(48, 2560)`, QSA qkv+gate `(6656, 2560)`, QSA out `(2560, 3072)`, indexer
   `(640, 2560)`, shared-expert gate/up `(640, 2560)`, lm_head `(124160, 2560)`, HC down+inject `(336, 10240)`,
   HC up `(10240, 320)`. cuBLAS on 48 SMs with M=1..4 often leaves 20–40% of bandwidth on the table; a tuned
   skinny kernel recovers it. Needs `nvidia-cutlass-dsl` (present, 4.6.2) and a warmup sweep of
   `SkinnyGemmConfig` per shape. Expected: a few % of step time (dense GEMMs are ~50% of bytes).
2. **Re-tune the QSA Triton profiles for GB10.** `qsa_sparse_paged_attention` picks `block_n/target_splits/
   num_warps` from a table "tuned on GB300" (160 SMs); on 48 SMs the decode profile `(16, 64 splits, 4 warps)`
   over-splits and pays a merge kernel. Same for `tiles_per_program` in `qsa_mqa_paged`. Cheap to sweep with
   a standalone micro-benchmark against the extracted kernels. Small absolute gain at short context; larger at
   64K+ contexts where attention time is visible.
3. **Try the SM12x-native fused MoE**: `ENABLE_EXPERT_PARALLEL=false` (see §4.1 for the script bug) and
   `--moe-backend flashinfer_b12x`. It is a single CuteDSL kernel that fuses dispatch, both GEMMs, SwiGLU and
   the top-k reduction with in-kernel bf16→FP4 activation quant, written for DGX Spark. Auto-selection is
   disabled "until the upstream CUTLASS SM121 MMA op guard is resolved", so it may fail to build on 12.1; the
   anemll dspark image on the worker runs with `VLLM_USE_B12X_MOE=1`, so a working variant exists. Per-GPU
   expert bytes are the same under TP-sharded experts (intermediate 640 → 320 per GPU), so this is a
   kernel-efficiency experiment, not a bandwidth one.
4. **FlashInfer GDN prefill on sm_121** (`qwen_gdn_linear_attn.py:_resolve_gdn_prefill_backend`): allow
   `is_device_capability_family(120)` with `head_k_dim == 128` and CUDA ≥ 13 (both true here). FlashInfer
   0.6.17 has `cp_delta_rule_dsl_sm120`. This is a prefill/TTFT item (§5), not decode.
5. **torch.compile** (`--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'`). All custom
   ops in the model register fake impls and the model classes carry `@support_torch_compile`, so it should
   trace. Gains at batch 1 are modest (fused elementwise/norm ops, fewer launches) but real at concurrency;
   cost is +2–5 min cold start and compile-cache writes to `~/.cache/vllm`. Keep mode 0 as the fallback.
6. **`--performance-mode interactivity`**: captures every size 1…32 instead of `[1,2,4,8,16,…]`, removing
   padding for odd batch sizes at MTP (5, 6, 7 tokens…). Free to try.

### 3.7 Things that will NOT help decode here (so nobody spends time on them)

- **KV-cache dtype / pool size**: batch-1 decode at short context reads ~0.1 GiB of state per step; the QSA
  backend only accepts bf16 anyway (`supported_kv_cache_dtypes = ["auto","bfloat16"]`).
- **PLE CPU offload**: on Spark the "CPU RAM" is the same LPDDR5x pool; it frees nothing and adds a
  cross-process semaphore per step.
- **`--mamba-ssm-cache-dtype bfloat16`**: saves ~50 MiB/step (<1%) and risks GDN state precision.
- **DeepEP low-latency all2all**: with DP=1 there is no token dispatch to accelerate; and NVSHMEM/IBGDA on an
  integrated GPU is unproven.
- **GPU clock locking**: memory-bound; the SM clock is not the limiter (`nvidia-smi -lgc` also needs root).

---

## 4. Stability, correctness and script issues found

### 4.1 `start.sh` builds `VLLM_ARGS` and then never uses it (real bug, silent)

Lines ~330–400 assemble `VLLM_ARGS` (including `EXTRA_VLLM_ARGS`, the `ENABLE_EXPERT_PARALLEL` guard and the
`MTP_NUM_SPECULATIVE_TOKENS -gt 0` guard), but both launch heredocs (`WORKER_SCRIPT`, `HEAD_SCRIPT`)
hard-code the flag list. Consequences today:

- `EXTRA_VLLM_ARGS` is **silently dropped**. The README's "`--kv-cache-memory` via `EXTRA_VLLM_ARGS`" recipe
  and every experiment in §3 that relies on it do nothing.
- `ENABLE_EXPERT_PARALLEL=false` is ignored (`--enable-expert-parallel --all2all-backend …` always emitted).
- `MTP_NUM_SPECULATIVE_TOKENS=0` emits `"num_speculative_tokens":0`, which vLLM rejects (`Field(gt=0)`), so
  "0 = disable" crashes at startup instead of disabling.

Fix: render `"${VLLM_ARGS[@]}"` into the heredocs (one shared array, per-node `--node-rank/--headless/--host`
appended), and keep `EXTRA_VLLM_ARGS` last. Also the head heredoc mounts `$HOME/.cache/huggingface` while
step 2 resolves `$HF_CACHE_DIR` (`HF_HOME`-aware); they diverge if `HF_HOME` is set.

### 4.2 `MODEL_ID` does not match the checkpoint on disk

`.env` has `MODEL_ID="local-inference-lab/Qwen3.8-Flash-Next-NVFP4"`, but the only NVFP4 snapshot in
`~/.cache/huggingface/hub` is `models--RadixArk--Qwen3.8-Flash-Next-NVFP4` (the FP8 one is
`models--Qwen--Qwen3.8-Flash-Next-FP8`). `./start.sh --no-download` will fail at step 2 ("Could not resolve
local cache path") and `./start.sh` will try to download 126 GiB again. Set `MODEL_ID=RadixArk/…` (or symlink
the hub directory) and re-run `check-weights.sh`.

### 4.3 Cluster is shared right now

Worker `spark2` runs `deepseek-v4-flash-vllm-dspark-1` (healthy, started minutes ago). `start.sh` does
`docker rm -f vllm-fn` but never checks for *other* GPU tenants. Add a preflight on both nodes:
`docker ps --filter ancestor=…` / `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and abort or
warn if the GPU is not idle; otherwise the first symptom is a `CUDA out of memory` during weight load, ten
minutes in.

### 4.4 No restart policy, no watchdog, no hang timeouts

`--restart` is unset on both containers and there is no health loop after launch. Recommended:

- `--restart unless-stopped` is risky for a 2-node TP job (a lone restarted rank hangs in NCCL init). Better: a
  small supervisor script on the head that polls `/health` every 30 s and runs `./stop.sh && ./start.sh --launch`
  on failure, with a cap on restarts.
- Pass `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (dspark image uses 1800), `TORCH_NCCL_DUMP_ON_TIMEOUT=1`,
  `TORCH_NCCL_ENABLE_MONITORING=1` so a stuck rank produces a flight-recorder dump instead of a silent hang.
- Keep `NCCL_DEBUG=WARN`; add `NCCL_DEBUG_SUBSYS=INIT,NET` on the first launch after any fabric change.

### 4.5 Regression test the token-0 "doom loop" on this stack

The sibling SGLang recipe documented a long-thinking decode collapse into token id 0 (`!`) on SM121 that then
poisoned later requests (`../Qwen3.8-Flash-Next-Dual-DGX-Sparks.BACKUP-sglang/logs/qwen38-doom-loop-bug-report.md`,
reproducer `../Qwen3.8-Flash/logs/qwen38_doom_loop_repro.py`). The root cause there was FlashInfer's TRT-LLM
sparse decode kernel; this vLLM image uses a different (Triton) QSA path, so it is *probably* not exposed, but it
has not been shown. Run the reproducer (1600-token forced thinking, T=0.6, then short follow-ups) as part of
every image/kernel change, and watch `finish_reason` + longest `!` run.

### 4.6 Other observations

- The README's "1M × 8 seqs is oversubscribed" note is correct; with `EXTRA_VLLM_ARGS` fixed, set
  `--kv-cache-memory` explicitly instead of tuning `GPU_MEMORY_UTILIZATION`.
- Prefix caching with the hybrid cache only hits at 1600-token block boundaries (`align` mode, block 1600), so
  multi-turn chat reuse is coarse. Not a bug; set expectations.
- `~/.cache/vllm/flashinfer_autotune_cache/0.6.17/…` and `modelinfos/*` are root-owned (container runs as
  root). Harmless for the server, but host-side tooling cannot read them; consider `--user` mapping or a
  `chown` in `stop.sh`.
- First request after launch triggers FlashInfer autotune (README). Send a warm-up request of each shape
  (1, 2, 4, 8 concurrent) from the supervisor before marking the service ready.
- Cold start is ~11 min, 458 s of which is streaming 126 GiB lazily over NFS to the worker. A local worker copy
  (`rsync` once) would cut it to a few minutes and removes the "do not stop the NFS container" hazard; the
  README's rationale (saving 126 GiB on a 4 TB NVMe) is weak.
- `EXTRA_VLLM_ARGS` is word-split (`SC2206` disabled); JSON with spaces will break. Prefer a bash array in `.env`
  or `IFS`-safe parsing once 4.1 is fixed.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` stays set even when YaRN is force-disabled; harmless but confusing.

---

## 5. Prefill / TTFT

- Prefill compute for an 8192-token chunk touches every expert (48 × 512 × 2.77 MB = 68 GB, 34 GB per GPU →
  ~140 ms of weight streaming) plus GDN chunk kernels (Triton FLA), QSA prefill (Triton, `block_n=64`,
  `splits=1`) and 96 all-reduces of `8192 × 2560 × bf16 = 42 MB` each. Over one 200 GbE link that is
  **~150–180 ms of communication per chunk**, a large fraction of chunk time. The second CX7 link (§3.5)
  halves it; it is the single most effective prefill change and needs no code.
- `MAX_NUM_BATCHED_TOKENS=8192` is reasonable; larger chunks do not reduce total all-reduce bytes.
- GDN prefill via FlashInfer's SM120 CP kernel (§3.6 item 4) is the kernel-level opportunity.
- Under concurrency, a long prefill chunk stalls decode for all resident streams (decode step ≈ 40 ms vs chunk
  ≈ 0.5–1 s). Set `--long-prefill-token-threshold 2048` (and optionally `--max-num-partial-prefills 2`) to
  cap per-step prefill work and keep per-stream decode latency stable; measure TTFT vs ITL trade-off.
- Multimodal prompts draft from text only (README); expect lower acceptance on image requests.

---

## 6. Measurement plan (do this before and after every change)

All commands run on the head with the server up. Use identical prompts and `seed`.

**Decode tok/s, batch 1 (greedy and at serving temperature):**

```bash
docker exec vllm-fn vllm bench serve --backend openai-chat --base-url http://localhost:8888 \
  --model qwen3.8-flash-next --dataset-name random --random-input-len 512 --random-output-len 512 \
  --num-prompts 8 --max-concurrency 1 --temperature 0 --seed 0 --percentile-metrics ttft,tpot,itl
```

Repeat with `--max-concurrency 2 4 8` for the concurrency knee, and with `--temperature 1.0 --top-p 0.95
--top-k 20` for the served sampling profile.

**Speculative acceptance (the number that decides MTP K):**

```bash
curl -s localhost:8888/metrics | grep -E 'spec_decode_num_(accepted|draft|drafts)|accepted_tokens_per_pos|preemption|kv_cache_usage'
```

**NCCL small-message latency and link bandwidth (ground truth for §3.5):**

```bash
docker run --rm --gpus all --network host --ipc host --device /dev/infiniband \
  -e NCCL_IB_HCA==rocep1s0f1 -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_DEBUG=INFO \
  vllm/vllm-openai:qwen38-flash-next bash -c 'git clone -q https://github.com/NVIDIA/nccl-tests && cd nccl-tests && make -j MPI=0 >/dev/null && ./build/all_reduce_perf -b 4K -e 64M -f 2 -g 1'
```

(run the matching command on the worker with `-e NCCL_IB_HCA==rocep1s0f0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0`,
both with `NCCL_COMM_ID=10.0.22.1:29500` and rank env for 2 processes; look for `GDRDMA`/`dmabuf` in the
transport lines and the 8 KB–32 KB latency column.)

**Per-step kernel breakdown (confirms the roofline model in §2):**

```bash
docker exec vllm-fn nsys profile -c cudaProfilerApi -o /root/.cache/vllm/decode --stats=true \
  python3 -c "import vllm;..."   # or attach: nsys profile --capture-range=cudaProfilerApi -p <worker pid>
```

Simpler: `VLLM_TORCH_PROFILER_DIR=/root/.cache/vllm/prof` in `EXTRA_DOCKER_ARGS`, then
`curl -X POST localhost:8888/start_profile` … `stop_profile`; open the trace and sum GEMM vs MoE vs NCCL time
for one step. If the GEMM + MoE kernels together are ≥ 30 ms of a ~40 ms step, §2 is confirmed and §3.1 is
the priority; if NCCL waits dominate, §3.5 is.

**A/B matrix (each row is one launch; keep everything else fixed):**

| # | Change | Expect |
|---|---|---|
| 1 | baseline (fix 4.1/4.2 first) | 55 tok/s @1 |
| 2 | `MAX_MODEL_LEN=262144`, `YARN_ENABLE=false` | +2–5% |
| 3 | `MTP_NUM_SPECULATIVE_TOKENS=2` / `4` | ± depending on per-pos acceptance |
| 4 | `index_share_for_mtp_iteration:true` | +1–3% |
| 5 | NCCL env set from §3.5 (+ dual NIC) | +3–8% decode, ~2× prefill comm |
| 6 | `--performance-mode interactivity` | 0–2% |
| 7 | compile mode 3 + FULL_AND_PIECEWISE | 0–8%, longer start |
| 8 | `./start-fp8.sh` (official FP8) | measure; roofline says ≤ +15% @1, worse at ×8 |
| 9 | concurrency 2/4/8 | up to ~2–3.5× aggregate |
| 10 | `FP8_DENSE=true` (hybrid checkpoint + overlay, §8) | +40–60% @1 (analytical) |
| 11 | `QSA_PROFILE=gb10` or benchmarked JSON (§8) / skinny GEMM on sm_121 / B12X MoE (§3.6) | each a few % |

---

## 7. Appendix

### 7.1 Per-step collective count (batch 1, MTP3)

| Source | Collectives / step | Payload |
|---|---|---|
| GDN `out_proj` all-reduce | 36 | 4 tok × 2560 × bf16 = 20 KB |
| Attention `o_proj` all-reduce | 12 | 20 KB |
| MoE TP all-reduce (after local experts + shared expert) | 48 | 20 KB |
| Logits all-gather (vocab-parallel lm_head) | 1 | 4 × 124160 × fp32 ≈ 2 MB |
| Draft passes (1 attn AR + 1 MoE AR + 1 logits gather) × 3 | 9 | 5 KB / 0.5 MB |
| **Total** | **≈ 106** | |

### 7.2 Files inside the image that the recommendations touch

```
/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py   (3.6-1)
/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py            (3.6-2)
/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/hyperconnection.py    (3.1 HC quant)
/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/qsa.py                (3.1 qkv quant)
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (3.6-4)
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py        (3.6-3)
```

Bind-mount them exactly like `files/ple_layer_patched.py` (extract with `docker cp`, patch, `-v host:container:ro`
on both nodes) so no image rebuild is needed.

### 7.3 Useful vLLM knobs present in this build

`--moe-backend {auto,flashinfer_cutlass,flashinfer_b12x,cutlass,marlin,emulation}`, `--linear-backend …`,
`--performance-mode {balanced,interactivity,throughput}`, `--async-scheduling/--no-async-scheduling`,
`--kv-cache-memory <bytes>`, `--mamba-cache-mode align`, `--long-prefill-token-threshold`,
`--max-num-partial-prefills`, `--override-generation-config`, `--additional-config '{"gdn_prefill_backend":…}'`,
env `VLLM_GDN_DECODE_KERNEL={cuda,triton}`, `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`,
`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`, `VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_TORCH_PROFILER_DIR`.

### 7.4 Checkpoint qualification (from the RadixArk snapshot)

GSM8K 97.27% (1283/1319), AIME26 pass@1 98.75%, both inside the bf16 reference band; MTP and all
non-routed tensors are source precision; PLE tables are Qwen's FP8 shards with a scalar scale (this is what
the `PLE_QUANT_OVERRIDE=fp8` shim honors).


---

## 8. Implemented on 2026-09-02 (follow-up to §3.1 and §1)

Both GPUs were occupied by another deployment (`deepseek-v4-flash-vllm-dspark-1`, 108 GB per node, host
RAM ~1.5 GB free) for the whole session, so everything below was built and validated **CPU-only**; the
GPU-side checks are listed as owed. Nothing was launched.

### 8.1 FP8-dense hybrid checkpoint (the "biggest lever")

| Item | Where | Status |
|---|---|---|
| Streaming converter (row-chunked, ~150 MB working set, runs in a 1.2 GB cgroup next to a live server) | `files/fp8dense/make_fp8_dense_checkpoint.py` | done |
| One-shot build + verify | `files/fp8dense/build.sh` | done |
| Checkpoint | `~/.cache/huggingface/hub/models--MiaAI-Lab--Qwen3.8-Flash-Next-NVFP4-FP8dense` (snapshot `fp8dense-7b719225`, `refs/main` points at it; huggingface_hub's offline path reads `refs/` without a hex check), 202 expert/PLE shards hard-linked to the source blobs, 4 bf16 shards rewritten (11.8 GB, index total 123 GiB) | **built** |
| Format checks: index/weight_map, 591 FP8 tensors each with a `[out]` fp32 `weight_scale`, hard links point at the source blobs, untouched tensors unchanged | `files/fp8dense/verify_fp8_dense_checkpoint.py` | **passed, 0 problems** |
| Quantization error (dequant vs bf16), all 591 tensors (`compute_quant_stats.py`) | `fp8_dense_quant_stats.json` in the snapshot | rel. RMSE min/median/max = 0.0240 / 0.0262 / 0.0268 (worst: the tiny `block_inject_weight` gates); cosine ≥ 0.9996 on samples |
| vLLM loader overlay: `modelopt.py` (mixed config dispatches `FP8_PER_CHANNEL_PER_TOKEN` / `FP8_PB_WO`, extra prefix spellings), `hyperconnection.py` (accepts `quant_config`), `model.py` (HC + lm_head get the config), `mtp.py` (drafter's shared lm_head + mixer) | `files/overlay/*.py` (+ `.orig`, `.diff`, `apply_patches.py`) | done |
| Quant-method resolution for every layer kind, both entry points (`ForConditionalGeneration` prefixes and `CausalLM`/MTP prefixes), using the real vLLM code with the overlay mounted | `files/fp8dense/test_quant_config_resolution.py` | **30/30 correct** |
| `start.sh` wiring: `FP8_DENSE=true` switches `MODEL_ID`, skips download, mounts the four files on both nodes | `start.sh`, `.env`, `.env.sample` | done |

What the loader will do at runtime (from the code, not yet observed): each dense linear becomes
`ModelOptFp8PcPtLinearMethod` → `init_fp8_linear_kernel(dynamic per-token act, static per-channel weight)`
→ first eligible kernel is `CutlassFP8ScaledMMLinearKernel` (Marlin is skipped on cc ≥ 89, FlashInfer's needs
per-tensor scales, B12X needs static per-tensor). The image ships non-`a` `sm_120` cubins for
`cutlass_scaled_mm_sm120_fp8` (checked with `cuobjdump --list-elf`), which GB10 (cc 12.1) can execute.

**Owed GPU validation (in order):**
1. `./start.sh --launch` with `FP8_DENSE=true`; confirm the log shows `Selected CutlassFP8ScaledMMLinearKernel
   for ModelOptFp8PcPtLinearMethod`, weights load without "unexpected"/"missing" key errors, and
   `Available KV cache memory` grows by ≈3.5 GiB per GPU.
2. Decode benchmark from §6 at batch 1/2/4/8 vs the NVFP4 baseline; expect ≈1.5× at batch 1.
3. Quality: the GSM8K quick eval used for the checkpoint's own qualification (`gsm8k_metrics.json` protocol),
   MTP acceptance per position (§3.2), and the token-0 loop reproducer (§4.5). If acceptance drops, rebuild
   with lm_head in bf16 (one line in `QUANT_PATTERNS`).

### 8.2 QSA kernels ("FlashAttention v2 is not what runs")

- README corrected: the full-attention layers run the model's own Triton QSA kernels; FA2 is nominal.
- `files/qsa_gb10/qsa.py` (bind-mounted overlay of `nvidia/ops/qsa.py`): launch profiles are now
  selectable (`VLLM_QSA_PROFILE=stock|gb10`, `VLLM_QSA_PROFILE_JSON=<file>`) for both the sparse GQA
  split-K kernel (block_n / splits / warps per `base_programs` bucket) and the MQA scorer
  (tiles per program / warps). `stock` is byte-for-byte the upstream behaviour.
- `files/qsa_gb10/bench_qsa_kernels.py`: runs the real kernels with this deployment's decode shapes
  (TP2: 12 q-heads × 1 kv-head × 256, page 1600, top-k width 2051, contexts 4K–128K, rows 1–32), sweeps
  block_n ∈ {16,32,64} × splits ∈ {1..64} × warps ∈ {2,4} and MQA tiles ∈ {1..16} × warps ∈ {1,2,4}, and
  writes the winning table as JSON for `QSA_PROFILE=<json>`.
- The `gb10` table (decode: block_n 32, 16 splits, 4 warps instead of 16/64/4) is a reasoned guess for
  48 SMs (≈1.3 waves instead of ≈5, one merge over 16 partials instead of 64). **It is unbenchmarked**:
  the GPU was busy. Default stays `stock`.

### 8.3 Launcher fixes shipped alongside

- `start.sh` renders one shared `VLLM_ARGS` array into both launch scripts: `EXTRA_VLLM_ARGS` is
  appended last (it was silently dropped before), `ENABLE_EXPERT_PARALLEL=false` and
  `MTP_NUM_SPECULATIVE_TOKENS=0` are honored, YaRN override is emitted once, the head mounts
  `$HF_CACHE_DIR` (HF_HOME-aware), and the rendered head script is saved as `.last_head_launch.sh`.
- GPU preflight on both nodes (`REQUIRE_IDLE_GPU`), overlay mount plumbing (scp to `/tmp/vllm-overlay`
  on the worker, `-v … :ro` on both).
- `.env` `MODEL_ID` fixed to the snapshot that actually exists (`RadixArk/…`).
