# Changelog

All notable changes to this serving recipe.

## [Unreleased] — 2026-08-28

### Added

- **NVFP4 KV cache** for QSA layers (`NVFP4_KV_CACHE=1`, `--kv-cache-dtype nvfp4`).
  Packed FP4 + per-block FP8 scales, no FP8 dequant workspace. QSA gather
  dequantizes on the way out. Measured on 2× GB10 TP2: **2,914,944 tokens /
  11.47 GB** (~3.1× the 925,504-token bf16 pool).
- `evals/nvfp4_kv_eval.py` / `./start.sh kv-eval` — passkey / needle-in-haystack
  reliability check against the live API. Quick suite on this cluster
  (2026-08-27): **RELIABLE through 128k as a single huge prompt** (0/50/100%).
  64k and 128k NIAH were 3/3 at every position. 256k same-turn 0%/50% hit
  the token-0 `!` loop; recency and a two-turn follow-up still retrieved.
- YaRN **1M context** as the script default (`CONTEXT_LENGTH=1048576`,
  factor 4.0 × native 262144). `start.sh` injects the rope override and
  `--max-prefill-tokens 2048`. Opt out with `CONTEXT_LENGTH=262144`.

### Changed

- `NVFP4_KV_CACHE` default is **1**. `0` selects **fp8_e4m3** KV (not bf16).
  The QSA Triton varlen fallback **and** the GQA prefill/chunk-prefill
  kernels (`sparse_gqa_fwd_interface_triton[_ck]`) upcast fp8 K/V to fp32
  before `tl.dot`. Without the GQA patch, the first extend with a prefix
  killed the worker (`Unsupported rhs dtype fp8e4nv`). `KV_CACHE_DTYPE=bf16`
  still forces bf16.
- `MEM_FRACTION_STATIC` default **0.80**. PLE (~26 GB/rank) is inside GB10's
  unified CUDA budget; **0.70 double-counted it** and starved KV/mamba
  ([issue #8](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/8)).
- `MAX_RUNNING_REQUESTS` default **28**. `CUDA_GRAPH_BS` is extended to match
  so decode steps above 16 stay graphed. (This cluster's YaRN `.env` at
  `MAMBA_FULL_MEMORY_RATIO=0.3` still caps at **14** mamba slots.)
- `CHUNKED_PREFILL_SIZE` default **1024**; clamped to 1024 when context > 262144
  (chunk 4096 at 300k history froze GB10).
- `--allow-auto-truncate` is **opt-in** (`ALLOW_AUTO_TRUNCATE=1`). Over-length
  prompts return 400 instead of silent trim.
- After boot, `start.sh` **refuses** if the KV pool cannot hold one advertised
  `--context-length` request (`ALLOW_SHORT_KV_POOL=1` to override).

### Fixed

- **SM121 QSA decode token-id-0** ([sglang#36537](https://github.com/sgl-project/sglang/issues/36537),
  [sglang#36806](https://github.com/sgl-project/sglang/pull/36806),
  [sglang#36845](https://github.com/sgl-project/sglang/pull/36845)). FlashInfer
  TRT-LLM sparse decode is numerically correct on exact SM120 but silently
  emits token id 0 (`!`) at long context on SM121/GB10. FA4 CuTe varlen does
  not compile for QSA's packed one-query shape on GB10. The derivative image
  now (1) forces `_resolve_trtllm_sparse_decode` to `None` on SM121 even if a
  newer base image re-enables it, and (2) routes
  `_resolve_flash_attn_varlen_func` to sglang#36845's Triton packed-varlen
  kernel (`sm121_varlen.py`, replaces `qsa_fa_fallback.py`). DSpark extra:
  fp8 K/V is still allowed (upcast in-kernel) so `NVFP4_KV_CACHE=0` works.
  Upstream validated exact NIAH at 120k/190k/210k on one GB10 with zero
  token id 0; rebuild the image (`KERNEL_PATCH=1`) before treating 256k as
  trusted on this cluster.
- Keyed-server readiness hang ([PR #7](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/7)): derive the
  effective `--api-key` (argparse last-wins, including `EXTRA_ARGS`) so
  `wait_ready` / `status` / `smoke` send `Authorization` instead of looping
  on 401. `--help` now prints the full header; keyed curl examples use a
  placeholder, never the secret.
- 2-node TP=2 hang at `shm_broadcast.wait_until_ready` ([issue #3](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/3) /
  [PR #9](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/pull/9)): pin
  `SGLANG_HOST_IP` to the ConnectX-7 IPs so the cross-node ZeroMQ MessageQueue
  does not advertise Wi-Fi/LAN. NCCL/Gloo/TP were already on CX7; this was the
  one subsystem still auto-detecting.
- NVFP4 decode **CUDA-graph capture**: hybrid pool defaulted `k_scale=1.0`
  (Python float), so `quantize()` did `torch.tensor(..., device=cuda)` during
  capture. The nvfp4 write path now uses on-device `k_scales_gpu[layer]`.
- `.env` is **first-assignment wins**. A leftover `NVFP4_KV_CACHE=0` above
  `=1` kept bf16; documented.

### Notes

- This cluster's live boot (YaRN, `MEM_FRACTION_STATIC=0.82`, NVFP4 KV):
  pool **2,914,944** tokens vs 900k then 1M advertised context; graphs
  captured `bs=[1…14]`; `allow_auto_truncate=False`.
- Native-262k 0.70 vs 0.80 A/B was not re-run here; the fail-closed pool
  check is what stops the 137k-token silent-truncate case from shipping.
- sglang#36556 (TRT-LLM sparse decode on all SM12x) remains **not** taken.
  sglang#36806 gates that path to SM100 + exact SM120; this recipe additionally
  hard-excludes SM121. Decode on GB10 uses the #36845 Triton packed-varlen
  fallback.
