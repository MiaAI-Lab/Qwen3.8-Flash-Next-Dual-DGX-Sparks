# Changelog

All notable changes to this serving recipe.

## [Unreleased] — 2026-08-27

### Added

- **NVFP4 KV cache** for QSA layers (`NVFP4_KV_CACHE=1`, `--kv-cache-dtype nvfp4`).
  Packed FP4 + per-block FP8 scales, no FP8 dequant workspace. QSA gather
  dequantizes on the way out. Measured on 2× GB10 TP2: **2,914,944 tokens /
  11.47 GB** (~3.1× the 925,504-token bf16 pool).
- `evals/nvfp4_kv_eval.py` / `./start.sh kv-eval` — passkey / needle-in-haystack
  reliability check against the live API. Quick suite on this cluster
  (2026-08-27T07:54Z): **11/11 PASS**, verdict **RELIABLE** through 16k
  (0/50/100%) plus a 2-turn radix follow-up (`cache_hit_rate` 0 → 0.970).
- YaRN **1M context** as the script default (`CONTEXT_LENGTH=1048576`,
  factor 4.0 × native 262144). `start.sh` injects the rope override and
  `--max-prefill-tokens 2048`. Opt out with `CONTEXT_LENGTH=262144`.

### Changed

- `NVFP4_KV_CACHE` default is **1** (opt out with `0` for bf16 KV). `fp8_e4m3`
  was tried as the off-path and **does not boot** on QSA (Triton fallback:
  q=bf16 vs k/v=fp8).
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
- sglang#36556 (TRTLLM sparse decode on SM12x) was **not** taken; the Triton
  varlen fallback stays.
