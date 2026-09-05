# Handoff — Qwen3.8-Flash-Next on DGX Spark

## 0. UPDATE 2026-09-04: single-Spark TP1 WORKS

`./start-tp1.sh` serves the local-inference-lab NVFP4 checkpoint on ONE Spark,
port **8888** (8890 while comfy-h3 was still enabled), 262,144 native context, MTP 3, without touching sysctls or
stopping any service (no sudo was available). Measured:

| | |
|---|---|
| KV cache | 685,825 tokens (MTP on, GMU 0.813, budget 98.9 GiB, `KV_TARGET_GIB=20 HOST_SLACK_GIB=5`); 418,013 at the default 12 GiB target |
| Decode, prose / code (temp 0, MTP 3) | 24.9 / 35.1 tok/s |
| Decode, no MTP | 22.0 tok/s |
| Needle at 90% depth of 128,947 / 201,447-token prompts | both correct; 72.5 s / 116.2 s total (~1,750 tok/s) |
| Host MemAvailable, minimum seen | 13.1 GiB at the 20 GiB KV target (MemFree ~0.8 GiB, the rest is page cache holding the mmap'd PLE table); 19.4 GiB at the 12 GiB target. Do not go below ~10 GiB: the cache is what keeps PLE lookups off the disk. |
| Startup to /health | ~10-12 min (see "known slow" below) |

Everything in §4-§7 below describing TP1 as "not working" is superseded.
Three independent bugs had to be fixed; §1-§3 and §6 remain valid background.

### What was wrong

1. **GB10 has no CUDA stream memory ops** (`CAN_USE_STREAM_MEM_OPS = 0`,
   measured). vLLM's PLE offload semaphore is `cuStreamWaitValue32` /
   `cuStreamWriteValue32`. On GB10 the wait returns success, the *next* kernel
   launch on that stream blocks the host thread, and captured graphs do not
   wait at all. That is the "spins forever after graph capture" deadlock.
   `files/patch_ple_offload.py` replaces it with a host-side handshake: the
   GPU worker sends the request and blocks until the CPU worker has
   synchronised its H2D copy and written a sequence number into shared memory.
   All stream-memop calls are now no-ops.
2. **Offload rows carried only the 4-bit codes** (80 B/head) while the GPU
   dequant expects codes + FP8 scales (90 B/head). Fixed in `patch_ple_layer.py`.
3. **The GPU-side placeholder never learned it was NVFP4.** Its constructor is
   skipped under offload, so `load_weights` never captured the quant method or
   global scale; the IPC buffer was bf16 instead of uint8 and dequant was
   bypassed → fluent gibberish, no error. Fixed by handing the placeholder its
   quant method from config, tolerating multi-call `load_weights`, and slicing
   the 2560-wide buffer to the 1440 valid bytes.

### How it fits without kernel tunables

- The 26.8 GiB PLE table is **memory-mapped** from a packed file
  (`~/.cache/vllm/ple_cache/…packed_u8`, built once by
  `files/build_ple_packed_table.py` in ~40 s, verified byte-for-byte against
  the shards). File-backed pages are evictable page cache, so the
  non-evictable footprint is ~77 GiB + KV, not ~104 GiB + KV.
- GPU budget is derived from (checkpoint − PLE) + 5.6 GiB overhead + KV target.
- Container cgroup cap (default budget + 10 GiB). **Correction to §1:** GPU
  parameter allocations are NOT charged to the container cgroup on GB10
  (72 GiB of weights showed up as host "used" with the container at 10 GiB).
  The cap bounds host-side memory only; vLLM's GMU bounds the GPU side.
- `files/memwatch.sh` kills the container if host MemAvailable < 6 GiB and
  logs a timeline to `logs/memwatch-<container>.log`.

### comfy-h3 — the real crash trigger

`comfy-h3.service` is a bash loop (runs as user mia) polling
`http://127.0.0.1:8888/v1/models`; as soon as anything answers there it
launches ComfyUI onto the GPU. It was disabled on 2026-09-04
(`systemctl disable --now comfy-h3.service`); the launcher now defaults to
8888 again and still refuses 8888 if the service is ever re-enabled.

### Known slow / next

- The GPU-worker placeholder still materialises every PLE shard tensor while
  draining its weight iterator (26 GiB read for nothing): ~7 min of the
  startup. Skipping shard names before `get_tensor` would cut startup to ~3 min.
- vLLM reported ~30 GiB of GPU headroom beyond the budget; `KV_TARGET_GIB`
  can go higher if host MemAvailable stays > ~15 GiB.
- FP8 KV cache is NOT supported by this model's QSA backend (`supported_kv_cache_dtypes = ["auto", "bfloat16"]` in qsa.py); bf16 KV is the only option.
- `bench/longctx.py` / `bench/decodebench.py` not yet re-run against TP1.

---


Session of 2026-09-03/04. Covers (A) a weights A/B on the working 2-node
deployment, and (B) an unfinished attempt to run the model on **one** Spark.

**Read §6 before launching anything. This session hard-crashed the host three
times.**

---

## 1. Hardware facts (measured, not from spec sheets)

| | |
|---|---|
| GPU | NVIDIA GB10, 1 per node |
| Memory | **121.69 GiB unified** (LPDDR5X) — CPU and GPU share one physical pool |
| Swap | 16 GiB swap file (`/swap.img`) |
| Nodes | `10.0.0.1` (head, spark1) + `10.0.0.2` (worker), ConnectX link |
| OS floor | ~6 GiB on a clean boot |

Three non-obvious properties, each of which cost us a crash or a wrong
conclusion:

1. **`MemFree` ≠ CUDA-visible memory.** CUDA sees `MemFree` minus the kernel
   reserve (`vm.min_free_kbytes` + watermarks). With a 4 GiB floor we measured
   `MemFree` 112 GiB but CUDA-visible only **100.68 GiB**.
2. **UVM GPU allocations are charged to the container's memory cgroup** as
   anonymous RSS. Confirmed by an OOM kill:
   `UVM GPU1 BH invoked oom-killer ... constraint=CONSTRAINT_MEMCG ...
   Killed process (VLLM::EngineCor) anon-rss:14712148kB`.
   So `docker --memory` *does* bound GPU memory here. Setting it too low
   strangles the engine mid-load.
3. **Exhausting the pool hangs the kernel — it does not raise an OOM.** The
   failure happens in NVIDIA driver context; no journal entries survive. No
   userspace guard is fast enough to catch it (see §6).

---

## 2. The two checkpoints

Both are "NVFP4" but they are **not** the same quantization.

| | `local-inference-lab/…NVFP4` | `RadixArk/…NVFP4` |
|---|---|---|
| On disk | 99G, 37 shards | 126G, 206 shards |
| Tensor bytes | 98.66 GiB | 125.9 GiB |
| Attention | **MXFP8** (8-bit) | **BF16** (unquantized) |
| PLE table | **NVFP4** 4-bit (25.6 GB U8) | **FP8** 8-bit (51.2 GB) |
| MoE experts | NVFP4 | NVFP4 (unchanged) |
| BF16 share | 4.0% | 11.8% |
| Published evals | none (README: "Work in progress") | GSM8K 97.27%, AIME26 pass@1 98.75% (n=8) |

The 27 GB difference is **entirely precision** — every changed component is
exactly 2× the bytes. RadixArk keeps the two most precision-sensitive parts
(attention, PLE) at higher precision. On priors it is the higher-fidelity
checkpoint; **this was not measured**, and the needle test cannot separate them
(both score 6/6).

`local-inference-lab` component breakdown (needed for any fitting exercise):

```
MoE experts        63.51 GiB   64.5%
PLE table          26.88 GiB   27.3%
attention           2.60 GiB
embed/lm_head       2.37 GiB
MTP draft           1.49 GiB
other + visual      1.69 GiB
```

---

## 3. Benchmark results (2-node TP2, the working config)

Config: TP2 + EP + MTP3, YaRN factor 4.0, `MAX_MODEL_LEN=1000000`,
`GMU=0.835`, `max-num-batched-tokens=8192`, port 8888.

KV cache: old **44.24 GiB / 3,246,130 tok**; new **30.37 GiB / 2,212,074 tok**.
Derived **KV cost ≈ 29,482 bytes/token (28.8 KiB)** — used by every fitting
calculation in this doc.

### Prefill — the two checkpoints are identical

| Context | old tok/s | new tok/s |
|---:|---:|---:|
| 32k | 2,808 | 2,754 |
| 64k | 2,654 | 2,550 |
| 128k | 2,432 | 2,432 |
| 256k | 2,171 | 2,152 |
| 400k | 1,955 | 1,964 |
| **600k** | **1,711** | **1,751** |

600k TTFT 350.6 s → 342.7 s. **>256k context verified working**: 6/6 needle
tests passed, including a needle at 95% depth in a 600k prompt (the model
quoted the correct entry numbers 001198 / 011998 / 022796, i.e. genuine
retrieval, not guessing).

Scaling is near-linear, not quadratic — 19× context costs only 1.64× in
per-token rate, consistent with this architecture's sparse/linear attention.

### Decode — RadixArk is consistently ~13–16% slower

Temp 0.0 @ 1k context (avg of 2 runs):

| Content | old | new | Δ |
|---|---:|---:|---:|
| prose | 42.9 | 37.0 | −13.8% |
| code | 51.5 | 43.4 | −15.6% |
| copy-from-context | 70.0 | 61.3 | −12.4% |

@ 600k: prose 39.9 → 34.7, code 50.0 → 42.4, copy 69.3 → 54.1.
Temp 0.8 @ 1k: prose 41.2 → 34.8, code 43.7 → 37.8.

Explanation: decode is memory-bandwidth bound, RadixArk moves more weight
bytes per step. Prefill at long context is compute bound, hence unaffected.

**Decode caveats — do not quote a single number:**
- Decode is ~flat across context (1k → 600k costs 2–8%). It varies far more by
  *content*: ~40 tok/s genuine prose vs ~70 tok/s copying from context, because
  MTP acceptance swings 62.8–92.6% (accepted length 2.88–3.78 of 4).
- The `entropy` task at temp 0 degenerates into repetition, which MTP predicts
  easily — its temp-0 numbers are junk. Read temp 0.8 (40.3 → 34.7).

### Other measurements

- Aggregate throughput vs concurrency (RadixArk): 1 → 46.1, 4 → 110.3,
  8 → **191.6 tok/s** (`max-num-seqs=8`).
- Prefix cache on a repeated 600k prompt: **336 s cold → 2.87–8.82 s warm**
  (~40–110×). Big win for multi-turn at long context.

Scripts: `bench/longctx.py`, `bench/decodebench.py` (restored into the repo;
they previously lived in `/tmp` and were lost to a reboot). Both salt prompts
against prefix caching. Note this vLLM build streams reasoning in
`delta["reasoning"]`, **not** `reasoning_content`.

---

## 4. Single-Spark (TP1): where it actually stands

**Not working. Do not treat `start-tp1.sh` as ready.**

What *is* proven:

- The model **loads and allocates KV on one Spark**. Best result:
  `Available KV cache memory: 3.05 GiB`, `GPU KV cache size: 105,371 tokens`
  at `GMU=0.872` with PLE offload — full model resident.
- Measured TP1 overhead (vLLM's own accounting): non-torch **3.37 GiB**,
  peak activation **1.92 GiB** @2048 batched tokens, CUDA graphs **0.12 GiB**
  = **5.41 GiB** total. (My original 7.9 GiB estimate was ~2.5 GiB pessimistic.)
- With PLE offload, GPU consumption measured **100.23–102.3 GiB**.

What is **not** working: the server never reached `/health` 200. After CUDA
graph capture completes, the GPU worker spins at 99% CPU forever inside
`torch.ops.vllm.ple_offload_wait`, while `EngineCore` logs
`No available shared memory broadcast block found in 60 seconds` every minute.
The offload worker registers cleanly (`Busy-loop started`, GPU worker 0
registered) and logs no errors. Unresolved — see §7.

### PLE offload: what it does and does not buy

`VLLM_PLE_CPU_OFFLOAD=1` moves the 26.88 GiB PLE table to a dedicated
subprocess. Two things to understand:

- It **does not reduce the GPU budget** on unified memory. The offload
  worker's host RAM still counts against CUDA device-free. An early run that
  appeared to need only 71.78 GiB was misleading — the table simply wasn't
  loaded by anyone.
- It **does** reduce load-phase UVM pressure (71.78 GiB through UVM instead of
  98.66, rest in pageable/swappable host RAM). That is its only real benefit
  here.

**It requires `--distributed-executor-backend mp`.** `spawn_ple_offload()` and
`wait_ple_offload_ready()` are called only from
`vllm/v1/executor/multiproc_executor.py`. Without it no offload worker spawns
and the GPU worker deadlocks silently.

---

## 5. Bugs found and fixed in `files/patch_ple_layer.py`

The NVFP4 PLE offload path was broken in three independent ways. All three
fixes are committed as new edits in the patch generator (which
`start-tp1.sh` re-runs on every launch, so editing the generated
`ple_layer_patched.py` directly would be overwritten).

1. **`get_offload_output_dtype` was dropped.** It appeared in the generator
   only as part of an *anchor*, so the replacement silently removed it. The
   IPC buffer then defaulted to bf16 while the table is packed uint8 →
   `RuntimeError: index_select(): self and result must have the same scalar
   type`. Restored, returning `torch.uint8` for NVFP4 (keyed on
   `_offload_weight_scale_2`) and `float8_e4m3fn` for FP8.
2. **`_offload_quant_method` was read but never assigned** (`ple_layer.py:960`),
   so the GPU-side NVFP4 dequant branch in `_dequantize_embeddings` was dead
   code. Now set in the NVFP4 branch of `load_weights`.
3. **The offload fast path assumed rows are `head_dim` wide.** NVFP4 rows are
   `head_dim/2 + head_dim/16` bytes. Slicing to `embedding_dim` and reshaping
   to `head_dim` gave `index_select` a wrongly-shaped `out=` tensor, so torch
   **resized it** — silently allocating a new tensor instead of writing into
   the shared IPC buffer, so the GPU side waited forever. Now derives
   `row_width = weight.shape[-1]` (identical to upstream when unquantized).

Fix 3's symptom was only a `UserWarning: An output with one or more elements
was resized`. Treat that warning as fatal in an IPC path.

---

## 6. CRASH SAFETY — read before launching

The host hard-crashed **three times** this session (hang, no logs, forced
reboot). Causes, in order:

1. **`comfy-h3.service` was running.** It is a permanent GPU co-tenant
   (`ComfyUI MiniMax-H3`, auto-starts on boot) that idles at ~170 MiB but
   allocates on demand. It must be stopped:
   `sudo systemctl stop comfy-h3.service`. `start-tp1.sh` now refuses to launch
   while it is active, and that check is deliberately **not** gated behind
   `REQUIRE_IDLE_GPU` — bypassing it once is what caused crash #1.
2. **Default kernel VM tunables.** The box ships with `swappiness=0` (16 GiB of
   swap effectively disabled), `min_free_kbytes=45167` (44 MB floor on a
   121 GiB machine) and `watermark_scale_factor=10` (reclaim starts at ~0.1%).
   Against UVM allocating GiB/second that is a configuration with no floor, no
   early warning and no escape valve.
3. **Full-GPU load even with guards.** Crash #3 happened *with* the tunables
   applied, loading all 98.66 GiB through UVM (PLE offload disabled). The
   guards help but are **not sufficient** for a full-GPU load of this
   checkpoint.

### Tunables that demonstrably help (runtime-only, reset on reboot)

```bash
sudo sysctl -w vm.min_free_kbytes=4194304      # 4 GiB floor
sudo sysctl -w vm.watermark_scale_factor=300   # reclaim at ~3%, not 0.1%
sudo sysctl -w vm.swappiness=30                # actually use the 16 GiB swap
sudo sysctl -w vm.vfs_cache_pressure=200
```

With these, six consecutive runs produced **zero** crashes: memory oscillated
against the 4 GiB floor and recovered, swap absorbed overshoot, and failures
arrived as readable `ValueError`s instead of reboots. **They are reset by
reboot — re-apply before every attempt.** Note they also *reduce* CUDA-visible
memory by the size of the reserve, which is a real trade-off.

### What does NOT protect the host

- A userspace polling watchdog (3 s interval) — the collapse from 7.7 GiB to
  crash happened inside one 25 s window.
- `docker --memory` sized generously — the system died before the container
  reached its limit.
- Dropping page cache — the pressure was UVM allocations, not cache.

---

## 7. Recommended next steps

**Highest value first.**

1. **Decide whether single-Spark is worth continuing.** The honest position:
   98.66 GiB of weights on a 121.69 GiB unified box has almost no margin. It
   *loads* and *allocates KV*, but has never served a request, and the failure
   mode is a host hang. The 2-node setup works today at 600k context.
2. **If continuing, debug the post-graph-capture deadlock** (§4). The next
   concrete step is a `py-spy dump` of the GPU worker to see exactly where it
   blocks. That needs `--cap-add SYS_PTRACE` on the container (it currently
   lacks it; `py-spy` inside the container fails with `Permission denied`).
   Suspicion: `ple_offload_wait` is a CUDA *stream* wait, so graph capture
   records it without executing — the deadlock only bites on first replay,
   implying a request-dispatch or semaphore-signalling gap.
3. **Do not pursue "shrink the checkpoint to fit"** without a smaller
   checkpoint. Single-Spark would want ≤ ~85 GiB of weights; neither available
   checkpoint qualifies (98.66 and 125.9 GiB).
4. **Unmeasured and worth knowing: is RadixArk actually better?** It costs
   ~15% decode and 32% of KV headroom. Nothing in this session measured
   quality. Cheapest discriminator is a **logprob/KL divergence comparison** on
   identical prompts (~30 min including two model swaps) — far more sensitive
   than GSM8K, which at 97.27% with n=1319 can only resolve a ≥1.3pp gap.
   AIME26 at n=8 would be the real test but is ~7 h *per checkpoint* at the
   measured 191.6 tok/s aggregate.

---

## 8. Repo state as left

**Modified / new:**

- `start-tp1.sh` — **new**, single-node TP1 launcher. Derives GMU from live
  `/proc/meminfo`, refuses configs that don't fit, blocks on `comfy-h3`,
  caps container memory, budgets the full checkpoint. Defaults: 65536 context,
  MTP off, PLE offload on, `mp` executor. **Not yet known-good.**
- `files/patch_ple_layer.py` — three new edits (§5).
- `files/ple_layer_patched.py` — regenerated output of the above.
- `.env` — `MODEL_ID` switched to `local-inference-lab/…` (old quant).
  Original at `.env.bak`. **This also changes the 2-node `start.sh`.** Switch
  back to `RadixArk/…` if you want the 2-node deployment on the new weights.
- `bench/longctx.py`, `bench/decodebench.py` — **new**, restored from `/tmp`.

**Machine state after the final crash:**

- Host rebooted; `vllm-fn-tp1` exited (255); `vllm-fn-nfs` running.
- `comfy-h3.service` **active again** (auto-starts on boot).
- All four sysctls **reset to defaults** — re-apply before any attempt.
- No container has a restart policy; nothing will come back on its own.

**To get back to a known-good server**, use the 2-node path:

```bash
./stop.sh          # clear both nodes
./start.sh --launch
```

---

## 9. Process notes / where I went wrong

Recorded so the next person doesn't repeat it:

- I dismissed PLE offload early as "a no-op on unified memory." Half right —
  it doesn't save physical memory — but I used that to rule it out entirely,
  when it was the right lever for load-phase pressure. Cost several hours.
- I removed `--distributed-executor-backend mp` from the TP1 script as
  "meaningless at TP=1." It is what spawns the PLE offload worker.
- I weakened the co-tenant guard to get past ComfyUI's 170 MiB idle
  allocation. That guard existed for exactly that process; crash #1 followed.
- I set `docker --memory 16g` reasoning that GPU allocations aren't cgroup
  charged. On GB10 they are, and it OOM-killed EngineCore at 14.7 GiB.
- I twice declared the problem impossible ("it can't work", "no safe loading
  margin") on the strength of estimates rather than measurements. Both times
  the measured numbers were materially better than my model of them.

General lesson: on this platform, measure `Free memory on device` from vLLM's
own startup log rather than reasoning from `free`/`nvidia-smi`. `nvidia-smi`
reports `[N/A]` for memory on GB10.
