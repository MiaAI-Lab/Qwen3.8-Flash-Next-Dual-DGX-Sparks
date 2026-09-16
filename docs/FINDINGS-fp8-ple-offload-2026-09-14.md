# FP8 PLE offload on 2x DGX Spark — findings (2026-09-14)

> **Co-authored with OpenCode (DeepSeek V4.1 Flash).** The whole FP8 + PLE-offload
> saga on this lane — the packed-table builder and launcher wiring, the first
> multi-node boot and the registration-deadlock autopsy, the node-local offload fix,
> the FP8 packed-table dtype fix, and the boot / sweep / 1M-needle validation — was
> implemented and driven agentically by OpenCode running DeepSeek V4.1 Flash on the pair.

Working branch: `ple-offload-fp8`. Baseline: FP8 lane, `PLE_OFFLOAD=false`,
`SKIP_PLE_PATCH=true`, no offload wiring in `start.sh`.

## What we built

- `files/build_ple_packed_table_fp8.py` — builds the packed table for an FP8
  checkpoint. Pure byte concatenation of the 128 equal-sized `F8_E4M3` shards
  (row = head_dim bytes, no scales to interleave); the NVFP4 sibling packs
  `cat(codes, scales)` into 90-byte rows. Stdlib only (the host has no numpy).
  Verified: 320,001,536 rows x 160 B = 47.68 GiB, built in 66 s.
- `start.sh` step 6c — applies `patch_ple_offload.py`, bind-mounts the patched
  worker/connector on both nodes, syncs + mounts the packed table, sets
  `VLLM_PLE_PACKED_TABLE_DIR`.
- `start-fp8.sh` — `SKIP_PLE_PATCH` made overridable (was hard `true`).

## What already existed (and was never wired)

The offload machinery is complete in `files/` but the TP2 `start.sh` never
invoked it: `patch_ple_offload.py` runs clean (all anchors unique), the FP8
output dtype is implemented in `worker.py::get_offload_output_dtype`, and
`_attach_packed_table` skips the width check for FP8 (no `packed_row_width` on
`Qwen3_8FlashNextPLEFp8EmbeddingMethod`). The TP1 lane (`tp1/start.sh`) wires
all of it properly, including a cgroup cap + `memwatch.sh` watchdog, because
PLE_OFFLOAD is mandatory there.

## First blocker

`vllm/v1/worker/gpu_worker.py::_validate_ple_offload_config` rejects
`parallel_config.nnodes != 1`:

    ValueError: VLLM_PLE_CPU_OFFLOAD does not support the requested
    configuration. Unsupported settings: nnodes=2

It is the ONLY failing check for our TP2 config (DP=mp, PP=1, PCP=1, DCP=1,
no ubatching, architecture allowed, no weight transfer).

## First multi-node boot: the FP8 format works

With the guard lifted (`VLLM_PLE_OFFLOAD_ALLOW_MULTINODE=1`, commit `ee12cb8`)
the pair booted past it. Proven working:

- FP8 packed table built (47.68 GiB, byte-exact) and visible in both containers.
- The offload worker logs, verbatim:

      PLE ...: using packed mmap table /var/tmp/qwen38-ple-packed/...packed_u8
      PLE ...: mmap table attached (320001536 rows x 160 B = 47.68 GiB)
      PLE weight loading complete.
      PleOffload: registered 1 PleOffloadLayer(s) (dp_rank=0, tp_rank=0, ipc_addr=ipc:///tmp/...)

So the FP8 packed format, the mmap attach and the FP8 output dtype are all
correct — the new builder works.

## The real blocker (corrected): the worker is node-local by design, the code around it is not

The first boot's failure was read as "two node-local workers, each waiting for
2 registrations". Reading `spawn_ple_offload`, `multiproc_executor.py` and
`accept_registrations` together shows something simpler: only ONE offload
worker ever existed.

- `spawn_ple_offload` spawns on GLOBAL rank 0 only (`self.rank != 0`).
- `multiproc_executor.py` gives node 1's only rank global rank 1
  (`global_start_rank = local_world_size * node_rank_within_dp`), so node 1
  never spawned a worker.
- `parallel_config._ple_offload_ipc_path` is generated per config
  (`get_open_zmq_ipc_path()` -> `ipc://<base>/<uuid4()>`), so node 1's
  connector connected to a path with no listener. A zmq PUSH `connect()` to a
  missing ipc path succeeds; the registration queues in the socket and is
  dropped at close (`linger=0`) with no error. The `PleOffload: registered
  ...` lines on both nodes are the CONNECTORS logging — they are not proof of
  two workers, and the two different ipc addresses are the two configs' paths,
  not two bound sockets.
- The one worker (node 0) waited for `num_workers = dp_size * tp_size = 2`
  registrations, received only rank 0's, and timed out:

      TimeoutError: PLE offload worker did not become ready within 600.0s

So the guard does encode a real invariant, but the invariant is finer than
"single node": the whole protocol assumes ONE process tree — a single worker
for all ranks, spawned from global rank 0, counted world-wide, with a single
request sender per DP group. Multi-node needs the per-node worker topology
that the worker-side docstrings already describe ("serve every local DP rank",
"one CPU offload process for all local DP and TP workers") but that the
spawn/accounting/request paths never implemented.

Three gaps, not one:

1. **spawn** — global rank 0 (`self.rank != 0`) instead of each node's first
   rank;
2. **accounting** — `num_workers = dp_size * tp_size` (world) and validation
   against global TP rank sets, instead of the node's local rank count and
   local rank sets;
3. **requests/inputs** — `_launch` sends only from `tp_rank == 0`, and
   `_pin_input_buffers` only runs there. On a second node the local rank would
   wait forever on a done flag nobody could set, and its worker would never
   receive a request — gaps 1+2 alone still deadlock.

## The fix on this branch

One offload worker per node, serving that node's local ranks:

- `spawn_ple_offload` spawns from each node's first DP0 rank
  (`rank == node_rank_within_dp * local_world_size`);
- `num_workers = local_world_size`;
- every registration carries its global `rank`; the worker validates that the
  received rank set equals the node's local rank range, rejects duplicate
  `(dp_rank, tp_rank)` slots, and picks the lowest-rank local member of each DP
  group as the input/request leader;
- the connector computes `is_local_leader` (lowest local rank of its DP group)
  and only the leader pins/stages input buffers and sends
  `PleOffloadRequest`. Every rank still blocks on its own done flag; inputs are
  TP-replicated, so the local copy is equivalent.

Single-node behaviour is unchanged: the leader is TP0 of each DP group and
`local_world_size == dp_size * tp_size`, so the registration set, the leader
and the request sender are exactly what they were.

The alternative — one worker serving both nodes over the fabric — is not
available today: CUDA IPC output buffers and file_system shared memory never
cross nodes, so it would need a different transport for both. Per-node
workers duplicate the CPU forward once per node (the table is already mmapped
locally on both); that is cheap next to the GPU forward and is the only
transport-correct option.

## Validation status

**Boot-validated on hardware (2026-09-14):** node-local registrations on both nodes,
`:8888` serving, real generations with zero worker errors, and a 1M-context needle run
(see the "UPDATE — validation boot" and "1M context" sections below). The single-node
(`nnodes=1`) regression, the remaining quality/soak items, and the optional 3-round/4-level
sweep were all closed on 2026-09-15 (see the depth-reasoning section below); no items remain
open.

**Safety rails worked:** cgroup cap held at 40.0 GiB; memwatch logged
`avail=44948MiB ... container=40956MiB` throughout; host MemAvailable stayed
~44 GiB. No host hang.

**Also fixed during the first boot (committed):**
- `HEAD_PLE_OFFLOAD_MOUNTS` was clobbered by a later `=` after the `+=`
  (gpu_worker mount lost) — reordered.
- The packed-table mount was added to the dead `DOCKER_ARGS` path instead of
  the live heredoc mount variables — moved to `HEAD/WORKER_PLE_OFFLOAD_MOUNTS`.

---

## UPDATE — validation boot (2026-09-14, same day)

The fix was booted on the pair (`PLE_OFFLOAD=true`, GPU_MEMORY_UTILIZATION=0.70,
TP2/2 nodes). Results:

**Topology fix: WORKS.**

- Both nodes spawned their own offload worker and served exactly one local
  registration:
  - node 0: `GPU worker 0 registered (dp_rank=0, tp_rank=0)` →
    `Registrations complete` → `Busy-loop started`;
  - node 1: `GPU worker 1 registered (dp_rank=0, tp_rank=1)` →
    `Registrations complete` → `Busy-loop started` (this node never had a
    worker before the fix).
- Boot reached `:8888` (`qwen3.8-flash-next-fp8`, max_model_len 262144),
  KV cache 1.58M tokens (6.05x concurrency at 262k).

**First real forward crashed — a second, unrelated bug found and fixed
(`96aa55d`).** The very first CPU forward on both nodes hit
`RuntimeError: index_select(): self and result must have the same scalar
type` in `ple_layer.py::forward_impl`'s packed-table branch: the mmapped
table is uint8 bytes, while the FP8 output buffer is float8_e4m3fn by design
(the GPU side bit-views the rows and dequantizes with the retained scale),
so `index_select(..., out=buffer)` is illegal. NVFP4 buffers are uint8 and
were unaffected. Fix: keep the zero-copy `out=` for uint8 buffers and
gather-then-bit-view (`rows.view(output.dtype)`) for float8 buffers. This
path had never executed anywhere before (previous boots never got past the
registration deadlock and the TP1 lane ships NVFP4).

**After the fix:** real generations served with zero worker errors on both
nodes across two requests (57 and 93 completion tokens); single-stream speed
in line with the ~47 tok/s thinking-on baseline.

**Multi-node DP gate (same commit):** `_validate_ple_offload_config` now
rejects `nnodes>1 && DP>1` with a clear message (a node-local worker only
serves one dp0 replica per node; that config previously died with a bare
`IndexError` in the connector).

## Sweep results (thinking on, EP A/B + chunk A/B; 2 rounds, levels 1 and 6)

| config             | mix c1 agg / TTFT | mix c6 agg / TTFT | long c1 agg / prefill | long c6 agg / TTFT |
|--------------------|-------------------|-------------------|-----------------------|--------------------|
| EP-on  chunk 4096  | 33.7 / 4.14 s     | 69.5 / 12.74 s    | 10.9 / 2196 t/s       | 14.8 / 11.39 s     |
| EP-off chunk 4096  | 36.2 / 4.10 s     | 75.8 / 11.38 s    | 11.8 / 2293 t/s       | 15.4 / 10.99 s     |
| EP-off chunk 8192  | 35.4 / 3.20 s     | 73.8 / 14.76 s    | 12.6 / 2425 t/s       | 16.6 / 10.70 s     |

- **EP-off is adopted** — it wins every cell (agg +7-9%), contrary to the
  earlier "−4% at TP2" note.
- **Chunk 8192 is kept for the long-context lane**: the flatlining long shape
  improves everywhere (+6-8% agg, TTFT, prefill +5.8%), at the cost of mix c6
  (−2 agg, +3.4 s TTFT). Tony's +59% prefill claim does not reproduce
  (+5.8%). Two rounds is thin; re-confirm with a 3-round, 4-level run before
  treating either chunk choice as final.
- `.env` now: `ENABLE_EXPERT_PARALLEL=false`, `MAX_NUM_BATCHED_TOKENS=8192`,
  `GPU_MEMORY_UTILIZATION=0.70`, `PLE_OFFLOAD=true` (deliberately
  uncommitted).

## 1M context on the FP8 lane (2026-09-14, evening)

First 1M-context run on this lane: FP8 weights + node-local PLE offload, booted with
`OVERRIDE_MAX_MODEL_LEN=1000000 OVERRIDE_YARN_ENABLE=true --no-download` (YaRN 4.0,
GMU 0.70, chunk 8192, EP off). Boot reports `GPU KV cache size: 1,670,658 tokens,
Maximum concurrency for 1,000,000 tokens per request: 1.67x` — a full 1M request fits.

Needle run with the repo's own `bench/longctx.py` (one prompt, three needles at
5% / 50% / 95% depth, salted to defeat prefix caching):

| metric | value |
|---|---|
| prompt tokens | 937,508 |
| TTFT (prefill wall) | 672.98 s |
| prefill speed | 1,393 tok/s |
| decode speed | 51.9 tok/s |
| needles | alpha / bravo / charlie — **all FOUND** |

**Scope: retrieval only, not quality.** This is a long-context needle retrieval pass;
**no quality evaluation has been run on this lane yet** — `bench/reasoning_check.py`,
AA-LCR and the thinking+tools agentic soak are still pending, and no claim about answer
quality at 1M should be made from this run.

Context for the numbers: her #41 validated the same depth class on **NVFP4** + YaRN
(937,525-token cold rung: 328 s ≈ 2,858 tok/s prefill, single needle at ~80% depth).
This run is three needles at three depths in one prompt, and it is the first
FP8 + offload 1M result anywhere. FP8 prefill here is ~2x slower (FP8 dense kernels +
the offload lane's GMU 0.70); decode at depth is healthy.

Two process notes carried over from the run:
- `max_tokens` under-budgeting looks like a retrieval failure (her #41 finding) — the
  first attempt used 256 and was restarted at 2048 before drawing any conclusion.
- Killing a streaming request leaves an engine-side zombie that keeps working (her
  issue #24, confirmed live): the aborted first attempt still consumed a full 937k
  prefill. Do not kill streaming tests; let them time out.

## Quality pass and the YaRN tax (2026-09-14/15)

Two quality measurements on the lane, both at the 1M/YaRN config unless noted:

**Reasoning/retrieval suite (`bench/reasoning_check.py`) — 12/12 PASS**: 8 reasoning
tasks, 2 math, plus needles at 8k and 64k; greedy (`temperature=0`), thinking off on
this lane. Every deterministic task correct.

**YaRN tax A/B** — same checkpoint, same serve script; lm-eval on-spec sampling
(thinking ON, temperature 1.0 / top_p 0.95 / top_k 20, seed 1234, 16k generation cap),
the 1M/YaRN config vs native 262k (YaRN off):

| cell | 1M / YaRN | native 262k |
|---|---|---|
| GSM8K n=50 flex / strict | 0.98 / 0.98 | 0.98 / 0.98 |
| IFEval n=80 inst loose | 0.9531 | 0.9375 |
| IFEval n=80 inst strict | 0.9453 | 0.9375 |
| IFEval n=80 prompt loose | 0.9250 | 0.9250 |
| IFEval n=80 prompt strict | 0.9125 | 0.9250 |

Read: the extension's short-context cost is below this protocol's resolution (+/-1 item
of 80, single seed; GSM8K saturates at 0.98). It does not show the seed-consistent
2-4 IFEval points Muse-Glimmer-30B measured for its own YaRN extension. A
higher-resolution tax number needs 5 seeds and a non-saturating cell; depth *reasoning*
quality remains unmeasured (the 937k needle proves retrieval, not reasoning at depth).

## Long-context load (2026-09-15)

Two-stage load on the 1M/YaRN lane (`lc_load.py`, filed in `~/q38bench/` on the head):
concurrent mid-depth requests and one deep single, each carrying a distinct needle at
50% depth; pass = the code echoed back exactly.

| stage | shape | result |
|---|---|---|
| 1 | 3 concurrent x ~400k tokens (1.2M total) | all 3 needles found; TTFT 206 / 406 / 609 s (one stream queued on capacity, then preempt-recomputed); wall 611 s |
| 2 | single x ~989k tokens | PASS; TTFT 667.7 s (1,481 tok/s prefill) |

Engine across both: 0 errors, 2 preemptions total (recovered, all answers exact),
KV integrity intact, no `shm_broadcast` warnings during serving (those are boot-warmup
only), pool drained to 0 after. Read: correctness holds at depth under load; the
limiter is capacity - ~2 concurrent 400k-class streams fit comfortably, the third
queues - and aggregate prefill does not scale with concurrency (~1,960 tok/s across
stage 1 vs ~1.5-2k single-stream): prefill-bound, consistent with the sweep.

### Table placement - what it costs

The FP8 lane runs the 47.68 GiB PLE table as a demand-paged mmap of the packed file on
each node's NVMe; loading it into RAM is not a slower option, it is a non-option:
62.5 GiB of weights + 47.68 GiB of table + KV exceeds the 121.6 GiB unified pool, and
the table alone exceeds the 40 GiB container cap.

For the same design family, Tony's TP2 ledger carries the clean RAM-vs-disk delta:
table resident (SPEED) 53.7 tok/s single-stream / 97.9 aggregate at six / KV 1.97M vs
table on disk (CONTEXT) 35.8 / 65.5 / KV 5.87M - the disk table costs ~33% of decode
and buys ~3x the KV pool. Ours is a different mechanism (mmap page-in vs his per-step
preadv gather), so the applicable number is the mmap's own cold-vs-warm cost. Measured
with the table pages dropped from the page cache on both nodes:

| phase | TTFT @ ~100k tokens | wall |
|---|---|---|
| prime | 40.3 s | 41.6 s |
| warm (pages cached) | 40.5 s | 42.4 s |
| cold (drop_caches both nodes) | 40.2 s | 42.0 s |

Read: no measurable cold-vs-warm penalty at this scale - page-in is below resolution
next to the prefill cost, consistent with a working set small relative to the page
cache (47.7 GiB file, 22-42 GiB of cache while serving). The ~33% figure belongs to
Tony's per-step gather (and to the compile-off pairing that profile requires), not to
demand paging as such. A deeper cold probe (500k+) was not run; at 1M the working set
grows, and that is the remaining open measurement.

## Depth reasoning, ≥95%-depth needles under load, and the thinking+tools soak (2026-09-15)

Three follow-ups, all on the same lane (FP8 weights, node-local PLE offload, 1M/YaRN,
EP off, chunk 8192, GMU 0.70).

**Needles at 95% depth *under load*.** `lc_load.py` gained a `--depth` argument (the
first version pinned the needle at 50% depth). Needles at **95% depth**:

| stage | shape | result |
|---|---|---|
| 1 | 6 concurrent x ~230k tokens (1.38M total) | **6/6 found** — TTFT 108 / 211 / 314 / 417 / 520 / 620 s (capacity-queued: ~2 running, up to 4 waiting); wall 622 s |
| 2 | single x ~989k tokens | **found** — TTFT 665.7 s, wall 667.7 s |

0 errors: retrieval holds at 95% depth with the lane at capacity, not only in a quiet
single stream.

**Depth reasoning — AA-LCR.** The long-context reasoning benchmark (ArtificialAnalysis),
run with an on-spec thinking protocol (temperature 1.0 / top_p 0.95 / top_k 20, seed 1234,
30k generation cap) and judged by an independent model (`gpt-5.6-sol`):

| cell | n | score |
|---|---|---|
| AA-LCR, 1M/YaRN, thinking on, seed 1234 | 100 | **0.81** |

This is the first AA-LCR run on this checkpoint (the README still owes a quality
re-evaluation). It is well clear of the retrieval floor: the answers are reasoned at
~100k-average document depth, not pattern-matched from a buried line.

**Long agentic thinking+tools soak (the sglang#36537 token-0 loop).** Two runs with
thinking ON and OpenAI tools enabled (`--tool-call-parser qwen3_coder`) — the combination
behind the reported token-ID-0 `!!!!` loop:

| run | turns | wall | context | result |
|---|---|---|---|---|
| 1 (2048 cap) | 100 | 30.3 min | 475 → 255k tokens | 0 degeneracy flags; reasoning novelty never collapses |
| 2 (8192 cap, reasoning persisted) | 60 | 22.6 min | 475 → 159k tokens | 1/60 turns capped; that turn is a **heavy tail** per `loop_detector`; aggregate no collapse |

Run 1's 14/100 capped turns are budget truncation, not a loop: raised to an 8192 cap in
run 2 only one turn capped, and its reasoning stays novel to the end (word 8-gram novelty
never falls below the loop threshold for three consecutive windows). **The token-0
degeneracy did not reproduce on this vLLM + FP8-KV lane across ~53 minutes and 160 turns
of agentic thinking+tools.**

**Single-node (`nnodes=1`) regression.** The patched files were booted at TP1 on one Spark
with the smaller NVFP4 checkpoint (`local-inference-lab/Qwen3.8-Flash-Next-NVFP4` — the
only Qwen3.8-Flash-Next checkpoint that fits one device; ~98.6 GiB on disk, PLE table
mmap-offloaded). The node-local topology degenerates to the single-node path exactly:

    waiting for 1 GPU worker registration(s)
    GPU worker 0 registered (dp_rank=0, tp_rank=0)
    Registrations complete (dp_size=1, tp_size=1)
    Busy-loop started.

Served `:8888`, KV 328,790 tokens (262,144 native), a chat completion and a thinking+tools
turn both OK. Scope: this exercises the topology/accounting/leader changes, which are
quantization-agnostic; the FP8 `ple_layer` dtype branch takes the uint8 path on an NVFP4
checkpoint, so the FP8-specific branch at TP1 stays untestable by construction (FP8 does
not fit one device).

**Optional sweep confirmation (3 rounds, 4 levels).** The shipped config (EP off, chunk
8192) re-run at higher fidelity — thinking on, levels 1/2/4/6 (agg tok/s / TTFT):

| level | mix | long |
|---|---|---|
| c1 | 39.7 / 3.13 s | 11.9 / 2.19 s |
| c2 | 51.9 / 5.99 s | 13.9 / 3.93 s |
| c4 | 66.6 / 7.99 s | 15.5 / 7.50 s |
| c6 | 85.0 / 22.87 s | 15.9 / 11.08 s |

Against the original 2-round / levels-1/6 numbers (mix c1 35.4, c6 73.8; long c1 12.6,
c6 16.6) the shipped config sits within run-to-run spread — mix slightly better, long
slightly lower — and the c2/c4 points fill in the curve. No regression; the config choice
stands.

## Status

- Branch `ple-offload-fp8` is pushed to the fork (`Capicua25x`) and tracks draft
  PR #54 to MiaAI-Lab; discussion thread: issue #55.
- Newest commits: the public-surface scrub (neutral node labels + pseudonymous
  committer identity, 2026-09-14) and the quality/YaRN-tax section above.
- Validated: boot topology, generations, EP/chunk sweep, 1M needle, quality suite,
  YaRN tax, long-context load (3x400k concurrent + 989k deep single), needles at 95%
  depth under load, AA-LCR depth reasoning (0.81), a 160-turn thinking+tools soak with
  no token-0 loop, the single-node (`nnodes=1`) regression at TP1, and the 3-round/4-level
  sweep confirmation. No items remain open.

## Credits

- **Tony ([tonyd2wild](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark))** —
  the EP/chunk experiments in this document were motivated by his per-knob ledger, and the
  concurrency sweep used for those numbers is built on his `bench_sweep.py` structure. The
  pair's DS4 deployment (what these Qwen windows swap with) also started as his
  `ds4-vision-tp2.sh`, adapted to this box.
- The lane, the offload machinery and `bench/longctx.py` are this repository's (MiaAI-Lab).
