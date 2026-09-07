# Settled results — dual-Spark Qwen3.8-Flash-Next tuning

All decode numbers: `bench/decodebench.py --decode 400 --contexts 1000`,
single stream, server otherwise idle, runs validated clean (exactly 1600
generation tokens) unless noted.

## Scope of the "output-safe" claim -- read this before repeating it

Independent adversarial review converged on the following correction, and it matters:

- **The DRAFTER changes are distribution-preserving.** Reduced draft vocabulary,
  quantized MTP layer, and draft-head storage balancing only alter the proposal q.
  Rejection sampling is exact for any q, so these cannot change the served
  distribution -- only acceptance rate.
- **FP8-dense is NOT in that category.** It quantizes the TARGET model -- attention,
  GDN projections, HyperConnection and the target lm_head all participate in the
  authoritative forward pass. Every request is affected, including requests with
  speculative decoding off. rel RMSE 0.026 / cosine 0.9997 bound the WEIGHT error;
  they do not bound generation quality, and small logit perturbations can flip greedy
  choices near ties. Treat FP8-dense as an explicit quality mode, not a free win.
- **Distribution-preserving is not seed-reproducible.** Changing which tokens are
  proposed changes accept/reject decisions and therefore RNG consumption, so the same
  seed can yield a different (equally valid) sequence. Three separate guarantees are
  worth stating separately: target distribution, greedy equivalence, seeded replay.
- **Task validation is still outstanding.** Run `bench/reasoning_check.py` plus the
  GSM8K/AIME harness. Do NOT gate on perplexity: on this model family AIME dropped
  86.7 -> 80.8 while long-context perplexity improved.

## Headline

| config | prose | code | entropy | copy | mean | acceptance |
|---|---|---|---|---|---|---|
| original baseline | 36.7 | 44.1 | 38.2 | 58.3 | 44.3 | 53.6% |
| + 8,600-id draft vocab | 39.6 | 35.4 | — | 53.9 | 42.4 | 34.3% |
| + 32,768-id draft vocab | 44.6 | 49.1 | 45.3 | 58.9 | 49.5 | ~50% |
| + 8192 batch tokens (R4) | 43.7 | 50.5 | 47.4 | 59.9 | 50.4 | 49.6% |
| + Marlin MoE | 47.1 | 54.3 | 46.9 | 59.7 | 52.0 | 52.2% |
| + FP8-dense (n=2) | 54.8 | 61.8 | 55.1 | 72.0 | 60.9 | 48.9-52.0% |
| + balanced draft head | 53.7 | 60.9 | 59.8 | 70.9 | 61.3 | 50.4% |
| **+ MTP quantized** (n=3, 800 tok) | **55.3** | **62.6** | **59.1** | **79.1** | **64.0** | 47.0-49.1% |

**Cumulative: +44% mean, +51% prose (36.7 -> 55.3).**

For comparison, this repo's own published `bench/decodebench.py` numbers:

| config | prose | code | entropy | copy |
|---|---|---|---|---|
| repo, full vocab | 40.0 | 43.2 | 40.1 | 59.1 |
| repo, best (`--balance-shards 2`) | 42.5 | 44.5 | 45.4 | 56.5 |
| **this work** | **55.3** | **62.6** | **59.1** | **79.1** |

i.e. **+30% prose and +41% code over the repo's best published config**, and past its
headline single-stream 54.4 tok/s. (Repo numbers are bf16 KV; these are fp8 KV.)

Measurement note: the last row uses `--decode 800` x3 runs rather than `--decode 400`
x1. At 400 tokens the run-to-run spread was +-3 tok/s, which is wider than several of
the effects being tested -- enough to make a real 4% change unreadable. Longer runs
plus repeats brought the spread to ~1.5 tok/s.

Note: the FP8-dense run has Marlin OFF. `VLLM_TEST_FORCE_FP8_MARLIN=1` was safe to
isolate only because there were no FP8 *linear* layers; FP8-dense creates them, so
the two must be re-tested together rather than assumed additive.

## Comparing against other published recipes — use steps/s, not tok/s

`bilikaz/qwen38-flash-next-cluster-recipe` reports **80 tok/s single-stream** and
**674 tok/s at 48 streams**. Those are not comparable to the numbers above, and the
reason is instructive.

Its reported per-position acceptance on prose is **0.91 / 0.85 / 0.80 / 0.71** (k=4),
i.e. mean accepted length **4.27**. Ours is 0.73 / 0.50 / 0.33 -> **2.44**. That 1.75x
ratio fully accounts for the tok/s gap. Their own README also reports **"~2.5 on long
reasoning"** -- essentially identical to our 2.44. So 80 tok/s is their favourable-prompt
figure, exactly the trap `bench/decodebench.py`'s own docstring warns about: *"copy from
context is the best case ... Do NOT quote a copy-heavy number as typical decode speed."*
An independent measurement report of this same model found a **27% spread** across runs
of a single prompt and concluded the same thing.

Normalising to engine steps -- the only prompt-independent metric:

| | tok/s | accepted length | **ms/step** | **steps/s** |
|---|---|---|---|---|
| bilikaz, single stream | 80 | 4.27 | 53.4 | 18.7 |
| **this work, single stream** | 55.3 | 2.44 | **44.1** | **22.7** |
| bilikaz, 48 streams | 674 | 4.27 | — | 157.8 |
| **this work, 48 streams** | 448 | 2.28 | — | **196.8** |

**This engine is ~21% faster per step single-stream and ~25% faster at 48 streams.**
The remaining difference is draft acceptance on their workload, not engine speed.

Config differences worth knowing (theirs -> ours): bf16 KV (they state the vendor QSA
guard refuses fp8; this repo patches that, so we run fp8 KV) ; `gpu-memory-utilization
0.70` vs our explicit `--kv-cache-memory`; a `hibrid47` checkpoint whose 95 GB n-gram
PLE table is re-quantized to NVFP4 and held resident and REPLICATED per box
(`MBX_PLE_REPLICATE=1`), where ours is FP8 and TP-sharded.

## Concurrency (max_num_seqs=48, prose, greedy, `bench/concbench.py`)

| streams | aggregate | per-stream | steps/s |
|---|---|---|---|
| 1 | — | 54.6 | 19.0 |
| 8 | 197.7 | 28.5 | 86.0 |
| 16 | 295.2 | 20.1 | 129.4 |
| 32 | 356.7 | 12.8 | 156.8 |
| 48 | **448.3** | 10.2 | 196.8 |

**Important for fleet/agent workloads:** the byte-cutting wins above are worth ~+50%
at concurrency 1 and very little by concurrency 8 -- weight reads amortize across the
batch. Published DGX Spark data shows the same shape (+27% at c1, +10% at c8, +0.2%
at c16). If the workload is many parallel streams, tune for aggregate and do not pay
for quantization complexity expecting it to show up there.

## What worked

**32,768-id corpus-ranked draft vocabulary (+11.7%).** The MTP drafter carries
its own BF16 lm_head over all 248,320 tokens and reads it once per draft step —
three of the four lm_head reads in an MTP=3 engine step. Slicing it to a
frequency-ranked subset cuts that. The size is a coverage/bandwidth trade:

| vocab | coverage | draft head/rank | result |
|---|---|---|---|
| 248,320 (full) | 100% | 0.59 GiB | baseline |
| 8,600 | too low | 0.04 GiB | **−4% overall, −20% on code** |
| 32,768 | 99.2% | 0.15 GiB | **+11.7%** |

The 8,600 version failed because coverage collapsed (acceptance 53.6% → 34.3%);
code suffers worst since rare identifiers fall outside a small vocabulary, and a
position-0 miss truncates the whole draft chain. Corpus used for ranking:
46 MB of Python source + 5 MB of docs + the model's own sampled output weighted
50x = 16.5M token occurrences, 60,407 distinct ids. Because 60,407 > 32,768,
every slot is genuinely corpus-ranked — no `--fill-to-size` merge-order guessing,
which was the flaw in the repo's earlier 65,536 attempt.

Rebuild with:

    docker run --rm --network none --entrypoint python3 \
      -v <corpusdir>:/corpus -v $PWD:/work \
      -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -w /work \
      vllm/vllm-openai:qwen38-flash-next \
      files/build_draft_vocab.py /corpus/code_corpus.txt /corpus/prose_corpus.txt \
        draft_corpus.jsonl:50 --model <snapshot> --size 32768 --out draft_vocab_32k.txt

(`--entrypoint python3` is required; the image entrypoint is the `vllm` CLI, so a
bare `python3 script.py` gets parsed as `vllm serve` arguments.)

**Right-sized KV cache (stability, not speed).** `--kv-cache-memory=24000000000`.
The pool was provisioned at 3.77M tokens while `max_num_seqs 8 x max_model_len
262144` caps reachable usage at 2.10M — ~15 GiB/node allocated but unaddressable.
`MemAvailable` was 2.1 GB with 4.4 GiB of swap in use; it is now 17–43 GB. This
almost certainly caused the first crash (a 55,710-token prefill stalled the
worker past the engine RPC timeout).

**MTP draft layer quantized (+5% mean over FP8-dense).** `mtp.*` was excluded from the
FP8-dense converter, leaving the whole draft layer BF16 -- ~0.59 GiB/step (~10%),
re-read on all 3 draft passes because `mtp_num_hidden_layers=1`. Now: dense + the 3
HyperConnections at per-channel FP8, and the 512 routed experts at FP8 block scales
(group 128) via `Fp8MoEMethod`. Measured rel RMSE 0.0239-0.0265; acceptance 47.0-49.1%
vs 48.9-52.0% before -- within noise.

**This one carries NO output-quality risk at all**, which is worth stating precisely:
rejection sampling is exact for ANY proposal distribution (Leviathan et al. Thm 1),
verified in this build's `_rejection_kernel` -- greedy accepts iff draft == target
argmax and writes the target argmax on rejection. Degrading the drafter can only lower
ACCEPTANCE, never change what the server emits. The one honest caveat: identical
*distribution*, not identical *samples* under a fixed seed.

Build: `MTP_FP8=true files/fp8dense/build.sh`, 78 s CPU-only, 7.4 GiB of new bytes
(everything else hard-linked). Run it independently on each node -- it is deterministic
and byte-identical, so a ~120 GiB rsync is unnecessary.

**Draft-head storage balancing (`VLLM_MTP_DRAFT_VOCAB_BALANCE=1`).** Byte-level BPE puts
frequent tokens at low ids, so slicing the draft vocabulary by the lm_head's own shard
range piled 32,406 of 32,768 rows onto rank 0 and 362 onto rank 1. A decode step waits
for the slowest rank, so rank 1 idled at the (value,index) all-gather while rank 0 read
~90x more bytes, three times per step. Now 16,384 rows each; the draft read halves
0.15 -> 0.08 GiB per draft step.

Correctness: `get_top_tokens` takes a local argmax then reduces with a (value, index)
all-gather, so it is correct for ANY row-to-rank assignment provided
`_draft_id_to_target_id` maps local row -> global id. WHICH tokens are in the draft
vocabulary (quality) and WHICH rank stores them (bandwidth) are independent decisions;
the original code tied them together. **This is NOT `build_draft_vocab.py --balance-shards`**,
which balances the VOCABULARY by padding it with merge-order ids and cost acceptance
56.5% -> 47.8%. Here the token set is unchanged, drafts are bit-identical, and
acceptance was unchanged (50.4%).

Honest caveat: the isolated measurement (61.33 vs a 59.9-61.9 baseline) sat inside
run-to-run noise. It is kept because it is free, provably correct and quality-neutral,
and because the byte saving is real and compounds with the other cuts -- not because a
win was measured in isolation.

**FP8-dense checkpoint (+17.5% over Marlin, +23% over the R4 baseline).**
Converts the bf16 dense half (GDN projections, HyperConnection, attention, lm_head)
to per-channel FP8, leaving the NVFP4 routed experts alone. Bytes/step ~7.8 -> ~5.6 GiB.

Quality: measured over all 591 converted tensors, rel RMSE min/median/max
**0.0240 / 0.0262 / 0.0268**, float64 cosine **0.99965-0.99968**. Crucially MTP
acceptance was UNHARMED (52.0% / 48.9% vs 49.6% before), which was the main risk --
the drafter's lm_head becomes FP8 too.

Build: `files/fp8dense/build.sh`, CPU-only, **72 seconds**, hard-links unchanged
shards so only ~11 GiB of new bytes hit disk. Rebuilt independently on both nodes,
byte-identical by sha256 -- no 123 GiB rsync needed.

Getting here required stacking FP8-dense with the draft vocab, which `start.sh` used
to refuse ("both overlay nvidia/mtp.py - pick one"). The two patch sets are
textually disjoint, but there is a real SEMANTIC collision: FP8-dense makes the
drafter's `lm_head` an FP8 tensor with per-row scales, while the draft-vocab patch
slices that head and runs `F.linear` on it -- which would have crashed ~10 minutes
into launch. `files/stack_mtp_fp8_draftvocab.py` bridges it by dequantizing the
slice with the matching row scales.

**Marlin MoE via `VLLM_TEST_FORCE_FP8_MARLIN=1` (+3.2% mean, +7.8% prose).**
`moe_backend=auto` resolves to FLASHINFER_CUTLASS, whose SM120 grouped GEMM has a
**128-row minimum M-tile** (cutlass_heuristic.cpp:595-635). Decode presents ~4 rows
per expert, so those tiles run ~3% full. Marlin's adaptive `block_size_m` drops to
**8 rows** at this shape and W4A16 skips the FP4 activation-quant kernel entirely.

Select it with the ENV VAR, never `--moe-backend marlin`: the backend name is global
and the MTP drafter's experts are BF16, which the unquantized oracle
(oracle/unquantized.py:176-179) refuses to build with marlin -> startup crash.

Costs: load time 300s -> 810s (Marlin repacks NVFP4 at load), and prefill gives up
FP4 tensor cores. Good trade for long decodes and rare restarts; re-evaluate if the
workload becomes prefill-heavy.

## What failed

**torch.compile mode 3 — hangs the box.** Inductor compiled for 103 minutes
without finishing; 104 consecutive 60s `shm_broadcast` stalls, then both
containers exited 255 on the RPC timeout. Early tell in the log:
`Not enough SMs to use max_autotune_gemm mode` — GB10's 48 SMs are under
Inductor's autotune threshold, so mode 3 could only have fused elementwise glue,
never retuned GEMMs. Low ceiling, fatal cost. **Stay on mode 0.** The hardcoded
mode 0 was correct, though no rationale had ever been recorded for it.

**GPU clock locking — no effect.** `nvidia-smi -lgc 2418,3003`: the GPU still ran
2463–2496 MHz at 33 W, unchanged. It self-limits well below the 3003 MHz "max",
and GB10 exposes no memory-clock control at all (`Memory: N/A`) — so SM clock was
never the limiter. Reverted.

**CPU pinning to the Cortex-X925 cores — no effect.** `taskset -apc 5-9,15-19`
(3.9 GHz X925 vs 2.808 GHz A725): mean 42.7 vs 44.3. Reverted.

**8,600-id draft vocabulary — net loss.** See above.

**`VLLM_MARLIN_USE_ATOMIC_ADD=1` — clear loss.** mean 45.95 vs 52.0 for plain
Marlin, below even the pre-Marlin baseline. Acceptance also moved 52.2% -> 46.4%,
which is expected: atomic reduction changes summation order, so logits differ
slightly and drafts land differently.

**GB10 QSA launch profile — loss (-6.5%).** Stacked with Marlin: mean 48.6 vs 52.0
for Marlin alone. The reasoning (upstream tables tuned on GB300's 160 SMs; GB10 has
48, so the decode profile over-launches ~5x CTAs and merges 64 split-K partials) was
plausible and is *documented in the patch itself* -- but it does not hold empirically
on this workload. The stacking machinery built to test it is still valuable: it is
what proved FP8-KV and a second ops/qsa.py overlay can coexist, and the same pattern
then unblocked FP8-dense + draft-vocab on mtp.py.

**PP=2 instead of TP=2 — arithmetically dead.** TP2 reads ~7.9 GiB/GPU in parallel
(~35 ms + 4.6 ms collectives); PP2 reads ~14.1 GiB sequentially (~63 ms). Break-even
would need each collective to cost >=240 us; measured GB10<->GB10 is ~40 us. At
concurrency 8 PP is ~0.75x TP2. The model *does* support PP (`SupportsPP` on
`Qwen3_8FlashNextForCausalLM`, model.py:589) -- it is simply the wrong trade.

**Sharding HyperConnection — net loss.** It is replicated by design
(`disable_tp=True` / `ReplicatedLinear`, hyperconnection.py:96-125) and issues ZERO
collectives. Sharding it would add ~97 collectives/step (~3.9 ms) to save ~0.6 GiB
of reads (~2.5 ms). Quantizing it instead (what FP8_DENSE does) halves the bytes
with no new collectives.

## Collective inventory (measured by source trace, not estimated)

**116 collectives per decode step**, ~4.6 ms at the published ~40 us GB10<->GB10
small all-reduce latency = ~10% of a 48 ms step:

| count | op | where |
|---|---|---|
| 1 | all-reduce | `embed_tokens` VocabParallelEmbedding |
| 36 | all-reduce | GDN `out_proj` RowParallelLinear |
| 12 | all-reduce | QSA `o_proj` RowParallelLinear |
| 48 | all-reduce | MoE late reduce (moe_runner.py:481-487) -- **ONE op, not two** |
| 1 | all-gather | target logits (0.99 MB/rank) |
| 18 | mixed | 3 MTP draft forwards |

96 of those are the structural TP minimum (2 per layer) and **no flag in this build
removes any of them**. Disabling EP does NOT change the count -- both paths issue the
same single late all-reduce. NCCL tuning caps at ~2% at concurrency 1.
**Conclusion: collectives are near-optimal. The remaining win is bytes.**

## Ruled out by analysis (don't re-investigate)

- **Interconnect.** RoCE v2 genuinely active (20.7M RDMA writes), link health
  pristine (0 CRC errors, 0 retransmits, 0 ECN marks). Decode uses **0.097%** of
  the 200 Gb/s link. All NCCL tuning combined caps at ~3–6%.
- **`--all2all-backend`.** Provably inert: `use_all2all_kernels` requires
  `dp_size>1 or pcp_size>1 or is_sequence_parallel` (fused_moe/config.py:1056)
  and this runs `dp_size=1`. The flag selects nothing.
- **CUDA graphs.** Correctly captured for the MTP shapes — 5 full + 5 prefill +
  4 decode graphs, sizes {4,8,16,24,32} tokens = {1,2,4,6,8} requests. Not a miss.
- **Async scheduling.** Already auto-enabled (`mtp` is in `EagleModelTypes`).
- **FP8 KV cache.** Safe here: both QSA indexer caches are hardcoded BF16
  (indexer_qsa.py:152-166), so sparse block selection is bit-identical to bf16.
  The upstream "quantized keys perturb block selection" warning cannot apply.

## Corrected roofline — read this before proposing "3x" wins

An earlier estimate in this effort (mine) put the memory roofline at ~16-18 ms/step
by counting only ~2.4B active params x 0.5 B/param. **That was wrong**: it ignored
the BF16 dense layers, which dominate. The real per-step read per GPU is ~9.2 GiB
(`docs/CLAUDE/fable5-1-report.md` §2, from the checkpoint's safetensors headers):

| component | GiB/step |
|---|---|
| lm_head x4 (1 verify + 3 MTP draft) | 2.36 |
| routed experts (NVFP4, EP-split) | ~2.1 |
| GDN projections (bf16) | 1.95 |
| HyperConnection (bf16, **replicated, not sharded**) | 1.19 |
| attention/indexer + MTP + router/shared/PLE + state | ~1.6 |

At 273 GB/s that is ~36 ms; at a realistic 235-245 GB/s achieved, **40-42 ms**.
Measured ~50 ms => **1.3x off roofline, not 3x.** Single-stream 100 tok/s is NOT
reachable by kernel tuning. The remaining ~8-10 ms is plausibly accounted for by
the ~144 small cross-node collectives per forward (see below) plus scheduling.

Consequence: **the only path to a large single-stream win is cutting bytes/step**,
i.e. the FP8-dense conversion. Kernel/flag tuning is worth ~1.3x at most.

## Why the GPU looks busy at 31 W

93% "utilization", ~31 W, 0% memory-controller utilization is the signature of an
**NCCL kernel spin-waiting on a network flag** — resident on an SM (so utilization
counts it), touching no DRAM, doing no math. TP=2 with one GPU per node means every
collective is an inter-node RoCE round trip: ~12 full-attention o_proj + 36
linear-attention out_proj all-reduces + 48 MoE blocks x2 = **~144 collectives per
target forward**, each carrying only 20-80 KB. Pure latency, no bandwidth — which is
exactly why the link sits at 0.097% utilization and is still worth several ms.
At a measured ~40 us per small inter-node all-reduce, ~6 ms/step (~12%).

## Verified image facts (do not re-investigate)

- **Rebuilding the image for sm_121a is a no-op.** Disassembling this image's
  FlashInfer MoE kernels vs the native `12.1a` build gives identical tensor-core
  instruction counts: `73056 HMMA / 26272 QMMA / 4736 OMMA` in both. The nonzero
  OMMA count also proves the routed experts run real FP4 tensor cores — this
  deployment is NOT on the silent Marlin/W4A16 fallback that causes most sm_121
  slowdowns.
- **This is the vendor-prescribed image.** Upstream vLLM support for the model
  merged to main 2026-08-31, five days after the latest stable release, so no
  release contains it. `eugr/spark-vllm-b12x` cannot load this architecture
  (its registry has no `Qwen3_8FlashNext`/`Qwen4Exp`).
- **SGLang / TRT-LLM are behind, not ahead.** SGLang's support PR is still open;
  TRT-LLM ships no SM120/121 cubins. SGLang was abandoned here over an SM121
  "doom loop" (decode collapsing to token id 0).
- **torch.compile mode 3 hanging is a known upstream bug**, not a local
  misconfiguration.

## Known open items

- **QSA kernels are tuned for GB300 (160 SMs), not GB10 (48).** Source comments
  say so explicitly (`qsa_ops_patched.py:694,927`). The decode path launches
  ~5x more CTAs than the machine can run, then merges 64 split-K partials.
  `files/qsa_gb10/qsa.diff` fixes this, but `start.sh:433` refuses `QSA_PROFILE`
  together with `KV_CACHE_DTYPE=fp8` ("both overlay ops/qsa.py — pick one").
  The two patches are applied sequentially and are probably stackable — there is
  precedent, since `patch_modelopt_fp8_block_moe` already stacks on the MXFP8
  patch.
- **Draft-vocab shard imbalance.** 32,406 of 32,768 ids landed on rank 0, because
  `--balance-shards` only balances `--fill-to-size` padding and we had none. A
  decode step waits for the slowest rank, so the effective bandwidth cut is 4x
  rather than 8x. Rebalancing is worth a little more.
- **Fused multi-step draft decode is unsupported** by the QSA backend
  (`speculator.py:117`), so each of the 3 draft steps rebuilds attention metadata
  on the CPU.
- **`generation_config.json` forces sampling.** The checkpoint ships
  `temperature=1.0, top_k=20, top_p=0.95`, applied to any request that omits
  temperature. Because `draft_sample_method` defaults to `greedy`, `draft_prob=1`
  and acceptance collapses to the target's own top-1 probability. Measured cost:
  **~16%** (T=0 vs T=1.0, clean paired trials). Fix without changing semantics:
  `draft_sample_method=probabilistic` — but it is mutually exclusive with the
  draft vocabulary, so it must beat +11.7% to be worth taking.
