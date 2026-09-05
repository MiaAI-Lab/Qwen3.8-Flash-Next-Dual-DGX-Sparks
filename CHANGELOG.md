# Changelog

Notable changes to this deployment. Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/).

## [Unreleased] — working tree since `4014cc6` (2026-08-31)

Everything below is **uncommitted**. The headline is the switch to
`nvidia/Qwen3.8-Flash-Next-NVFP4` with MTP speculative decoding working, **FP8 KV as the new
default** (1.70× the cache, no measured quality cost), an opt-in reduced-vocabulary MTP drafter,
and a large body of single-Spark / offload / dense-FP8 tooling that had accumulated untracked.

Two patches in this release are adapted from
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
(AGPL-3.0-or-later), a single-Spark TP=1 recipe. Its FP8-KV approach is in turn credited there to
`lancelind/qwen3.8-Flash-DGX` (Apache-2.0).

### Added — nvidia NVFP4 checkpoint support (2026-09-05)

Three checkpoint-specific gaps had to be closed before this checkpoint would serve. All three
are applied automatically by `start.sh`; none modify the HuggingFace cache (patched files are
written to `files/` and bind-mounted into the container).

- **`files/detect_ple_dtype.py`** — derives the PLE table dtype when
  `text_config.ple_embedding_dtype` is absent. This checkpoint declares its FP8 per-tensor PLE
  table only in `quantization_config.config_groups`, but the patched `ple_layer.py` dispatches on
  the `text_config` key. Without this the 51 GB PLE table fails to load. Resolves
  `nvidia/…` → `float8_e4m3fn`, and correctly returns nothing for `local-inference-lab/…` and
  `RadixArk/…`, which already declare the key.
- **`files/patch_checkpoint_config.py`** — adds MTP absolute-index aliases
  (`mtp.layers.0.…` → `mtp.layers.48.…`, i.e. `num_hidden_layers + i`). vLLM builds the MTP draft
  layer at the absolute index and matches quantization metadata by exact string with no
  renumbering; this checkpoint records only the relative index. Patches **both** `config.json` and
  the legacy `hf_quant_config.json` — vLLM falls back to the legacy file on the draft-model path,
  so patching only `config.json` fails identically. Also provides a `--mtp-moe-algo` preflight mode.
- **`files/patch_modelopt_fp8_block_moe.py`** — teaches
  `ModelOptMixedPrecisionConfig.get_quant_method` to build `FP8_BLOCK_SCALES` routed experts.
  This checkpoint's MTP routed experts are 128×128 block-scaled FP8, but the dispatch handles only
  `FP8` / `NVFP4` / `W4A16_NVFP4` / `MXFP8` and returns `None` otherwise — yielding a silently
  *unquantized* MoE that dies ~7 min into loading with
  `Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`. The patch routes to
  vLLM's own `Fp8MoEMethod` with `weight_block_size` read from the checkpoint's `group_size`; it
  refuses to guess a block shape, since a wrong one applies misaligned scales silently instead of
  failing. Patches `files/modelopt_patched.py` in place and is idempotent, so it stacks on the
  MXFP8 patch.

  > This gap is **not** image-specific. Upstream vLLM has no `FP8_BLOCK_SCALES` branch either —
  > not at commit `d4d703ca` (which the model card recommends, and which only fixes FP8 PLE
  > loading, #54882) nor on current `main`. Upgrading vLLM does not enable MTP here.

- `start.sh` now preflights the MTP expert algorithm and fails in **seconds** if it sees an
  algorithm the dispatch cannot build, instead of ~7 minutes into weight loading.

### Added — FP8 KV cache and reduced-vocabulary drafting (2026-09-05)

- **FP8 KV cache — now the default.** `files/patch_qsa_fp8_kv.py` teaches the QSA kernels to read
  an FP8-e4m3 cache: per-tensor K/V scales are applied *after* the tensor-core dots, which is
  algebraically equivalent before rounding and avoids materialising FP32 dequantisation tiles.
  Inert unless the dtype is fp8 (`KV_QUANT_MODE` is a `tl.constexpr`, so the BF16 path compiles to
  the same code as before). `start.sh` applies it automatically at step 4f whenever
  `KV_CACHE_DTYPE` starts with `fp8`. Vendored from the single-Spark recipe; applied to this
  image's sources unchanged, every anchor matching, so the two images ship identical QSA kernels.

  | | `auto` (bf16) | **`fp8`** (new default) |
  |---|---|---|
  | KV pool | 31.70 GiB | 32.02 GiB |
  | Cache size | 2,131,159 tok | **3,652,200 tok** |
  | Concurrency at 262,144 | 8.13× | **13.93×** |
  | Tokens per GiB | 67,229 | **114,060** (1.70×) |
  | `bench/reasoning_check.py` | 12/12 | 12/12, no regressions |
  | decode (prose/code/entropy/copy) | 42.5 / 44.5 / 45.4 / 56.5 | 40.0 / 47.7 / 47.0 / 56.4 |

  1.70× rather than 2× because the QSA indexer and compressor stay BF16 — only the main K/V halve.
  The upstream patch warns that a long-reasoning benchmark regressed 6/6 → 2/6 on the reference
  implementation, because quantised keys perturb *which blocks the sparse indexer selects*, not
  merely the attention output. **That did not reproduce here** — 12/12 including needle recall at
  13k and 104k prompt tokens — but 12 tasks is a smoke test, not a benchmark, and the warning
  concerns long reasoning chains this suite does not stress. Re-validate on your own workload.

- **Reduced-vocabulary MTP drafting (`MTP_DRAFT_VOCAB`, off by default)** —
  `files/patch_mtp_draft_vocab.py`. This checkpoint's vocabulary is 248,320 tokens and the MTP
  drafter carries its *own* BF16 `ParallelLMHead` over all of it (`tie_word_embeddings` false):
  1.18 GiB, 0.59 GiB per GPU at TP=2, read once per draft step — three of the four lm_head reads in
  an MTP=3 engine step. The patch slices that head to a frequency-ranked subset for drafting only;
  `compute_logits` keeps the full vocabulary. Output-safe by construction: drafts are verified
  against the target model, so a token outside the subset is rejected exactly like any other bad
  draft. Reached only via `get_top_tokens`, so `start.sh` also sets `use_local_argmax_reduction`
  (which independently cuts the draft all-gather from O(vocab_size) to O(2·tp_size) per token).

  > **TP-aware rewrite.** The upstream patch refuses to engage at `tp_size != 1` — the reduced head
  > there is a plain matmul, not a vocab-parallel one. This version slices each rank's own shard by
  > id range and reduces the per-rank winners with the same `(value, index)` all-gather vLLM's
  > `LogitsProcessor.get_top_tokens` uses. `files/test_draft_vocab.py` checks the slicing and the
  > cross-rank reduction against a full-vocabulary argmax restricted to the draft set.

  Measured (bf16 KV, 400 decoded tokens at 1k context, greedy):

  | draft vocab | draft head/rank | prose / code / entropy / copy | acceptance |
  |---|---|---|---|
  | full 248,320 | 0.59 GiB | 40.0 / 43.2 / 40.1 / **59.1** | **56.5%** |
  | 65,536, lowest-id fill | 0.31 GiB | 40.3 / 44.1 / 42.3 / 65.0 | 53.2% |
  | 65,536, `--balance-shards 2` | **0.16 GiB** | **42.5 / 44.5 / 45.4** / 56.5 | 47.8% |

  **A real trade, not a free win**, and left off by default for that reason: balancing helps prose,
  code and entropy, but acceptance falls 56.5% → 47.8% and the `copy` task ends up below the
  full-vocabulary baseline. A vocab-parallel lm_head splits by id range and a decode step waits for
  the slowest rank, so an unbalanced vocabulary saves bandwidth on one rank and none on the other —
  the lowest-id fill put 65,392 of 65,536 ids on rank 0.

  > **Known limitation.** Only ~8,300 of those 65,536 ids are corpus-ranked. The corpus was 85
  > documents / 49,141 token occurrences → 8,305 distinct ids, with coverage saturating by 16,384;
  > `--fill-to-size` padded the rest with the lowest-numbered unseen ids on the theory that
  > byte-level BPE is built in merge order. That is a **proxy, not a measurement**, and is the most
  > likely explanation for the acceptance loss. Untested: a larger corpus, and the corpus-only
  > ~8.3k vocabulary (which would cut the draft head to ~0.02 GiB per rank).

- **Vocabulary tooling** — `files/build_draft_vocab.py` (vendored, then extended with
  `--fill-to-size` and `--balance-shards`; reports coverage at several cut sizes so you tune on
  coverage rather than size) and `bench/gen_draft_corpus.py`, which samples the served model's own
  output distribution — the distribution the drafter must predict, which is neither the prompts nor
  a generic text corpus.

- **Measurement harnesses** — `bench/mtp_accept.py` snapshots and diffs vLLM's spec-decode
  counters, because cumulative acceptance mixes in every request the server has ever served
  (including other clients on the LAN); always measure a delta. `bench/reasoning_check.py` runs 10
  deterministic reasoning tasks plus needle recall at configurable depths, greedy and
  machine-graded, with `--compare` to diff against a saved baseline and a non-zero exit on
  regression. Built specifically because FP8 KV's failure mode is a wrong answer, not a slow one.

### Added — infrastructure and tooling (untracked before this changelog)

- **NFS weight sharing over ConnectX (opt-in)** — `files/nfs-share.sh`, `files/nfs-server/`,
  gated behind `NFS_SHARE` (default `false`) or the `--nfs` / `--no-nfs` flags. When on, the head
  exports its HuggingFace cache and the worker mounts it read-only as `/root/.cache/huggingface`,
  keeping **no local checkpoint copy**. `stop.sh --nfs` tears the share down (default leaves it up
  so the next `--launch` does not rebuild it).
- **`download.sh`** — fetch weights onto the head only (`--fp8` for the official FP8 checkpoint,
  or an explicit `org/repo`).
- **PLE CPU offload** — `files/patch_ple_offload.py`, `files/ple_offload/`,
  `files/build_ple_packed_table.py`. Packs the NVFP4 PLE table into one flat mmap-able file so the
  26.8 GiB table lives in evictable page cache rather than RSS. Includes the GB10 fix for
  `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 0`: the stock design synchronises via
  `cuStreamWaitValue32`, which on GB10 stalls the host thread and is not honoured inside captured
  CUDA graphs, hanging the GPU worker outright.
- **MXFP8 fallback** — `files/patch_modelopt_mxfp8.py`. FlashInfer's `mm_mxfp8` only accepts
  weights with `N,K >= 128` and `% 32 == 0`; layers outside that envelope now fall back to the
  BF16 emulation kernel instead of raising. Limits were measured on GB10 (sm_121) and are
  independent of token count.
- **PLE mixed-precision loading** — `files/patch_ple_layer.py`, porting the PLE quant dispatch
  from vLLM PR #53899 (`qwen4_exp`) onto the `qwen3_8_flash_next` layer shipped in the image.
  The stock image only selects the FP8 path when the *whole* checkpoint is `Fp8Config`.
- **FP8-dense variant** — `files/fp8dense/`, `files/overlay/`, gated behind `FP8_DENSE=true`
  (`FP8_DENSE_MODEL_ID`, default `MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense`). Includes
  checkpoint construction, verification, quant-stat computation, and a quant-config-resolution test.
- **QSA launch profiles** — `files/qsa_gb10/`, gated behind `QSA_PROFILE` (`stock` | `gb10` | path
  to JSON). `bench_qsa_kernels.py` benchmarks the real Triton QSA kernels at this deployment's
  decode shapes and writes the best table as JSON.
- **Single-GPU deployment** — `start-tp1.sh` and the self-contained `tp1/` tree (TP=1 on one
  Spark, serving the smaller local-inference-lab NVFP4 checkpoint). `start-fp8.sh` serves the
  official FP8 checkpoint at native 262144 context.
- **Benchmarks** — `bench/decodebench.py`, `bench/longctx.py`.
- **Memory watchdog** — `files/memwatch.sh`. On unified memory an exhausted pool hangs the kernel
  rather than raising an OOM, so this kills the container when host `MemAvailable` crosses a floor,
  behind the container's cgroup cap. Also logs a 5 s memory timeline for post-mortems.
- **Docs** — `docs/HANDOFF-single-spark.md`, `docs/CLAUDE/fable5-1-report.md`, `html/` (rendering
  probes and a 100-shot visual sample set).

### Changed

- **`.env` / `.env.sample`** — `MODEL_ID` is now `nvidia/Qwen3.8-Flash-Next-NVFP4` (133 GB /
  11 shards); previous checkpoints are kept as commented alternatives. `MAX_MODEL_LEN` dropped
  1000000 → 262144 and `YARN_ENABLE` true → false (see the YaRN fix below).
  `MTP_NUM_SPECULATIVE_TOKENS=3`. **`KV_CACHE_DTYPE` changed `auto` → `fp8`** (also in the
  `start.sh` fallback). New keys: `MM_ENCODER_TP_MODE`, `QSA_PROFILE`, `REQUIRE_IDLE_GPU`,
  `FP8_DENSE`, `NFS_SHARE`, `MTP_DRAFT_VOCAB` (`PLE_OFFLOAD` already existed). The live `.env` on
  this box is gitignored; it additionally sets `NFS_SHARE=true`, since this cluster's worker keeps
  no local checkpoint copy.
- **`start.sh`** — new steps for the reduced draft vocabulary (4e) and the FP8 KV patch (4f),
  alongside checkpoint-config patching (4g), FP8-dense overlay (4b), QSA profile overlay (4d) and
  the FP8-block-MoE patch (6b); the step letters were renumbered because three pairs collided.
  `add_overlay` now refuses two overlays on one container path, and refuses a basename clash as
  well — worker overlays land in a flat `/tmp/vllm-overlay`, so a clash would have one file quietly
  overwrite the other there. New `extract_from_image` helper. The speculative config gains
  `use_local_argmax_reduction` when a draft vocabulary is set. Step 3 now branches on `NFS_SHARE`:
  rsync to the worker's own HF cache by default (skipped when the checkpoint is already there),
  or the NFS export when sharing is enabled. Added `--nfs` / `--no-nfs`; `--launch` also skips the
  sync step. Both the worker's weight mount and the launch summary follow the same toggle.
- **`check-weights.sh`** — verifies the worker's local checkpoint copy by default, and the NFS
  volume only when `NFS_SHARE=true`; remediation hints name the matching command.
- **`stop.sh`** — added `--nfs` / `--all` to also stop the head share and remove the worker volume,
  with a 15 s timeout (kernel NFS in Docker can ignore SIGKILL when rpcbind is in D-state) and
  `--help`.
- **`README.md`** — new sections for the nvidia checkpoint, the MTP measurements, "NFS weight
  sharing (optional)" (with an rsync-vs-NFS trade-off table), "Reduced-vocabulary MTP drafting",
  and a rewritten Performance section (concurrency sweep + prefill), measured with
  [sparkDash](https://github.com/MiaAI-Lab/sparkDash) and credited there. "KV cache budget" was
  rebuilt around the fp8 defaults: new pool/token figures, an fp8-vs-bf16 table, per-GPU accounting
  re-read from the running container, and the byte model extended with the fp8 line (13,312
  B/token vs 26,624 bf16) explaining why measured overhead rises from 14% to 41% of the pure-
  attention figure. The previous batch-1 content-mix table was measured on a different container
  and different weights; it is retained in a collapsed "superseded" block. All intra-document
  anchors verified to resolve.
- **`.gitignore`** — excludes `.last_head_launch.sh` and the `start.sh`-generated
  `files/config_patched.json` / `files/hf_quant_config_patched.json`.

### Fixed

- **YaRN was a silent no-op.** `start.sh` emitted `--hf-overrides '{"rope_parameters":{…}}'` at the
  *top level*. vLLM's `ModelConfig._apply_dict_overrides` only recurses into keys that are
  themselves nested configs; for `qwen4_exp` the parent exposes `rope_parameters` as a plain dict,
  so the override was `setattr`'d onto the parent and never reached `text_config`, which the model
  actually reads. Verified in-image: after the override, `text_config.rope_parameters` still read
  `rope_type: "default"`. **Every earlier "1M context" run was serving 1M positions on unscaled
  rope.** The override is now nested under `text_config` (alongside `ple_embedding_dtype`), so
  `YARN_ENABLE=true` changes rope for the first time — 1M context is therefore **unvalidated** on
  this kit and `YARN_ENABLE` is left `false`.
- `start.sh` force-disables YaRN when `MAX_MODEL_LEN <= 262144`, where it has no benefit and costs
  accuracy.
- **The README's "fp8 KV is not supported by this checkpoint" was wrong.** It is not a checkpoint
  limitation: the stock QSA kernels declare `supported_kv_cache_dtypes = ["auto", "bfloat16"]` and
  raise `Qwen3.8-Flash-Next QSA requires a BF16 main KV cache`. With the kernels patched, this
  checkpoint runs an FP8 cache fine. The gotcha now says so and points at `KV_CACHE_DTYPE` rather
  than telling readers to avoid the flag.
- **Documented: `.env` silently beats the environment.** `start.sh` sources `.env` after reading
  the environment, so `KV_CACHE_DTYPE=fp8 ./start.sh` is ignored for any key `.env` already
  defines — it cost a full launch cycle here before the launch summary's `KV dtype:` line caught
  it. Recorded as a gotcha; the precedence itself is unchanged.

### Measured (2-Spark kit, TP2+EP, `MAX_MODEL_LEN=262144`, `GMU=0.835`)

MTP speculative decoding, batch-1 greedy:

| | MTP off | MTP=3 |
|---|---|---|
| Decode | 24.5 tok/s | **52.1 tok/s (2.13×)** |
| Draft acceptance | — | 72.8% (823/1131) |
| Acceptance by position | — | 89% / 74.5% / 60% |
| Weights resident | 62.72 GiB | 64.3 GiB |
| KV cache | 2,663,445 tok (10.16×) | 2,131,159 tok (8.13×) |

The decaying per-position acceptance curve is the check that matters — a wrong block shape would
show up as near-random acceptance, not as a crash.

The KV-cache row above is the **bf16** measurement, kept because it is the like-for-like
MTP off/on comparison; the shipped fp8 default now gives 3,652,200 tokens (13.93×).

Decode under concurrency (prose) and prefill throughput are tabulated in the README's Performance
section, measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) on fp8 KV *plus* the
balanced 65,536 draft vocabulary (not a default): aggregate scales 3.8× from ×1 (54.4 tok/s) to ×8
(207.0 tok/s), TTFT at ×1 is 160 ms, and prefill peaks at 2962.2 tok/s (32k), shedding 8% by 128k.
Because both levers moved at once, those deltas are not individually attributable — the per-lever
numbers are in the two tables above. Low concurrency and TTFT improved (×1 +4.4%, TTFT −32%);
×4 regressed 6.2%.

Regression suite passing with MTP on: chat, streaming, tool calling, vision, and long context
(43,281 tokens, needle recovered).

---

## Prior history

See `git log`. The last commit before this working tree is `4014cc6`
(*start.sh: reject non-numeric MAX_MODEL_LEN before the YaRN guard*, 2026-08-31), which followed
`6e08722` — the replacement of the SGLang deployment with this vLLM TP2+EP+MTP3 dual-DGX-Spark
bring-up.
