# Adaptive MTP depth on TP2: benchmark report

Results collected on 2026-09-22 (UTC+8). This report accompanies an opt-in, default-off port of the measured MTP strategy.

## Summary

This experiment changes **MTP speculative-decoding strategy only**: select draft depth K=1/2/3/4 from observed acceptance and step cost, instead of always drafting three tokens. It does not change model weights, quantization precision, context scaling, or target verification/sampling rules. It introduces no additional lossy approximation to the target model.

The paired results support improvements in Chinese prose, repetitive code, structured output, and mixed rolling concurrency. English prose and thinking are generally near baseline, with small or statistically unresolved differences. **We do not claim a speedup on every workload.**

Highlights, using paired geometric-mean changes:

- Chinese prose: **+18.92% single-stream**, **+23.04% at concurrency 4**.
- Repetitive Python code: **+5.73% to +15.38%**, across concurrency 1/2/4/8.
- Structured counting: **+7.88% to +13.05%**, across concurrency 1/2/4/8.
- Chinese prose with a 30,459-token prefix: **+14.35%**.
- Mixed rolling concurrency 4: **+4.79% end-to-end throughput**.
- English prose single-stream: **+2.08% point estimate**, but the confidence interval crosses zero; treat this as unresolved variation, not an established speedup.

## What was compared

| Component | Fixed baseline | Adaptive candidate |
|---|---|---|
| Hardware | 2 × DGX Spark, TP2 + EP | Same |
| Target | NVIDIA Qwen3.8-Flash-Next-NVFP4 | Same weights and precision |
| Draft vocabulary | 47,149 tokens, local argmax | Same |
| IndexShare | Enabled | Enabled |
| Allocated/captured maximum draft depth | 4 | 4 |
| Actual draft depth | Fixed K=3 | Online K=1/2/3/4 |
| Target verification / sampling | Existing implementation | Unchanged |
| Context | Native 262,144; YaRN off | Same |
| KV / SSM dtype | FP8 KV / BF16 SSM | Same |
| Maximum sequences / batched tokens | 8 / 8,192 | Same |
| Model load | One continuously loaded service | Same load; no restart between arms |

Deployment repository revision: `d2f54b78c0d2f9d74ac61aa56200e3c40fac3f22`.
Runtime version: `v0.1.dev20073+g8e685d198` (ModelRunnerV2).
Both nodes used image digest `sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`.
Controller SHA-256: `9c8a781aee7e9a98bc7bfd40f6032ce10b3509b6765f509f6f14649d670c42f0`.

**Scope:** both arms already contain IndexShare and the max-K4-capable runner. This isolates adaptive depth; it is **not** a measurement of the complete patch against unmodified upstream defaults, nor an independent measurement of IndexShare.

## Method

Based on [Mia's sweep](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/blob/main/bench/sweep.py), [sparkDash prompts](https://github.com/MiaAI-Lab/sparkDash/blob/main/src/shared/llmPrompts.js), and [sparkDash decode timing](https://github.com/MiaAI-Lab/sparkDash/blob/main/server/collectors/DecodeBench.js):

- Four upstream prompt types: English hash-map explanation, repetitive Python clamp functions, GPU-metrics JSON, and counting. Each tested at concurrency 1/2/4/8.
- Added Chinese single-stream/concurrency 4, long-prefix Chinese, and English coding thinking.
- Each of the 20 cells has **six paired repetitions**. Balanced arm order: AB/BA/AB/BA/BA/AB. Cell order is deterministically shuffled within each repetition.
- **600 output tokens per request**, temperature 0, top_p 1, no seed override, forced length. Thinking disabled except for the thinking extension (`xhigh`).
- A separate **32-token warmup at the same concurrency** precedes every arm; warmups are excluded from reported results.
- Concurrent upstream cases retain the original `(stream i/N)` prompt suffix.
- Three additional paired repetitions use **four rolling slots and 16 mixed jobs per wave**, with English/Chinese/code/JSON output lengths 320/640/512/256. A completed job is immediately replaced.
- Total: **246 measured waves, 900 measured requests, 523,872 output tokens**, over **72.75 minutes**. Another 828 warmup requests / 26,496 warmup tokens are excluded.
- All measured waves were isolated by request/token-counter checks; **zero preemptions, no incomplete streams, no service restarts**. The policy was frozen during the run.

For ordinary cells, aggregate decode TPS is `sum(completion_tokens - 1) / (latest last-content arrival - earliest first-content arrival)`. At concurrency 1 this is single-stream decode TPS. It excludes initial waiting/prefill. Rolling concurrency instead uses total completed output tokens divided by the end-to-end wall-clock interval, including initial waiting.

All repetitions are retained. Throughput columns below are arithmetic means; percentage changes are the geometric mean of within-pair ratios, so they can differ slightly from the ratio of displayed means. Intervals are 10,000-resample paired bootstrap 95% intervals. They are **per-cell, not multiplicity-adjusted**, and six repetitions (three for rolling) are still a limited sample. An interval crossing zero is labeled **unresolved variation**, not a demonstrated regression or proof of equivalence.

## Results: upstream prompt matrix

Unit: aggregate output tok/s. Concurrent results are total throughput, not per-stream speed.

| Workload | Concurrency | Fixed K=3 | Adaptive | Paired change | 95% interval | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| English prose | 1 | 56.26 | 57.46 | +2.08% | −1.53% to +5.64% | Unresolved variation |
| English prose | 2 | 94.98 | 92.23 | −2.94% | −5.85% to +0.66% | Unresolved variation |
| English prose | 4 | 148.32 | 148.66 | +0.26% | −4.07% to +4.80% | Unresolved variation |
| English prose | 8 | 230.84 | 233.31 | +1.07% | +0.26% to +2.12% | Small improvement |
| Python code | 1 | 75.20 | 84.35 | **+12.16%** | +11.66% to +12.83% | Improvement |
| Python code | 2 | 129.31 | 149.40 | **+15.38%** | +11.38% to +19.96% | Improvement |
| Python code | 4 | 225.09 | 252.72 | **+12.28%** | +9.91% to +14.82% | Improvement |
| Python code | 8 | 385.81 | 407.90 | **+5.73%** | +3.22% to +7.85% | Improvement |
| JSON | 1 | 65.57 | 67.44 | +2.80% | +0.31% to +5.41% | Small improvement |
| JSON | 2 | 114.65 | 119.85 | +4.51% | +2.57% to +6.50% | Improvement |
| JSON | 4 | 192.92 | 197.77 | +2.49% | −0.54% to +4.97% | Unresolved variation |
| JSON | 8 | 314.83 | 319.64 | +1.52% | −0.13% to +3.21% | Unresolved variation |
| Structured counting | 1 | 75.28 | 85.10 | **+13.05%** | +12.50% to +13.59% | Improvement |
| Structured counting | 2 | 132.76 | 148.48 | **+11.84%** | +10.89% to +12.81% | Improvement |
| Structured counting | 4 | 219.96 | 245.53 | **+11.62%** | +10.21% to +13.10% | Improvement |
| Structured counting | 8 | 358.49 | 386.74 | **+7.88%** | +5.76% to +9.57% | Improvement |

The individual English C2 pair 95.49 → 94.95 tok/s is a −0.56% observation. It is not, by itself, evidence of a regression. The complete six-pair result is reported above without discarding that pair or hiding the negative point estimate.

## Results: extended workloads

| Workload | Concurrency | Fixed K=3 | Adaptive | Paired change | 95% interval | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Chinese prose | 1 | 25.62 | 30.46 | **+18.92%** | +14.37% to +22.82% | Improvement |
| Chinese prose | 4 | 71.19 | 87.60 | **+23.04%** | +22.32% to +23.71% | Improvement |
| Chinese, 30,459-token prefix | 1 | 28.52 | 32.57 | **+14.35%** | +8.76% to +21.65% | Improvement |
| English coding thinking | 1 | 41.61 | 41.88 | +0.71% | −2.37% to +3.67% | Unresolved variation |
| Mixed rolling jobs, **end-to-end TPS** | 4 | 91.79 | 96.17 | **+4.79%** | +3.57% to +7.14% | Improvement; three pairs |

These prompts are synthetic, not private harness traces. Long-prefix decode results do not measure uncached prefill throughput. Mixed rolling jobs exercise changing batch membership but are not a broad long-duration serving benchmark.

## Why it helps

The controller estimates expected emitted tokens per step, `1 + sum(P(accepted prefix >= j))`, divided by observed step cost. It uses position-conditional acceptance probabilities, treats unobserved draft positions as censored rather than rejected, and keeps periodic exploration to avoid permanently locking into a shallow depth. Decisions use a rolling window, hysteresis, and settled decode steps; prefill and depth transitions are excluded from the estimator.

Single-stream counter/timing decomposition:

| Workload | Mean K, fixed → adaptive | Accepted draft tokens/step | Effective step ms | Explanation |
|---|---:|---:|---:|---|
| Chinese prose | 3.00 → 1.20 | 0.331 → 0.272 | 51.98 → 41.78 | Avoid expensive low-yield tail drafts |
| Python code | 3.00 → 3.84 | 2.997 → 3.825 | 53.04 → 57.04 | More accepted tokens offset a modestly dearer step |
| English prose | 3.00 → 3.43 | 1.928 → 2.105 | 52.00 → 54.00 | Extra acceptance largely competes with extra cost |
| English thinking | 3.00 → 2.52 | 1.166 → 1.070 | 52.05 → 49.42 | Lower cost and fewer accepted tokens largely offset |

Effective step time is wall-clock decode duration divided by counted draft steps, **not CUDA-event kernel timing**. These are descriptive counters, not independent causal ablations of every policy component. One K is shared by the scheduled batch; changing batch membership resets the estimator.

## Quality and numerical precision

**All changes evaluated here are MTP strategy/scheduling optimizations, not model compression or lower-precision computation.** Target/draft weights, quantization settings, vocabulary, and the target acceptance/verification/sampling rules are unchanged. Draft proposals are not accepted by a relaxed heuristic. Under the standard exact-verification assumptions, changing proposal depth preserves the target sampling distribution: the optimization does not intentionally trade model quality for speed.

This statement is about the algorithm and unchanged precision configuration, not a claim that this throughput experiment proves identical accuracy on every task. Different verification shapes/batching can expose existing floating-point differences; **bitwise-identical output is not promised**. No comprehensive quality evaluation was run.

Offline sanity checks on stored outputs, without executing generated code:

- Complete clamp function bodies matched the reference AST in **90/90 fixed and 90/90 adaptive code responses**; the final forcibly truncated function was excluded.
- Requiring function numbering to start at `clamp_00` gave **90/90 fixed and 89/90 adaptive**. The one adaptive response started at `clamp_16` with otherwise matching bodies; it used the `(stream 2/8)` suffix. This observed output difference is retained, not hidden or asserted to be a proven quality loss.
- A strict counting-from-one check passed **24/90 in both arms**. Concurrent suffixes frequently coincided with shifted starting numbers (for example, both arms started at 25 for one `stream 2/8` case). These throughput prompts are therefore unsuitable as a standalone accuracy score.

## Run integrity and artifacts

- 246/246 measured waves had matching request/token counters; zero preemptions.
- Both serving containers remained running with restart count 0 and unchanged start times; health check returned 200.
- Passive GPU samples: median temperatures 72°C / 71°C; maximum 77°C / 76°C. Median SM clocks 2496 / 2515 MHz. Available host memory stayed above approximately 3.28 / 6.75 GiB. Samples include warmups and inter-wave gaps; they do not constitute a formal thermal-throttling test.
- Portable benchmark: [run.py](../../../bench/mtp_adaptive/run.py); frozen settings: [protocol.json](protocol.json). The complete synthetic prompt set and long-prefix generator are in the runner.
- All 246 per-wave numeric records, with request/token counts and acceptance counters: [waves.csv](waves.csv). No rows were dropped. Generated text and local host diagnostics are not bundled into this compact dataset.
- Read-only reproduction: [analyze.py](../../../bench/mtp_adaptive/analyze.py); completion metadata: [state.json](state.json).

## Packaging and verification

The feature is **opt-in and default-off**: `MTP_ADAPTIVE=false` and `MTP_INDEX_SHARE=false` retain the existing launcher behavior. Enabling adaptive depth requires both flags true and `MTP_NUM_SPECULATIVE_TOKENS=4`. See [configuration](../../adaptive-mtp.md).

The five generated scheduler/runner/graph overlays match the measured files byte-for-byte. The controller's selection, observation, and reset methods are unchanged; packaging replaces the deployment-specific control path, makes adaptive mode the fallback when the feature is enabled without a control file, refuses a simultaneous batch-size Dynamic SD policy, and corrects a comment describing the cost regularizer. Source SHA-256 checks reject unreviewed images, and the same overlay set is copied to both ranks. No model, quantization, network, authentication, or cache-policy changes are included.

CPU policy/patch tests, source-backed overlay hash comparisons, inert-default/invalid-option launcher checks, and shell syntax checks are included. The portable runner and numeric dataset reproduce every table entry without sending inference requests when using `analyze.py`. **The packaging itself was not used for another GPU reload or benchmark**; GPU evidence comes from the measured prototype with identical generated runtime overlays and policy methods.

The small/unresolved differences are reported as variation, not established regressions or proven gains. The per-cell confidence intervals do not establish universal non-inferiority, and the proposal does not advertise an across-the-board speedup.
