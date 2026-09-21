# Adaptive MTP depth (experimental, TP2)

This is an opt-in port of the acceptance/cost policy measured in the
[paired benchmark report](benchmarks/adaptive-mtp-20260922/report.md). It changes
MTP strategy, not model weights, quantization precision, or the target's
verification/sampling rules. No additional lossy approximation is introduced.
Exact verification preserves the target distribution in theory; floating-point
results from different batch/verification shapes need not be bitwise identical.

## Enable

### IndexShare alone (fixed depth)

`MTP_INDEX_SHARE=true` exposes vLLM's existing
`index_share_for_mtp_iteration` speculative-config option: reuse the QSA index
selection across MTP draft steps. This PR wires the native implementation into
the launcher; it does not introduce a new IndexShare kernel. It can be enabled
independently while keeping `MTP_ADAPTIVE=false` and fixed MTP3. No adaptive
overlays are loaded in that configuration. There is no separate IndexShare
on/off ablation in the published report.

### IndexShare with adaptive depth

Edit `.env` (as with the other knobs, `.env` takes precedence over the shell):

```bash
MTP_ADAPTIVE=true
MTP_INDEX_SHARE=true
MTP_NUM_SPECULATIVE_TOKENS=4
```

Then launch normally **when you intend to reload the model**. These are not
hot-enable flags: the runner must allocate the maximum depth and capture the
required shapes at startup. `4` is the maximum; actual decode depth varies over
1/2/3/4. Default settings remain fixed MTP3 with both new flags off, with no
adaptive overlays/imports/environment injected.

The current port is restricted to TP2, MTP, ModelRunnerV2, max K=4, IndexShare on,
and the source fingerprints from `v0.1.dev20073+g8e685d198`. The patch generator
refuses unknown image sources. Both nodes must have the same image digest.
Generated originals are cached by image digest, not by mutable image tag. All
patch generation/checks happen before this launcher's serving-container removal.

`start.sh` uses the existing overlay copy/mount mechanism for both ranks, with
unique basenames. Target/draft model modules, quantization, and rejection sampler
are not patched. The scheduler, V2 runner, autoregressive/MTP draft loop, and
decode graph capture are taught the active K. The unfused draft loop is used;
this QSA backend already uses that path at baseline.

No weights are downloaded by the feature itself. Maximum-K buffers/graphs can
consume more memory than stock max-K3; inspect the normal startup KV allocation
rather than assuming an unchanged KV pool. The published A/B kept max-K4
allocation identical in both arms, so it does not measure this memory delta.

## Optional hot control, once loaded

With the feature enabled, a missing control file starts in adaptive mode. The
head's normal cache mount exposes the host file
`~/.cache/vllm/mtp-adaptive/control.json` inside the container at
`/root/.cache/vllm/mtp-adaptive/control.json`. Only the head scheduler makes the
choice; the scheduled depth is propagated to workers. No worker control file
needs synchronizing.

Example contents for a same-load fixed-depth comparison:

```json
{"mode": "fixed", "k": 3}
```

To resume adaptive selection:

```json
{"mode": "adaptive"}
```

Write by atomic replacement, not by truncating an actively read file. Changes
are polled at most once per second. Invalid/missing updates retain the last
valid mode; **deleting the file does not revert a previously read fixed mode**.
The optional `policy` object accepts `window`, `interval`, `margin`,
`probe_steps`, and `probe_interval`; defaults are 64/16/0.04/8/96. Leave these
unchanged to reproduce the reported policy. A direct-container user may override
the container-side path with `QWEN_MTP_CONTROL`.

The prototype used a dated control path and a pre-created adaptive control file.
The packaged controller changes only the path/configuration fallback (missing
file now means adaptive), an incompatible-Dynamic-SD guard, a comment, and license header. Its selection/observation
methods are unchanged. The five generated runtime overlays have the exact SHA-256
values of the measured files; the source-backed CPU test checks this.

## Policy and limits

- Estimate expected emitted tokens as `1 + sum(P(accepted prefix >= j))` and
  divide by observed step cost. No prompt text, language, or benchmark name is
  consulted.
- Shorter drafts censor unobserved tail positions; they are not tail rejections.
- TP2 bootstrap costs are 40/46/51/56 ms, rescaled to the current request/batch
  and replaced by observed medians. Monotonic cost regularization is a policy
  heuristic, not a physical claim that a larger GPU shape can never be faster.
- Reconsider every 16 valid steps, use 4% switching hysteresis, and occasionally
  probe a longer tail. Ignore prefill, batch changes, and transition timing.
- A scheduled batch shares one depth. Batch membership changes reset the
  estimator; this is not independent per-request K. Do not combine it with a
  second dynamic-speculation controller through `EXTRA_VLLM_ARGS`; the factory
  refuses a simultaneous batch-size Dynamic SD table.

Validation covered synthetic greedy text, concurrency 1/2/4/8, a 30k-token
prefix, and mixed rolling jobs. Stochastic sampling, other images/hardware,
multimodal requests, and long-running real harness workloads were not benchmarked.

## CPU-only checks (no model reload)

```bash
python3 -m unittest discover -s tests -p 'test_mtp_adaptive*.py'
bash tests/test_mtp_adaptive_launch.sh
for f in start.sh stop.sh check-weights.sh files/mtp_adaptive/launch.sh; do bash -n "$f"; done
```

To also check generated bytes, point `MTP_TEST_SOURCE_ROOT` at a **pristine vLLM
package tree extracted from the supported image**, not the currently overlaid
serving container. For example, after a feature-enabled launch, the image-keyed
`files/mtp_adaptive/generated/<image-sha>/original` directory contains that tree:

```bash
MTP_TEST_SOURCE_ROOT=/path/to/pristine/vllm \
  python3 -m unittest discover -s tests -p 'test_mtp_adaptive*.py'
```

## Reproduce the measurements

The opt-in benchmark in `bench/mtp_adaptive/` changes only the hot control file;
it does **not** launch/restart/restore a model. Run it on the head during an
otherwise idle period. It stops on foreign traffic, failed requests, preemption,
or an output-directory `STOP` file and leaves the last tested mode in place.
Do not run it alongside real user requests. See its `--help` before starting.

Committed per-wave numeric data and a read-only analysis tool are under
`docs/benchmarks/adaptive-mtp-20260922/` and `bench/mtp_adaptive/`. They contain
only synthetic benchmark metrics, not credentials, host addresses, or user
conversation content. The full matrix, unresolved differences, and quality
limitations are deliberately retained in the report.
