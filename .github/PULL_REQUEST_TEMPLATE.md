## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this kit, a good description usually covers:
      * which .env knob or script behavior changes
      * whether the change affects measured numbers (KV cache budget, TTFT, decode
        throughput, concurrency) and, if so, the expected direction
      * why the change is safe on both DGX Spark nodes

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the [issue tracker](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues).
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this kit that usually means:

      * `bash -n start.sh stop.sh check-weights.sh` (syntax check)
      * `shellcheck start.sh stop.sh check-weights.sh` if available
      * an actual launch on the cluster: `./start.sh --no-download --launch` and the
        resulting `docker logs vllm-fn 2>&1 | grep -E "Available KV cache memory|GPU KV cache size"`,
        or at minimum a dry-run on a node you have
      * if behavior changed, the measured numbers with the new settings (see README
        "Performance" / "KV cache budget" for the format used)

-->

---

## Checklist:

<!--

    Thanks for contributing to Mia's AI Lab!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and `.env.sample` and kept my changes consistent with them.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (e.g. script syntax check, a launch, or a re-measurement).
- [ ] I updated the README and/or `.env.sample` if a `.env` knob, default, or measured number changed.
- [ ] Defaults in `.env.sample` still work out of the box; a new knob has a sane fallback like the existing ones.
