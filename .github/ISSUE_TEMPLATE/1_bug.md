---
name: Bug report
about: The model kit is not behaving as expected, is producing an error, or the numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README and .env reference first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's ".env Reference" and "KV cache
     budget" sections list the knobs that affect behavior, and the defaults shipped
     in .env.sample. -->

- Hardware / nodes: <!-- e.g. 2x DGX Spark (GB10, 128 GB unified memory), or 1x DGX Station, ... -->
- Interconnect: <!-- e.g. ConnectX RoCE/IB, 10GbE, ... -->
- Image / vLLM version: <!-- `docker images | grep vllm` or `docker inspect vllm-fn` -->
- Model (`MODEL_ID`): <!-- e.g. RadixArk/Qwen3.8-Flash-Next-NVFP4 -->
- `start.sh` flags used: <!-- e.g. `./start.sh --no-download --launch` -->
- Relevant `.env` values: <!-- e.g. MAX_MODEL_LEN, KV_CACHE_DTYPE, GPU_MEMORY_UTILIZATION, TENSOR_PARALLEL_SIZE, MTP_NUM_SPECULATIVE_TOKENS, PLE_OFFLOAD -->

---

## Steps to Reproduce

<!-- Please include full steps to reproduce so that we can reproduce the problem. -->

1. Run `./start.sh` <!-- describe the flags and what it printed up to the failure -->
2. ... <!-- describe steps to demonstrate bug -->
3. ... <!-- for example "curl /v1/models returns a different served name than the one set in .env" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add any other context about the problem here: a minimal request to reproduce bad output, JSON responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.
-->

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 256
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the lines with the backticks, and mention
      whether it came from `start.sh`, `docker logs vllm-fn`, or a client.

      Common culprits worth checking before filing:
        * `CUDA out of memory` right after launch -> drop_caches was skipped or
          GPU_MEMORY_UTILIZATION / MAX_NUM_SEQS are too high (see README "Gotchas").
        * NCCL hanging mid-init -> IB_HCA points at a port cabled to another cluster.
        * `kv_cache_dtype` mismatch -> this checkpoint only supports bf16 KV cache.
-->

```

```

</details>

<!--
      Consider also attaching screenshots and/or videos to better illustrate the issue.

      You can upload them directly on GitHub.
      Beware that video file size is limited to 10MB.
-->
