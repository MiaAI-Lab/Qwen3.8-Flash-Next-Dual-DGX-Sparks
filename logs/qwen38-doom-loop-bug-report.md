# [Bug] Qwen3.8-Flash-Next long thinking decode enters a token-0 `!` loop and poisons later requests on SM121 TP2

## Summary

`RadixArk/Qwen3.8-Flash-Next-NVFP4` served by the current dual-DGX-Spark recipe can begin emitting the decoded token-0 signature (`!`) continuously during a long thinking-enabled generation. The request still returns HTTP 200 and consumes the full token budget.

After one corrupted long request, later short requests can also enter the same loop. In the observed run, the recipe's smoke command printed hundreds of exclamation marks and still exited successfully. Restarting both tensor-parallel ranks cleared the degraded state.

This reproduces with FP8 KV after the recipe's sglang#36806 + #36845 SM121 fallback fixes, so it is not limited to NVFP4 KV or the disabled TRT-LLM sparse-decode path.

## Environment

| Component | Value |
| --- | --- |
| Hardware | 2x NVIDIA DGX Spark GB10, SM121, TP=2 |
| OS | Ubuntu 24.04 LTS |
| Kernel | `6.17.0-1031-nvidia` |
| NVIDIA driver | `580.173.02` |
| Fabric | Direct ConnectX-7 RoCEv2, 200 Gb, MTU 9000 |
| Recipe | `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` |
| Upstream recipe commit | `344f9d0d5e9523d8398fa2804d5a3e123fd3d21a` |
| Relevant recipe fix | `6acd773` (sglang#36806 + #36845 SM121 Triton packed-varlen fallback) |
| Local security patch | `45e008390289c5c60519ed3a9ea2ae9bfe6b748c` |
| Base image | `lmsysorg/sglang:qwen38flashnext@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4` |
| Patched image | `qwen38-flashnext-dspark:local` on both nodes |
| Model revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Served model | `Qwen3.8-Flash-Next-NVFP4` |
| Context | 900,000 tokens via YaRN from 262,144 |
| KV cache | FP8 E4M3 (`NVFP4_KV_CACHE=0`) |
| Speculation | NEXTN 3 steps / top-k 1 / 4 draft tokens |
| Concurrency | `MAX_RUNNING_REQUESTS=6` |
| Mamba allocation | `MAMBA_FULL_MEMORY_RATIO=0.18`; 35 state slots on the reproducing boot |
| CUDA graphs | Exact batch sizes `1 2 3 4 5 6` |
| Radix cache | Enabled |
| Attention / sampling | FlashInfer; SM121 QSA decode routed to `sm121_varlen.py` |

Both containers used `restart: unless-stopped` and had zero restarts before the reproduction.

## Reproducer

The attached `qwen38_doom_loop_repro.py` uses only the Python standard library. It does not print the API key or response text. It reports token counts and repeated-character statistics only.

Run from the operator Mac against a freshly started server on SparkA:

```bash
ssh -o BatchMode=yes jessebunch@192.168.1.110 \
  'cd ~/Qwen3.8-Flash-Next-Dual-DGX-Sparks && export API_KEY="$(perl -ne '\''print $1 if /^API_KEY=(.*)$/'\'' .env)" && python3 -' \
  < ~/Desktop/qwen38_doom_loop_repro.py
```

A loop is considered reproduced when the combined `reasoning_content` and `content` contain at least 16 consecutive `!` characters. The script exits `1` when reproduced, `0` when all attempts are clean, and `2` on a request/protocol failure.

The core request is:

```json
{
  "model": "Qwen3.8-Flash-Next-NVFP4",
  "messages": [
    {
      "role": "user",
      "content": "Develop a detailed architecture and rollout plan for a reliable distributed inference service, including failure recovery, observability, security, capacity planning, and validation."
    }
  ],
  "temperature": 0.6,
  "max_tokens": 1600,
  "min_tokens": 1600,
  "ignore_eos": true,
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

## Observed Result

On 2026-08-29, the first attempt after a clean boot produced:

```text
long attempt 1: completion_tokens=1600, longest_bang_run=480, replacement_chars=0
REPRODUCED: decoded token-0 exclamation loop detected
```

The response began coherently, then ended with 480 consecutive `!` characters. The API stayed healthy and returned HTTP 200.

A later short recipe smoke request then returned a few normal words followed by hundreds of consecutive `!` characters. The smoke command exited zero because it validates HTTP/JSON success but does not reject token-0 output. This matches the stateful degradation reported in recipe issue #11: a corrupted or aborted long request can leave later work vulnerable until both ranks restart.

## Expected Result

- The 1,600-token generation remains coherent and contains no token-0 repetition.
- A failed or aborted generation releases all scheduler, KV, Mamba, and speculative-decoding state.
- Subsequent unrelated requests remain healthy without restarting the TP pair.
- The API should not return HTTP 200 for a completion known to contain invalid repeated token IDs.

## Controls

The following passed on the same image and configuration before the forced long probe:

- Authenticated recipe smoke test
- `/v1/models` with `max_model_len=900000`
- Structured `get_weather({"city":"Paris"})` tool call
- Vision request
- Six simultaneous 128-token requests
- Five deterministic 400-token thinking-disabled requests
- Public keyless request rejected with HTTP 401
- OpenCode request through the public tunnel
- Zero container restarts on both ranks

Per-request `chat_template_kwargs: {"enable_thinking": false}` avoids the observed loop. FP8 KV does not eliminate it. Thinking remains enabled by default in this deployment by operator choice.

## Why This Appears Distinct From The Fixed SM121 Path

The patched image explicitly keeps FlashInfer TRT-LLM sparse decode disabled on exact SM121 and routes QSA packed one-query varlen work through the sglang#36845 Triton fallback. Startup, short structured tools, vision, and long-context retrieval had already validated that route. This failure occurs later in a long free-form thinking decode and becomes sticky across requests.

Relevant upstream threads:

- sglang issue #36537: thinking/tool token-ID-0 loop
- sglang PR #36806: exact-SM120 TRT-LLM routing; SM121 excluded
- sglang PR #36845: SM121 Triton packed-varlen fallback
- recipe issue #19: token-0 loop and KV-dtype observations
- recipe issue #11: long-context arena/scheduler state not released
- recipe issue #22: single-stream latency behavior on the same recipe commit

## Requested Investigation

1. Capture the first layer/operation that produces token ID 0 during the long decode, before parser/detokenizer handling.
2. Compare target logits, speculative draft acceptance, QSA output, and Mamba state immediately before the first repeated zero.
3. Verify CUDA-graph row/state reuse across a completed or aborted long request and the following request.
4. Add a server-side guard that fails a request rather than streaming an unbounded repeated token-ID-0 sequence.
5. Add a regression that runs a forced 1,600-token thinking generation followed by an unrelated short request on SM121 TP2.

## Logs

On SparkA:

```text
/home/jessebunch/Qwen3.8-Flash-Next-Dual-DGX-Sparks/.sglang.log
/home/jessebunch/Qwen3.8-Flash-Next-Dual-DGX-Sparks/.sglang-worker.log
```

The deployment was restarted after reproduction so the currently running logs describe the clean post-reproduction boot. Preserve logs from a future repro before restarting if synchronized rank traces are needed.
