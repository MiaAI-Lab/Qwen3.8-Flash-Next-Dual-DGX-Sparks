#!/usr/bin/env python3
"""Aggregate-throughput sweep: N concurrent streams, steady-state decode.

Single-stream tok/s is the wrong metric for a batch workload. Weight bytes per
engine step are nearly flat in batch size, so aggregate throughput scales far
better than per-stream does -- and for agent/autoresearch fleets aggregate is
what you are actually buying.

Reports per-stream and aggregate, plus engine steps/s (aggregate tok/s divided
by mean accepted length), which is the only number comparable ACROSS prompts
and deployments -- tok/s alone compares workloads, not engines.
"""
import argparse, json, statistics, threading, time, urllib.request

BASE, MODEL = "http://localhost:8888", "qwen3.8-flash-next"
PROMPT = ("Write a flowing, continuous essay about the history of maritime "
          "navigation. Use ordinary narrative prose, no lists, no headings.")


def one(n_tok, temp, out, idx):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": n_tok, "min_tokens": n_tok, "ignore_eos": True,
            "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # Count tokens from the usage field, not from stream chunks: a chunk is not
    # a token (it can carry several, or none), which under-counts badly.
    t0 = time.time(); ttft = None; t_last = t0; usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            s = raw.decode().strip()
            if not s.startswith("data: "):
                continue
            d = s[6:]
            if d == "[DONE]":
                break
            o = json.loads(d)
            if o.get("usage"):
                usage = o["usage"]
            for ch in o.get("choices", []):
                if ch.get("delta") is None:
                    continue
                if ttft is None:
                    ttft = time.time() - t0
                t_last = time.time()
    ntok = (usage or {}).get("completion_tokens", n_tok)
    # decode time excludes prefill, matching decodebench.py
    dec = max(t_last - t0 - (ttft or 0.0), 1e-9)
    out[idx] = (ttft or 0.0, dec, ntok)


def spec_snapshot():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=20) as r:
            body = r.read().decode()
    except Exception:
        return {}
    import re
    want = ("vllm:spec_decode_num_drafts_total",
            "vllm:spec_decode_num_accepted_tokens_total")
    out = {}
    for line in body.splitlines():
        m = re.match(r"^([a-z_:]+)\{([^}]*)\}\s+([0-9.eE+-]+)$", line.strip())
        if m and m.group(1) in want:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="1,2,4,8")
    ap.add_argument("--decode", type=int, default=400)
    ap.add_argument("--temp", default="0.0")
    a = ap.parse_args()

    print(f"{'streams':>7} {'aggregate':>11} {'per-stream':>11} {'TTFT':>8} "
          f"{'acc_len':>8} {'steps/s':>8}")
    print("-" * 60)
    for n in [int(x) for x in a.streams.split(",")]:
        before = spec_snapshot()
        out = [None] * n
        ths = [threading.Thread(target=one, args=(a.decode, float(a.temp), out, i))
               for i in range(n)]
        t0 = time.time()
        for t in ths: t.start()
        for t in ths: t.join()
        wall = time.time() - t0
        after = spec_snapshot()

        toks = sum(o[2] for o in out if o)
        agg = toks / wall
        per = statistics.mean([o[2] / o[1] for o in out if o])
        ttft = statistics.mean([o[0] for o in out if o])
        dd = after.get("vllm:spec_decode_num_drafts_total", 0) - \
             before.get("vllm:spec_decode_num_drafts_total", 0)
        da = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - \
             before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        acc_len = (1 + da / dd) if dd else float("nan")
        print(f"{n:>7} {agg:>8.1f} t/s {per:>8.1f} t/s {ttft:>7.2f}s "
              f"{acc_len:>8.2f} {agg/acc_len:>8.1f}")


if __name__ == "__main__":
    main()
