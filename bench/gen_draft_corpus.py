#!/usr/bin/env python3
"""Sample the served model's own output distribution into a corpus .jsonl.

files/build_draft_vocab.py needs the distribution the DRAFTER has to predict,
which is the model's output, not a generic text corpus and not the prompts.
This drives the live endpoint over a spread of task types and writes one JSON
object per completion.

  python3 bench/gen_draft_corpus.py --out corpus.jsonl --per-prompt 3
"""
import argparse, json, random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

TOPICS = [
    "distributed systems", "the French Revolution", "protein folding", "monetary policy",
    "CUDA kernels", "Baroque music", "climate modelling", "Roman engineering",
    "graph theory", "sourdough fermentation", "orbital mechanics", "copyright law",
    "immunology", "compiler optimisation", "Ottoman history", "tidal energy",
]
LANGS = ["Python", "Rust", "C++", "TypeScript", "Go", "SQL", "Bash"]

def prompts():
    out = []
    for t in TOPICS:
        out.append(f"Write three detailed paragraphs about {t}.")
        out.append(f"Explain {t} to an expert, then list four open problems.")
        out.append(f"Summarise the key debates in {t}.")
    for l in LANGS:
        out.append(f"Write a well-commented {l} module implementing an LRU cache with tests.")
        out.append(f"Write idiomatic {l} to parse a large CSV and aggregate by key. Explain the tradeoffs.")
    out += [
        "Solve step by step: a train leaves at 3pm going 80km/h, another at 4pm going 100km/h. When do they meet?",
        "Prove that the square root of 2 is irrational, rigorously.",
        "Write a JSON schema for an e-commerce order, then three example documents.",
        "Draft a professional email declining a vendor proposal, with reasons.",
        "Write a short story about a lighthouse keeper who finds a radio.",
        "Compare REST and gRPC for a latency-sensitive internal API.",
        "Write a detailed bug report for an intermittent race condition.",
        "Outline a 6-week curriculum for learning linear algebra.",
        "用中文详细解释量子纠缠及其在通信中的应用。",
        "Explain the tradeoffs of speculative decoding in LLM inference serving.",
    ]
    return out

def call(args, prompt):
    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": 0.95,
    }).encode()
    req = urllib.request.Request(args.url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            d = json.load(r)
        return d["choices"][0]["message"].get("content") or ""
    except Exception as exc:                      # noqa: BLE001
        print(f"  ! {type(exc).__name__}: {exc}", file=sys.stderr)
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--out", default="draft_corpus.jsonl")
    ap.add_argument("--per-prompt", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    jobs = [p for p in prompts() for _ in range(args.per_prompt)]
    random.shuffle(jobs)
    print(f"{len(jobs)} completions at concurrency {args.concurrency}...")
    done = chars = 0
    with open(args.out, "w") as fh, ThreadPoolExecutor(args.concurrency) as pool:
        for text in pool.map(lambda p: call(args, p), jobs):
            done += 1
            if text:
                chars += len(text)
                fh.write(json.dumps({"text": text}) + "\n")
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)}  {chars:,} chars", flush=True)
    print(f"wrote {args.out}: {chars:,} characters")

if __name__ == "__main__":
    main()
