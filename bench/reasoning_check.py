#!/usr/bin/env python3
"""Deterministic reasoning + long-context checks with machine-checkable answers.

Exists because FP8 KV is a *quality* trade on this model, not just a capacity
one: quantised keys perturb which blocks the QSA indexer selects, so the failure
mode is a wrong answer, not a slower one. Run this at KV_CACHE_DTYPE=auto,
record the score, then run it again after switching to fp8.

  python3 bench/reasoning_check.py --out bf16.json
  python3 bench/reasoning_check.py --out fp8.json --compare bf16.json

Greedy (temperature 0) so reruns are comparable. Each task is graded by
substring match against accepted answers, after stripping the reasoning block.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

# (name, prompt, [accepted answer substrings, lowercased])
TASKS = [
    ("arith_chain",
     "Compute step by step: ((17 * 23) + 149) / 4 - 37. Give the final numeric answer last.",
     ["98"]),
    ("trains",
     "Train A leaves at 15:00 travelling 80 km/h. Train B leaves the same station "
     "at 16:00 travelling 100 km/h on the same track. At what clock time does B "
     "catch A? Give the time as HH:MM.",
     ["20:00", "8:00 pm", "8 pm"]),
    ("logic_grid",
     "Alice, Bob and Carol have a cat, a dog and a bird, in some order. Alice does "
     "not have the bird. Bob does not have the dog and does not have the bird. "
     "Who has the bird? Answer with just the name.",
     ["carol"]),
    ("sqrt2",
     "Is the square root of 2 rational or irrational? Answer with one word.",
     ["irrational"]),
    ("counting",
     "How many times does the letter 'r' appear in 'strawberry raspberry'? "
     "Give a single integer.",
     ["6"]),
    ("units",
     "A tank with a 240 litre capacity starts full. It drains at 3 litres per "
     "minute for 20 minutes, then is refilled at 8 litres per minute for 15 "
     "minutes, overflow being discarded. How many litres are in it now? "
     "Give a number.",
     ["240"]),
    ("date",
     "If today is Wednesday, what day of the week is it in 100 days? "
     "Answer with the weekday name only.",
     ["friday"]),
    ("modular",
     "What is 7^100 mod 13? Give a single integer.",
     ["9"]),
    ("ordering",
     "Sort these from smallest to largest and give only the sorted list: "
     "0.9, 0.85, 0.099, 0.891, 0.1.",
     ["0.099, 0.1, 0.85, 0.891, 0.9"]),
    ("negation",
     "Every glorp is a frimp. Some frimps are not blicks. Does it follow that "
     "some glorps are not blicks? Answer yes or no, then one sentence.",
     ["no"]),
]

FILLER = ("Record {i:05d}: the depot logged a routine variance in the northbound "
          "inventory reconciliation for quarter {q}.\n")


def needle_task(depth_tokens: int, secret: str):
    """Long-context recall: a secret buried in filler at ~depth_tokens in."""
    lines = [FILLER.format(i=i, q=(i % 4) + 1) for i in range(depth_tokens // 16)]
    mid = len(lines) // 2
    lines.insert(mid, f"Record {mid:05d}: the archive passphrase is {secret}.\n")
    body = "".join(lines)
    return (f"{body}\nWhat is the archive passphrase? Answer with the passphrase only.",
            [secret.lower()])


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    return text


def ask(url, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return content, reasoning, time.time() - t0, d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8888/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out")
    ap.add_argument("--compare")
    ap.add_argument("--needles", default="8000,64000",
                    help="comma-separated approximate context depths, or empty to skip")
    args = ap.parse_args()

    tasks = list(TASKS)
    for i, depth in enumerate([d for d in args.needles.split(",") if d.strip()]):
        secret = f"BRIGHT-HARBOR-{9174 + i}"
        prompt, accepted = needle_task(int(depth), secret)
        tasks.append((f"needle_{depth}", prompt, accepted))

    results = {}
    passed = 0
    for name, prompt, accepted in tasks:
        try:
            content, reasoning, dt, usage = ask(
                args.url, args.model, prompt, args.max_tokens, args.timeout)
        except Exception as exc:                        # noqa: BLE001
            print(f"  {name:<14} ERROR {type(exc).__name__}: {exc}")
            results[name] = {"ok": False, "error": str(exc)}
            continue
        hay = strip_reasoning(content).lower()
        ok = any(a in hay for a in accepted)
        passed += ok
        results[name] = {
            "ok": ok,
            "answer_tail": content.strip()[-160:],
            "seconds": round(dt, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_chars": len(reasoning),
        }
        print(f"  {name:<14} {'PASS' if ok else 'FAIL'}  {dt:6.1f}s  "
              f"{usage.get('prompt_tokens', '?')} prompt tok")
        if not ok:
            print(f"       got: {content.strip()[-160:]!r}")

    total = len(tasks)
    print(f"\nscore: {passed}/{total}")
    if args.out:
        json.dump({"passed": passed, "total": total, "results": results},
                  open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")

    if args.compare:
        base = json.load(open(args.compare))
        print(f"\nbaseline {args.compare}: {base['passed']}/{base['total']}")
        regressions, fixes = [], []
        for name, cur in results.items():
            was = base["results"].get(name, {}).get("ok")
            if was and not cur["ok"]:
                regressions.append(name)
            elif was is False and cur["ok"]:
                fixes.append(name)
        print(f"regressions: {regressions or 'none'}")
        print(f"newly passing: {fixes or 'none'}")
        if regressions:
            sys.exit(1)


if __name__ == "__main__":
    main()
