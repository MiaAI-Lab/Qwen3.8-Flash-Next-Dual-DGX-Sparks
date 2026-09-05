#!/usr/bin/env python3
"""Snapshot / diff vLLM's MTP acceptance counters.

Cumulative acceptance mixes every request the server has ever seen, including
other clients. Always measure a *delta* around the workload you care about:

  python3 bench/mtp_accept.py --save before.json
  python3 bench/decodebench.py ...
  python3 bench/mtp_accept.py --since before.json
"""
import argparse
import json
import re
import urllib.request

URL = "http://localhost:8888/metrics"
WANTED = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    "vllm:generation_tokens_total",
)
LINE = re.compile(r"^([a-z_:]+)\{([^}]*)\}\s+([0-9.eE+-]+)$")


def scrape(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        body = r.read().decode()
    out = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        m = LINE.match(line.strip())
        if not m:
            continue
        name, labels, value = m.groups()
        if name not in WANTED:
            continue
        pos = re.search(r'position="(\d+)"', labels)
        key = f"{name}[{pos.group(1)}]" if pos else name
        out[key] = out.get(key, 0.0) + float(value)
    return out


def report(cur, base=None):
    def get(k):
        v = cur.get(k, 0.0)
        return v - base.get(k, 0.0) if base else v

    drafts = get("vllm:spec_decode_num_drafts_total")
    dtok = get("vllm:spec_decode_num_draft_tokens_total")
    atok = get("vllm:spec_decode_num_accepted_tokens_total")
    gen = get("vllm:generation_tokens_total")

    label = "delta" if base else "cumulative"
    print(f"MTP acceptance ({label}):")
    if dtok <= 0:
        print("  no draft tokens in this window")
        return
    print(f"  drafts            {drafts:,.0f}")
    print(f"  draft tokens      {dtok:,.0f}")
    print(f"  accepted tokens   {atok:,.0f}")
    print(f"  acceptance        {100.0 * atok / dtok:.1f}%")
    if drafts > 0:
        print(f"  accepted/draft    {atok / drafts:.3f}")
    for i in range(8):
        k = f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]"
        if k not in cur:
            break
        v = get(k)
        if drafts > 0:
            print(f"    position {i}      {v:,.0f}  ({100.0 * v / drafts:.1f}%)")
    if gen:
        print(f"  generation tokens {gen:,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--save", help="write a snapshot to this path")
    ap.add_argument("--since", help="report the delta against this snapshot")
    args = ap.parse_args()

    cur = scrape(args.url)
    if args.save:
        json.dump(cur, open(args.save, "w"))
        print(f"snapshot -> {args.save}")
        return
    base = json.load(open(args.since)) if args.since else None
    report(cur, base)


if __name__ == "__main__":
    main()
