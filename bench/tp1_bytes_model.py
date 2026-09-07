#!/usr/bin/env python3
"""Bytes-per-step model: ONE Spark (TP=1, whole checkpoint resident) vs the 2-node
TP=2+EP server, for the FP8-dense + quantized-MTP + NVFP4-PLE checkpoint.

Pure python, no GPU. Component sizes come from the checkpoint's safetensors headers
(pass --snapshot to recompute; defaults are the measured values of
MiaAI-Lab/Qwen3.8-Flash-Next-NVFP4-FP8dense-MTP[-PLE4], 2026-09-07). The model is
calibrated on the measured TP=2 numbers from autoresearch/RESULTS.md and then
applied to TP=1, so the TP=1 numbers are PREDICTIONS with the arithmetic shown.

Per engine step with MTP k=3 and B streams: one target forward over T = 4B tokens
(1 bonus + 3 drafts each) and three draft forwards over B tokens.
  read once per forward, split by TP (column/row/vocab-parallel): lm_head, GDN,
      attention, shared experts, PLE projections; the 32k draft head x3
  replicated on every rank (never split): HyperConnection, norms/gates/router
  routed experts: only experts touched by the step are read. With top-10 routing
      over 512 experts, uniform routing touches U(T) = 512*(1-(1-10/512)^T) per
      layer; real routing is skewed, so a --skew factor (<1) scales U at T>4.
      TP=2+EP owns 256 experts per rank -> each rank reads ~U/2 of them.
  GDN recurrent state: 36 layers x 48 heads x 128x128 bf16 = 56.6 MB per stream,
      read+written once per target forward, sharded by heads under TP.
  KV cache reads at ~1k context are negligible and are ignored.
Time per step = bytes / BW + collectives (TP=2 only) + fixed overhead O, where BW
and O are fitted from the c=1 TP=2 measurement (44.1 ms/step) and the collective
cost is the measured 116 ops x ~40 us = 4.6 ms (RESULTS.md).
"""
import argparse
import json
import math
import os
import re
import struct

GIB = 2 ** 30

# GiB, from safetensors headers of the FP8dense-MTP(-PLE4) snapshot (bench: this file --snapshot)
DEFAULT = {
    "lm_head": 0.59, "gdn": 1.96, "attn": 0.59, "shared": 0.22, "ple_proj": 0.06,
    "hc": 0.60, "other": 0.12, "experts": 63.28, "mtp_experts": 2.34, "mtp_dense": 0.09,
    "embed": 1.18, "visual": 0.84, "ple_table": 26.82,
}
N_EXPERTS, TOPK, N_MOE_LAYERS = 512, 10, 48
DRAFT_HEAD_GIB = 32768 * 2560 * 2 / GIB      # 32k-row bf16 draft lm_head slice (dequantized)
GDN_STATE_GIB = 36 * 48 * 128 * 128 * 2 / GIB  # per stream, bf16
MTP_K = 3


def component_sizes(snapshot):
    idx = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    hdr = {}
    def H(f):
        if f not in hdr:
            with open(os.path.join(snapshot, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr[f] = json.loads(fh.read(n))
        return hdr[f]
    c = {k: 0.0 for k in DEFAULT}
    for name, f in idx.items():
        m = H(f)[name]; b = m["data_offsets"][1] - m["data_offsets"][0]
        if "ngram_embedding" in name: k = "ple_table"
        elif re.search(r"\.mlp\.experts\.", name) and name.startswith("model."): k = "experts"
        elif name.startswith("mtp."): k = "mtp_experts" if ".experts." in name else "mtp_dense"
        elif "visual" in name: k = "visual"
        elif "embed_tokens" in name: k = "embed"
        elif name.startswith("lm_head"): k = "lm_head"
        elif ".linear_attn." in name: k = "gdn"
        elif "hyper_connection" in name: k = "hc"
        elif ".self_attn." in name: k = "attn"
        elif ".shared_expert" in name: k = "shared"
        elif ".ple." in name: k = "ple_proj"
        else: k = "other"
        c[k] += b / GIB
    return c


def touched_experts(tokens, skew):
    u = N_EXPERTS * (1 - (1 - TOPK / N_EXPERTS) ** tokens)
    return u * (skew if tokens > 4 else 1.0)


def bytes_per_step(c, tp, streams, skew, ep=True):
    T = streams * (MTP_K + 1)
    once = (c["lm_head"] + c["gdn"] + c["attn"] + c["shared"] + c["ple_proj"]) / tp
    once += c["hc"] + c["other"]                         # replicated
    draft = MTP_K * (DRAFT_HEAD_GIB / tp + c["mtp_dense"] / tp)
    experts = c["experts"] * touched_experts(T, skew) / N_EXPERTS / (tp if ep else 1)
    draft_experts = MTP_K * c["mtp_experts"] * touched_experts(streams, skew) / N_EXPERTS / (tp if ep else 1)
    gdn_state = 2 * GDN_STATE_GIB * streams / tp
    return {"once": once, "draft_head+dense": draft, "experts": experts,
            "draft_experts": draft_experts, "gdn_state": gdn_state,
            "total": once + draft + experts + draft_experts + gdn_state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", help="recompute component sizes from this snapshot")
    ap.add_argument("--tp2-ms", type=float, default=44.1, help="measured TP=2 c=1 ms/step (RESULTS.md: 55.3 tok/s / 2.44 accepted... 22.7 steps/s)")
    ap.add_argument("--collective-ms", type=float, default=4.6, help="116 collectives x 40 us")
    ap.add_argument("--bw", type=float, default=230.0, help="assumed achieved GB/s (peak 273); O is fitted from it")
    ap.add_argument("--skew", type=float, default=0.8, help="routing skew: fraction of uniform unique experts at T>4")
    ap.add_argument("--accepted", type=float, default=2.44, help="mean accepted length c=1 prose (code: 2.76)")
    ap.add_argument("--accepted-conc", type=float, default=2.28, help="mean accepted length at concurrency")
    ap.add_argument("--per-stream-ms", type=float, default=2.6,
                    help="empirical per-stream cost beyond --stream-knee streams (attention over KV, sampler, "
                         "scheduler): fitted so TP=2 at 32/48 streams matches 356.7/448.3 tok/s")
    ap.add_argument("--stream-knee", type=int, default=16)
    a = ap.parse_args()
    c = component_sizes(a.snapshot) if a.snapshot else dict(DEFAULT)
    resident = sum(c.values())
    print("component GiB:", {k: round(v, 2) for k, v in c.items()})
    print(f"resident checkpoint: {resident:.2f} GiB (PLE table {c['ple_table']:.2f}); one Spark = 121.69 GiB unified\n")

    bw = a.bw * 1e9 / GIB / 1000  # GiB per ms
    b2 = bytes_per_step(c, 2, 1, a.skew)
    O = a.tp2_ms - b2["total"] / bw - a.collective_ms
    print(f"calibration (TP=2, c=1): {b2['total']:.2f} GiB/rank/step at {a.bw:.0f} GB/s = {b2['total']/bw:.1f} ms "
          f"+ {a.collective_ms} ms collectives + O = {a.tp2_ms} ms  =>  fixed overhead O = {O:.1f} ms")
    print("  per-rank breakdown:", {k: round(v, 2) for k, v in b2.items()})
    b1 = bytes_per_step(c, 1, 1, a.skew)
    t1 = b1["total"] / bw + O
    print(f"\nTP=1, c=1: {b1['total']:.2f} GiB/step = {b1['total']/bw:.1f} ms + 0 collectives + O {O:.1f} = {t1:.1f} ms/step")
    print("  breakdown:", {k: round(v, 2) for k, v in b1.items()})
    print(f"  extra bytes vs TP=2 rank: +{b1['total']-b2['total']:.2f} GiB = +{(b1['total']-b2['total'])/bw:.1f} ms; "
          f"saved collectives: -{a.collective_ms} ms; net {t1 - a.tp2_ms:+.1f} ms/step")
    for name, acc, ref in (("prose", a.accepted, 55.3), ("code", 2.76, 62.6)):
        print(f"  predicted {name}: {1000/t1:.1f} steps/s x {acc} = {1000/t1*acc:.1f} tok/s  (TP=2 measured {ref}; "
              f"ratio {1000/t1*acc/ref:.2f}); if O shrinks 25% at TP=1: {1000/(b1['total']/bw+0.75*O)*acc:.1f}")
    print("\nConcurrency (aggregate tok/s). TP=2 measured: 8 streams 197.7, 16: 295.2, 32: 356.7, 48: 448.3")
    print(f"(per-stream term {a.per_stream_ms} ms beyond {a.stream_knee} streams applied to BOTH; each TP=1 box carries half the streams)")
    print(f"{'streams/box':>11} {'TP=2 pred':>10} {'TP=2 meas':>10} | {'TP=1/box':>9} {'2x TP=1':>8} {'vs TP=2 @ same total streams':>30}")
    meas = {1: 54.6, 8: 197.7, 16: 295.2, 32: 356.7, 48: 448.3}
    for s in (1, 4, 8, 16, 24, 48):
        bb2 = bytes_per_step(c, 2, 2 * s, a.skew)["total"]
        coll = a.collective_ms * (1 + 0.02 * 2 * s)   # payload grows ~linearly, latency-dominated
        t2 = bb2 / bw + coll + O + a.per_stream_ms * max(0, 2 * s - a.stream_knee)
        agg2 = 1000 / t2 * 2 * s * (a.accepted if s == 1 else a.accepted_conc)
        bb1 = bytes_per_step(c, 1, s, a.skew)["total"]
        tt1 = bb1 / bw + O + a.per_stream_ms * max(0, s - a.stream_knee)
        agg1 = 1000 / tt1 * s * (a.accepted if s == 1 else a.accepted_conc)
        m = meas.get(2 * s, float("nan"))
        print(f"{s:>11} {agg2:>10.0f} {m:>10} | {agg1:>9.0f} {2*agg1:>8.0f} {2*agg1/agg2-1:>+29.0%} (bytes/rank {bb1:.1f} vs {bb2:.1f} GiB)")
    print("\nKV headroom on one Spark (fp8 KV ~14.4 KiB/token + bf16 indexer caches):")
    for kv in (5, 6, 8):
        print(f"  {kv} GiB KV ~ {kv*GIB/(17*1024)/1000:.0f}k tokens")


if __name__ == "__main__":
    main()
