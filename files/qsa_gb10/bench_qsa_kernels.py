#!/usr/bin/env python3
"""Micro-benchmark the Qwen3.8-Flash-Next QSA Triton kernels on this GPU and emit the
best launch profile as JSON for VLLM_QSA_PROFILE_JSON.

Runs the real kernels from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa with the
deployment's decode shapes (TP2: 12 query heads x 1 KV head x head_dim 256, page 1600,
top-k width 2048+3) over a sweep of (block_n, splits, warps) and of MQA scorer tiles.
Needs an idle GPU (run inside the image: docker run --gpus all ... python3 bench_qsa_kernels.py).

  python3 bench_qsa_kernels.py --tp 2 --rows 4 --ctx 8192 65536 262144 --out /root/.cache/vllm/qsa_gb10.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import torch

os.environ.setdefault("VLLM_QSA_PROFILE", "stock")
from vllm.models.qwen3_8_flash_next.nvidia.ops import qsa as qsa_ops  # noqa: E402


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6  # us


def make_decode_case(rows, ctx, tp, page_size=1600, head_dim=256, topk=2048, compress=4, device="cuda"):
    num_q_heads = 24 // tp
    num_kv_heads = max(1, 2 // tp)
    pages_per_req = (ctx + page_size - 1) // page_size
    num_reqs = rows
    num_pages = pages_per_req * num_reqs + 1
    q = torch.randn(rows, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(1, num_pages, device=device, dtype=torch.int32).view(num_reqs, pages_per_req)
    token_to_req = torch.arange(rows, device=device, dtype=torch.int32)
    width = topk + compress - 1
    # random sparse selection inside [0, ctx)
    idx = torch.randint(0, ctx, (rows, width), device=device, dtype=torch.int32)
    idx, _ = idx.sort(dim=1)
    out = torch.empty_like(q)
    # indexer side: compressed key cache (1 head, dim 128), positions/seq_lens
    cpage = page_size // compress
    kc = torch.randn(num_pages, cpage, 1, 128, device=device, dtype=torch.bfloat16)
    cq = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
    ctable = torch.arange(1, num_pages, device=device, dtype=torch.int32).view(num_reqs, pages_per_req)
    positions = torch.full((rows,), ctx - 1, device=device, dtype=torch.int32)
    seq_lens = torch.full((num_reqs,), ctx, device=device, dtype=torch.int32)
    return dict(q=q, k_cache=k_cache, v_cache=v_cache, idx=idx, block_table=block_table, token_to_req=token_to_req,
                out=out, kc=kc, cq=cq, ctable=ctable, positions=positions, seq_lens=seq_lens, topk=topk, compress=compress)


def bench_sparse(case, profile_row):
    qsa_ops._QSA_PROFILE = {"sparse": {b: profile_row for b in ("small", "lt32", "le256", "le512", "large")}}
    return timeit(lambda: qsa_ops.qsa_sparse_paged_attention(
        case["q"], case["k_cache"], case["v_cache"], case["idx"], case["block_table"], case["token_to_req"], case["out"]))


def bench_mqa(case, tiles, warps):
    qsa_ops._QSA_PROFILE = {"mqa": {"tiles_small": tiles, "tiles_large": tiles, "num_warps": warps}}
    return timeit(lambda: qsa_ops.qsa_select_paged_tokens(
        case["cq"], case["kc"], case["ctable"], case["token_to_req"], case["positions"], case["seq_lens"],
        case["topk"], case["compress"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 32])
    ap.add_argument("--ctx", type=int, nargs="+", default=[4096, 32768, 131072])
    ap.add_argument("--out", default="qsa_profile.json")
    a = ap.parse_args()
    assert hasattr(qsa_ops, "_QSA_PROFILE"), "run with the qsa_gb10 overlay mounted"
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} SMs={p.multi_processor_count} cc={p.major}.{p.minor}")

    sparse_grid = [(bn, sp, w) for bn in (16, 32, 64) for sp in (1, 2, 4, 8, 16, 32, 64) for w in (2, 4)]
    results = {}
    best_per_bucket = {}
    for rows in a.rows:
        for ctx in a.ctx:
            case = make_decode_case(rows, ctx, a.tp)
            base_programs = rows * case["k_cache"].shape[2]
            bucket = ("small" if base_programs <= 8 else "lt32" if base_programs < 32 else
                      "le256" if base_programs <= 256 else "le512" if base_programs <= 512 else "large")
            stock = {"small": (16, 64, 4), "lt32": (16, 32, 4), "le256": (64, 8, 2), "le512": (64, 4, 2), "large": (64, 1, 2)}[bucket]
            t_stock = bench_sparse(case, stock)
            best = (t_stock, stock)
            for cfg in sparse_grid:
                try:
                    t = bench_sparse(case, cfg)
                except Exception as e:  # noqa: BLE001
                    print(f"  rows={rows} ctx={ctx} cfg={cfg} failed: {str(e)[:80]}")
                    continue
                if t < best[0]:
                    best = (t, cfg)
            m_stock = bench_mqa(case, 1 if rows <= 32 else 8, 2)
            m_best = (m_stock, (1 if rows <= 32 else 8, 2))
            for tiles, w in itertools.product((1, 2, 4, 8, 16), (1, 2, 4)):
                t = bench_mqa(case, tiles, w)
                if t < m_best[0]:
                    m_best = (t, (tiles, w))
            print(f"rows={rows:3d} ctx={ctx:7d} bucket={bucket:5s} sparse: stock {t_stock:8.1f}us -> best {best[0]:8.1f}us {best[1]} | "
                  f"mqa+topk: stock {m_stock:8.1f}us -> best {m_best[0]:8.1f}us tiles/warps={m_best[1]}")
            results[f"rows={rows},ctx={ctx}"] = {"bucket": bucket, "sparse_stock_us": t_stock, "sparse_best_us": best[0],
                                                 "sparse_best": best[1], "mqa_stock_us": m_stock, "mqa_best_us": m_best[0],
                                                 "mqa_best": m_best[1]}
            prev = best_per_bucket.get(bucket)
            # keep the config with the best geometric mean across the contexts seen for this bucket
            best_per_bucket.setdefault(bucket, []).append((ctx, best[1], best[0], t_stock))
    profile = {"sparse": {}, "mqa": {"tiles_small": 1, "tiles_large": 8, "num_warps": 2}, "_results": results,
               "_device": f"{p.name} SMs={p.multi_processor_count}"}
    for bucket, entries in best_per_bucket.items():
        # pick the most frequently winning config; ties -> the one with the largest speedup
        from collections import Counter
        cnt = Counter(e[1] for e in entries)
        profile["sparse"][bucket] = list(cnt.most_common(1)[0][0])
    for b in ("small", "lt32", "le256", "le512", "large"):
        profile["sparse"].setdefault(b, list({"small": (16, 64, 4), "lt32": (16, 32, 4), "le256": (64, 8, 2),
                                              "le512": (64, 4, 2), "large": (64, 1, 2)}[b]))
    json.dump(profile, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
