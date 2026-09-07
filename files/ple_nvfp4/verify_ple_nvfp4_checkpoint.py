#!/usr/bin/env python3
"""Verify an NVFP4-PLE snapshot (built by make_ple_nvfp4_checkpoint.py) without a GPU.

Checks
  1. config: text_config.ple_embedding_dtype == "nvfp4"; quantization_config identical
     to the source except the comment (dense/MTP/expert dispatch unchanged).
  2. index: every non-PLE source tensor still present with identical dtype/shape and
     living in the same (hard-linked) file; every PLE shard replaced by U8 codes
     [rows, D/2] + F8_E4M3 block scales [rows, D/16]; exactly one F32 weight_scale_2;
     the old global BF16 `ngram_embedding.weight_scale` is gone (it would collide
     with the per-row block-scale parameter in the NVFP4 loader).
  3. every non-PLE safetensors file is the same inode as the source (hard link).
  4. dequantization error on sampled row ranges of several shards, using the
     runtime's own _dequant_nvfp4_rows from files/ple_layer_patched.py when it can
     be imported (falls back to an inline copy of the same formula), against the
     FP8 source rows. Also checks block scales never exceed 448 / are finite.
Exit code 1 on any problem.
"""
import argparse
import json
import os
import re
import struct
import sys

import torch

FP8 = torch.float8_e4m3fn
BLOCK = 16
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def read(path, base, info):
    s, e = info["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + s)
        return bytearray(fh.read(e - s))


def load_runtime_dequant(path):
    """exec the runtime's _dequant_nvfp4_codes/_rows + _FP4_VALUES out of the patched
    ple_layer.py (it has relative imports, so it cannot be imported as a module)."""
    src = open(path).read()
    start = src.index("_FP4_VALUES = (")
    end = src.index("def _get_shared_nvfp4_outer_scale")
    ns = {"torch": torch, "_NVFP4_BLOCK_SIZE": BLOCK}
    exec(compile(src[start:end], path, "exec"), ns)
    return ns["_dequant_nvfp4_rows"], tuple(ns["_FP4_VALUES"])


def inline_dequant_rows(packed_rows, head_dim, scale_2, output_dtype, lut):
    half = head_dim // 2
    packed = packed_rows[..., :half]
    scale = packed_rows[..., half:].contiguous().view(FP8)
    codes = torch.stack([packed & 0xF, (packed >> 4) & 0xF], dim=-1).reshape(*packed.shape[:-1], half * 2)
    magnitude = lut[(codes & 0x7).long()]
    sign = ((codes >> 3) & 1).to(torch.float32)
    fp4 = (magnitude * (1 - 2 * sign)).reshape(*packed.shape[:-1], -1, BLOCK)
    out = fp4 * scale.float().unsqueeze(-1) * scale_2.float()
    return out.reshape(*packed.shape[:-1], half * 2).to(output_dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="FP8-PLE source snapshot")
    ap.add_argument("--dst", required=True, help="NVFP4-PLE snapshot")
    ap.add_argument("--shards", default="0,1,37,64,100,127")
    ap.add_argument("--rows", type=int, default=200000, help="rows sampled per shard")
    ap.add_argument("--max-rel", type=float, default=0.12)
    ap.add_argument("--ple-layer", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ple_layer_patched.py"))
    a = ap.parse_args()
    problems = 0

    src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    dst_idx = json.load(open(os.path.join(a.dst, "model.safetensors.index.json")))["weight_map"]
    src_cfg = json.load(open(os.path.join(a.src, "config.json")))
    dst_cfg = json.load(open(os.path.join(a.dst, "config.json")))
    prefixes = sorted({k[: k.index(".shard_")] for k in src_idx if ".ngram_embedding.shard_" in k})
    assert len(prefixes) == 1, prefixes
    p = prefixes[0]

    # 1. config
    if dst_cfg.get("text_config", dst_cfg).get("ple_embedding_dtype") != "nvfp4":
        print("BAD CONFIG: text_config.ple_embedding_dtype != nvfp4"); problems += 1
    sq = dict(src_cfg["quantization_config"]); dq = dict(dst_cfg["quantization_config"])
    sq.pop("comment", None); dq.pop("comment", None)
    if sq != dq:
        print("BAD CONFIG: quantization_config changed beyond the comment"); problems += 1
    stc = dict(src_cfg["text_config"]); dtc = dict(dst_cfg["text_config"])
    stc.pop("ple_embedding_dtype"); dtc.pop("ple_embedding_dtype")
    if stc != dtc:
        print("BAD CONFIG: text_config changed beyond ple_embedding_dtype"); problems += 1
    print("config ok: ple_embedding_dtype=nvfp4, quantization_config unchanged")

    # 2. index
    shard_re = re.compile(re.escape(p) + r"\.shard_(\d+)\.(weight|weight_scale)$")
    hdrs = {}
    def H(root, f):
        key = os.path.join(root, f)
        if key not in hdrs:
            hdrs[key] = header(key)
        return hdrs[key]
    n_src_shards = 0
    D = None
    for name, f in src_idx.items():
        m = shard_re.match(name)
        if m and m.group(2) == "weight":
            n_src_shards += 1
            info = H(a.src, f)[0][name]
            D = info["shape"][1]
            i = int(m.group(1))
            rows = info["shape"][0]
            for suffix, dtype, shape in ((".weight", "U8", [rows, D // 2]), (".weight_scale", "F8_E4M3", [rows, D // BLOCK])):
                dn = f"{p}.shard_{i}{suffix}"
                if dn not in dst_idx:
                    print("MISSING", dn); problems += 1; continue
                dinfo = H(a.dst, dst_idx[dn])[0].get(dn)
                if dinfo is None or dinfo["dtype"] != dtype or dinfo["shape"] != shape:
                    print("BAD PLE ENTRY", dn, dinfo, "expected", dtype, shape); problems += 1
            continue
        if name == f"{p}.weight_scale":
            if name in dst_idx:
                print("STALE global FP8 weight_scale still in index:", name); problems += 1
            continue
        if name not in dst_idx:
            print("MISSING", name); problems += 1; continue
        if dst_idx[name] != f:
            print("MOVED", name, f, "->", dst_idx[name]); problems += 1; continue
        sinfo = H(a.src, f)[0][name]; dinfo = H(a.dst, f)[0].get(name)
        if dinfo is None or (sinfo["dtype"], sinfo["shape"]) != (dinfo["dtype"], dinfo["shape"]):
            print("CHANGED UNTOUCHED TENSOR", name); problems += 1
    ws2 = f"{p}.weight_scale_2"
    if ws2 not in dst_idx:
        print("MISSING", ws2); problems += 1
    else:
        info = H(a.dst, dst_idx[ws2])[0][ws2]
        if info["dtype"] != "F32" or info["shape"] not in ([], [1]):
            print("BAD weight_scale_2", info); problems += 1
    n_dst_shards = sum(1 for k in dst_idx if shard_re.match(k) and k.endswith(".weight"))
    print(f"index ok: {n_src_shards} FP8 shards -> {n_dst_shards} NVFP4 shard pairs, D={D}")
    if n_src_shards != n_dst_shards:
        problems += 1
    # every index entry exists in its file
    for name, f in dst_idx.items():
        if name not in H(a.dst, f)[0]:
            print("INDEX ENTRY NOT IN FILE", name, f); problems += 1

    # 3. hard links
    same = 0
    for f in sorted(set(src_idx.values())):
        if f.startswith("model-plefp8-"):
            continue
        s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
        if os.path.exists(d) and os.stat(s).st_ino == os.stat(d).st_ino:
            same += 1
        else:
            print("NOT HARDLINKED", f); problems += 1
    print(f"hard-linked shards ok: {same}")

    # 4. dequant error, using the runtime's own code when importable
    deq_rows = None
    try:
        deq_rows, vals = load_runtime_dequant(a.ple_layer)
        assert vals == FP4_VALUES, vals
        print("dequant: using runtime _dequant_nvfp4_rows exec'd from", a.ple_layer)
    except Exception as exc:  # noqa: BLE001
        print(f"dequant: runtime source unavailable ({type(exc).__name__}: {str(exc)[:120]}); using inline copy")
        deq_rows = inline_dequant_rows
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32)
    byte2val = torch.arange(256, dtype=torch.uint8).view(FP8).float()
    # scales
    sws_name = f"{p}.weight_scale"
    sh, sb = H(a.src, src_idx[sws_name]); sinfo = sh[sws_name]
    raw = read(os.path.join(a.src, src_idx[sws_name]), sb, sinfo)
    ws = (torch.frombuffer(raw, dtype=torch.bfloat16) if sinfo["dtype"] == "BF16" else torch.frombuffer(raw, dtype=torch.float32)).float().item()
    dh, db = H(a.dst, dst_idx[ws2]); ws2_v = torch.frombuffer(read(os.path.join(a.dst, dst_idx[ws2]), db, dh[ws2]), dtype=torch.float32).reshape(())
    if not torch.isfinite(ws2_v) or ws2_v.item() <= 0:
        print("BAD weight_scale_2 value", ws2_v); problems += 1
    print(f"global scales: fp8 weight_scale={ws:.6g} nvfp4 weight_scale_2={ws2_v.item():.6g} (ratio {ws2_v.item()/ws:.4f})")
    worst = 0.0
    for i in [int(x) for x in a.shards.split(",")]:
        wn, sn, fn = f"{p}.shard_{i}.weight", f"{p}.shard_{i}.weight_scale", f"{p}.shard_{i}.weight"
        if wn not in dst_idx or sn not in dst_idx:
            continue
        dh, db = H(a.dst, dst_idx[wn]); sh, sb = H(a.src, src_idx[fn])
        rows = sh[fn]["shape"][0]
        n = min(a.rows, rows)
        r0 = (rows - n) // 2 if i % 2 else 0     # middle of odd shards, start of even ones
        with open(os.path.join(a.dst, dst_idx[wn]), "rb") as fh:
            fh.seek(db + dh[wn]["data_offsets"][0] + r0 * (D // 2)); codes = torch.frombuffer(bytearray(fh.read(n * (D // 2))), dtype=torch.uint8).view(n, D // 2)
            fh.seek(db + dh[sn]["data_offsets"][0] + r0 * (D // BLOCK)); scales = torch.frombuffer(bytearray(fh.read(n * (D // BLOCK))), dtype=torch.uint8).view(n, D // BLOCK)
        with open(os.path.join(a.src, src_idx[fn]), "rb") as fh:
            fh.seek(sb + sh[fn]["data_offsets"][0] + r0 * D); ref = byte2val[torch.frombuffer(bytearray(fh.read(n * D)), dtype=torch.uint8).long()].view(n, D) * ws
        sf = scales.view(FP8).float()
        if not torch.isfinite(sf).all() or sf.max().item() > 448:
            print("BAD BLOCK SCALES in shard", i, sf.max().item()); problems += 1
        packed_rows = torch.cat([codes, scales], dim=-1)      # exactly what the runtime lookup returns
        out = deq_rows(packed_rows, D, ws2_v, torch.float32, lut)
        rel = ((out - ref).double().norm() / ref.double().norm()).item()
        cos = torch.nn.functional.cosine_similarity(out.double().flatten(), ref.double().flatten(), dim=0).item()
        mx = (out - ref).abs().max().item()
        worst = max(worst, rel)
        print(f"shard {i:3d} rows {r0}..{r0+n}: rel_rmse={rel:.4f} cos={cos:.6f} max_abs_err={mx:.4g} (ref absmax {ref.abs().max().item():.4g})")
        if rel > a.max_rel:
            print("HIGH ERROR shard", i); problems += 1
    stats_path = os.path.join(a.dst, "ple_nvfp4_quant_stats.json")
    if os.path.exists(stats_path):
        st = json.load(open(stats_path))
        errs = sorted(v["rel_rmse"] for k, fs in st.items() if k != "global" for v in fs.values())
        if errs:
            print(f"build stats: {len(errs)} shards rel_rmse min/median/max = {errs[0]:.4f} / {errs[len(errs)//2]:.4f} / {errs[-1]:.4f}")
        if len(errs) != n_src_shards:
            print("build stats cover", len(errs), "shards, expected", n_src_shards); problems += 1
    print("PROBLEMS:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
