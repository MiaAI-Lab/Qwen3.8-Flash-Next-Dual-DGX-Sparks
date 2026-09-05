#!/usr/bin/env python3
"""Verify a hybrid NVFP4+FP8-dense snapshot without a GPU.

Checks: index/weight_map consistency, every quantized tensor has an F8_E4M3 weight and
a matching per-channel F32 weight_scale, hard-linked shards are byte-identical to the
source (inode equality), and dequantization error on a sample of tensors is small.
"""
import argparse
import json
import os
import struct
import sys

import torch


def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--samples", type=int, default=6)
    a = ap.parse_args()
    src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    dst_idx = json.load(open(os.path.join(a.dst, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(a.dst, "config.json")))["quantization_config"]
    ql = cfg["quantized_layers"]
    fp8_layers = {k for k, v in ql.items() if v["quant_algo"] == "FP8_PER_CHANNEL_PER_TOKEN"}
    problems = 0

    # 1. every source tensor still exists (or was replaced by weight+scale)
    for name in src_idx:
        if name not in dst_idx:
            print("MISSING", name); problems += 1
    for lay in fp8_layers:
        for suffix in (".weight", ".weight_scale"):
            if lay + suffix not in dst_idx:
                print("MISSING", lay + suffix); problems += 1

    # 2. untouched shards are the same inode (hard link)
    same = 0
    for f in sorted(set(src_idx.values())):
        if f.startswith("model-bf16-"):
            continue
        s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
        if os.path.exists(d) and os.stat(s).st_ino == os.stat(d).st_ino:
            same += 1
        else:
            print("NOT HARDLINKED", f); problems += 1
    print(f"hard-linked shards ok: {same}")

    # 3. dtype/shape checks + dequant error samples in the rewritten shards
    checked = 0
    samples = 0
    for f in sorted({v for k, v in dst_idx.items() if v.startswith("model-bf16-")}):
        hdr, data_start = header(os.path.join(a.dst, f))
        src_hdr, src_start = header(os.path.join(a.src, f))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            base = name[: -len(".weight")] if name.endswith(".weight") else None
            if base in fp8_layers and name.endswith(".weight"):
                sc = hdr.get(base + ".weight_scale")
                if info["dtype"] != "F8_E4M3" or sc is None or sc["dtype"] != "F32" or sc["shape"] != [info["shape"][0]]:
                    print("BAD QUANT ENTRY", name, info, sc); problems += 1
                checked += 1
                if samples < a.samples and (checked % 97 == 1):
                    # dequantize and compare against source bf16
                    s0, s1 = info["data_offsets"]; c0, c1 = sc["data_offsets"]
                    with open(os.path.join(a.dst, f), "rb") as fh:
                        fh.seek(data_start + s0); q = torch.frombuffer(bytearray(fh.read(s1 - s0)), dtype=torch.float8_e4m3fn).view(info["shape"])
                        fh.seek(data_start + c0); scale = torch.frombuffer(bytearray(fh.read(c1 - c0)), dtype=torch.float32)
                    o0, o1 = src_hdr[name]["data_offsets"]
                    with open(os.path.join(a.src, f), "rb") as fh:
                        fh.seek(src_start + o0); x = torch.frombuffer(bytearray(fh.read(o1 - o0)), dtype=torch.bfloat16).view(info["shape"]).float()
                    deq = q.float() * scale[:, None]
                    rel = ((deq - x).norm() / x.norm()).item()
                    cos = torch.nn.functional.cosine_similarity(deq.flatten(), x.flatten(), dim=0).item()
                    print(f"sample {name}: rel_err={rel:.4f} cos={cos:.6f} shape={info['shape']}")
                    samples += 1
                    if rel > 0.06:
                        print("HIGH ERROR", name); problems += 1
            elif name in src_hdr:
                if (info["dtype"], info["shape"]) != (src_hdr[name]["dtype"], src_hdr[name]["shape"]):
                    print("CHANGED UNTOUCHED TENSOR", name); problems += 1
    print(f"fp8 tensors checked: {checked} (expected {len(fp8_layers)})")
    if checked != len(fp8_layers):
        problems += 1
    print("PROBLEMS:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
