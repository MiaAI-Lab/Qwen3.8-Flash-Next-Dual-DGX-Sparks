#!/usr/bin/env python3
"""Verify a hybrid NVFP4+FP8-dense snapshot without a GPU.

Checks: index/weight_map consistency, every quantized tensor has an F8_E4M3 weight and
a matching per-channel F32 weight_scale, hard-linked shards are byte-identical to the
source (inode equality), and dequantization error on a sample of tensors is small.
"""
import argparse
import json
import os
import re
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
    ap.add_argument("--replaced-shard-prefix", default=None, metavar="PREFIX",
                    help="source shard files starting with PREFIX were deliberately replaced "
                         "(e.g. model-plefp8- after files/ple_nvfp4): skip their hard-link and "
                         "tensor-presence checks; files/ple_nvfp4/verify_ple_nvfp4_checkpoint.py covers them")
    a = ap.parse_args()
    replaced = {f for f in set(json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"].values())
                if a.replaced_shard_prefix and f.startswith(a.replaced_shard_prefix)}
    src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    dst_idx = json.load(open(os.path.join(a.dst, "model.safetensors.index.json")))["weight_map"]
    full_cfg = json.load(open(os.path.join(a.dst, "config.json")))
    cfg = full_cfg["quantization_config"]
    ql = cfg["quantized_layers"]
    num_hidden_layers = int(full_cfg.get("text_config", full_cfg)["num_hidden_layers"])
    def is_runtime_alias(k):  # mtp.layers.<num_hidden_layers+i>.*: vLLM prefix, no tensor of its own
        m = re.match(r"^mtp\.layers\.(\d+)\.", k)
        return bool(m) and int(m.group(1)) >= num_hidden_layers
    fp8_layers = {k for k, v in ql.items() if v["quant_algo"] == "FP8_PER_CHANNEL_PER_TOKEN"
                  and not is_runtime_alias(k)}
    # MTP routed experts re-emitted per expert as block FP8 (checkpoint-index keys only)
    block_bases = {k: v for k, v in ql.items() if v["quant_algo"] == "FP8_BLOCK_SCALES"
                   and k.endswith(".mlp.experts") and f"{k}.0.gate_proj.weight" in dst_idx}
    problems = 0

    # 0. metadata: MTP entries need the vLLM runtime-index alias; no blanket mtp.* exclusion
    mtp_keys = [k for k in ql if re.match(r"^mtp\.layers\.(\d+)\.", k) and int(re.match(r"^mtp\.layers\.(\d+)\.", k).group(1)) < num_hidden_layers]
    for k in mtp_keys:
        m = re.match(r"^mtp\.layers\.(\d+)\.", k)
        alias = f"mtp.layers.{num_hidden_layers + int(m.group(1))}." + k[m.end():]
        if ql.get(alias) != ql[k]:
            print("MISSING ALIAS", alias); problems += 1
    if any(k.startswith("mtp.") for k in ql) and any(x in ("mtp.*", "model.mtp.*") for x in cfg["ignore"]):
        print("BAD IGNORE: mtp.* excluded although MTP layers are quantized"); problems += 1
    for k, v in block_bases.items():
        if v.get("group_size") != 128:
            print("BAD BLOCK group_size", k, v); problems += 1
    print(f"metadata ok: {len(mtp_keys)} MTP entries aliased, {len(block_bases)} block-FP8 MoE layers")

    # 1. every source tensor still exists (or was replaced by weight+scale / per-expert tensors)
    expanded = set()
    for base in block_bases:
        for fused in ("gate_up_proj", "down_proj"):
            expanded.add(f"{base}.{fused}")
    for name in src_idx:
        if name not in dst_idx and name not in expanded and src_idx[name] not in replaced:
            print("MISSING", name); problems += 1
    for lay in fp8_layers:
        for suffix in (".weight", ".weight_scale"):
            if lay + suffix not in dst_idx:
                print("MISSING", lay + suffix); problems += 1

    # 2. untouched shards are the same inode (hard link)
    same = 0
    for f in sorted(set(src_idx.values())):
        if f.startswith("model-bf16-") or f in replaced:
            continue
        s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
        if os.path.exists(d) and os.stat(s).st_ino == os.stat(d).st_ino:
            same += 1
        else:
            print("NOT HARDLINKED", f); problems += 1
    print(f"hard-linked shards ok: {same}" + (f" ({len(replaced)} replaced shards skipped)" if replaced else ""))

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

    # 3b. per-channel MTP dense tensors: all of them (few, small) get a dequant check
    for lay in sorted(l for l in fp8_layers if l.startswith("mtp.")):
        name = lay + ".weight"
        f = dst_idx[name]
        hdr, data_start = header(os.path.join(a.dst, f))
        src_hdr, src_start = header(os.path.join(a.src, src_idx[name]))
        info, sc = hdr[name], hdr[lay + ".weight_scale"]
        s0, s1 = info["data_offsets"]; c0, c1 = sc["data_offsets"]
        with open(os.path.join(a.dst, f), "rb") as fh:
            fh.seek(data_start + s0); q = torch.frombuffer(bytearray(fh.read(s1 - s0)), dtype=torch.float8_e4m3fn).view(info["shape"])
            fh.seek(data_start + c0); scale = torch.frombuffer(bytearray(fh.read(c1 - c0)), dtype=torch.float32)
        o0, o1 = src_hdr[name]["data_offsets"]
        with open(os.path.join(a.src, src_idx[name]), "rb") as fh:
            fh.seek(src_start + o0); x = torch.frombuffer(bytearray(fh.read(o1 - o0)), dtype=torch.bfloat16).view(info["shape"]).float()
        deq = q.float() * scale[:, None]
        rel = ((deq - x).norm() / x.norm()).item()
        print(f"mtp dense {name}: rel_err={rel:.4f} shape={info['shape']}")
        if rel > 0.06:
            print("HIGH ERROR", name); problems += 1

    # 4. block-FP8 MTP experts: dtype/shape of every per-expert tensor, dequant error on a sample
    for base in sorted(block_bases):
        n_exp = 0
        while f"{base}.{n_exp}.gate_proj.weight" in dst_idx:
            n_exp += 1
        shard = dst_idx[f"{base}.0.gate_proj.weight"]
        hdr, data_start = header(os.path.join(a.dst, shard))
        src_hdr, src_start = header(os.path.join(a.src, src_idx[f"{base}.gate_up_proj"]))
        gu_shape = src_hdr[f"{base}.gate_up_proj"]["shape"]   # [E, 2I, H]
        dn_shape = src_hdr[f"{base}.down_proj"]["shape"]      # [E, H, I]
        E, I2, H = gu_shape; I = I2 // 2
        if n_exp != E or dn_shape != [E, H, I]:
            print("BAD EXPERT COUNT/SHAPE", base, n_exp, gu_shape, dn_shape); problems += 1
        want = {"gate_proj": [I, H], "up_proj": [I, H], "down_proj": [H, I]}
        bad = 0
        for i in range(n_exp):
            for proj, shp in want.items():
                w = hdr.get(f"{base}.{i}.{proj}.weight"); sc = hdr.get(f"{base}.{i}.{proj}.weight_scale_inv")
                if (w is None or sc is None or w["dtype"] != "F8_E4M3" or w["shape"] != shp
                        or sc["dtype"] != "F32" or sc["shape"] != [shp[0] // 128, shp[1] // 128]
                        or dst_idx.get(f"{base}.{i}.{proj}.weight") != shard):
                    bad += 1
        print(f"block-fp8 experts {base}: {n_exp} experts x 3 projections checked, {bad} bad")
        problems += bad
        # sample experts: dequantize and compare to the fused bf16 source slice
        gu_off = src_hdr[f"{base}.gate_up_proj"]["data_offsets"][0]
        dn_off = src_hdr[f"{base}.down_proj"]["data_offsets"][0]
        worst = 0.0
        for i in [0, 1, n_exp // 2, n_exp - 1]:
            with open(os.path.join(a.src, src_idx[f"{base}.gate_up_proj"]), "rb") as fh:
                fh.seek(src_start + gu_off + i * I2 * H * 2)
                gu = torch.frombuffer(bytearray(fh.read(I2 * H * 2)), dtype=torch.bfloat16).view(I2, H).float()
                fh.seek(src_start + dn_off + i * H * I * 2)
                dn = torch.frombuffer(bytearray(fh.read(H * I * 2)), dtype=torch.bfloat16).view(H, I).float()
            ref = {"gate_proj": gu[:I], "up_proj": gu[I:], "down_proj": dn}
            for proj, x in ref.items():
                w = hdr[f"{base}.{i}.{proj}.weight"]; sc = hdr[f"{base}.{i}.{proj}.weight_scale_inv"]
                with open(os.path.join(a.dst, shard), "rb") as fh:
                    fh.seek(data_start + w["data_offsets"][0]); q = torch.frombuffer(bytearray(fh.read(w["data_offsets"][1] - w["data_offsets"][0])), dtype=torch.float8_e4m3fn).view(w["shape"])
                    fh.seek(data_start + sc["data_offsets"][0]); s = torch.frombuffer(bytearray(fh.read(sc["data_offsets"][1] - sc["data_offsets"][0])), dtype=torch.float32).view(sc["shape"])
                r, c = w["shape"]
                deq = (q.float().view(r // 128, 128, c // 128, 128) * s[:, None, :, None]).reshape(r, c)
                rel = ((deq - x).norm() / x.norm()).item()
                worst = max(worst, rel)
                if rel > 0.06:
                    print("HIGH ERROR", f"{base}.{i}.{proj}", rel); problems += 1
        print(f"  sampled experts 0,1,{n_exp // 2},{n_exp - 1}: worst rel_err={worst:.4f}")
    print("PROBLEMS:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
