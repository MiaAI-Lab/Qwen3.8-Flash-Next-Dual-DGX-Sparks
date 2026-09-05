#!/usr/bin/env python3
"""Recompute per-tensor dequantization error (relative RMSE) for every FP8 tensor of a hybrid
snapshot against the bf16 source, streaming row chunks (CPU, <300 MB working set)."""
import argparse, json, os, struct, sys
import torch

def header(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n

ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
a = ap.parse_args()
dst_idx = json.load(open(os.path.join(a.dst, "model.safetensors.index.json")))["weight_map"]
src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
ql = json.load(open(os.path.join(a.dst, "config.json")))["quantization_config"]["quantized_layers"]
fp8 = sorted(k for k, v in ql.items() if v["quant_algo"] == "FP8_PER_CHANNEL_PER_TOKEN")
stats = {}
hdr_cache = {}
def H(path):
    if path not in hdr_cache: hdr_cache[path] = header(path)
    return hdr_cache[path]
for i, lay in enumerate(fp8):
    name = lay + ".weight"
    dpath = os.path.join(a.dst, dst_idx[name]); spath = os.path.join(a.src, src_idx[name])
    dh, d0 = H(dpath); sh, s0 = H(spath)
    rows, cols = dh[name]["shape"]
    q0, q1 = dh[name]["data_offsets"]; c0, c1 = dh[lay + ".weight_scale"]["data_offsets"]; x0, x1 = sh[name]["data_offsets"]
    with open(dpath, "rb") as fh:
        fh.seek(d0 + c0); scale = torch.frombuffer(bytearray(fh.read(c1 - c0)), dtype=torch.float32)
    num = den = 0.0
    step = max(1, (8 << 20) // cols)
    with open(dpath, "rb") as fd, open(spath, "rb") as fs:
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            fd.seek(d0 + q0 + r0 * cols); q = torch.frombuffer(bytearray(fd.read((r1 - r0) * cols)), dtype=torch.float8_e4m3fn).view(r1 - r0, cols)
            fs.seek(s0 + x0 + r0 * cols * 2); x = torch.frombuffer(bytearray(fs.read((r1 - r0) * cols * 2)), dtype=torch.bfloat16).view(r1 - r0, cols).float()
            deq = q.float() * scale[r0:r1, None]
            num += (deq - x).pow(2).sum().item(); den += x.pow(2).sum().item()
    stats[name] = {"rows": rows, "cols": cols, "rel_rmse": (num / max(den, 1e-30)) ** 0.5}
    if i % 100 == 0: print(f"{i}/{len(fp8)}", flush=True)
json.dump(stats, open(os.path.join(a.dst, "fp8_dense_quant_stats.json"), "w"), indent=1)
errs = sorted(v["rel_rmse"] for v in stats.values())
print(f"tensors={len(errs)} rel_rmse min={errs[0]:.4f} median={errs[len(errs)//2]:.4f} max={errs[-1]:.4f}")
for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["rel_rmse"])[:5]: print(f"  {v['rel_rmse']:.4f} {k}")
