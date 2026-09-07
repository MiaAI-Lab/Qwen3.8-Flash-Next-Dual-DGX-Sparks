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
# mtp.layers.<num_hidden_layers+i>.* entries are vLLM runtime-prefix aliases without tensors of their own
fp8 = sorted(k for k, v in ql.items() if v["quant_algo"] == "FP8_PER_CHANNEL_PER_TOKEN" and k + ".weight" in dst_idx)
missing = sorted(k for k, v in ql.items() if v["quant_algo"] == "FP8_PER_CHANNEL_PER_TOKEN"
                 and k + ".weight" not in dst_idx and not k.startswith("mtp.layers."))
assert not missing, f"quantized_layers entries without tensors: {missing[:5]}"
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
errs = sorted(v["rel_rmse"] for v in stats.values())
print(f"tensors={len(errs)} rel_rmse min={errs[0]:.4f} median={errs[len(errs)//2]:.4f} max={errs[-1]:.4f}")
for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["rel_rmse"])[:5]: print(f"  {v['rel_rmse']:.4f} {k}")
mtp = {k: v for k, v in stats.items() if k.startswith("mtp.")}
if mtp:
    print("MTP dense (per-channel):")
    for k, v in sorted(mtp.items(), key=lambda kv: -kv[1]["rel_rmse"]): print(f"  {v['rel_rmse']:.4f} {k} {v['rows']}x{v['cols']}")

# Block-FP8 MTP routed experts (per-expert weight + weight_scale_inv vs the fused bf16 source)
for base in sorted(k for k, v in ql.items() if v["quant_algo"] == "FP8_BLOCK_SCALES"
                   and k.endswith(".mlp.experts") and f"{k}.0.gate_proj.weight" in dst_idx):
    bs = ql[base]["group_size"]
    shard = dst_idx[f"{base}.0.gate_proj.weight"]; dh, d0 = H(os.path.join(a.dst, shard))
    sh, s0 = H(os.path.join(a.src, src_idx[f"{base}.gate_up_proj"]))
    E, I2, Hd = sh[f"{base}.gate_up_proj"]["shape"]; I = I2 // 2
    gu_off = sh[f"{base}.gate_up_proj"]["data_offsets"][0]; dn_off = sh[f"{base}.down_proj"]["data_offsets"][0]
    acc = {p: [0.0, 0.0, 0.0, -1] for p in ("gate_proj", "up_proj", "down_proj")}  # num, den, worst, worst_expert
    with open(os.path.join(a.dst, shard), "rb") as fd, open(os.path.join(a.src, src_idx[f"{base}.gate_up_proj"]), "rb") as fs:
        for i in range(E):
            fs.seek(s0 + gu_off + i * I2 * Hd * 2); gu = torch.frombuffer(bytearray(fs.read(I2 * Hd * 2)), dtype=torch.bfloat16).view(I2, Hd).float()
            fs.seek(s0 + dn_off + i * Hd * I * 2); dn = torch.frombuffer(bytearray(fs.read(Hd * I * 2)), dtype=torch.bfloat16).view(Hd, I).float()
            for proj, x in (("gate_proj", gu[:I]), ("up_proj", gu[I:]), ("down_proj", dn)):
                w = dh[f"{base}.{i}.{proj}.weight"]; sc = dh[f"{base}.{i}.{proj}.weight_scale_inv"]
                fd.seek(d0 + w["data_offsets"][0]); q = torch.frombuffer(bytearray(fd.read(w["data_offsets"][1] - w["data_offsets"][0])), dtype=torch.float8_e4m3fn).view(w["shape"])
                fd.seek(d0 + sc["data_offsets"][0]); s = torch.frombuffer(bytearray(fd.read(sc["data_offsets"][1] - sc["data_offsets"][0])), dtype=torch.float32).view(sc["shape"])
                r, c = w["shape"]
                deq = (q.float().view(r // bs, bs, c // bs, bs) * s[:, None, :, None]).reshape(r, c)
                num = (deq - x).pow(2).sum().item(); den = x.pow(2).sum().item()
                rel = (num / max(den, 1e-30)) ** 0.5
                acc[proj][0] += num; acc[proj][1] += den
                if rel > acc[proj][2]: acc[proj][2], acc[proj][3] = rel, i
            if i % 128 == 0: print(f"  experts {i}/{E}", flush=True)
    print(f"MTP routed experts {base} ({E} experts, {bs}x{bs} block FP8):")
    for proj, (num, den, worst, wi) in acc.items():
        rel = (num / max(den, 1e-30)) ** 0.5
        stats[f"{base}[{proj}]"] = {"experts": E, "block": bs, "rel_rmse": rel, "worst_expert_rel_rmse": worst, "worst_expert": wi}
        print(f"  {proj:10s} rel_rmse={rel:.4f}  worst expert {wi}: {worst:.4f}")
json.dump(stats, open(os.path.join(a.dst, "fp8_dense_quant_stats.json"), "w"), indent=1)
