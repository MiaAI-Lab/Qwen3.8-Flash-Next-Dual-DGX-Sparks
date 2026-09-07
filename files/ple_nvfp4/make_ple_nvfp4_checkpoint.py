#!/usr/bin/env python3
"""Requantize the PLE n-gram table of a hybrid snapshot from FP8 to NVFP4.

Why: the FP8 PLE table is 47.7 GiB (40% of the FP8-dense+MTP checkpoint). On ONE
DGX Spark (121.69 GiB unified) that checkpoint does not fit resident, so the
published single-Spark path memory-maps the table from host page cache through a
CPU offload worker. NVFP4 packs the same 320M rows x 160 dims as 4-bit E2M1 codes
(uint8 [rows, 80]) + E4M3 block scales over 16 elements ([rows, 10]) + one fp32
global scale: 90 B/row instead of 160 B -> 26.8 GiB, i.e. ~98.7 GiB total, which
is the size local-inference-lab's NVFP4 checkpoint already loads on one Spark.
The PLE is a SPARSE lookup (a few rows per token), so this buys residency /
capacity, not bandwidth.

Exact layout consumed by the patched vLLM PLE layer (files/patch_ple_layer.py,
Qwen3_8FlashNextPLENVFp4EmbeddingMethod / _dequant_nvfp4_codes):
  <prefix>.ngram_embedding.shard_<i>.weight        U8       [rows, head_dim/2]
      byte j = code[2j] | code[2j+1] << 4 ; code = sign<<3 | index into
      (0, 0.5, 1, 1.5, 2, 3, 4, 6)
  <prefix>.ngram_embedding.shard_<i>.weight_scale  F8_E4M3  [rows, head_dim/16]
  <prefix>.ngram_embedding.weight_scale_2          F32      []   (global)
  dequant = fp4 * weight_scale * weight_scale_2
and the checkpoint must declare text_config.ple_embedding_dtype = "nvfp4".
The old single global `ngram_embedding.weight_scale` ([1] BF16) must NOT be
emitted: the NVFP4 loader routes it to AutoWeightsLoader, where it would collide
with the per-row block-scale parameter of the same name.

Scale convention (ModelOpt NVFP4): weight_scale_2 = amax_global / (6 * 448),
block scale = amax_block / (6 * weight_scale_2) rounded to E4M3 (<= 448 by
construction), codes = RNE(x / (block scale * weight_scale_2)) on the E2M1 grid.
amax_global is measured in a cheap first pass over the raw FP8 bytes.

Everything except the PLE shards is hard-linked from --src (safetensors) or
copied (small json/tokenizer files); config.json and the index are rewritten.
Streaming, CPU-only, ~1 GiB working set: safe next to a live server.

Usage (inside the vLLM image, CUDA_VISIBLE_DEVICES=""):
  make_ple_nvfp4_checkpoint.py --src <fp8dense-mtp snapshot> --dst <new snapshot> [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fp8dense"))
from make_fp8_dense_checkpoint import SafetensorsWriter, read_header  # noqa: E402

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
BLOCK = 16
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
BYTE2VAL = torch.arange(256, dtype=torch.uint8).view(FP8).float()  # NaN at 0x7F / 0xFF
NAN_MAG = 0x7F
ROWS_PER_CHUNK = 65536
PLE_FILE_PREFIX = "model-plefp8-"
OUT_FILE_PREFIX = "model-plenvfp4-"


def e2m1_index_rne(a: torch.Tensor) -> torch.Tensor:
    """|q| in [0, 6] -> index into E2M1, round-to-nearest, ties to even index."""
    hi = torch.bucketize(a, MID, right=True)  # MID[hi-1] <= a < MID[hi]: ties go up
    prev = MID[(hi - 1).clamp(min=0)]
    tie = (hi > 0) & (a == prev)
    return torch.where(tie & ((hi & 1) == 1), hi - 1, hi)


def quantize_chunk(raw: torch.Tensor, g: float):
    """raw: uint8 [R, D] FP8 bytes. g = weight_scale_2 / weight_scale (fp8 units).
    Returns packed codes [R, D/2] u8, block scales [R, D/16] as u8 view, and
    (err_num, err_den, max_abs_err) in fp8 units."""
    R, D = raw.shape
    v = BYTE2VAL[raw.long()]                       # fp8 units; x = v * weight_scale
    blk = v.view(R, D // BLOCK, BLOCK)
    amax = blk.abs().amax(dim=-1)                  # [R, D/16]
    s = (amax / (6.0 * g)).to(FP8)                 # E4M3 block scale, <= 448 by construction
    sf = s.float()
    denom = sf * g
    inv = torch.where(denom > 0, 1.0 / denom, torch.zeros_like(denom))
    q = blk * inv.unsqueeze(-1)
    k = e2m1_index_rne(q.abs().clamp(max=6.0))     # [R, D/16, 16] int64
    neg = q < 0
    code = (k | (neg.to(torch.int64) << 3)).to(torch.uint8).view(R, D)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    mag = E2M1[k]
    deq = (torch.where(neg, -mag, mag) * denom.unsqueeze(-1)).view(R, D)
    diff = (deq - v).double()
    return packed.contiguous(), s.view(torch.uint8).contiguous(), (
        diff.pow(2).sum().item(), v.double().pow(2).sum().item(), diff.abs().max().item())


def fadvise_dontneed(fh, start: int, length: int) -> None:
    try:
        os.posix_fadvise(fh.fileno(), start, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="FP8-dense(+MTP) snapshot with the FP8 PLE table")
    ap.add_argument("--dst", required=True, help="output snapshot directory (created)")
    ap.add_argument("--resume", action="store_true", help="keep complete output PLE shards")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rows-per-chunk", type=int, default=ROWS_PER_CHUNK)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if os.path.realpath(src) == os.path.realpath(dst):
        ap.error("--dst must differ from --src")
    os.makedirs(dst, exist_ok=True)
    t0 = time.time()

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wmap = index["weight_map"]
    ple_files = sorted({f for f in set(wmap.values()) if f.startswith(PLE_FILE_PREFIX)})
    if not ple_files:
        sys.exit(f"no {PLE_FILE_PREFIX}* shards in {src}")
    prefixes = sorted({k[: k.index(".shard_")] for k in wmap if ".ngram_embedding.shard_" in k})
    if len(prefixes) != 1:
        sys.exit(f"expected exactly one PLE table, found {prefixes}")
    prefix = prefixes[0]  # ...ple.ple_embedding.ngram_embedding
    ws_name = f"{prefix}.weight_scale"
    ws2_name = f"{prefix}.weight_scale_2"
    if ws_name not in wmap:
        sys.exit(f"missing global FP8 scale {ws_name}")
    for f in ple_files:  # PLE files must hold nothing but the table
        hdr, _ = read_header(os.path.join(src, f))
        extra = [k for k in hdr if k != "__metadata__" and not k.startswith(prefix + ".")]
        if extra:
            sys.exit(f"{f} carries non-PLE tensors {extra[:3]}; not supported")

    # global FP8 scale
    hdr, base = read_header(os.path.join(src, wmap[ws_name]))
    m = hdr[ws_name]
    with open(os.path.join(src, wmap[ws_name]), "rb") as fh:
        fh.seek(base + m["data_offsets"][0])
        raw = fh.read(m["data_offsets"][1] - m["data_offsets"][0])
    if m["dtype"] == "BF16":
        ws = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().item()
    elif m["dtype"] == "F32":
        ws = torch.frombuffer(bytearray(raw), dtype=torch.float32).item()
    else:
        sys.exit(f"unexpected dtype for {ws_name}: {m['dtype']}")
    assert ws > 0 and ws == ws, ws

    # shard inventory (mirror the source file layout)
    shard_re = re.compile(re.escape(prefix) + r"\.shard_(\d+)\.weight$")
    layout: dict[str, list[int]] = {}
    shapes: dict[int, tuple[int, int]] = {}
    for name, f in wmap.items():
        mm = shard_re.match(name)
        if mm:
            layout.setdefault(f, []).append(int(mm.group(1)))
    n_shards = sum(len(v) for v in layout.values())
    for f in ple_files:
        hdr, _ = read_header(os.path.join(src, f))
        for i in layout[f]:
            info = hdr[f"{prefix}.shard_{i}.weight"]
            if info["dtype"] != "F8_E4M3" or len(info["shape"]) != 2:
                sys.exit(f"shard {i} is {info['dtype']} {info['shape']}, expected F8_E4M3 [rows, dim]")
            shapes[i] = tuple(info["shape"])
    D = shapes[0][1]
    assert D % BLOCK == 0 and all(s[1] == D for s in shapes.values()), shapes
    assert sorted(shapes) == list(range(n_shards)), sorted(shapes)[:5]
    print(f"PLE table {prefix}: {n_shards} shards x {shapes[0][0]} rows x {D} (FP8, global scale {ws:.6g}) "
          f"in {len(ple_files)} files; output {D // 2 + D // BLOCK} B/row", flush=True)

    # pass 1: global amax over raw bytes (E4M3 magnitude is monotonic in the low 7 bits)
    stats_path = os.path.join(dst, "ple_nvfp4_quant_stats.json")
    stats = json.load(open(stats_path)) if (args.resume and os.path.exists(stats_path)) else {}
    if "global" not in stats:
        max_byte = 0
        nan_count = 0
        total = 0
        for f in ple_files:
            hdr, base = read_header(os.path.join(src, f))
            with open(os.path.join(src, f), "rb") as fh:
                for i in layout[f]:
                    s0, s1 = hdr[f"{prefix}.shard_{i}.weight"]["data_offsets"]
                    fh.seek(base + s0)
                    pos = base + s0
                    remaining = s1 - s0
                    while remaining:
                        piece = fh.read(min(remaining, 256 << 20))
                        t = torch.frombuffer(bytearray(piece), dtype=torch.uint8) & 0x7F
                        nan_count += int((t == NAN_MAG).sum())
                        max_byte = max(max_byte, int(t.max()))
                        fadvise_dontneed(fh, pos, len(piece))
                        pos += len(piece)
                        remaining -= len(piece)
                        total += len(piece)
            print(f"  amax pass {f}: max magnitude byte so far 0x{max_byte:02x} "
                  f"({BYTE2VAL[max_byte].item():.1f}), {total / 2**30:.1f} GiB, {time.time() - t0:.0f}s", flush=True)
        if nan_count:
            sys.exit(f"FP8 table contains {nan_count} NaN bytes; refusing")
        amax_v = BYTE2VAL[max_byte].item()
        g = amax_v / (6.0 * FP8_MAX)          # weight_scale_2 in fp8 units
        stats["global"] = {"fp8_weight_scale": ws, "amax_fp8_units": amax_v, "amax": amax_v * ws,
                           "weight_scale_2": g * ws, "block": BLOCK, "head_dim": D,
                           "shards": n_shards, "rows_per_shard": shapes[0][0], "elements": total}
        json.dump(stats, open(stats_path, "w"), indent=1)
    gstat = stats["global"]
    g = gstat["weight_scale_2"] / ws
    print(f"amax = {gstat['amax_fp8_units']:.1f} fp8-units -> weight_scale_2 = {gstat['weight_scale_2']:.6g} "
          f"(= {g:.6g} x weight_scale)", flush=True)

    # pass 2: quantize file by file (mirroring the source file layout)
    new_wmap = {k: v for k, v in wmap.items() if not v.startswith(PLE_FILE_PREFIX)}
    ws2_file = OUT_FILE_PREFIX + wmap[ws_name][len(PLE_FILE_PREFIX):]
    for f in ple_files:
        out_f = OUT_FILE_PREFIX + f[len(PLE_FILE_PREFIX):]
        entries: list[tuple[str, str, list[int]]] = []
        for i in layout[f]:
            rows = shapes[i][0]
            entries.append((f"{prefix}.shard_{i}.weight", "U8", [rows, D // 2]))
            entries.append((f"{prefix}.shard_{i}.weight_scale", "F8_E4M3", [rows, D // BLOCK]))
        if out_f == ws2_file:
            entries.append((ws2_name, "F32", []))
        for name, _, _ in entries:
            new_wmap[name] = out_f
        expected = sum((4 if not sh else int(torch.tensor(sh).prod())) for _, _, sh in entries)
        dst_path = os.path.join(dst, out_f)
        if args.resume and os.path.exists(dst_path) and out_f in stats:
            hdr, base = read_header(dst_path)
            if os.path.getsize(dst_path) == base + expected:
                print(f"resume: keeping {out_f}", flush=True)
                continue
        hdr, base = read_header(os.path.join(src, f))
        meta = hdr.get("__metadata__")
        writer = SafetensorsWriter(dst_path, entries, meta)
        fstat: dict = {}
        with open(os.path.join(src, f), "rb") as fh:
            for i in layout[f]:
                rows = shapes[i][0]
                s0, _ = hdr[f"{prefix}.shard_{i}.weight"]["data_offsets"]
                num = den = 0.0
                mx = 0.0
                t1 = time.time()
                for r0 in range(0, rows, args.rows_per_chunk):
                    r1 = min(rows, r0 + args.rows_per_chunk)
                    fh.seek(base + s0 + r0 * D)
                    raw = torch.frombuffer(bytearray(fh.read((r1 - r0) * D)), dtype=torch.uint8).view(r1 - r0, D)
                    packed, scales, (n_, d_, m_) = quantize_chunk(raw, g)
                    writer.write(f"{prefix}.shard_{i}.weight", packed)
                    writer.write(f"{prefix}.shard_{i}.weight_scale", scales)
                    num += n_
                    den += d_
                    mx = max(mx, m_)
                    fadvise_dontneed(fh, base + s0 + r0 * D, (r1 - r0) * D)
                fstat[f"shard_{i}"] = {"rows": rows, "rel_rmse": (num / max(den, 1e-30)) ** 0.5,
                                       "max_abs_err": mx * ws, "seconds": round(time.time() - t1, 1)}
                print(f"  {out_f} shard {i}: rel_rmse={fstat[f'shard_{i}']['rel_rmse']:.4f} "
                      f"({time.time() - t1:.0f}s, {time.time() - t0:.0f}s total)", flush=True)
        if out_f == ws2_file:
            writer.write(ws2_name, torch.tensor([gstat["weight_scale_2"]], dtype=torch.float32))
        writer.close()
        with open(dst_path, "rb") as fh:
            fadvise_dontneed(fh, 0, os.path.getsize(dst_path))
        stats[out_f] = fstat
        json.dump(stats, open(stats_path, "w"), indent=1)

    # link / copy everything else
    linked = copied = 0
    skip = {"config.json", "model.safetensors.index.json", "ple_nvfp4_quant_stats.json"}
    for x in sorted(os.listdir(src)):
        s = os.path.join(src, x)
        d = os.path.join(dst, x)
        if os.path.isdir(s) or x in skip or x.startswith(PLE_FILE_PREFIX):
            continue
        if x.endswith(".safetensors"):
            target = os.path.realpath(s)
            if os.path.lexists(d):
                if os.path.exists(d) and os.path.samefile(d, target):
                    linked += 1
                    continue
                os.unlink(d)
            try:
                os.link(target, d)
            except OSError as exc:
                print(f"  hardlink failed for {x} ({exc}); using symlink", flush=True)
                os.symlink(target, d)
            linked += 1
        else:
            shutil.copyfile(os.path.realpath(s), d)
            copied += 1
    print(f"linked {linked} safetensors, copied {copied} small files", flush=True)

    # configs
    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = cfg.get("text_config", cfg)
    tc["ple_embedding_dtype"] = "nvfp4"
    qc = cfg.get("quantization_config", {})
    note = ("; PLE n-gram table requantized FP8 -> NVFP4 (E2M1 codes, 16-wide E4M3 block scales, "
            "fp32 global weight_scale_2; text_config.ple_embedding_dtype=nvfp4) by "
            "files/ple_nvfp4/make_ple_nvfp4_checkpoint.py")
    if note.strip("; ") not in qc.get("comment", ""):
        qc["comment"] = qc.get("comment", "") + note
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    total = 0
    for f in sorted(set(new_wmap.values())):
        total += os.path.getsize(os.path.join(dst, f))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_wmap.items()))},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    errs = [v["rel_rmse"] for k, fs in stats.items() if k != "global" for v in fs.values()]
    errs.sort()
    print(f"done in {time.time() - t0:.0f}s: {n_shards} PLE shards -> NVFP4, rel_rmse min/median/max = "
          f"{errs[0]:.4f} / {errs[len(errs) // 2]:.4f} / {errs[-1]:.4f}; index total_size {total / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
