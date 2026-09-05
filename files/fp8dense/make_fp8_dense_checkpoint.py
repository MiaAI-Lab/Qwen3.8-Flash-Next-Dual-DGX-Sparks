#!/usr/bin/env python3
"""Build a hybrid checkpoint: RadixArk NVFP4 routed experts + FP8 (per-output-channel,
E4M3) dense projections, packaged as a ModelOpt MIXED_PRECISION checkpoint for vLLM.

Why: on 2x DGX Spark at batch 1 the decode step is memory-bandwidth bound and the
bf16 dense projections (GDN, attention, HyperConnection, shared expert, lm_head) are
~6.9 of the ~9.2 GiB streamed per step. Per-channel FP8 halves those bytes without
calibration data (activations are quantized dynamically per token at runtime).

The script is streaming and memory-frugal (row chunks), so it can run next to a
live vLLM instance. It never touches the expert / PLE shards: those are hard-linked.

Usage (inside the vLLM image, CPU only):
  python3 make_fp8_dense_checkpoint.py --src <snapshot dir> --dst <snapshot dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import struct
import sys
import time

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8).max)  # 448.0
ROW_CHUNK_BYTES = 16 * 1024 * 1024  # bf16 bytes per processed chunk

# Dense projections that become FP8_PER_CHANNEL_PER_TOKEN. Everything else in the
# bf16 shards is copied verbatim (norms, conv1d, A_log, dt_bias, in_proj_a/b, router
# gate, shared_expert_gate, indexer, PLE projections, embeddings, vision, MTP).
QUANT_PATTERNS = [
    "model.language_model.layers.*.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.*.linear_attn.in_proj_z.weight",
    "model.language_model.layers.*.linear_attn.out_proj.weight",
    "model.language_model.layers.*.self_attn.q_proj.weight",
    "model.language_model.layers.*.self_attn.k_proj.weight",
    "model.language_model.layers.*.self_attn.v_proj.weight",
    "model.language_model.layers.*.self_attn.o_proj.weight",
    "model.language_model.layers.*.attn_hyper_connection.input_mix_weight_down.weight",
    "model.language_model.layers.*.attn_hyper_connection.block_inject_weight.weight",
    "model.language_model.layers.*.attn_hyper_connection.input_mix_weight_up.weight",
    "model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_down.weight",
    "model.language_model.layers.*.mlp_hyper_connection.block_inject_weight.weight",
    "model.language_model.layers.*.mlp_hyper_connection.input_mix_weight_up.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
    "model.language_model.layers.*.mlp.shared_expert.gate_proj.weight",
    "model.language_model.layers.*.mlp.shared_expert.up_proj.weight",
    "model.language_model.layers.*.mlp.shared_expert.down_proj.weight",
    "lm_head.weight",
]

# Modules that stay bf16 and must be declared excluded for the ModelOpt loader.
EXCLUDE_MODULES = [
    "model.language_model.embed_tokens",
    "model.embed_tokens",
    "mtp.*",
    "model.mtp.*",
    "model.visual.*",
    "*.ple.*",
    "*.mlp.gate",
    "*.mlp.shared_expert_gate",
    "*.self_attn.indexer.*",
    "*.self_attn.q_norm",
    "*.self_attn.k_norm",
    "*.linear_attn.in_proj_a",
    "*.linear_attn.in_proj_b",
    "*.linear_attn.in_proj_ba",
    "*.linear_attn.conv1d",
    "*.linear_attn.norm",
    "*.hc_norm",
    "model.language_model.hyper_connection_mixer.block_inject_weight",
]

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I64": 8, "I32": 4, "BOOL": 1}
TORCH_DTYPE = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
               "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "I64": torch.int64,
               "I32": torch.int32, "BOOL": torch.bool}


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    return hdr, 8 + n


def matches_quant(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in QUANT_PATTERNS)


class SafetensorsWriter:
    """Streaming safetensors writer: header first (sizes known up-front), then data."""

    def __init__(self, path: str, entries: list[tuple[str, str, list[int]]], metadata: dict | None):
        self.path = path
        header: dict = {}
        offset = 0
        self.offsets: dict[str, tuple[int, int]] = {}
        for name, dtype, shape in entries:
            nbytes = DTYPE_BYTES[dtype]
            for d in shape:
                nbytes *= d
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
            self.offsets[name] = (offset, offset + nbytes)
            offset += nbytes
        if metadata:
            header["__metadata__"] = metadata
        hbytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
        pad = (8 - len(hbytes) % 8) % 8
        hbytes += b" " * pad
        self.fh = open(path + ".tmp", "wb")
        self.fh.write(struct.pack("<Q", len(hbytes)))
        self.fh.write(hbytes)
        self.data_start = 8 + len(hbytes)
        self.total = offset
        self.written: dict[str, int] = {}

    def write(self, name: str, chunk: torch.Tensor) -> None:
        start, end = self.offsets[name]
        pos = self.written.get(name, 0)
        buf = chunk.contiguous().view(torch.uint8).numpy().tobytes() if chunk.dtype != torch.uint8 else chunk.contiguous().numpy().tobytes()
        assert start + pos + len(buf) <= end, name
        self.fh.seek(self.data_start + start + pos)
        self.fh.write(buf)
        self.written[name] = pos + len(buf)

    def close(self) -> None:
        for name, (start, end) in self.offsets.items():
            assert self.written.get(name, 0) == end - start, f"short write for {name}"
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        os.replace(self.path + ".tmp", self.path)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric FP8 E4M3: w = round(x / s), s = amax/448."""
    xf = x.float()
    amax = xf.abs().amax(dim=1)
    scale = (amax / FP8_MAX).clamp_min(1e-12)
    q = (xf / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale


def convert_shard(src_path: str, dst_path: str, stats: dict, dry_run: bool) -> list[tuple[str, str]]:
    hdr, data_start = read_header(src_path)
    meta = hdr.pop("__metadata__", None)
    entries: list[tuple[str, str, list[int]]] = []
    plan: list[tuple[str, dict, bool]] = []
    for name, info in hdr.items():
        q = matches_quant(name)
        if q:
            assert info["dtype"] == "BF16" and len(info["shape"]) == 2, (name, info)
            entries.append((name, "F8_E4M3", info["shape"]))
            entries.append((name[: -len(".weight")] + ".weight_scale", "F32", [info["shape"][0]]))
        else:
            entries.append((name, info["dtype"], info["shape"]))
        plan.append((name, info, q))
    if dry_run:
        return [(n, d) for n, d, _ in entries]

    writer = SafetensorsWriter(dst_path, entries, meta)
    with open(src_path, "rb") as fh:
        for name, info, q in plan:
            s, e = info["data_offsets"]
            shape = info["shape"]
            if not q:
                # verbatim copy in 64 MiB pieces
                fh.seek(data_start + s)
                remaining = e - s
                while remaining:
                    piece = fh.read(min(remaining, 64 << 20))
                    writer.write(name, torch.frombuffer(bytearray(piece), dtype=torch.uint8))
                    remaining -= len(piece)
                continue
            rows, cols = shape
            row_bytes = cols * 2
            rows_per_chunk = max(1, ROW_CHUNK_BYTES // row_bytes)
            scale_name = name[: -len(".weight")] + ".weight_scale"
            err_num = 0.0
            err_den = 0.0
            fh.seek(data_start + s)
            for r0 in range(0, rows, rows_per_chunk):
                r1 = min(rows, r0 + rows_per_chunk)
                raw = fh.read((r1 - r0) * row_bytes)
                x = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(r1 - r0, cols)
                qv, sc = quantize_rows(x)
                writer.write(name, qv)
                writer.write(scale_name, sc)
                deq = qv.float() * sc[:, None]
                err_num += (deq - x.float()).pow(2).sum().item()
                err_den += x.float().pow(2).sum().item()
            stats[name] = {"rows": rows, "cols": cols, "rel_rmse": (err_num / max(err_den, 1e-30)) ** 0.5}
    writer.close()
    return [(n, d) for n, d, _ in entries]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="RadixArk NVFP4 snapshot directory")
    ap.add_argument("--dst", required=True, help="output snapshot directory (created)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep already-written bf16 shard replacements")
    args = ap.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    os.makedirs(dst, exist_ok=True)

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    files = sorted(set(index["weight_map"].values()))
    bf16_files = [f for f in files if f.startswith("model-bf16-")]
    other_files = [f for f in files if f not in bf16_files]

    # 1. hard-link untouched shards (experts NVFP4, PLE FP8) + tokenizer/config extras
    linked = 0
    link_names = other_files + [x for x in os.listdir(src)
                                if not x.endswith(".safetensors") and x not in
                                ("config.json", "hf_quant_config.json", "model.safetensors.index.json")]
    for f in link_names:
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.isdir(s):
            continue
        # HF cache snapshots are symlinks into ../../blobs; link the *blob* itself,
        # otherwise a hard link of the relative symlink dangles in the new repo dir.
        target = os.path.realpath(s)
        if os.path.lexists(d):
            if os.path.exists(d) and os.path.samefile(d, target):
                linked += 1
                continue
            if not args.dry_run:
                os.unlink(d)  # stale/dangling entry from an earlier run
        if not args.dry_run:
            try:
                os.link(target, d)
            except OSError as exc:  # e.g. cross-device or protected_hardlinks
                print(f"  hardlink failed for {f} ({exc}); using symlink", flush=True)
                os.symlink(target, d)
            if not os.path.exists(d):
                raise RuntimeError(f"could not link {f} into {dst}")
        linked += 1
    print(f"linked {linked} untouched files ({len(link_names)} expected)", flush=True)

    # 2. rewrite the bf16 shards
    new_weight_map = {k: v for k, v in index["weight_map"].items() if v not in bf16_files}
    stats: dict = {}
    quantized_layers: dict[str, dict] = {}
    t0 = time.time()
    for f in bf16_files:
        dst_shard = os.path.join(dst, f)
        if args.resume and os.path.exists(dst_shard):
            print(f"resume: keeping existing {f}", flush=True)
            entries = convert_shard(os.path.join(src, f), dst_shard, stats, dry_run=True)
        else:
            print(f"converting {f} ...", flush=True)
            entries = convert_shard(os.path.join(src, f), dst_shard, stats, args.dry_run)
        for name, dtype in entries:
            new_weight_map[name] = f
            if dtype == "F8_E4M3" and name.endswith(".weight") and matches_quant(name):
                quantized_layers[name[: -len(".weight")]] = {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}
        print(f"  done ({time.time() - t0:.0f}s elapsed)", flush=True)

    # routed experts stay NVFP4: one entry per MoE layer using the vLLM RoutedExperts prefix
    layer_ids = sorted({int(m.group(1)) for k in index["weight_map"]
                        if (m := re.match(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.", k))})
    for i in layer_ids:
        quantized_layers[f"model.language_model.layers.{i}.mlp.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
        # ModelOpt-style child keys so prefix lookups also resolve
        for child in ("gate_up_proj", "down_proj"):
            quantized_layers[f"model.language_model.layers.{i}.mlp.experts.{child}"] = {"quant_algo": "NVFP4", "group_size": 16}

    # 3. configs
    cfg = json.load(open(os.path.join(src, "config.json")))
    qc = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "producer": {"name": "modelopt", "version": "0.46.0"},
        "ignore": EXCLUDE_MODULES,
        "quantized_layers": quantized_layers,
        "comment": "NVFP4 routed experts from RadixArk/Qwen3.8-Flash-Next-NVFP4; dense projections "
                   "requantized to FP8 E4M3 per-output-channel (dynamic per-token activations) by "
                   "files/fp8dense/make_fp8_dense_checkpoint.py",
    }
    cfg["quantization_config"] = qc
    hfq = {
        "producer": {"name": "modelopt", "version": "0.46.0"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "group_size": 16,
            "exclude_modules": EXCLUDE_MODULES,
            "quantized_layers": quantized_layers,
        },
    }
    if not args.dry_run:
        json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
        json.dump(hfq, open(os.path.join(dst, "hf_quant_config.json"), "w"), indent=2)
        total = 0
        for f in sorted(set(new_weight_map.values())):
            total += os.path.getsize(os.path.join(dst, f))
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_weight_map.items()))},
                  open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
        stats_path = os.path.join(dst, "fp8_dense_quant_stats.json")
        if stats or not os.path.exists(stats_path):
            json.dump(stats, open(stats_path, "w"), indent=1)
    nq = sum(1 for v in quantized_layers.values() if v["quant_algo"].startswith("FP8"))
    print(f"quantized {nq} dense linears to FP8 per-channel; {len(layer_ids)} MoE layers kept NVFP4")
    if stats:
        worst = sorted(stats.items(), key=lambda kv: -kv[1]["rel_rmse"])[:8]
        print("worst relative RMSE (dequant vs bf16):")
        for n, s in worst:
            print(f"  {s['rel_rmse']:.4f}  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
