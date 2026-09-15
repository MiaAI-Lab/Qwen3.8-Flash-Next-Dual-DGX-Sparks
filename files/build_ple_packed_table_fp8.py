#!/usr/bin/env python3
"""Build a packed FP8 PLE table file for memory-mapped CPU offload.

Sibling of build_ple_packed_table.py (which is NVFP4-only: it hard-asserts the
weight shards are U8 packed codes plus FP8 block scales, and concatenates
cat(codes, scales) into a 90-byte row).

The FP8 checkpoint needs no packing math: each row is already a contiguous
[head_dim] run of float8_e4m3fn bytes, and the 128 shards are equal-sized and
contiguous, so the packed file is simply the shard bytes concatenated in shard
order. Row r of shard i lands at index i*rows_per_shard + r, which is exactly
what the offload worker's mmap + index_select path expects.

Consumed by files/ple_offload/worker.py::_attach_packed_table, which for FP8
finds no `packed_row_width` on the quant method and therefore skips the width
check, validating only shape[0] (rows) and the file size.

Stdlib only (no numpy): the host runs bare python3, and this is pure byte
concatenation, so shards are streamed with preadv-style chunked reads.

Usage: build_ple_packed_table_fp8.py <snapshot_dir> <out_dir>
"""
import json, os, struct, sys, time

snap, out_dir = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]

prefix_of = set()
for k in idx:
    if ".ngram_embedding.shard_0.weight" in k and not k.endswith("weight_scale"):
        prefix_of.add(k[: k.index(".shard_0.weight")])
if not prefix_of:
    sys.exit("no PLE ngram_embedding shards in index")

_bases = {}
def meta_of(name):
    fname = idx[name]
    if fname not in _bases:
        with open(os.path.join(snap, fname), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        _bases[fname] = (h, 8 + n)
    h, base = _bases[fname]
    m = h[name]
    return os.path.join(snap, fname), base + m["data_offsets"][0], m["dtype"], tuple(m["shape"])

os.makedirs(out_dir, exist_ok=True)
for prefix in sorted(prefix_of):
    vname = prefix
    if vname.startswith("model.language_model."):
        vname = "language_model.model." + vname[len("model.language_model."):]
    shards = sorted({int(k[len(prefix) + len(".shard_"):].split(".")[0])
                     for k in idx if k.startswith(prefix + ".shard_")})
    assert shards == list(range(len(shards))), shards
    path0, off0, dt, shape = meta_of(f"{prefix}.shard_0.weight")
    if dt != "F8_E4M3":
        sys.exit(f"build_ple_packed_table_fp8: expected F8_E4M3 shards for the FP8 "
                 f"checkpoint, got {dt} (use build_ple_packed_table.py for NVFP4)")
    rows, width = shape
    out_name = os.path.join(out_dir, vname + ".packed_u8")
    meta = {"rows_per_shard": rows, "num_shards": len(shards), "row_width": width,
            "codes_width": width, "scales_width": 0, "total_rows": rows * len(shards),
            "dtype": "F8_E4M3",
            "snapshot": os.path.basename(os.path.normpath(snap))}
    if os.path.exists(out_name) and os.path.getsize(out_name) == rows * len(shards) * width:
        print("exists:", out_name); continue
    want = rows * len(shards) * width
    print(f"building {out_name}: {len(shards)} shards x {rows} rows x {width} B = "
          f"{want/2**30:.2f} GiB", flush=True)
    t0 = time.time()
    tmp = out_name + ".tmp"
    CH = 1 << 26  # 64 MiB copy buffer
    with open(tmp, "wb") as out:
        for i in shards:
            p, off, d, sh = meta_of(f"{prefix}.shard_{i}.weight")
            assert d == "F8_E4M3" and sh == (rows, width), (i, d, sh)
            need = rows * width
            with open(p, "rb", buffering=0) as src:
                src.seek(off)
                while need > 0:
                    chunk = src.read(min(CH, need))
                    if not chunk:
                        sys.exit(f"short read on {p}")
                    out.write(chunk)
                    need -= len(chunk)
            if i % 16 == 0:
                print(f"  shard {i}/{len(shards)} {time.time()-t0:.0f}s", flush=True)
    got = os.path.getsize(tmp)
    assert got == want, (got, want)
    os.rename(tmp, out_name)
    json.dump(meta, open(out_name + ".json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
