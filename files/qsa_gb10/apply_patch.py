#!/usr/bin/env python3
"""Patch ops/qsa.py so the QSA Triton launch profiles can be overridden per device.

The stock tables were "tuned on GB300" (160 SMs). GB10 has 48 SMs. This patch keeps
the stock behaviour by default and adds:
  * VLLM_QSA_PROFILE=gb10  -> a conservative 48-SM table (fewer split-K partials,
                              fewer merge passes; see GB10_PROFILES below)
  * VLLM_QSA_PROFILE_JSON=/path.json -> table produced by bench_qsa_kernels.py
The tables are looked up by the same (base_programs, block_m) buckets as upstream.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "qsa.py.orig")).read()

old_head = '''_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024
'''
new_head = '''_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024

# ---------------------------------------------------------------------------
# [qsa_gb10 overlay] device-specific launch profiles.
#
# Upstream tile tables were tuned on GB300 (160 SMs). A DGX Spark GB10 has 48 SMs,
# so the decode profile (block_n=16, 64 split-K partials, 4 warps) launches ~5x more
# CTAs than the machine can run concurrently and then pays a merge kernel over 64
# partials. Selected by VLLM_QSA_PROFILE={stock,gb10} or a JSON file from
# bench_qsa_kernels.py via VLLM_QSA_PROFILE_JSON.
# ---------------------------------------------------------------------------
import json as _json
import os as _os

# (block_n, target_splits, num_warps) for the sparse GQA kernel, keyed by the same
# base_programs buckets used below; and (tiles_per_program_small, tiles_per_program_large)
# for the MQA scorer.
GB10_PROFILES = {
    "sparse": {
        "small": (32, 16, 4),  # base_programs <= small_profile_limit  (decode, 1-2 reqs)
        "lt32": (32, 8, 4),    # base_programs < 32                     (decode, up to 8 reqs)
        "le256": (64, 4, 2),
        "le512": (64, 2, 2),
        "large": (64, 1, 2),
    },
    "mqa": {"tiles_small": 2, "tiles_large": 8, "num_warps": 2},
}


def _load_qsa_profile():
    path = _os.environ.get("VLLM_QSA_PROFILE_JSON", "").strip()
    if path:
        with open(path) as fh:
            return _json.load(fh)
    name = _os.environ.get("VLLM_QSA_PROFILE", "stock").strip().lower()
    if name == "gb10":
        return GB10_PROFILES
    return None


_QSA_PROFILE = _load_qsa_profile()
'''
assert src.count(old_head) == 1
src = src.replace(old_head, new_head)

old_mqa = '''    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
'''
new_mqa = '''    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    mqa_warps = 2
    if _QSA_PROFILE is not None and "mqa" in _QSA_PROFILE:  # [qsa_gb10 overlay]
        _m = _QSA_PROFILE["mqa"]
        tiles_per_program = _m["tiles_small"] if q.shape[0] <= 32 else _m["tiles_large"]
        mqa_warps = int(_m.get("num_warps", 2))
'''
assert src.count(old_mqa) == 1
src = src.replace(old_mqa, new_mqa)

old_mqa_launch = '''        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks
'''
new_mqa_launch = '''        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=mqa_warps,
    )
    return logits, visible_blocks
'''
assert src.count(old_mqa_launch) == 1
src = src.replace(old_mqa_launch, new_mqa_launch)

old_sparse = '''    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
'''
new_sparse = '''    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
        _bucket = "small"
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
        _bucket = "lt32"
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
        _bucket = "le256"
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
        _bucket = "le512"
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
        _bucket = "large"
    if _QSA_PROFILE is not None and "sparse" in _QSA_PROFILE:  # [qsa_gb10 overlay]
        block_n, target_splits, partial_warps = (
            int(v) for v in _QSA_PROFILE["sparse"][_bucket]
        )
'''
assert src.count(old_sparse) == 1
src = src.replace(old_sparse, new_sparse)
open(os.path.join(HERE, "qsa.py"), "w").write(src)
print("patched qsa.py")
