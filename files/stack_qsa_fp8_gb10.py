#!/usr/bin/env python3
"""Stack the qsa_gb10 launch-profile overlay ON TOP OF the FP8-KV overlay.

Both patch vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py. They collide on
exactly ONE anchor (the MQA scorer's Triton launch tail, where the FP8 patch
inserts `KV_QUANT_MODE=kv_quant_mode,` immediately above `num_warps=2,`).
Everything else is disjoint.

Order matters: FP8 first, then GB10 with a KV_QUANT_MODE-aware anchor.

Run from the repo root, AFTER files/patch_qsa_fp8_kv.py has produced
files/qsa_ops_patched.py (start.sh step 4f does this), or standalone:

    python3 files/patch_qsa_fp8_kv.py          # -> files/qsa_ops_patched.py
    python3 stack_qsa_fp8_gb10.py <repo_root>  # -> files/qsa_gb10/qsa.py  (stacked)

Then mount files/qsa_gb10/qsa.py at .../nvidia/ops/qsa.py (INSTEAD of
files/qsa_ops_patched.py) and keep mounting files/qsa_nvidia_patched.py at
.../nvidia/qsa.py, and export VLLM_QSA_PROFILE / VLLM_QSA_PROFILE_JSON.
"""
import os
import shutil
import subprocess
import sys

repo = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
files = os.path.join(repo, "files")
gb10 = os.path.join(files, "qsa_gb10")

fp8_out = os.path.join(files, "qsa_ops_patched.py")
if not os.path.exists(fp8_out):
    sys.exit(f"missing {fp8_out}; run files/patch_qsa_fp8_kv.py first")

# 1. Feed the FP8 output in as the gb10 patcher's "original".
shutil.copyfile(fp8_out, os.path.join(gb10, "qsa.py.orig"))

# 2. Make the one colliding anchor KV_QUANT_MODE-aware, in place, idempotently.
ap = os.path.join(gb10, "apply_patch.py")
s = open(ap).read()
MARK = "_KVQ = "
if MARK not in s:
    old = """old_mqa_launch = '''        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks
'''"""
    new = """_KVQ = '        KV_QUANT_MODE=kv_quant_mode\\n' if 'KV_QUANT_MODE=kv_quant_mode' in src else ''
_KVQ = _KVQ.replace('kv_quant_mode', 'kv_quant_mode,') if _KVQ else ''
old_mqa_launch = '''        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
''' + _KVQ + '''        num_warps=2,
    )
    return logits, visible_blocks
'''"""
    assert s.count(old) == 1, f"apply_patch.py anchor A3 not found ({s.count(old)})"
    s = s.replace(old, new)

    old2 = """new_mqa_launch = '''        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=mqa_warps,
    )
    return logits, visible_blocks
'''"""
    new2 = """new_mqa_launch = '''        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
''' + _KVQ + '''        num_warps=mqa_warps,
    )
    return logits, visible_blocks
'''"""
    assert s.count(old2) == 1
    s = s.replace(old2, new2)
    open(ap, "w").write(s)
    print("apply_patch.py: MQA-launch anchor made KV_QUANT_MODE-aware")
else:
    print("apply_patch.py: already KV_QUANT_MODE-aware")

# 3. Run it. Writes files/qsa_gb10/qsa.py = FP8-KV + GB10 profiles.
subprocess.check_call([sys.executable, "apply_patch.py"], cwd=gb10)

# 4. Sanity: parses, and carries BOTH feature sets.
import ast
src = open(os.path.join(gb10, "qsa.py")).read()
ast.parse(src)
for tok, why in (("KV_QUANT_MODE", "fp8-kv"), ("_QSA_PROFILE", "gb10 profile"),
                 ("mqa_warps", "gb10 mqa warps"), ("_bucket", "gb10 sparse buckets"),
                 ("_qsa_as_fp8", "fp8 cache reinterpret")):
    assert tok in src, f"stacked qsa.py lost {why} ({tok})"
print("stacked OK ->", os.path.join(gb10, "qsa.py"))
