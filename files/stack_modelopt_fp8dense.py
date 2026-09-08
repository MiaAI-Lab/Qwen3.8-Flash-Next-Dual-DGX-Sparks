#!/usr/bin/env python3
"""Fold the FP8-dense overlay hunks INTO files/modelopt_patched.py.

Three patches claim .../quantization/modelopt.py:
  1. files/patch_modelopt_mxfp8.py          -> files/modelopt_patched.py   (always, start.sh 6b)
  2. files/patch_modelopt_fp8_block_moe.py  -> same file, in place         (always, start.sh 6b)
  3. files/overlay/modelopt.diff            -> files/overlay/modelopt.py   (FP8_DENSE=true, 4c)

start.sh mounts (1+2) at that container path unconditionally and, with
FP8_DENSE=true, ALSO mounts (3) there: docker refuses duplicate mount points, so
FP8_DENSE could never launch. The three touch disjoint regions (verified with
`patch --dry-run --fuzz=0`: all four fp8dense hunks apply to the (1+2) output
with a pure line offset, no fuzz), so this applies (3)'s hunks on top of (1+2),
in place, and start.sh keeps mounting files/modelopt_patched.py alone.

Run AFTER patch_modelopt_mxfp8.py and patch_modelopt_fp8_block_moe.py:

    python3 files/stack_modelopt_fp8dense.py [<repo_root>]

Idempotent (marker: fp8_pcpt_config). Fails loudly if any hunk does not apply,
and asserts the result parses and still carries all three patches' markers.
"""
import ast
import os
import shutil
import subprocess
import sys

repo = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
files = os.path.join(repo, "files")
target = os.path.join(files, "modelopt_patched.py")
diff = os.path.join(files, "overlay", "modelopt.diff")

if not os.path.isfile(target):
    sys.exit(f"missing {target}; run files/patch_modelopt_mxfp8.py first")
if not os.path.isfile(diff):
    sys.exit(f"missing {diff}")

src = open(target).read()
for tok, why in (("ModelOptMxFp8EmulationLinearMethod", "MXFP8 fallback (patch_modelopt_mxfp8.py)"),
                 ("FP8_BLOCK_SCALES", "FP8 block MoE (patch_modelopt_fp8_block_moe.py)")):
    if tok not in src:
        sys.exit(f"{target} lacks {why} ({tok}); stack order is mxfp8 -> fp8_block_moe -> fp8dense")

if "fp8_pcpt_config" in src:
    print("modelopt_patched.py: already carries the fp8dense hunks")
else:
    if shutil.which("patch") is None:
        sys.exit("GNU `patch` is required (apt install patch)")
    tmp = target + ".stack.tmp"
    with open(diff) as fh:
        proc = subprocess.run(
            ["patch", "-p1", "--fuzz=0", "--forward", "--no-backup-if-mismatch", "-o", tmp, target],
            stdin=fh, capture_output=True, text=True,
        )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        if os.path.exists(tmp):
            os.unlink(tmp)
        sys.exit(f"fp8dense modelopt hunks did not apply on top of mxfp8+fp8_block_moe (rc={proc.returncode})")
    os.replace(tmp, target)
    print("modelopt_patched.py: fp8dense hunks stacked on mxfp8 + fp8_block_moe")

# Sanity: parses, and carries ALL THREE feature sets.
src = open(target).read()
ast.parse(src)
for tok, why in (("ModelOptMxFp8EmulationLinearMethod", "mxfp8 fallback"),
                 ("FP8_BLOCK_SCALES", "fp8 block-scales MoE"),
                 ("fp8_pcpt_config", "fp8dense per-channel config"),
                 ("ModelOptFp8PcPtLinearMethod(self.fp8_pcpt_config)", "fp8dense dispatch"),
                 ('candidates.append("lm_head")', "fp8dense lm_head prefix")):
    assert tok in src, f"stacked modelopt_patched.py lost {why} ({tok})"
assert src.count("[fp8dense overlay]") == 3, src.count("[fp8dense overlay]")
print("stacked OK ->", target)
