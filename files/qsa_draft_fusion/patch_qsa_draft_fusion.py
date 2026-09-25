#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Verifier for the QSA draft-fusion overlay (backport of vllm#58449).
# Fails closed: the pristine image source must match the pinned vLLM commit
# byte-for-byte before the pinned patch is applied, so the fused overlay can
# never be carried onto a drifted base.
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIG = HERE / "orig" / "qsa_cache.py"
PATCH = HERE / "pr58449.patch"
OUT = HERE / "qsa_cache.py"
ORIG_SHA256 = "c6651a0cf27e2ffb23b90eafe21e044df5609ce976f52ebffa7a4fbea688fc04"
OUT_SHA256 = "ee658ad3d7b32995c8483540f2e43370244734821fce12ed078f01708a5c5139"

import tempfile
with tempfile.TemporaryDirectory(dir=HERE) as td:  # same filesystem: replace() stays atomic
    work = Path(td)
    # Fresh copy (not hardlink): git apply rewrites the target in place.
    (work / "qsa_cache.py").write_bytes(ORIG.read_bytes())
    if hashlib.sha256((work / "qsa_cache.py").read_bytes()).hexdigest() != ORIG_SHA256:
        sys.exit("qsa_draft_fusion: orig/qsa_cache.py does not match the pinned vLLM source")
    env = {**os.environ, "GIT_INDEX_FILE": str(work / ".apply-index")}
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=work, env=env, check=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=work, env=env, check=True)
    if hashlib.sha256((work / "qsa_cache.py").read_bytes()).hexdigest() != OUT_SHA256:
        sys.exit("qsa_draft_fusion: patch output hash mismatch")
    os.replace(work / "qsa_cache.py", OUT)
print("qsa_draft_fusion: patched qsa_cache.py (verified against pinned hashes)")
