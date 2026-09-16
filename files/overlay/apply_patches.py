#!/usr/bin/env python3
"""Regenerate the FP8-dense loader overlay from the image plus the recorded diffs.

    files/overlay/<name>.diff  (committed, source of truth)
  + <name>.py extracted verbatim from the vLLM image  -> files/overlay/<name>.py.orig
  = files/overlay/<name>.py                            (bind-mounted by start.sh 4c)

for name in model, hyperconnection, modelopt, mtp. Every hunk must apply with
--fuzz=0 (context must match exactly); an offset is fine, a fuzzy match is not.

NOTE: two of the four outputs are NOT what start.sh should mount when other
patches also claim the same container file:
  * modelopt.py  - files/modelopt_patched.py (MXFP8 fallback + FP8_BLOCK_SCALES
                   MoE) is ALWAYS mounted at that path by step 6b. Use
                   files/stack_modelopt_fp8dense.py to fold this overlay's hunks
                   into files/modelopt_patched.py instead of mounting both.
  * mtp.py       - with MTP_DRAFT_VOCAB set, use files/stack_mtp_fp8_draftvocab.py,
                   which stacks the reduced-vocabulary drafter on top of this file.

Idempotent: an output carrying the "[fp8dense overlay]" marker is left alone.
Usage: python3 files/overlay/apply_patches.py [--force]
Env:   IMAGE (default vllm/vllm-openai:qwen38-flash-next)
"""
import ast
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.environ.get("IMAGE", "vllm/vllm-openai:qwen38-flash-next")
VLLM_PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
MARKER = "[fp8dense overlay]"

# name -> (path inside the image, hunks in the diff, "[fp8dense overlay]" marker lines it adds)
TARGETS = {
    "model": (f"{VLLM_PKG}/models/qwen3_8_flash_next/nvidia/model.py", 3, 4),
    "hyperconnection": (f"{VLLM_PKG}/models/qwen3_8_flash_next/nvidia/hyperconnection.py", 4, 1),
    "modelopt": (f"{VLLM_PKG}/model_executor/layers/quantization/modelopt.py", 4, 3),
    "mtp": (f"{VLLM_PKG}/models/qwen3_8_flash_next/nvidia/mtp.py", 2, 2),
}


def extract_from_image(container_path: str, dest: str) -> None:
    if os.path.isfile(dest):
        return
    cid = subprocess.check_output(["docker", "create", IMAGE, "/bin/true"], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{container_path}", dest])
    finally:
        subprocess.call(["docker", "rm", cid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.isfile(dest):
        sys.exit(f"apply_patches: failed to extract {container_path} from {IMAGE}")


def apply(name: str, force: bool) -> str:
    container_path, n_hunks, n_marks = TARGETS[name]
    orig = os.path.join(HERE, f"{name}.py.orig")
    diff = os.path.join(HERE, f"{name}.diff")
    out = os.path.join(HERE, f"{name}.py")
    if not os.path.isfile(diff):
        sys.exit(f"apply_patches: missing {diff}")
    if not force and os.path.isfile(out) and MARKER in open(out).read():
        print(f"{name}.py: already carries {MARKER}; skipping")
        return out
    extract_from_image(container_path, orig)
    if shutil.which("patch") is None:
        sys.exit("apply_patches: GNU `patch` is required (apt install patch)")
    tmp = out + ".tmp"
    with open(diff) as fh:
        proc = subprocess.run(
            ["patch", "-p1", "--fuzz=0", "--forward", "--no-backup-if-mismatch",
             "-o", tmp, orig],
            stdin=fh, capture_output=True, text=True,
        )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        if os.path.exists(tmp):
            os.unlink(tmp)
        sys.exit(f"apply_patches: {name}.diff did not apply cleanly to the image's {name}.py "
                 f"(rc={proc.returncode}); the image may have changed")
    src = open(tmp).read()
    ast.parse(src)
    if src.count(MARKER) != n_marks:
        os.unlink(tmp)
        sys.exit(f"apply_patches: {name}.py carries {src.count(MARKER)} '{MARKER}' markers, "
                 f"expected {n_marks}")
    os.replace(tmp, out)
    os.chmod(out, 0o644)  # `patch -o` creates 0600; the worker copy must stay readable
    print(f"{name}.py: {n_hunks} hunks applied -> {out}")
    return out


def main() -> None:
    force = "--force" in sys.argv[1:]
    for name in TARGETS:
        apply(name, force)


if __name__ == "__main__":
    main()
