#!/usr/bin/env python3
"""Print the PLE table dtype a checkpoint needs declared, or nothing.

The patched ple_layer.py picks its PLE embedding method from
`text_config.ple_embedding_dtype`. Checkpoints are inconsistent about setting it:

  local-inference-lab/Qwen3.8-Flash-Next-NVFP4  ple_embedding_dtype = "nvfp4"
  RadixArk/Qwen3.8-Flash-Next-NVFP4             ple_embedding_dtype = "float8_e4m3fn"
  nvidia/Qwen3.8-Flash-Next-NVFP4               absent — declared only in
                                                quantization_config.config_groups

All three are MIXED_PRECISION, so neither the Fp8Config nor the
ModelOptNvFp4Config branch in _get_ple_embedding_quant_method fires; without the
key the PLE table gets no quant method and its FP8 rows fail to load.

When the key is missing we recover it from the config group that targets
`...ple.ple_embedding.ngram_embedding` (num_bits 8 -> FP8, 4 -> NVFP4), and
start.sh feeds it back in via --hf-overrides. Prints nothing when the checkpoint
already declares the key, or when no PLE group is found (nothing to override).

Usage: detect_ple_dtype.py <model dir with config.json>
"""
import json
import os
import sys


def detect(config_path: str) -> str:
    with open(config_path) as fh:
        config = json.load(fh)

    text_config = config.get("text_config", config)
    if text_config.get("ple_embedding_dtype"):
        return ""  # already declared — no override needed

    groups = config.get("quantization_config", {}).get("config_groups", {})
    for group in groups.values():
        targets = group.get("targets", [])
        if not any("ple_embedding.ngram_embedding" in t for t in targets):
            continue
        num_bits = (group.get("weights") or {}).get("num_bits")
        if num_bits == 8:
            return "float8_e4m3fn"
        if num_bits == 4:
            return "nvfp4"
        raise SystemExit(
            f"detect_ple_dtype: PLE group has unsupported num_bits={num_bits!r}"
        )
    return ""  # unquantized PLE table (or none) — leave the config alone


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <model dir>", file=sys.stderr)
        sys.exit(2)
    path = os.path.join(sys.argv[1], "config.json")
    if not os.path.isfile(path):
        print(f"detect_ple_dtype: no config.json at {path}", file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(detect(path))
