#!/usr/bin/env python3
"""Add the missing MTP layer-index aliases to a checkpoint's quantization config.

vLLM instantiates the MTP draft layers at *absolute* indices continuing the main
stack: with num_hidden_layers=48 the single MTP layer is built with the runtime
prefix `mtp.layers.48.mlp.experts`. ModelOptMixedPrecisionConfig._resolve_quant_algo
matches those prefixes against `quantized_layers` by exact string, with no
renumbering.

Checkpoints disagree on which name they record:

  local-inference-lab/Qwen3.8-Flash-Next-NVFP4   mtp.layers.0.*  AND  mtp.layers.48.*
  nvidia/Qwen3.8-Flash-Next-NVFP4                mtp.layers.0.*  only

With only the `.0.` name the lookup misses, the MTP MoE is built unquantized,
and loading dies at:

  AttributeError: Layer mtp.layers.48.mlp.experts has no parameter
  'w2_weight_scale_inv' for checkpoint weight
  'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'

BOTH files must be patched: `quantized_layers` appears in config.json's
`quantization_config` *and* in the legacy hf_quant_config.json, and vLLM reads
the legacy file when it is present. Patching only config.json is not enough.

Outputs go next to this script; start.sh bind-mounts them over the snapshot's
copies in the container, so the HF cache is never modified.

Usage: patch_checkpoint_config.py <snapshot dir> <output dir>
Prints the space-separated basenames that needed patching (empty if none).
"""
import json
import os
import re
import sys

MTP_RE = re.compile(r"^mtp\.layers\.(\d+)\.")


def _aliaser(num_hidden_layers: int):
    def alias(name: str) -> str | None:
        match = MTP_RE.match(name)
        if not match:
            return None
        index = int(match.group(1))
        if index >= num_hidden_layers:
            return None  # already an absolute index
        return f"mtp.layers.{num_hidden_layers + index}." + name[match.end() :]

    return alias


def add_aliases(quant_config: dict, num_hidden_layers: int) -> bool:
    """Add absolute-index MTP aliases in place. Returns True if anything changed."""
    alias = _aliaser(num_hidden_layers)
    changed = False

    layers = quant_config.get("quantized_layers")
    if isinstance(layers, dict):
        for name, info in list(layers.items()):
            aliased = alias(name)
            if aliased and aliased not in layers:
                layers[aliased] = info
                changed = True

    # Mirror into config_groups so the file stays self-consistent for anything
    # reading targets rather than quantized_layers.
    for group in (quant_config.get("config_groups") or {}).values():
        targets = group.get("targets")
        if not isinstance(targets, list):
            continue
        for name in list(targets):
            aliased = alias(name)
            if aliased and aliased not in targets:
                targets.append(aliased)
                changed = True

    return changed


# RoutedExperts algos ModelOptMixedPrecisionConfig.get_quant_method can build.
# Anything else resolves to a *silently unquantized* MoE, which then dies at load
# with "has no parameter 'w2_weight_scale_inv'".
# FP8_BLOCK_SCALES is supported only because files/patch_modelopt_fp8_block_moe.py
# adds that branch; stock vLLM (image and upstream main) would build an
# unquantized MoE and die at load.
SUPPORTED_MOE_ALGOS = {"FP8", "NVFP4", "W4A16_NVFP4", "MXFP8", "FP8_BLOCK_SCALES"}


def mtp_moe_algo(snapshot_dir: str) -> str:
    """quant_algo of the MTP routed experts, or "" when they are unquantized."""
    for name in ("config.json", "hf_quant_config.json"):
        path = os.path.join(snapshot_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            doc = json.load(fh)
        quant = doc.get("quantization_config") or doc.get("quantization") or {}
        for key, info in (quant.get("quantized_layers") or {}).items():
            if MTP_RE.match(key) and key.endswith(".mlp.experts"):
                return str(info.get("quant_algo", "")).upper()
    return ""


def main(snapshot_dir: str, out_dir: str) -> None:
    config_path = os.path.join(snapshot_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"patch_checkpoint_config: no config.json at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as fh:
        config = json.load(fh)

    text_config = config.get("text_config", config)
    num_hidden_layers = text_config.get("num_hidden_layers")
    if not isinstance(num_hidden_layers, int):
        raise SystemExit("patch_checkpoint_config: num_hidden_layers missing")

    patched: list[str] = []

    quant_config = config.get("quantization_config")
    if isinstance(quant_config, dict) and add_aliases(quant_config, num_hidden_layers):
        with open(os.path.join(out_dir, "config_patched.json"), "w") as fh:
            json.dump(config, fh, indent=2)
        patched.append("config.json")

    # Legacy sidecar — vLLM prefers it when present, so it needs the same fix.
    legacy_path = os.path.join(snapshot_dir, "hf_quant_config.json")
    if os.path.isfile(legacy_path):
        with open(legacy_path) as fh:
            legacy = json.load(fh)
        if add_aliases(legacy.get("quantization", legacy), num_hidden_layers):
            with open(os.path.join(out_dir, "hf_quant_config_patched.json"), "w") as fh:
                json.dump(legacy, fh, indent=2)
            patched.append("hf_quant_config.json")

    print(" ".join(patched))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--mtp-moe-algo":
        algo = mtp_moe_algo(sys.argv[2])
        sys.stdout.write(algo)
        # exit 3 = MTP experts use an algo the mixed-precision MoE dispatch
        # cannot build, so speculative decoding cannot work on this checkpoint.
        sys.exit(3 if algo and algo not in SUPPORTED_MOE_ALGOS else 0)
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <snapshot dir> <output dir>", file=sys.stderr)
        print(f"       {sys.argv[0]} --mtp-moe-algo <snapshot dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
