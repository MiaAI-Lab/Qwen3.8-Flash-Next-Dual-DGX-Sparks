#!/usr/bin/env python3
"""Teach ModelOptMixedPrecisionConfig to build FP8_BLOCK_SCALES routed experts.

nvidia/Qwen3.8-Flash-Next-NVFP4 quantizes its MTP routed experts as 128x128
block-scaled FP8 (model card; config.json group_1 / quantized_layers). But
ModelOptMixedPrecisionConfig.get_quant_method only builds RoutedExperts for
FP8 / NVFP4 / W4A16_NVFP4 / MXFP8 and returns None for anything else, so the MTP
MoE is built *unquantized* and loading dies with:

    AttributeError: Layer mtp.layers.48.mlp.experts has no parameter
    'w2_weight_scale_inv' for checkpoint weight
    'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'

This gap is not specific to the image: upstream vLLM has no FP8_BLOCK_SCALES
branch either, at the commit the model card recommends (d4d703ca) or on main.

vLLM already ships the machinery — Fp8MoEMethod sets block_quant from
Fp8Config.weight_block_size and then names its scales `weight_scale_inv`, which
is exactly what the checkpoint stores. This patch only wires the dispatch.

Operates in place on files/modelopt_patched.py (the output of
patch_modelopt_mxfp8.py), so both patches stack.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "modelopt_patched.py")

HELPER = '''    def _fp8_block_scales_config(self, prefix: str):
        """Build an Fp8Config for an FP8_BLOCK_SCALES layer.

        The block shape is read from the checkpoint's own group_size rather than
        assumed: a wrong block shape does not fail loudly, it silently applies
        misaligned scales, so guessing is worse than refusing.
        """
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        info = None
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            info = self.quantized_layers.get(candidate)
            if info is None:
                prefix_dot = candidate + "."
                for key, value in self.quantized_layers.items():
                    if key.startswith(prefix_dot):
                        info = value
                        break
            if info is not None:
                break

        group_size = (info or {}).get("group_size")
        if not isinstance(group_size, int) or group_size <= 0:
            raise ValueError(
                f"FP8_BLOCK_SCALES layer {prefix} declares no usable group_size "
                f"in quantized_layers (got {group_size!r}); refusing to guess a "
                "block shape."
            )
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[group_size, group_size],
        )

'''

ANCHOR_HELPER = """    @staticmethod
    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:
"""

ANCHOR_DISPATCH = """            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None
"""

DISPATCH = """            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "FP8_BLOCK_SCALES":
                # Imported lazily: modelopt.py deliberately does not import fp8.py
                # at module scope.
                from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

                logger.info_once(
                    "Routed experts %s use FP8_BLOCK_SCALES; building them with "
                    "Fp8MoEMethod (block-quantized).",
                    prefix,
                )
                return Fp8MoEMethod(
                    quant_config=self._fp8_block_scales_config(prefix),
                    layer=layer,
                )
            return None
"""


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    if src.count(old) != 1:
        raise AssertionError(f"fp8-block patch: {what} anchor count={src.count(old)}")
    return src.replace(old, new)


def patch() -> None:
    src = open(TARGET).read()
    if "FP8_BLOCK_SCALES" in src:
        print("already patched", TARGET)
        return
    src = _replace_once(src, ANCHOR_HELPER, HELPER + ANCHOR_HELPER, "helper")
    src = _replace_once(src, ANCHOR_DISPATCH, DISPATCH, "moe dispatch")
    open(TARGET, "w").write(src)
    print("ok", TARGET)


if __name__ == "__main__":
    if not os.path.isfile(TARGET):
        print(f"ERROR: missing {TARGET} (run patch_modelopt_mxfp8.py first)", file=sys.stderr)
        sys.exit(1)
    patch()
