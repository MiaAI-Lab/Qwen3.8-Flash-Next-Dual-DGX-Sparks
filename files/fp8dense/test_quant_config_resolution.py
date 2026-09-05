#!/usr/bin/env python3
"""CPU-only check that vLLM (with the overlay) resolves the right quant method for every
kind of layer prefix in the hybrid checkpoint. Run inside the image with the overlay
modelopt.py bind-mounted and CUDA_VISIBLE_DEVICES="".
"""
import json
import sys

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptQuantConfigBase,
)
from vllm.model_executor.models.utils import WeightsMapper

cfg_path = sys.argv[1]
qc = json.load(open(cfg_path))["quantization_config"]
assert ModelOptMixedPrecisionConfig.override_quantization_method(qc, None) == "modelopt_mixed"
cfg = ModelOptMixedPrecisionConfig.from_config(qc)
assert isinstance(cfg, ModelOptMixedPrecisionConfig), type(cfg)
assert hasattr(cfg, "fp8_pcpt_config"), "overlay modelopt.py not active"

# Same mappers/packed mappings the model classes declare (see nvidia/model.py, qwen3_5.py).
packed = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight", "_input_mix_padding"],
}
cond_gen_mapper = WeightsMapper(orig_to_new_prefix={
    "model.visual.": "visual.",
    "model.language_model.": "language_model.model.",
    "lm_head.": "language_model.lm_head.",
})
causal_lm_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})


def resolve(cfg, prefix):
    if cfg.is_layer_excluded(prefix):
        return "EXCLUDED"
    return cfg._resolve_quant_algo(prefix) or "UNQUANTIZED"


expect = {
    # runtime prefix -> expected outcome
    "language_model.model.layers.0.linear_attn.in_proj_qkvz": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.0.linear_attn.in_proj_ba": "EXCLUDED",
    "language_model.model.layers.0.linear_attn.out_proj": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.3.self_attn.qkv_proj": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.3.self_attn.o_proj": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.3.self_attn.indexer.index_qk_proj": "EXCLUDED",
    "language_model.model.layers.3.attn_hyper_connection.input_mix_weight_down_block_inject": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.3.attn_hyper_connection.input_mix_weight_up": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.3.mlp_hyper_connection.input_mix_weight_down_block_inject": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.hyper_connection_mixer.input_mix_weight_down": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.hyper_connection_mixer.input_mix_weight_up": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.3.mlp.gate": "EXCLUDED",
    "language_model.model.layers.3.mlp.shared_expert_gate": "EXCLUDED",
    "language_model.model.layers.3.mlp.shared_expert.gate_up_proj": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.3.mlp.shared_expert.down_proj": "FP8_PER_CHANNEL_PER_TOKEN",
    "language_model.model.layers.3.mlp.experts": "NVFP4",
    "language_model.model.layers.1.ple.key_proj": "EXCLUDED",
    "language_model.model.layers.1.ple.value_proj": "EXCLUDED",
    "language_model.model.embed_tokens": "EXCLUDED",
    "language_model.lm_head": "FP8_PER_CHANNEL_PER_TOKEN",
    "visual.blocks.0.attn.qkv": "EXCLUDED",
    "visual.merger.linear_fc1": "EXCLUDED",
}
expect_causal = {  # text-only entry point / draft model spellings
    "model.layers.0.linear_attn.in_proj_qkvz": "FP8_PER_CHANNEL_PER_TOKEN",
    "model.layers.3.mlp.experts": "NVFP4",
    "lm_head": "FP8_PER_CHANNEL_PER_TOKEN",
    "mtp.layers.48.self_attn.qkv_proj": "EXCLUDED",
    "mtp.layers.48.attn_hyper_connection.input_mix_weight_down_block_inject": "EXCLUDED",
    "mtp.layers.48.mlp.experts": "EXCLUDED",
    "mtp.hyper_connection_mixer.input_mix_weight_down": "EXCLUDED",
    "mtp.fc_hidden": "EXCLUDED",
}
bad = 0
for mapper, table, label in ((cond_gen_mapper, expect, "ConditionalGeneration"), (causal_lm_mapper, expect_causal, "CausalLM/MTP")):
    c = ModelOptMixedPrecisionConfig.from_config(qc)
    c.packed_modules_mapping = packed
    c.apply_vllm_mapper(mapper.get_unstacked_mapper())
    for prefix, want in table.items():
        got = resolve(c, prefix)
        flag = "ok " if got == want else "BAD"
        if got != want:
            bad += 1
        print(f"[{label}] {flag} {prefix:90s} -> {got}")
print("FAILURES:", bad)
sys.exit(1 if bad else 0)
