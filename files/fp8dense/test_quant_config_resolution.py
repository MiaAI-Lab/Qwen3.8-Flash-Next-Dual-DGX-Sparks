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
}
# MTP draft model: vLLM builds it at the absolute index mtp.layers.<num_hidden_layers>
# (nvidia/mtp.py) and only remaps exclude_modules, so quantized_layers must already
# carry that spelling. Three checkpoint generations: MTP fully bf16 (mtp.* excluded),
# MTP dense FP8 with bf16 experts, MTP dense FP8 + block-FP8 experts.
mtp_bf16 = "mtp.*" in qc["ignore"]
mtp_experts = qc["quantized_layers"].get("mtp.layers.48.mlp.experts", {}).get("quant_algo", "").upper()
PCPT = "EXCLUDED" if mtp_bf16 else "FP8_PER_CHANNEL_PER_TOKEN"
expect_causal.update({
    "mtp.layers.48.self_attn.qkv_proj": PCPT,
    "mtp.layers.48.self_attn.o_proj": PCPT,
    "mtp.layers.48.attn_hyper_connection.input_mix_weight_down_block_inject": PCPT,
    "mtp.layers.48.attn_hyper_connection.input_mix_weight_up": PCPT,
    "mtp.layers.48.mlp_hyper_connection.input_mix_weight_down_block_inject": PCPT,
    "mtp.layers.48.mlp.shared_expert.gate_up_proj": PCPT,
    "mtp.layers.48.mlp.shared_expert.down_proj": PCPT,
    "mtp.hyper_connection_mixer.input_mix_weight_down": PCPT,
    "mtp.hyper_connection_mixer.input_mix_weight_up": PCPT,
    "mtp.fc_hidden": PCPT,
    "mtp.fc_embedding": PCPT,
    "mtp.layers.48.mlp.experts": "EXCLUDED" if (mtp_bf16 or not mtp_experts) else mtp_experts,
    # always bf16 in the draft, whatever generation
    "mtp.layers.48.mlp.gate": "EXCLUDED",
    "mtp.layers.48.mlp.shared_expert_gate": "EXCLUDED",
    "mtp.layers.48.self_attn.indexer.index_qk_proj": "EXCLUDED",
    "mtp.hyper_connection_mixer.block_inject_weight": "EXCLUDED",
})
bad = 0
for mapper, table, label in ((cond_gen_mapper, expect, "ConditionalGeneration"), (causal_lm_mapper, expect_causal, "CausalLM/MTP")):
    c = ModelOptMixedPrecisionConfig.from_config(qc)
    c.packed_modules_mapping = packed
    c.apply_vllm_mapper(mapper.get_unstacked_mapper())
    if label == "CausalLM/MTP":
        # exactly what nvidia/mtp.py _make_draft_vllm_config does to the draft's config
        import re
        c.exclude_modules = [re.sub(r"(?<=\.layers\.)\d+", lambda m: str(48 + int(m.group(0))), x)
                             if x.startswith("mtp.") else x for x in c.exclude_modules]
        # and the FP8_BLOCK_SCALES branch must find a usable block size
        if mtp_experts == "FP8_BLOCK_SCALES":
            fcfg = c._fp8_block_scales_config("mtp.layers.48.mlp.experts")
            assert fcfg.weight_block_size == [128, 128], fcfg.weight_block_size
            print("[CausalLM/MTP] ok  _fp8_block_scales_config(mtp.layers.48.mlp.experts) ->", fcfg.weight_block_size)
    for prefix, want in table.items():
        got = resolve(c, prefix)
        flag = "ok " if got == want else "BAD"
        if got != want:
            bad += 1
        print(f"[{label}] {flag} {prefix:90s} -> {got}")
print("FAILURES:", bad)
sys.exit(1 if bad else 0)
