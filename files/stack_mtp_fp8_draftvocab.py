#!/usr/bin/env python3
"""Stack the reduced-vocabulary MTP drafter ON TOP OF the FP8-dense mtp.py overlay.

Both patch vllm/models/qwen3_8_flash_next/nvidia/mtp.py:

  files/overlay/mtp.diff (FP8_DENSE)        2 hunks, constructor only:
    H1  Qwen3_8FlashNextMultiTokenPredictor.__init__: hyper_connection_mixer gets
        quant_config=draft_vllm_config.quant_config          (pristine lines 224-228)
    H2  Qwen3_8FlashNextMTP.__init__: ParallelLMHead gets quant_config=self.quant_config
                                                             (pristine lines 391-395)
  files/patch_mtp_draft_vocab.py (MTP_DRAFT_VOCAB)  5 anchors:
    A1  "from vllm.compilation.decorators import support_torch_compile\\n"  (line 22)
    A2  "from vllm.distributed import get_pp_group\\n"                       (line 24)
    A3  "def _remap_ignored_layers(\\n"                                      (line 58)
    A4  the compute_logits method                                            (lines 427-430)
    A5  "        return loader.load_weights(remap_weight_names())\\n"        (line 444)

Textually the two are DISJOINT (no anchor of one lies inside a hunk of the
other, and every anchor stays unique after the other patch is applied), so
either order yields the same file. They collide SEMANTICALLY on one object:
H2 turns lm_head.weight into an FP8-E4M3 tensor with a separate per-row
`weight_scale`, while the drafter's `_attach_draft_vocab` slices lm_head.weight
by row and runs it through F.linear, which has no float8 kernel (and would drop
the scales even if it had). Untouched, that fails on the first draft step,
~10 minutes after launch. So the stack is: FP8 hunks first (they do not care
what the drafter does), the drafter's five anchors second, then ONE extra
edit inside `_attach_draft_vocab`: when the slice is float8, dequantize it with
the matching rows of weight_scale into the model dtype. The slice is taken in
load_weights, before process_weights_after_loading transposes the head, so
weight is still [vocab_shard, hidden] and weight_scale [vocab_shard] there.

Result: files/overlay/mtp_draftvocab.py (mount INSTEAD of files/overlay/mtp.py
and INSTEAD of files/mtp_patched.py). Without VLLM_MTP_DRAFT_VOCAB in the
environment it behaves exactly like files/overlay/mtp.py.

    python3 files/overlay/apply_patches.py            # -> files/overlay/mtp.py (FP8 hunks)
    python3 files/stack_mtp_fp8_draftvocab.py [root]  # -> files/overlay/mtp_draftvocab.py

Idempotent. Asserts the output parses and carries marker tokens from BOTH
patches plus the dequant bridge; refuses to write anything else.
"""
import ast
import importlib.util
import os
import subprocess
import sys

repo = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
files = os.path.join(repo, "files")
overlay = os.path.join(files, "overlay")
fp8_mtp = os.path.join(overlay, "mtp.py")
out = os.path.join(overlay, "mtp_draftvocab.py")
dv_script = os.path.join(files, "patch_mtp_draft_vocab.py")

STACK_MARK = "[fp8dense+draftvocab stack]"

# 1. FP8-dense base: files/overlay/mtp.py must exist and carry both hunks.
if not (os.path.isfile(fp8_mtp) and "[fp8dense overlay]" in open(fp8_mtp).read()):
    print("files/overlay/mtp.py missing or unpatched; running files/overlay/apply_patches.py")
    subprocess.check_call([sys.executable, os.path.join(overlay, "apply_patches.py")])
base = open(fp8_mtp).read()
assert base.count("[fp8dense overlay]") == 2, "files/overlay/mtp.py must carry exactly the 2 fp8dense hunks"
assert "_attach_draft_vocab" not in base, "files/overlay/mtp.py already contains the drafter; refusing to stack twice"

# 2. Run the draft-vocab patcher with the FP8 file as its "original". Its five
#    anchors are reused verbatim (no duplication here); only ORIG/OUT are redirected.
spec = importlib.util.spec_from_file_location("patch_mtp_draft_vocab", dv_script)
dv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dv)
for anchor in (
    "from vllm.compilation.decorators import support_torch_compile\n",
    "from vllm.distributed import get_pp_group\n",
    "def _remap_ignored_layers(\n",
    "        return self.logits_processor(self.lm_head, hidden_states)\n",
    "        return loader.load_weights(remap_weight_names())\n",
):
    assert base.count(anchor) == 1, f"draft-vocab anchor no longer unique in FP8 mtp.py: {anchor!r}"
dv.ORIG = fp8_mtp
dv.OUT = out
if os.path.isfile(out) and STACK_MARK in open(out).read():
    print("mtp_draftvocab.py: already stacked")
else:
    if os.path.isfile(out):
        os.unlink(out)  # stale partial output: the patcher's own idempotency check would keep it
    dv.main()          # sys.exit()s loudly on any missing/non-unique anchor

    # 3. The one semantic collision: dequantize an FP8 head slice with its per-row scales.
    src = open(out).read()
    old_slice = '''    model.register_buffer(
        "_draft_lm_head_weight",
        weight.data.index_select(0, rows).contiguous(),
        persistent=False,
    )
'''
    new_slice = '''    sliced = weight.data.index_select(0, rows).contiguous()
    if sliced.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # %s the FP8-dense checkpoint stores lm_head as
        # FP8 E4M3 + per-output-channel fp32 weight_scale (ModelOptFp8PcPtLinearMethod).
        # F.linear has no float8 path and the scales must not be dropped, so the
        # draft slice is dequantized once, here, into the model dtype: 32k rows x
        # 2560 x 2 B = 160 MiB total, still ~4x below the full FP8 shard read.
        scale = getattr(lm_head, "weight_scale", None)
        if scale is None or scale.dim() != 1 or scale.shape[0] != weight.shape[0]:
            logger.warning(
                "MTP draft vocab: FP8 lm_head without a per-row weight_scale "
                "(%%s); skipping.", None if scale is None else tuple(scale.shape),
            )
            return
        out_dtype = getattr(getattr(lm_head, "quant_method", None), "out_dtype", None)
        if not isinstance(out_dtype, torch.dtype):
            out_dtype = torch.bfloat16
        sliced = (
            sliced.float() * scale.data.index_select(0, rows).float()[:, None]
        ).to(out_dtype).contiguous()
    model.register_buffer("_draft_lm_head_weight", sliced, persistent=False)
''' % STACK_MARK
    assert src.count(old_slice) == 1, f"slice anchor count={src.count(old_slice)}"
    src = src.replace(old_slice, new_slice)

    old_log = "    cut_gib = model._draft_lm_head_weight.numel() * esize / 2**30\n"
    new_log = ("    cut_gib = (model._draft_lm_head_weight.numel()\n"
               "               * model._draft_lm_head_weight.element_size() / 2**30)\n")
    assert src.count(old_log) == 1, f"log anchor count={src.count(old_log)}"
    src = src.replace(old_log, new_log)
    open(out, "w").write(src)

# 4. Sanity: parses, and carries BOTH feature sets plus the bridge.
src = open(out).read()
ast.parse(src)
checks = (
    ("quant_config=draft_vllm_config.quant_config,  # [fp8dense overlay]", "fp8dense H1 (mixer quant_config)"),
    ("quant_config=self.quant_config,  # [fp8dense overlay]", "fp8dense H2 (lm_head quant_config)"),
    ("from vllm.distributed.communication_op import tensor_model_parallel_all_gather", "draftvocab A2 import"),
    ("def _attach_draft_vocab(", "draftvocab A3 helper"),
    ("    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:", "draftvocab A4 method"),
    ("        _attach_draft_vocab(self)\n        return loaded\n", "draftvocab A5 load hook"),
    ("_draft_id_to_target_id", "draftvocab id map"),
    ("torch.float8_e4m3fn, torch.float8_e5m2", "fp8 dequant bridge"),
    (STACK_MARK, "stack marker"),
)
for tok, why in checks:
    assert tok in src, f"stacked mtp_draftvocab.py lost {why} ({tok!r})"
assert src.count("[fp8dense overlay]") == 2
assert "    def compute_logits(" in src and src.count("self.logits_processor(self.lm_head, hidden_states)") == 1
print("stacked OK ->", out)
