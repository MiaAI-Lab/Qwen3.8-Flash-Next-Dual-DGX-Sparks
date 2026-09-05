#!/usr/bin/env python3
"""Patch nvidia/mtp.py for reduced-vocabulary drafting (FR-Spec style), TP-aware.

Adapted from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark (AGPL-3.0-or-later),
which implements this for TP=1 only. This version supports vocab-parallel
lm_heads, which is what this 2-Spark TP=2 deployment needs.

Why it pays here: this checkpoint's vocabulary is 248,320 tokens and the MTP
drafter carries its OWN BF16 ParallelLMHead over all of it (tie_word_embeddings
is false) -- 248320 x 2560 x 2 B = 1.18 GiB, or 0.59 GiB per GPU at TP=2, read
once per draft step. At MTP=3 that is three of the four lm_head reads in an
engine step, ~2.4 GiB of the ~9.2 GiB a single-stream step moves per GPU.

Draft sampling is greedy (the speculator only builds draft_logits for the
"probabilistic" method), so the drafter needs an argmax and nothing else. An
argmax over a frequency-ranked subset is wrong only for tokens outside the
subset, and a wrong draft is *rejected by the target model at verification*,
exactly like any other bad draft. This trades acceptance for bandwidth and
cannot change what the server emits.

The reduced head is reached only through get_top_tokens, which the speculator
calls when use_local_argmax_reduction is set. compute_logits keeps the full
head, so every other path keeps full-vocabulary behaviour.

TP notes (the part that differs from upstream): each rank's lm_head holds the
org-vocab rows [org_vocab_start_index, org_vocab_end_index). Each rank slices
its own shard to the draft ids landing in its range, takes a local argmax, and
the winners are reduced across ranks with the same (value, index) all-gather
vLLM's own LogitsProcessor.get_top_tokens uses -- O(batch * 2 * tp_size)
instead of O(batch * vocab_size).

Both the logit scale and the tanh soft cap are monotonic and applied
identically on every rank, so skipping them leaves both the local argmax and
the cross-rank comparison unchanged.

Inputs:  files/mtp_patched.py.orig   (nvidia/mtp.py from the image)
Outputs: files/mtp_patched.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "mtp_patched.py.orig")
OUT = os.path.join(HERE, "mtp_patched.py")

DRAFT_VOCAB_BLOCK = '''

def _attach_draft_vocab(model: nn.Module) -> None:
    """Slice this rank's lm_head shard down to VLLM_MTP_DRAFT_VOCAB's token ids.

    The file is one integer token id per line (see files/build_draft_vocab.py).
    The full head is left in place and untouched; only get_top_tokens below
    reads the slice.
    """
    path = os.environ.get("VLLM_MTP_DRAFT_VOCAB", "").strip()
    if not path:
        return
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dim() != 2:
        logger.warning("MTP draft vocab: no 2-D lm_head weight; skipping.")
        return
    scale = getattr(model.logits_processor, "scale", 1.0)
    if scale <= 0.0:
        logger.warning("MTP draft vocab: non-positive logit scale; skipping.")
        return
    shard = getattr(lm_head, "shard_indices", None)
    if shard is None:
        logger.warning("MTP draft vocab: lm_head has no shard_indices; skipping.")
        return

    org_vocab = int(getattr(lm_head, "org_vocab_size", weight.shape[0]))
    try:
        with open(path) as handle:
            ids = sorted({int(line) for line in handle if line.strip()})
    except OSError as exc:
        logger.warning("MTP draft vocab: cannot read %s (%s); skipping.", path, exc)
        return
    ids = [i for i in ids if 0 <= i < org_vocab]
    if not ids or len(ids) >= org_vocab:
        logger.warning(
            "MTP draft vocab: %d usable ids against a %d vocabulary; skipping.",
            len(ids), org_vocab,
        )
        return

    # Keep only the ids this rank actually owns, as local row offsets.
    start = int(shard.org_vocab_start_index)
    end = int(shard.org_vocab_end_index)
    local_ids = [i for i in ids if start <= i < end]
    rows = torch.tensor([i - start for i in local_ids], dtype=torch.long,
                        device=weight.device)
    model.register_buffer(
        "_draft_lm_head_weight",
        weight.data.index_select(0, rows).contiguous(),
        persistent=False,
    )
    model.register_buffer(
        "_draft_id_to_target_id",
        torch.tensor(local_ids, dtype=torch.long, device=weight.device),
        persistent=False,
    )
    tp_size = int(getattr(lm_head, "tp_size", 1))
    esize = weight.element_size()
    full_gib = weight.numel() * esize / 2**30
    cut_gib = model._draft_lm_head_weight.numel() * esize / 2**30
    logger.info(
        "MTP draft vocab: %d of %d tokens (%.1f%%), %d on this rank of %d; "
        "draft lm_head shard %.2f -> %.2f GiB per draft step",
        len(ids), org_vocab, 100.0 * len(ids) / org_vocab, len(local_ids),
        tp_size, full_gib, cut_gib,
    )

'''

GET_TOP_TOKENS = '''
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft token ids, over the reduced vocabulary when present.

        Mirrors LogitsProcessor.get_top_tokens: local argmax per rank, then a
        (value, index) all-gather instead of an all-gather of full logits. The
        logit scale and soft cap are monotonic and identical on every rank, so
        skipping them changes neither the local argmax nor the cross-rank
        comparison; _attach_draft_vocab refuses to engage when the scale is not
        positive.
        """
        weight = getattr(self, "_draft_lm_head_weight", None)
        if weight is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

        num_tokens = hidden_states.shape[0]
        if weight.shape[0] == 0:
            # This rank owns none of the draft vocabulary: contribute a losing bid.
            local_max_vals = torch.full((num_tokens,), float("-inf"),
                                        dtype=torch.float32, device=hidden_states.device)
            global_indices = torch.zeros((num_tokens,), dtype=torch.long,
                                         device=hidden_states.device)
        else:
            logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
            local_max_vals, local_max_indices = logits.max(dim=-1)
            global_indices = self._draft_id_to_target_id[local_max_indices]

        tp_size = int(getattr(self.lm_head, "tp_size", 1))
        if tp_size == 1:
            return global_indices.to(torch.int64)

        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(num_tokens, tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

'''


def patch(edits):
    src = open(ORIG).read()
    for i, (old, new) in enumerate(edits):
        count = src.count(old)
        if count != 1:
            sys.exit(f"mtp_patched: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"mtp_patched: patched source does not parse: {exc}")
    open(OUT, "w").write(src)
    print("patched mtp_patched.py")


def main():
    if not os.path.isfile(ORIG):
        sys.exit(f"ERROR: missing {ORIG} (start.sh extracts it from the image)")
    if os.path.isfile(OUT) and "_attach_draft_vocab" in open(OUT).read():
        print("mtp_patched.py already patched")
        return
    patch([
        # os, a logger, and the all-gather primitive the reduction needs.
        (
            "from vllm.compilation.decorators import support_torch_compile\n",
            "import os\n\n"
            "from vllm.compilation.decorators import support_torch_compile\n"
            "from vllm.logger import init_logger\n",
        ),
        (
            "from vllm.distributed import get_pp_group\n",
            "from vllm.distributed import get_pp_group\n"
            "from vllm.distributed.communication_op import "
            "tensor_model_parallel_all_gather\n",
        ),
        # Module scope, before the first helper: the MTP class itself sits under
        # a @support_torch_compile decorator, so nothing can go between.
        (
            "def _remap_ignored_layers(\n",
            "logger = init_logger(__name__)\n"
            + DRAFT_VOCAB_BLOCK
            + "\ndef _remap_ignored_layers(\n",
        ),
        # get_top_tokens beside compute_logits, which stays full-vocabulary.
        (
            "    def compute_logits(\n"
            "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
            "    ) -> torch.Tensor | None:\n"
            "        return self.logits_processor(self.lm_head, hidden_states)\n",
            "    def compute_logits(\n"
            "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
            "    ) -> torch.Tensor | None:\n"
            "        return self.logits_processor(self.lm_head, hidden_states)\n"
            + GET_TOP_TOKENS,
        ),
        # Slice once the real weights are in.
        (
            "        return loader.load_weights(remap_weight_names())\n",
            "        loaded = loader.load_weights(remap_weight_names())\n"
            "        _attach_draft_vocab(self)\n"
            "        return loaded\n",
        ),
    ])


if __name__ == "__main__":
    main()
