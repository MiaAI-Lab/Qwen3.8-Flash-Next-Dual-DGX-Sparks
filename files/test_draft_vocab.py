#!/usr/bin/env python3
"""CPU test for files/patch_mtp_draft_vocab.py's shard slicing and TP reduction.

Runs the *real* patched functions against a fake vocab-parallel lm_head, with
the collective stubbed, and compares the result to a full-vocabulary argmax
restricted to the draft id set. A wrong local-row or id mapping produces valid
tokens from the wrong rows, so it cannot be caught by "does it load" -- only by
checking the ids.

Run inside the image with the patched mtp.py mounted:

  docker run --rm --entrypoint bash \\
    -v $PWD/files/mtp_patched.py:$VLLM/models/qwen3_8_flash_next/nvidia/mtp.py:ro \\
    -v $PWD/files/test_draft_vocab.py:/tmp/t.py:ro \\
    vllm/vllm-openai:qwen38-flash-next -lc 'python3 /tmp/t.py'
"""
import os
import random
import sys
import tempfile

import torch

VOCAB, HIDDEN, TP, TOKENS = 100, 8, 2, 6


class FakeShardIndices:
    def __init__(self, start, end):
        self.org_vocab_start_index = start
        self.org_vocab_end_index = end


class FakeLMHead:
    def __init__(self, weight, start, end):
        self.weight = weight
        self.shard_indices = FakeShardIndices(start, end)
        self.org_vocab_size = VOCAB
        self.tp_size = TP


class FakeLogitsProcessor:
    scale = 1.0


class FakeModel:
    """Just enough of nn.Module for register_buffer + attribute access."""

    def __init__(self, lm_head):
        self.lm_head = lm_head
        self.logits_processor = FakeLogitsProcessor()

    def register_buffer(self, name, tensor, persistent=True):
        setattr(self, name, tensor)


class Captured(Exception):
    def __init__(self, pair):
        self.pair = pair


def main():
    from vllm.models.qwen3_8_flash_next.nvidia import mtp as m

    if not hasattr(m, "_attach_draft_vocab"):
        sys.exit("FAIL: mtp.py is not the patched build")

    torch.manual_seed(0)
    random.seed(0)
    full_weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32)
    hidden = torch.randn(TOKENS, HIDDEN, dtype=torch.float32)

    # A scattered draft vocabulary, deliberately unbalanced across the shards.
    draft_ids = sorted(random.sample(range(VOCAB), 23))
    fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    fh.write("".join(f"{i}\n" for i in draft_ids))
    fh.close()
    os.environ["VLLM_MTP_DRAFT_VOCAB"] = fh.name

    shard = VOCAB // TP
    models = []
    for rank in range(TP):
        start, end = rank * shard, (rank + 1) * shard
        head = FakeLMHead(full_weight[start:end].clone(), start, end)
        model = FakeModel(head)
        m._attach_draft_vocab(model)
        models.append(model)

        # The slice must hold exactly this rank's draft ids, from the right rows.
        expect_ids = [i for i in draft_ids if start <= i < end]
        got_ids = model._draft_id_to_target_id.tolist()
        assert got_ids == expect_ids, f"rank {rank} ids {got_ids} != {expect_ids}"
        if expect_ids:
            assert torch.equal(
                model._draft_lm_head_weight, full_weight[expect_ids]
            ), f"rank {rank} sliced the wrong rows"
        print(f"  rank {rank}: {len(expect_ids)} of {len(draft_ids)} draft ids, "
              f"rows match")

    # Capture each rank's local (value, index) bid without completing the call.
    fn = m.Qwen3_8FlashNextMTP.get_top_tokens
    real_gather = m.tensor_model_parallel_all_gather
    pairs = []
    for model in models:
        m.tensor_model_parallel_all_gather = lambda t, dim=-1: (_ for _ in ()).throw(
            Captured(t)
        )
        try:
            fn(model, hidden)
            sys.exit("FAIL: expected the stubbed collective to fire")
        except Captured as exc:
            pairs.append(exc.pair)

    # Now let the reduction see both ranks, exactly as all_gather(dim=-1) would.
    gathered = torch.cat(pairs, dim=-1)
    m.tensor_model_parallel_all_gather = lambda t, dim=-1: gathered
    got = fn(models[0], hidden)
    m.tensor_model_parallel_all_gather = real_gather

    # Ground truth: argmax over the full head, restricted to the draft ids.
    logits = hidden @ full_weight.T
    mask = torch.full((VOCAB,), float("-inf"))
    mask[draft_ids] = 0.0
    expected = (logits + mask).argmax(dim=-1)

    assert got.dtype == torch.int64, f"dtype {got.dtype}"
    assert torch.equal(got, expected), f"\n got      {got.tolist()}\n expected {expected.tolist()}"
    print(f"  reduction: {got.tolist()} == restricted full-vocab argmax")

    # Every returned id must be inside the draft vocabulary.
    assert set(got.tolist()) <= set(draft_ids), "returned an id outside the draft vocab"

    # A rank owning no draft ids must still lose cleanly rather than crash.
    head = FakeLMHead(full_weight[:shard].clone(), 0, shard)
    empty = FakeModel(head)
    empty.register_buffer("_draft_lm_head_weight", torch.zeros(0, HIDDEN))
    empty.register_buffer("_draft_id_to_target_id", torch.zeros(0, dtype=torch.long))
    m.tensor_model_parallel_all_gather = lambda t, dim=-1: (_ for _ in ()).throw(
        Captured(t)
    )
    try:
        fn(empty, hidden)
        sys.exit("FAIL: expected the stubbed collective to fire")
    except Captured as exc:
        assert torch.isinf(exc.pair[:, 0]).all() and (exc.pair[:, 0] < 0).all(), \
            "empty rank did not bid -inf"
    m.tensor_model_parallel_all_gather = real_gather
    print("  empty rank: bids -inf, no crash")

    os.unlink(fh.name)
    print("\nPASS")


if __name__ == "__main__":
    main()
