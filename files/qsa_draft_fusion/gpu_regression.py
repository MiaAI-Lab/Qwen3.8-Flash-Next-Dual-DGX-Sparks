# SPDX-License-Identifier: Apache-2.0
# From vLLM PR58449 at848dbf89ecb00408351e00c39689d3a8c2f1a4c9, standalone execution.
from types import SimpleNamespace
import torch
# Follow engine import order to avoid standalone config import cycles.
from vllm.engine.arg_utils import EngineArgs
from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
def test_qsa_draft_decode_metadata_update_matches_rebuild(layout: str) -> None:
    """A fused draft step's in-place update equals a fresh build at seq_lens + 1."""
    device = torch.device("cuda")
    compress_ratio = 4 if layout != "plain" else 1

    def make_builder() -> QSAMetadataBuilder:
        builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
        builder.compress_ratio = compress_ratio
        builder.reorder_batch_threshold = 1
        builder.is_circular_buffer = layout == "circular"
        builder.kv_cache_spec = SimpleNamespace(block_size=4)
        builder.storage_block_size = 16 if layout == "compressed" else 4
        builder.token_to_req_buffer = torch.empty(8, dtype=torch.int32, device=device)
        builder.slot_mapping_buffer = torch.empty(8, dtype=torch.int64, device=device)
        builder.logical_positions_buffer = torch.empty(
            8, dtype=torch.int64, device=device
        )
        builder.visible_blocks_buffer = torch.empty(8, dtype=torch.int32, device=device)
        builder.request_capacity = 8
        builder.k_work_metadata_buffer = torch.empty(
            8 if layout == "compressed" else 0, 2, dtype=torch.int32, device=device
        )
        return builder

    query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    def make_common(seq_lens: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            num_actual_tokens=3,
            num_reqs=3,
            max_query_len=1,
            max_seq_len=int(seq_lens.max()),
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens,
            slot_mapping=torch.tensor([5, 9, 20], dtype=torch.int64, device=device),
            block_table_tensor=torch.tensor(
                [[3, 1], [0, 4], [2, 5]], dtype=torch.int32, device=device
            ),
            token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
        )

    # 7 -> 8 crosses a compression boundary, so visible_blocks must change.
    seq_lens = torch.tensor([7, 11, 16], dtype=torch.int32, device=device)
    builder = make_builder()
    metadata = builder.build(0, make_common(seq_lens))
    positions_before = metadata.logical_positions.clone()

    seq_lens.add_(1)
    builder.update_draft_decode_metadata(metadata)
    expected = make_builder().build(0, make_common(seq_lens.clone()))

    assert not torch.equal(metadata.logical_positions, positions_before)
    for name in ("token_to_req", "logical_positions", "visible_blocks", "slot_mapping"):
        torch.testing.assert_close(getattr(metadata, name), getattr(expected, name))
    torch.testing.assert_close(metadata.k_work_metadata, expected.k_work_metadata)


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'GPU test requires CUDA'
    for layout in ('plain', 'compressed', 'circular'):
        test_qsa_draft_decode_metadata_update_matches_rebuild(layout)
        torch.cuda.synchronize()
        print('QSA_DRAFT_METADATA_PASS', layout, flush=True)
