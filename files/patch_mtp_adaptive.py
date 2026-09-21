# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the measured variable-depth MTP overlays from the pinned image.

CPU-only. Does not import vLLM/torch, change the input tree, or start containers.
Original vLLM SPDX headers are preserved in generated files. Refuse unknown
sources instead of silently applying a stale overlay after an image update.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMPORT = 'from vllm.v1.core.sched.qwen_mtp_adaptive import ENABLED as QWEN_MTP_ADAPTIVE, runtime_depth\n'
TARGETS = {
    "qwen_scheduler_adaptive.py": "v1/core/sched/scheduler.py",
    "qwen_model_runner_adaptive.py": "v1/worker/gpu/model_runner.py",
    "qwen_autoregressive_adaptive.py": "v1/worker/gpu/spec_decode/autoregressive/speculator.py",
    "qwen_mtp_speculator_adaptive.py": "v1/worker/gpu/spec_decode/mtp/speculator.py",
    "qwen_cudagraph_adaptive.py": "v1/worker/gpu/cudagraph_utils.py"
}
SOURCE_SHA256 = {
    "v1/core/sched/scheduler.py": "c710f49e41e974e5b7b8f1cd2f5fb9722523f4da0165252e16351199ebd03124",
    "v1/worker/gpu/model_runner.py": "f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8",
    "v1/worker/gpu/spec_decode/autoregressive/speculator.py": "575f39930f7b3a89402c385885d598416137b72e51fea83f2320a3212b5b99e1",
    "v1/worker/gpu/spec_decode/mtp/speculator.py": "1fcffbf5e5a85e4c901bd71c65a73da814e7273627276dbdfebf367720a0bc1a",
    "v1/worker/gpu/cudagraph_utils.py": "ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c"
}

def replace(src, old, new, count=1):
    actual = src.count(old)
    if actual != count:
        raise ValueError(f"Source anchor mismatch: expected {count}, got {actual}: {old[:100]!r}")
    return src.replace(old, new)


def build(source_root, out):
    source_root, out = Path(source_root).resolve(), Path(out).resolve()
    if out == source_root or source_root in out.parents:
        raise ValueError("Output must be outside the original source tree")
    for name, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unsupported vLLM source {name}: {actual}; expected {expected}. "
                             "Disable MTP_ADAPTIVE or review the patch for this image.")
    pending, manifest = {}, {}

    def save(name, source, text):
        ast.parse(text)
        pending[name] = text
        manifest[name] = dict(
            original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            patched_sha256=hashlib.sha256(text.encode()).hexdigest())

    p = source_root / 'v1/core/sched/scheduler.py'
    s = p.read_text(encoding='utf-8')
    s = replace(s, 'import time\n', 'import time\nfrom vllm.v1.core.sched.qwen_mtp_adaptive import create_controller\n')
    s = replace(s, '        self.dynamic_sd_lookup: list[int] | None = None\n',
                '        self.qwen_mtp_controller = create_controller(vllm_config)\n        self.dynamic_sd_lookup: list[int] | None = None\n')
    s = replace(s, '        scheduled_encoder_input_stats = None\n',
                '        if self.qwen_mtp_controller is not None:\n            num_spec_tokens_to_schedule = self.qwen_mtp_controller.choose(num_scheduled_tokens)\n\n        scheduled_encoder_input_stats = None\n')
    s = replace(s, '        sampled_token_ids = model_runner_output.sampled_token_ids\n',
                '        if self.qwen_mtp_controller is not None:\n            self.qwen_mtp_controller.observe(scheduler_output, model_runner_output)\n        sampled_token_ids = model_runner_output.sampled_token_ids\n')
    # Do not insert max-K speculative padding into requests joining variable-K batches.
    s = replace(s, '(self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)',
                '(self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None and self.qwen_mtp_controller is None)')
    save('qwen_scheduler_adaptive.py', p, s)

    p = source_root / 'v1/worker/gpu/model_runner.py'
    s = replace(p.read_text(encoding='utf-8'), 'import functools\n', 'import functools\n' + IMPORT)
    s = replace(s, '        if not dummy_run:\n            # Update the request states.\n',
                '        if not dummy_run:\n            if QWEN_MTP_ADAPTIVE and self.speculator is not None:\n                self.speculator._qwen_active_steps = runtime_depth(scheduler_output.num_spec_tokens_to_schedule, self.num_speculative_steps)\n            # Update the request states.\n')
    s = replace(s, '            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n',
                '            self.req_states.draft_tokens[input_batch.idx_mapping, :draft_tokens.shape[1]] = draft_tokens\n')
    s = replace(s, '                self.req_states.draft_tokens[input_batch.idx_mapping],\n',
                '                self.req_states.draft_tokens[input_batch.idx_mapping, :getattr(self.speculator, "_qwen_active_steps", self.num_speculative_steps)],\n')
    save('qwen_model_runner_adaptive.py', p, s)

    p = source_root / 'v1/worker/gpu/spec_decode/autoregressive/speculator.py'
    s = replace(p.read_text(encoding='utf-8'), 'from typing import Any\n', 'from typing import Any\n' + IMPORT)
    s = replace(s, '    def _configure_fused_multi_step_decode(self) -> None:\n',
                '    def _configure_fused_multi_step_decode(self) -> None:\n        if QWEN_MTP_ADAPTIVE:\n            self.use_fused_multi_step_decode = False\n            return\n')
    s = replace(s, '        if self.num_speculative_steps == 1:\n            # Early exit.\n',
                '        if getattr(self, "_qwen_active_steps", self.num_speculative_steps) == 1:\n            # Early exit.\n')
    s = replace(s, '        return self.draft_tokens[:num_reqs]\n',
                '        return self.draft_tokens[:num_reqs, :getattr(self, "_qwen_active_steps", self.num_speculative_steps)]\n')
    # Only the non-fused runtime loop; leave capture-time fused loop untouched.
    s = replace(s, '        for step in range(1, self.num_speculative_steps):\n            # Rebuild every step',
                '        for step in range(1, getattr(self, "_qwen_active_steps", self.num_speculative_steps)):\n            # Rebuild every step')
    save('qwen_autoregressive_adaptive.py', p, s)

    p = source_root / 'v1/worker/gpu/spec_decode/mtp/speculator.py'
    s = replace(p.read_text(encoding='utf-8'),
                '        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:\n',
                '        if self.share_mtp_topk_indices and getattr(self, "_qwen_active_steps", self.num_speculative_steps) > 1:\n')
    save('qwen_mtp_speculator_adaptive.py', p, s)

    p = source_root / 'v1/worker/gpu/cudagraph_utils.py'
    s = replace(p.read_text(encoding='utf-8'), 'from collections import defaultdict\n', 'from collections import defaultdict\n' + IMPORT)
    s = replace(s, '        capture_varlen_decode = (\n',
                '        if QWEN_MTP_ADAPTIVE and self.decode_query_len == self.vllm_config.num_speculative_tokens + 1:\n            decode_query_lens = list(range(2, self.decode_query_len + 1))\n\n        capture_varlen_decode = (\n')
    save('qwen_cudagraph_adaptive.py', p, s)

    controller = ROOT / 'mtp_adaptive/qwen_mtp_adaptive.py'
    save(controller.name, controller, controller.read_text(encoding='utf-8'))
    # Publish only after every source and every generated AST has passed.
    out.mkdir(parents=True, exist_ok=True)
    for name, text in pending.items():
        (out / name).write_text(text, encoding='utf-8', newline='\n')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                     encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-sources', action='store_true')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.list_sources:
        for output, target in TARGETS.items():
            print(f'{output}|{target}')
        return
    if args.source_root is None or args.output_dir is None:
        parser.error('--source-root and --output-dir are required')
    try:
        manifest = build(args.source_root, args.output_dir)
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(1, f'adaptive MTP patch refused: {exc}\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
