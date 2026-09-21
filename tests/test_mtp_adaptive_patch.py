# SPDX-License-Identifier: AGPL-3.0-or-later
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'files'))
import patch_mtp_adaptive as patch

EXPECTED_PATCHED = {
    'qwen_scheduler_adaptive.py': '85ccc2257a7f4582f873001ff7d997216d4bd5a849d0b1d03f82c90d53d5dbc8',
    'qwen_model_runner_adaptive.py': '083a7b36980d4e2d29168633abf6dba3f12d9773e17ad49b7c24aedf0735be28',
    'qwen_autoregressive_adaptive.py': '56e9a3e3aecdf9ed0162cd07d3582d2796b04e332543e130ceb703b7ca3005c3',
    'qwen_mtp_speculator_adaptive.py': '1eb9ae7a7c461d77c9711bcb7209f349bb4f9bc177ebc69f5190e3fe05b00546',
    'qwen_cudagraph_adaptive.py': '67364b87c45bfb0be55ebf15bc0dee49ccea9fedb43b7cb9e1ce54e03e5ac1bf',
}


class PatchTests(unittest.TestCase):
    def test_anchor_mismatch_and_ambiguity_refused(self):
        for text in ('absent', 'anchor anchor'):
            with self.assertRaises(ValueError):
                patch.replace(text, 'anchor', 'new')
        self.assertEqual(patch.replace('an anchor', 'anchor', 'update'), 'an update')

    def test_unknown_image_does_not_publish_any_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d) / 'original', Path(d) / 'out'
            for target in patch.TARGETS.values():
                p = root / target
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('# unknown image\n')
            with self.assertRaisesRegex(ValueError, 'Unsupported vLLM source'):
                patch.build(root, out)
            self.assertFalse(out.exists())

    def test_original_source_tree_cannot_be_an_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'outside'):
                patch.build(Path(d), Path(d) / 'out')

    def test_unique_worker_basenames(self):
        self.assertEqual(len(set(patch.TARGETS)), 5)
        self.assertNotIn('qwen_mtp_adaptive.py', patch.TARGETS)

    @unittest.skipUnless(os.environ.get('MTP_TEST_SOURCE_ROOT'), 'optional pinned-image source fixture')
    def test_generated_overlays_match_measured_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            patch.build(Path(os.environ['MTP_TEST_SOURCE_ROOT']), Path(d) / 'out')
            for name, expected in EXPECTED_PATCHED.items():
                data = (Path(d) / 'out' / name).read_bytes()
                ast.parse(data)
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected, name)


if __name__ == '__main__':
    unittest.main()
