# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
WIN_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = os.environ.get('BASH_FOR_TESTS') or (str(WIN_BASH) if os.name == 'nt' and WIN_BASH.exists() else shutil.which('bash'))


@unittest.skipUnless(BASH, 'bash unavailable')
class LauncherJSONTests(unittest.TestCase):
    def render(self, share, vocab, k):
        script = (ROOT / 'start.sh').read_text(encoding='utf-8')
        start = script.index('    # JSON args: use printf')
        end = script.index('    VLLM_ARGS+=("--compilation-config"', start)
        fragment = script[start:end]
        prefix = (f'set -euo pipefail\nVLLM_ARGS=()\nMTP_INDEX_SHARE={share}\n'
                  f'MTP_NUM_SPECULATIVE_TOKENS={k}\nMTP_DRAFT_VOCAB={vocab}\n')
        result = subprocess.run([BASH, '-c', prefix + fragment + '\nprintf "%s\\n" "${VLLM_ARGS[1]}"'],
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout.strip().strip("'"))

    def test_default_json_unchanged(self):
        self.assertEqual(self.render('false', 'vocab.txt', 3),
                         dict(method='mtp', num_speculative_tokens=3, use_local_argmax_reduction=True))

    def test_indexshare_json_with_reduced_vocab(self):
        self.assertEqual(self.render('true', 'vocab.txt', 4),
                         dict(method='mtp', num_speculative_tokens=4,
                              use_local_argmax_reduction=True, index_share_for_mtp_iteration=True))

    def test_full_vocab_json(self):
        self.assertEqual(self.render('true', "''", 4),
                         dict(method='mtp', num_speculative_tokens=4, index_share_for_mtp_iteration=True))


if __name__ == '__main__':
    unittest.main()
