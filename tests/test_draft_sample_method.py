#!/usr/bin/env python3
"""CPU-only checks for the probabilistic-drafting knob: the real spec-config
JSON block and the real validation lines are extracted from start.sh and
executed — no launcher run, no GPU, no Docker."""
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "start.sh").read_text()
GUARDS = SOURCE[SOURCE.index('# How the MTP drafter picks tokens'):
                SOURCE.index('# Refuse to launch when another process')]
BLOCK_START = SOURCE.index('        _SPEC_EXTRA=""')
BLOCK_START = SOURCE.rindex('    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then', 0, BLOCK_START)
BLOCK_END = SOURCE.index('    VLLM_ARGS+=("--compilation-config"', BLOCK_START)
BLOCK = SOURCE[BLOCK_START:BLOCK_END]
# Values these lines read but define earlier in the real script.
PRELUDE = (
    'SCRIPT_DIR=/repo; V030="${V030:-false}"; MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-}"\n'
    'MTP_DISABLE_BLOCK_DROP="${MTP_DISABLE_BLOCK_DROP:-0}"\n'
    'MTP_INDEX_SHARE="${MTP_INDEX_SHARE:-false}"\n'
)


def run(script, **env_over):
    env = {"PATH": os.defpath, **env_over}
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c",
         'set -eu; err(){ printf "%s\\n" "$*" >&2; exit 42; }; VLLM_ARGS=()\n'
         + PRELUDE + GUARDS + "\n" + BLOCK + '\nprintf "%s\\n" "${VLLM_ARGS[@]}"'],
        env=env, capture_output=True, text=True)


def spec_json(result):
    lines = result.stdout.splitlines()
    return json.loads(lines[lines.index("--speculative-config") + 1].strip("'"))


class DraftSampleMethodTests(unittest.TestCase):
    def test_greedy_default_payload_unchanged(self):
        r = run("", MTP_NUM_SPECULATIVE_TOKENS="3", MTP_DISABLE_BLOCK_DROP="1",
                MTP_INDEX_SHARE="true", V030="false")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(spec_json(r), {"method": "mtp", "num_speculative_tokens": 3,
                                        "disable_eagle_block_drop": True,
                                        "index_share_for_mtp_iteration": True})

    def test_probabilistic_payload_and_rejection_default(self):
        r = run("", MTP_NUM_SPECULATIVE_TOKENS="3", V030="true",
                MTP_DRAFT_SAMPLE_METHOD="probabilistic")
        self.assertEqual(r.returncode, 0, r.stderr)
        spec = spec_json(r)
        self.assertEqual(spec["draft_sample_method"], "probabilistic")
        self.assertNotIn("rejection_sample_method", spec)  # vLLM default: standard
        self.assertNotIn("use_local_argmax_reduction", spec)

    def test_greedy_never_emits_the_key(self):
        for lane in ["true", "false"]:
            r = run("", MTP_NUM_SPECULATIVE_TOKENS="3", V030=lane,
                    MTP_DRAFT_SAMPLE_METHOD="greedy")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("draft_sample_method", spec_json(r))

    def test_validation_guards(self):
        cases = [
            ({"MTP_DRAFT_SAMPLE_METHOD": "sideways"}, "must be greedy or probabilistic"),
            ({"MTP_DRAFT_SAMPLE_METHOD": "probabilistic", "V030": "false"}, "requires the vLLM 0.30 lane"),
            ({"MTP_DRAFT_SAMPLE_METHOD": "probabilistic", "V030": "true",
              "MTP_DRAFT_VOCAB": "files/vocab.txt"}, "conflicts with MTP_DRAFT_VOCAB"),
        ]
        for env, message in cases:
            with self.subTest(**env):
                r = run("", MTP_NUM_SPECULATIVE_TOKENS="3", **env)
                self.assertEqual(r.returncode, 42)
                self.assertIn(message, r.stderr)

    def test_draft_vocab_greedy_path_still_gets_local_argmax(self):
        r = run("", MTP_NUM_SPECULATIVE_TOKENS="3", V030="true",
                MTP_DRAFT_VOCAB="files/vocab.txt")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(spec_json(r)["use_local_argmax_reduction"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
