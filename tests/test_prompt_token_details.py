#!/usr/bin/env python3
"""CPU-only base-argv checks; never execute the launchers or contact a GPU."""
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAG = "--enable-prompt-tokens-details"


class PromptTokenDetailsTests(unittest.TestCase):
    def test_dual_and_tp1_base_args(self):
        for name in ("start.sh", "tp1/start.sh"):
            with self.subTest(launcher=name):
                source = (ROOT / name).read_text()
                blocks = re.findall(
                    r'^\s*VLLM_ARGS=\(\)\n((?:[ \t]*VLLM_ARGS\+=\([^\n]*\)\n)+)',
                    source, re.M,
                )
                self.assertEqual(len(blocks), 1)
                argv = subprocess.check_output(
                    ["bash", "--noprofile", "--norc", "-c",
                     'VLLM_ARGS=()\n' + blocks[0] + '\nprintf "%s\\0" "${VLLM_ARGS[@]}"'],
                    env={"PATH": os.defpath},
                ).decode().split("\0")[:-1]
                self.assertEqual(argv.count(FLAG), 1)
                self.assertNotIn("--enable-prompt-token-details", source)
                self.assertNotIn("--no-enable-prompt-tokens-details", source)
                # Both the head and worker consume this one shared argv string.
                self.assertEqual(source.count('VLLM_ARGS_STR="${VLLM_ARGS[*]}"'), 1)
                self.assertEqual(source.count('$VLLM_ARGS_STR \\'), 2 if name == "start.sh" else 1)


if __name__ == "__main__":
    unittest.main()
