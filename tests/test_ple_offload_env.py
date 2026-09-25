#!/usr/bin/env python3
"""CPU-only tests of PLE placement in the actual two-rank launch templates.

Extracts the real resolution block and both real heredoc templates from
start.sh, renders them with a sanitized environment, and executes the
rendered worker/head scripts with a shell-function `docker` stub. Nothing
here contacts a GPU, SSH, the Docker daemon or the network.
"""
import os
from pathlib import Path
import re
import subprocess
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "start.sh").read_text()
DEFAULTS = SOURCE[SOURCE.index('PLE_OFFLOAD="${PLE_OFFLOAD:-false}"'):
                  SOURCE.index('# Vision MLP')]
RENDER = SOURCE[SOURCE.index('    # PLE table placement'):
               SOURCE.index('    # Write worker launch script')]
TEMPLATES = re.findall(
    r'cat > "\$(WORKER|HEAD)_SCRIPT" <<LAUNCH_EOF\n(.*?)\nLAUNCH_EOF',
    SOURCE, re.S,
)


class PleOffloadTests(unittest.TestCase):
    def shell(self, value, script, extra=None):
        # Do not inherit the operator's .env, credentials or runtime settings.
        env = {"PATH": os.defpath, **(extra or {})}
        if value is not None:
            env["PLE_OFFLOAD"] = value
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-c",
             'set -eu; err(){ printf "%s\n" "$*" >&2; exit 42; };\n'
             + DEFAULTS + "\n" + RENDER + "\n" + script],
            env=env, capture_output=True, text=True,
        )

    def test_explicit_boolean_and_default_resolution(self):
        for value, expected in [(None, "0"), ("", "0"), ("false", "0"), ("true", "1")]:
            with self.subTest(value=value):
                result = self.shell(value, 'printf "%s" "$PLE_OFFLOAD_ENV"')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "-e VLLM_PLE_CPU_OFFLOAD=" + expected)

    def test_docker_args_mirror_renders_explicitly(self):
        # The head-side DOCKER_ARGS mirror must carry the same unconditional
        # rendering as the rank heredocs.
        self.assertIn('DOCKER_ARGS+=(-e "VLLM_PLE_CPU_OFFLOAD=$PLE_CPU_OFFLOAD_VALUE")', SOURCE)
        self.assertNotIn('if [[ "$PLE_OFFLOAD" == "true" ]]; then', SOURCE)

    def test_invalid_values_fail_before_rendering(self):
        for value in ["0", "1", "False", "TRUE", "garbage", "false "]:
            with self.subTest(value=value):
                result = self.shell(value, 'printf "SHOULD_NOT_RENDER"')
                self.assertEqual(result.returncode, 42)
                self.assertIn("PLE_OFFLOAD must be true or false", result.stderr)
                self.assertNotIn("SHOULD_NOT_RENDER", result.stdout)

    def test_both_executed_rank_commands_and_unchanged_engine_arguments(self):
        self.assertEqual([role for role, _ in TEMPLATES], ["WORKER", "HEAD"])
        for lane in ["false", "true"]:
            for role, template in TEMPLATES:
                observed = []
                for value, expected in [(None, "0"), ("", "0"), ("false", "0"), ("true", "1")]:
                    with self.subTest(lane=lane, role=role, value=value):
                        names = set(re.findall(r'\$\{?([A-Z][A-Z0-9_]*)', template))
                        env = {name: "" for name in names}
                        env.update(HOME="/home/head", REMOTE_HOME="/home/worker",
                                   IMAGE="example-image", MODEL_ID="example-model", V030=lane)
                        rendered = self.shell(value, "cat <<LAUNCH_EOF\n" + template + "\nLAUNCH_EOF", env)
                        self.assertEqual(rendered.returncode, 0, rendered.stderr)
                        # Execute only the rendered script with a shell-function
                        # docker stub: no daemon, SSH, GPU or network contact.
                        executed = subprocess.run(
                            ["bash", "--noprofile", "--norc", "-c",
                             'set -eu; docker(){ printf "%s\\0" "$@"; };\n' + rendered.stdout],
                            env={"PATH": os.defpath}, capture_output=True, check=True,
                        )
                        argv = executed.stdout.decode().split("\0")[:-1]
                        placement = [a for a in argv if a.startswith("VLLM_PLE_CPU_OFFLOAD=")]
                        self.assertEqual(placement, ["VLLM_PLE_CPU_OFFLOAD=" + expected])
                        index = argv.index(placement[0])
                        self.assertEqual(argv[index - 1], "-e")
                        self.assertEqual(argv[argv.index("--node-rank") + 1], "1" if role == "WORKER" else "0")
                        observed.append(argv[:index - 1] + argv[index + 1:])
                # The placement flag is the only launch difference between the
                # four resolutions on this rank; nothing else moves.
                self.assertTrue(all(a == observed[0] for a in observed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
