#!/usr/bin/env python3
"""CPU-only regression checks for the persistent Triton compile-cache mount.

WHAT THESE TESTS ARE
    start.sh renders the worker and head launch scripts through two
    unquoted LAUNCH_EOF heredocs and then *executes them for real*: the
    worker script is fed to ``ssh ... bash -s`` and the head script runs with
    ``bash <file>``. This suite reproduces that exact execution path on a
    CPU-only machine by running start-v030.sh/start.sh with stubbed
    ``ssh``, ``docker``, ``scp``, ``rsync``, ``curl``, ``sleep`` and
    ``nvidia-smi`` binaries. The stub docker records every ``docker run``
    argv, so the assertions below read the **actual** rendered-and-executed
    launch commands — mounts, environment, and engine arguments — rather
    than a regex over the template text.

WHAT THESE TESTS ARE NOT
    No container is started, no GPU is touched, nothing leaves the machine,
    and no timing claim is made. The render probe proves launch
    *composition*, not a real two-node launch, and it measures no speed or
    startup benefit. The stock (day-0) lane gets static template checks only,
    because rendering it end-to-end would need the image's real vLLM sources
    for the PLE / MXFP8 extraction-and-patch steps; the v0.30 lane skips
    those patches, so it renders fully under stubs.

Run from the repo root:
    python3 -m unittest tests.test_persistent_triton_cache -v
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Exact image ID the stub `docker image inspect` reports (sha256 IDs contain
# a colon, which the launcher sanitises out of the host directory name).
STUB_IMAGE_ID = "sha256:aaaabbbbccccddddeeeeffff0000111122223333444455556666777788889999"
OTHER_IMAGE_ID = "sha256:9999888877776666555544443333222211110000ffffeeeeddddccccbbbbaaaa0001"
TEST_IMAGE = "vllm/vllm-openai:test-image"

TRITON_DIR = "/root/.triton"
FLASHINFER_AUTOTUNE = "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/fi_autotune"
MODEL_TOKEN = "testorg/testmodel"


def sanitized(key: str) -> str:
    """Mirror of the launcher's sanitizer: keep [A-Za-z0-9._-], replace rest."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", key)


BASE_ENV = {
    "HEAD_IP": "127.0.0.1",
    "WORKER_IP": "127.0.0.2",
    "WORKER_USER": "testuser",
    "IFACE": "lo",
    "IB_HCA": "=rocestub0",
    "IB_GID_INDEX": "3",
    "MODEL_ID": MODEL_TOKEN,
    "SERVED_MODEL_NAME": "testmodel",
    "MAX_MODEL_LEN": "262144",
    "YARN_ENABLE": "false",
    "GPU_MEMORY_UTILIZATION": "0.80",
    "MAX_NUM_SEQS": "1",
    "MAX_NUM_BATCHED_TOKENS": "256",
    "PORT": "18888",
    "TENSOR_PARALLEL_SIZE": "2",
    "ENABLE_EXPERT_PARALLEL": "true",
    "MTP_NUM_SPECULATIVE_TOKENS": "3",
    "KV_CACHE_DTYPE": "auto",
    "MM_ENCODER_TP_MODE": "data",
    "MTP_DRAFT_VOCAB": "",
    "QSA_PROFILE": "stock",
    "REQUIRE_IDLE_GPU": "false",
    "EVICT_PAGE_CACHE": "false",
    "IMAGE": TEST_IMAGE,
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    # start-v030.sh honours a pre-set OVERRIDE_IMAGE; pin it so the render
    # exercises this fixture's stubbed image reference on both lanes.
    "OVERRIDE_IMAGE": TEST_IMAGE,
    "MASTER_PORT": "50001",
    "NFS_SHARE": "false",
    "PLE_OFFLOAD": "false",
    "FP8_DENSE": "false",
}

# ---------------------------------------------------------------------------
# Stubs. Every "remote" command executes locally with HOME rewritten to the
# fake worker home, so worker ~/.cache paths are real, checkable directories
# that stay distinct from the fake head home.
# ---------------------------------------------------------------------------
SSH_STUB = """#!/usr/bin/env bash
# Stub ssh for tests/test_persistent_triton_cache.py: drops ssh option args
# and the target host, runs the command string locally as the worker.
export HOME="$FAKE_WORKER_HOME"
skip=0
cmd=""
for a in "$@"; do
    if (( skip )); then skip=0; continue; fi
    case "$a" in
        -o) skip=1; continue ;;
    esac
    case "$a" in
        *@127.0.0.2|127.0.0.2) continue ;;
    esac
    cmd+="${cmd:+ }$a"
done
if [[ "$cmd" == *'echo "$HOME"'* ]]; then echo "$FAKE_WORKER_HOME"; exit 0; fi
if [[ "$cmd" == *"bash -s"* ]]; then exec bash -s; fi
case "$cmd" in
    *"docker image inspect"*|python3*|"test -d"*|"du -sh"*|"docker rm"*|mkdir\\ *|"ref="*|*"docker pull"*)
        exec bash -c "$cmd" ;;
esac
echo "STUB-SSH-UNHANDLED: $cmd" >&2
exit 127
"""

DOCKER_STUB = """#!/usr/bin/env bash
# Stub docker: `docker run` argv is appended to $RUN_LOG.$$ for the test to
# read back; the inspect/ps/rm/pull queries start.sh makes are answered here.
case "${1:-}" in
    run)
        printf '%s\\n' "${@:2}" >> "$RUN_LOG.$$"
        exit 0
        ;;
    image)
        if [[ "${2:-}" == inspect && -n "${IMAGE_ID:-}" ]]; then
            echo "$IMAGE_ID"; exit 0
        fi
        exit 1
        ;;
    ps)   echo vllm-fn; exit 0 ;;
    logs) exit 0 ;;
    rm)   exit 0 ;;
    pull) exit 0 ;;
    *)    exit 0 ;;
esac
"""

CURL_STUB = """#!/usr/bin/env bash
printf '200'
exit 0
"""

SLEEP_STUB = """#!/usr/bin/env bash
exit 0
"""

RSYNC_STUB = """#!/usr/bin/env bash
# A real sync is never needed: the worker fixture snapshot is pre-seeded.
exit 0
"""

SCP_STUB = """#!/usr/bin/env bash
exit 0
"""

NVSMI_STUB = """#!/usr/bin/env bash
exit 0
"""


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.write_text(text)
    if executable:
        path.chmod(0o755)


class RenderProbe:
    """One isolated execution of the real launcher under stubs."""

    def __init__(self, *, env_overrides: dict, image_id: str | None):
        self.work = Path(tempfile.mkdtemp(prefix="triton-render-"))
        self.head_home = self.work / "head_home"
        self.worker_home = self.work / "worker_home"
        self.bin = self.work / "bin"
        self.runlog = self.work / "docker-run"
        self._run(env_overrides, image_id)

    # -- fixture --------------------------------------------------------
    def _seed_hub_copy(self, home: Path) -> None:
        repo = home / ".cache/huggingface/hub/models--testorg--testmodel"
        snap = repo / "snapshots/rev1"
        snap.mkdir(parents=True)
        (repo / "refs").mkdir(parents=True)
        (repo / "refs/main").write_text("rev1")
        (snap / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a": "model-00001.safetensors"}}'
        )
        (snap / "model-00001.safetensors").write_text("weights")
        (snap / "config.json").write_text(
            '{"text_config": {"num_hidden_layers": 48}}'
        )

    def _run(self, env_overrides: dict, image_id: str | None) -> None:
        self.bin.mkdir(parents=True)
        _write(self.bin / "ssh", SSH_STUB, executable=True)
        _write(self.bin / "docker", DOCKER_STUB, executable=True)
        _write(self.bin / "curl", CURL_STUB, executable=True)
        _write(self.bin / "sleep", SLEEP_STUB, executable=True)
        _write(self.bin / "rsync", RSYNC_STUB, executable=True)
        _write(self.bin / "scp", SCP_STUB, executable=True)
        _write(self.bin / "nvidia-smi", NVSMI_STUB, executable=True)
        (self.bin / "python3").symlink_to(sys.executable)
        self._seed_hub_copy(self.head_home)
        self._seed_hub_copy(self.worker_home)

        app = self.work / "app"
        app.mkdir()
        for name in ("start.sh", "start-v030.sh"):
            shutil.copyfile(ROOT / name, app / name)
            (app / name).chmod(0o755)
        shutil.copytree(ROOT / "files", app / "files")
        (app / "download.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        env = dict(BASE_ENV, **env_overrides)
        quoted = []
        for k, v in env.items():
            quoted.append(f"{k}='{str(v).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'")
        (app / ".env").write_text("\n".join(quoted) + "\n")

        proc_env = {
            "PATH": os.pathsep.join([str(self.bin), os.defpath]),
            "HOME": str(self.head_home),
            "FAKE_WORKER_HOME": str(self.worker_home),
            "IMAGE_ID": image_id or "",
            "RUN_LOG": str(self.runlog),
            "TMPDIR": str(self.work),
        }
        # The v0.30 lane is the one that renders end-to-end without the
        # image's real sources (day-0 overlays are all skipped there).
        self.proc = subprocess.run(
            ["bash", str(app / "start-v030.sh"), "--no-download"],
            cwd=str(app),
            env=proc_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.app = app
        self.runs = {}
        for part in sorted(self.work.glob("docker-run.*")):
            argv = part.read_text().splitlines()
            if "--node-rank" in argv:
                rank = argv[argv.index("--node-rank") + 1]
                self.runs[rank] = argv

    def cleanup(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

    # -- recorded-launch accessors --------------------------------------
    def volume_mounts(self, rank: str):
        argv = self.runs[rank]
        return [argv[i + 1] for i, a in enumerate(argv)
                if a == "-v" and i + 1 < len(argv)]

    def env_flags(self, rank: str):
        argv = self.runs[rank]
        return [argv[i + 1] for i, a in enumerate(argv)
                if a == "-e" and i + 1 < len(argv)]

    def triton_host_dir(self, rank: str) -> str | None:
        for m in self.volume_mounts(rank):
            if m.endswith(":" + TRITON_DIR):
                return m[: -len(TRITON_DIR) - 1]
        return None

    def engine_args(self, rank: str):
        """`vllm serve` argv from the model token onward (image excluded)."""
        argv = self.runs[rank]
        i = argv.index(MODEL_TOKEN)
        return argv[i:]


_PROBES: dict[str, RenderProbe] = {}


def probe(name: str, **kwargs) -> RenderProbe:
    if name not in _PROBES:
        _PROBES[name] = RenderProbe(**kwargs)
    p = _PROBES[name]
    return p


@unittest.skipUnless(
    (ROOT / "start.sh").is_file() and (ROOT / "start-v030.sh").is_file(),
    "launchers not present",
)
class RenderedLaunches(unittest.TestCase):
    """Assertions on docker-run argv produced by executing the real heredocs."""

    @classmethod
    def tearDownClass(cls) -> None:
        for p in _PROBES.values():
            p.cleanup()
        _PROBES.clear()

    def base(self) -> RenderProbe:
        p = probe("base", env_overrides={}, image_id=STUB_IMAGE_ID)
        self.assertEqual(
            p.proc.returncode, 0,
            f"launcher render run failed:\n{p.proc.stdout}\n{p.proc.stderr}",
        )
        self.assertIn("0", p.runs, "head launch (rank 0) never executed")
        self.assertIn("1", p.runs, "worker launch (rank 1) never executed")
        return p

    # -- Triton cache mounts -------------------------------------------
    def test_head_executed_launch_mounts_triton_at_root_triton(self):
        p = self.base()
        expected = f"{p.head_home}/.cache/triton/{sanitized(STUB_IMAGE_ID)}:{TRITON_DIR}"
        self.assertIn(expected, p.volume_mounts("0"))

    def test_worker_executed_launch_mounts_triton_at_root_triton(self):
        p = self.base()
        expected = f"{p.worker_home}/.cache/triton/{sanitized(STUB_IMAGE_ID)}:{TRITON_DIR}"
        self.assertIn(expected, p.volume_mounts("1"))

    def test_both_nodes_pin_triton_cache_dir_env(self):
        p = self.base()
        for rank in ("0", "1"):
            self.assertIn(
                f"TRITON_CACHE_DIR={TRITON_DIR}", p.env_flags(rank),
                f"rank {rank} must pin TRITON_CACHE_DIR at the mount target",
            )

    def test_head_and_worker_triton_homes_are_distinct(self):
        p = self.base()
        head, worker = p.triton_host_dir("0"), p.triton_host_dir("1")
        self.assertIsNotNone(head, "head has no Triton mount")
        self.assertIsNotNone(worker, "worker has no Triton mount")
        self.assertTrue(str(head).startswith(f"{p.head_home}/"),
                        f"head cache must live under the head HOME: {head}")
        self.assertTrue(str(worker).startswith(f"{p.worker_home}/"),
                        f"worker cache must live under the worker HOME: {worker}")
        self.assertNotEqual(head, worker,
                            "head and worker must not share one cache dir")

    def test_cache_dirs_are_precreated_on_both_nodes(self):
        p = self.base()
        key = sanitized(STUB_IMAGE_ID)
        self.assertTrue((p.head_home / ".cache/triton" / key).is_dir(),
                        "head cache dir must exist before docker run")
        self.assertTrue((p.worker_home / ".cache/triton" / key).is_dir(),
                        "worker cache dir must exist before docker run")

    # -- isolation by exact image identity --------------------------------
    def test_different_image_id_gets_a_different_cache_lane(self):
        a = self.base()
        b = probe("other-id", env_overrides={}, image_id=OTHER_IMAGE_ID)
        self.assertEqual(b.proc.returncode, 0, b.proc.stderr)
        b_lane = b.triton_host_dir("0")
        self.assertIsNotNone(b_lane)
        self.assertEqual(sanitized(OTHER_IMAGE_ID), Path(b_lane).name)
        self.assertNotEqual(a.triton_host_dir("0"), b_lane)

    def test_image_tag_mutation_with_same_id_reuses_the_default_lane(self):
        # The key is the locally resolved image ID, not the mutable tag: an
        # overridden reference for identical content keeps the same lane.
        p = probe("tag-mutate",
                  env_overrides={"OVERRIDE_IMAGE": "vllm/vllm-openai:mutant-tag"},
                  image_id=STUB_IMAGE_ID)
        self.assertEqual(p.proc.returncode, 0, p.proc.stderr)
        self.assertEqual(f"{p.head_home}/.cache/triton/{sanitized(STUB_IMAGE_ID)}",
                         p.triton_host_dir("0"))

    def test_unresolvable_image_id_falls_back_to_the_image_reference(self):
        # No `docker image inspect` answer (fresh/odd node): the lane is still
        # deterministic and still sanitised (no ':' path segments).
        p = probe("no-id", env_overrides={}, image_id=None)
        self.assertEqual(p.proc.returncode, 0, p.proc.stderr)
        lane = p.triton_host_dir("0")
        self.assertIsNotNone(lane)
        self.assertEqual(f"{p.head_home}/.cache/triton/{sanitized(TEST_IMAGE)}", lane)
        self.assertNotIn(":", lane.rsplit("/", 1)[-1])

    def test_default_lane_matches_sanitized_image_id(self):
        # The sanitizer contract: sha256:... becomes one safe path segment,
        # and the rendered lane uses exactly this key.
        key = sanitized(STUB_IMAGE_ID)
        self.assertEqual(
            "sha256-aaaabbbbccccddddeeeeffff0000111122223333444455556666777788889999",
            key)
        self.assertNotIn(":", key)
        p = self.base()
        lane = p.triton_host_dir("0")
        self.assertIsNotNone(lane)
        self.assertTrue(lane.endswith("/" + key))

    # -- what this change must NOT touch -----------------------------------
    def test_vllm_cache_mount_preserved_on_both_nodes(self):
        p = self.base()
        self.assertIn(f"{p.head_home}/.cache/vllm:/root/.cache/vllm",
                      p.volume_mounts("0"))
        self.assertIn(f"{p.worker_home}/.cache/vllm:/root/.cache/vllm",
                      p.volume_mounts("1"))

    def test_v030_flashinfer_autotune_cache_stays_in_tmp(self):
        p = self.base()
        for rank in ("0", "1"):
            self.assertIn(FLASHINFER_AUTOTUNE, p.env_flags(rank))
        for rank in ("0", "1"):
            for m in p.volume_mounts(rank):
                self.assertNotIn("/tmp/fi_autotune", m,
                                 "FlashInfer /tmp autotune stays container-local")

    def test_engine_arguments_unchanged_and_cache_free(self):
        p = self.base()
        head, worker = p.engine_args("0"), p.engine_args("1")
        self.assertEqual(MODEL_TOKEN, head[0])
        self.assertEqual("0", head[head.index("--node-rank") + 1])
        self.assertEqual("1", worker[worker.index("--node-rank") + 1])
        self.assertIn("--headless", worker)
        self.assertIn("--host", head)
        self.assertIn("--port", head)
        self.assertIn("--tensor-parallel-size", head)
        self.assertIn("--enable-expert-parallel", head)
        # Engine argv for these exact .env values — nothing cache-related
        # may appear in it, and the v0.30 engine defaults must stand.
        joined_head = " ".join(head)
        joined_worker = " ".join(worker)
        for blob in (joined_head, joined_worker):
            self.assertNotIn("triton", blob.lower())
        self.assertIn("--kv-cache-dtype auto", joined_head)
        self.assertIn('"cudagraph_mode":"FULL_DECODE_ONLY"', joined_head)
        self.assertIn('"num_speculative_tokens":3', joined_head)
        self.assertNotIn("--hf-overrides", joined_head)


class StaticLauncherChecks(unittest.TestCase):
    """Template checks that also cover the stock (day-0) lane, which cannot
    render end-to-end under stubs without the real image sources."""

    def setUp(self):
        self.source = (ROOT / "start.sh").read_text()
        self.templates = re.findall(
            r"<<LAUNCH_EOF\n(.*?)\nLAUNCH_EOF", self.source, re.S)
        self.assertEqual(2, len(self.templates),
                         "expected exactly two launch heredocs (worker, head)")

    def test_triton_mount_and_env_present_once_in_every_heredoc(self):
        for i, tmpl in enumerate(self.templates):
            self.assertEqual(
                1, tmpl.count(f":{TRITON_DIR}"),
                f"heredoc {i} must mount the Triton cache exactly once",
            )
            self.assertIn(f"-e TRITON_CACHE_DIR={TRITON_DIR} \\", tmpl,
                          f"heredoc {i} must pin TRITON_CACHE_DIR")
            self.assertRegex(tmpl, r"-v \$\w*TRITON_DIR:" + re.escape(TRITON_DIR))

    def test_triton_dirs_follow_each_nodes_own_home(self):
        # Head uses the head's $HOME, worker the worker's resolved $HOME
        # (REMOTE_HOME) — same layout as the existing ~/.cache/vllm mounts.
        self.assertRegex(self.source,
                         r'HEAD_TRITON_DIR="\$HOME/\.cache/triton/\$TRITON_KEY"')
        self.assertRegex(self.source,
                         r'WORKER_TRITON_DIR="\$REMOTE_HOME/\.cache/triton/\$TRITON_KEY"')

    def test_cache_key_prefers_resolved_image_id_with_tag_fallback(self):
        self.assertRegex(self.source, r'TRITON_KEY_RAW="\$\{LOCAL_ID:-\$IMAGE\}"')
        self.assertRegex(self.source,
                         r"tr -c 'A-Za-z0-9\._-' '-'")

    def test_vllm_and_flashinfer_settings_textually_untouched(self):
        # ~/.cache/vllm mounts survive on every path that mentions them:
        # the head heredoc, the worker heredoc, and the DOCKER_ARGS mirror.
        worker_tmpl, head_tmpl = self.templates
        self.assertIn("-v $REMOTE_HOME/.cache/vllm:/root/.cache/vllm", worker_tmpl)
        self.assertIn("-v $HOME/.cache/vllm:/root/.cache/vllm", head_tmpl)
        self.assertIn('-v $HOME/.cache/vllm:/root/.cache/vllm', self.source)
        self.assertEqual(
            1, self.source.count(FLASHINFER_AUTOTUNE),
            "the v0.30 FlashInfer /tmp autotune setting must stay exactly once",
        )
        self.assertIn(
            'DOCKER_ARGS+=("-v $HEAD_TRITON_DIR:/root/.triton")', self.source,
            "the DOCKER_ARGS mirror stays consistent with the head heredoc",
        )

    def test_vllm_args_block_builds_no_cache_arguments(self):
        # Extract the whole shared engine-argv block — from `VLLM_ARGS=()`
        # up to the VLLM_ARGS_STR join — including the conditional
        # `[[ ... ]] && VLLM_ARGS+=(...)` lines, and execute it in bash.
        match = re.search(
            r"^([ \t]*)VLLM_ARGS=\(\)\n(.*?)^\1VLLM_ARGS_STR=",
            self.source, re.M | re.S,
        )
        self.assertIsNotNone(match, "VLLM_ARGS block not found")
        block = match.group(2) if match else ""
        # .env.sample defaults for the knobs the block reads.
        presets = {
            "SERVED_MODEL_NAME": "qwen3.8-flash-next",
            "TENSOR_PARALLEL_SIZE": "2",
            "GPU_MEMORY_UTILIZATION": "0.835",
            "MAX_NUM_SEQS": "8",
            "MAX_NUM_BATCHED_TOKENS": "8192",
            "MAX_MODEL_LEN": "262144",
            "KV_CACHE_DTYPE": "fp8",
            "MAMBA_SSM_CACHE_DTYPE": "bfloat16",
            "MM_ENCODER_TP_MODE": "data",
            "HEAD_IP": "10.0.0.1",
            "MASTER_PORT": "50000",
            "ENABLE_EXPERT_PARALLEL": "true",
            "MTP_NUM_SPECULATIVE_TOKENS": "3",
            "MTP_DISABLE_BLOCK_DROP": "1",
            "MTP_INDEX_SHARE": "true",
            "MTP_DRAFT_VOCAB": "",
            "V030": "false",
            "PLE_EMBEDDING_DTYPE": "",
            "YARN_ENABLE": "false",
            "YARN_FACTOR": "4.0",
            "EXTRA_VLLM_ARGS": "",
        }
        exports = "".join(
            f'{k}={subprocess.list2cmdline([v])}\n' for k, v in presets.items()
        )
        argv = subprocess.check_output(
            ["bash", "--noprofile", "--norc", "-c",
             exports + "VLLM_ARGS=()\n" + block
             + '\nprintf "%s\\0" "${VLLM_ARGS[@]}"'],
            env={"PATH": os.pathsep.join(
                [str(Path(sys.executable).parent), os.defpath])},
            text=True,
        ).split("\0")[:-1]
        self.assertIn("--tensor-parallel-size", argv)
        self.assertIn("2", argv)
        self.assertIn("--compilation-config", argv)
        self.assertIn("--mamba-ssm-cache-dtype", argv)
        self.assertNotIn("triton", " ".join(argv).lower())
        self.assertNotIn("--triton", " ".join(argv).lower())
        self.assertNotIn("/root/.triton", " ".join(argv))


if __name__ == "__main__":
    unittest.main(verbosity=2)
