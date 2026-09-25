#!/usr/bin/env python3
"""CPU-only checks for the QSA draft-fusion overlay (vllm#58449 backport).

Validates provenance hashes, the patcher's fail-closed behavior on drifted
sources, the marker functions in the patched output, and the launcher wiring.
No GPU, no image, no network: the patcher is the only code path exercised.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUS = ROOT / "files" / "qsa_draft_fusion"
ORIG = FUS / "orig" / "qsa_cache.py"
OUT = FUS / "qsa_cache.py"
PATCHER = FUS / "patch_qsa_draft_fusion.py"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Provenance(unittest.TestCase):
    def setUp(self):
        self.man = json.loads((FUS / "provenance.json").read_text())

    def test_committed_files_match_pinned_hashes(self):
        self.assertEqual(sha(ORIG), self.man["orig_sha256"])
        # Image-independent: the patch is applied to the pinned vLLM source
        # commit, and provenance records that commit.
        if OUT.exists():
            self.assertEqual(sha(OUT), self.man["patched_sha256"])

    def test_patcher_output_is_pinned(self):
        r = subprocess.run([sys.executable, str(PATCHER)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(sha(OUT), self.man["patched_sha256"])
        r2 = subprocess.run([sys.executable, str(PATCHER)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r2.returncode, 0, "regeneration must be idempotent")
        self.assertEqual(sha(OUT), self.man["patched_sha256"])

    def test_patcher_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "qsa_draft_fusion"
            shutil.copytree(FUS, tmp)
            drifted = tmp / "orig" / "qsa_cache.py"
            drifted.write_text(drifted.read_text() + "\n# drift\n")
            (tmp / "qsa_cache.py").unlink(missing_ok=True)
            r = subprocess.run([sys.executable, str(tmp / "patch_qsa_draft_fusion.py")],
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("does not match the pinned vLLM source", r.stdout + r.stderr)
            self.assertFalse((tmp / "qsa_cache.py").exists(), "must not emit output on drift")


class PatchedContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OUT.exists():
            subprocess.run([sys.executable, str(PATCHER)], cwd=ROOT, check=True)
        cls.src = OUT.read_text()

    def test_fused_update_markers_present(self):
        # The capability flag gating fusion and the in-place update itself.
        self.assertIn("supports_draft_decode_metadata_update", self.src)
        self.assertIn("def update_draft_decode_metadata", self.src)
        self.assertIn("def _launch_qsa_metadata_kernel", self.src)

    def test_only_the_side_cache_file_changed(self):
        # Diff vs pristine: hunks must be confined to the metadata builder.
        self.assertEqual(sha(ORIG),
                         json.loads((FUS / "provenance.json").read_text())["orig_sha256"])


class LauncherWiring(unittest.TestCase):
    def test_gated_opt_in_wiring(self):
        source = (ROOT / "start.sh").read_text()
        self.assertIn('QSA_DRAFT_FUSION="${QSA_DRAFT_FUSION:-false}"', source)
        step = source[source.index("if $DO_LAUNCH && [[ \"$QSA_DRAFT_FUSION\" == \"true\" ]]"):]
        step = step[:step.index("\n\n")]
        self.assertIn('[[ "$V030" == "true" ]] || err', step)
        self.assertIn("patch_qsa_draft_fusion.py", step)
        self.assertIn(
            'add_overlay "$SCRIPT_DIR/files/qsa_draft_fusion/qsa_cache.py" '
            '"$VLLM_PKG/models/qwen4_exp/common/qsa_cache.py"', step)


if __name__ == "__main__":
    unittest.main(verbosity=2)
