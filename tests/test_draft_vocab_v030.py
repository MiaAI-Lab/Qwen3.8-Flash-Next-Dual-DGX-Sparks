import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
FILES = REPO / "files"
ORIG = FILES / "mtp_v030_patched.py.orig"


@unittest.skipUnless(ORIG.is_file(), "no extracted v0.30 mtp.py; run ./start-v030.sh --launch once")
class DraftVocabV030(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for name in ("patch_mtp_draft_vocab_v030.py", "patch_mtp_draft_vocab.py", "mtp_v030_patched.py.orig"):
            shutil.copy(FILES / name, self.tmp / name)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run([sys.executable, str(self.tmp / "patch_mtp_draft_vocab_v030.py")], capture_output=True, text=True)

    def test_patches_get_top_tokens_and_attach(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.tmp / "mtp_v030_patched.py").read_text()
        ast.parse(src)
        self.assertIn("def get_top_tokens(self, hidden_states", src)
        self.assertIn("_attach_draft_vocab(self)", src)
        self.assertEqual(src.count("logger = init_logger(__name__)"), 1)

    def test_rerun_gives_same_output(self):
        self.run_patch()
        first = (self.tmp / "mtp_v030_patched.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.tmp / "mtp_v030_patched.py").read_text(), first)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.tmp / "mtp_v030_patched.py.orig"
        orig.write_text(orig.read_text().replace("return loader.load_weights(", "return  loader.load_weights("))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.tmp / "mtp_v030_patched.py").exists())


class LaunchWrapper(unittest.TestCase):
    def test_wrapper_sets_the_lane(self):
        src = (REPO / "start-v030.sh").read_text()
        for needle in ("export V030=true", "vllm/vllm-openai:v0.30.0", "OVERRIDE_KV_CACHE_DTYPE:-fp8", 'exec "$SCRIPT_DIR/start.sh"'):
            self.assertIn(needle, src)


if __name__ == "__main__":
    unittest.main()
