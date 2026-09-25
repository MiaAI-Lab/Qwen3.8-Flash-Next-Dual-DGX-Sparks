import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "files" / "patch_block_drop.py"
ORIG = REPO / "files" / "block_drop" / "orig"
MOUNTED = ("config/speculative.py", "v1/core/kv_cache_utils.py", "v1/core/sched/scheduler.py")
LISTED = subprocess.run([sys.executable, str(PATCH), "--list"], capture_output=True, text=True).stdout.split()
HAVE_SOURCES = bool(LISTED) and all((ORIG / f).is_file() for f in LISTED)


@unittest.skipUnless(HAVE_SOURCES, "no extracted image sources; launch once with MTP_DISABLE_BLOCK_DROP=1")
class BlockDropOverlay(unittest.TestCase):
    def test_mounted_files_are_patched(self):
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, str(PATCH), str(ORIG), out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for f in MOUNTED:
                src = (Path(out) / f).read_text()
                ast.parse(src)
                self.assertNotEqual(src, (ORIG / f).read_text(), f)
                self.assertIn("eagle_block_drop", src, f)

    def test_mounted_basenames_are_unique(self):
        names = [Path(f).name for f in MOUNTED]
        self.assertEqual(len(names), len(set(names)))


class LauncherWiring(unittest.TestCase):
    def test_spec_config_carries_both_keys(self):
        src = (REPO / "start.sh").read_text()
        self.assertIn('_SPEC_EXTRA+=\',"disable_eagle_block_drop":true\'', src)
        self.assertIn('_SPEC_EXTRA+=\',"index_share_for_mtp_iteration":true\'', src)
        self.assertEqual(src.count('"$MTP_NUM_SPECULATIVE_TOKENS" "$_SPEC_EXTRA")'), 2)

    def test_overlays_only_the_core_files(self):
        src = (REPO / "start.sh").read_text()
        line = next(l for l in src.splitlines() if "for f in config/speculative.py" in l)
        self.assertEqual(line.strip().removeprefix("for f in ").removesuffix("; do").split(), list(MOUNTED))


if __name__ == "__main__":
    unittest.main()
