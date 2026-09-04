"""Tests for verify-weights.py.

Runs against a tiny fake model directory and a stubbed manifest so no network
access is needed, plus an integration path that maps a "snapshots/<rev>/..."
local layout onto manifest relative paths.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# verify-weights.py has a hyphen, so it cannot be imported by name; load it
# from its path instead.
import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "verify_weights", os.path.join(REPO_ROOT, "verify-weights.py"))
vw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vw)  # noqa: E402


def make_file(path, size, content=None):
    """Write a file of the requested size; content must fill the size exactly."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        if content is not None:
            f.write(content)
        else:
            f.write(b"\x00" * size)
    return path


class VerifyWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ok_files(self):
        """Expected use: all files present with matching size and hash."""
        data = b"hello world"
        make_file(os.path.join(self.tmp, "config.json"), len(data), data)
        manifest = {
            "config.json": {"type": "file", "size": len(data), "lfs": {"oid": ""}},
            "model-00001.safetensors": {
                "type": "file",
                "size": len(data),
                "lfs": {"oid": "sha256:" + vw.file_sha256(make_file(
                    os.path.join(self.tmp, "model-00001.safetensors"), len(data), data))},
            },
        }
        missing, size_mismatch, hash_mismatch = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(missing, [])
        self.assertEqual(size_mismatch, [])
        self.assertEqual(hash_mismatch, [])

    def test_missing_file(self):
        """Failure case: a file in the manifest is absent locally."""
        make_file(os.path.join(self.tmp, "config.json"), 5)
        manifest = {
            "config.json": {"type": "file", "size": 5, "lfs": {"oid": ""}},
            "model.safetensors": {"type": "file", "size": 10, "lfs": {"oid": "sha256:abc"}},
        }
        missing, _, _ = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(missing, ["model.safetensors"])

    def test_size_mismatch(self):
        """Failure case: same path, different size."""
        make_file(os.path.join(self.tmp, "config.json"), 100)
        manifest = {
            "config.json": {"type": "file", "size": 50, "lfs": {"oid": ""}},
        }
        _, size_mismatch, _ = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(size_mismatch, ["config.json"])

    def test_hash_mismatch(self):
        """Failure case: same size, different content hash."""
        data_a = b"A" * 16
        data_b = b"B" * 16
        make_file(os.path.join(self.tmp, "model.safetensors"), 16, data_b)
        manifest = {
            "model.safetensors": {
                "type": "file",
                "size": 16,
                "lfs": {"oid": "sha256:" + vw.file_sha256(make_file(
                    os.path.join(self.tmp, "expected.safetensors"), 16, data_a))},
            },
        }
        _, _, hash_mismatch = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(hash_mismatch, ["model.safetensors"])

    def test_snapshot_prefix_stripped(self):
        """Edge case: standard hub layout uses snapshots/<rev>/ prefix."""
        data = b"weights"
        # Build the manifest with plain relative paths.
        oid = "sha256:" + vw.file_sha256(make_file(
            os.path.join(self.tmp, "snapshots/abc123", "model.safetensors"),
            len(data), data))
        manifest = {
            "model.safetensors": {"type": "file", "size": len(data), "lfs": {"oid": oid}},
        }
        missing, size_mismatch, hash_mismatch = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual((missing, size_mismatch, hash_mismatch), ([], [], []))

    def test_ignores_download_artifacts(self):
        """Edge case: lock files and download artifacts are not treated as missing."""
        make_file(os.path.join(self.tmp, "model.safetensors"), 10)
        make_file(os.path.join(self.tmp, "model.safetensors.lock"), 3)
        make_file(os.path.join(self.tmp, "download", "junk.incomplete"), 3)
        manifest = {
            "model.safetensors": {"type": "file", "size": 10, "lfs": {"oid": ""}},
        }
        missing, _, _ = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(missing, [])

    def test_manifest_save_and_load(self):
        """Expected use: a saved manifest round-trips and drives verification."""
        data = b"weights"
        manifest = {
            "model.safetensors": {
                "type": "file",
                "size": len(data),
                "lfs": {"oid": "sha256:" + vw.file_sha256(make_file(
                    os.path.join(self.tmp, "model.safetensors"), len(data), data))},
            },
        }
        manifest_path = os.path.join(self.tmp, "manifest.json")
        vw.save_manifest_file(manifest_path, manifest)
        loaded = vw.load_manifest_file(manifest_path)
        self.assertEqual(loaded, manifest)
        missing, size_mismatch, hash_mismatch = vw.verify(self.tmp, loaded, workers=2)
        self.assertEqual((missing, size_mismatch, hash_mismatch), ([], [], []))


if __name__ == "__main__":
    unittest.main()
