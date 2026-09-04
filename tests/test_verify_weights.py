"""Tests for verify-weights.py.

Runs against tiny fake model directories and stubbed manifests / HTTP responses
so no network access is needed. Covers the expected path, every failure mode the
verifier can report, and the resolution logic for the standard hub layout.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

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


def shas(root, files):
    """Write the given files under root and return a valid manifest for them.

    files maps relative path -> content bytes/str. Returns the manifest entries
    (with correct size and LFS SHA-256) for those files.
    """
    manifest = {}
    for rel, content in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(content, str):
            content = content.encode()
        with open(path, "wb") as f:
            f.write(content)
        manifest[rel] = {"type": "file", "size": len(content),
                         "lfs": {"oid": "sha256:" + vw.file_sha256(path)}}
    return manifest


class VerifyWeightsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def kinds(self, problems):
        """Return the multiset of problem kinds as a list."""
        return sorted(p.kind for p in problems)

    def test_ok_files(self):
        """Expected use: all files present with matching size and hash."""
        make_file(os.path.join(self.tmp, "config.json"), 5, b"hello")
        manifest = shas(self.tmp, {"config.json": b"hello"})
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_missing_file(self):
        """Failure case: a file in the manifest is absent locally."""
        manifest = shas(self.tmp, {"config.json": b"hello"})
        manifest["model.safetensors"] = {"type": "file", "size": 10, "lfs": {"oid": ""}}
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(self.kinds(problems), ["MISSING"])
        self.assertEqual(problems[0].rel, "model.safetensors")

    def test_size_mismatch(self):
        """Failure case: same path, different size."""
        manifest = shas(self.tmp, {"model.safetensors": b"A" * 15})
        with open(os.path.join(self.tmp, "model.safetensors"), "wb") as f:
            f.write(b"B" * 16)  # same path, different size than the manifest
        problems = vw.verify(self.tmp, manifest, workers=2)
        # The size differs AND the hash differs; both are reported.
        self.assertEqual(self.kinds(problems), ["HASH", "SIZE"])
        self.assertIn("15 bytes", problems[0].detail)
        self.assertIn("16", problems[0].detail)

    def test_hash_mismatch(self):
        """Failure case: same size, different content hash."""
        manifest = shas(self.tmp, {"model.safetensors": b"A" * 16})
        with open(os.path.join(self.tmp, "model.safetensors"), "wb") as f:
            f.write(b"B" * 16)  # same size, different content
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(self.kinds(problems), ["HASH"])
        self.assertEqual(problems[0].rel, "model.safetensors")

    def test_multiple_problems_reported(self):
        """Failure case: one bad run reports every problem, not just the first."""
        make_file(os.path.join(self.tmp, "present.safetensors"), 4, b"WXYZ")
        make_file(os.path.join(self.tmp, "wrong-size.safetensors"), 4, b"WXYZ")
        manifest = {
            "present.safetensors": {"type": "file", "size": 4,
                                    "lfs": {"oid": "sha256:" + vw.file_sha256(
                                        os.path.join(self.tmp, "present.safetensors"))}},
            "wrong-size.safetensors": {"type": "file", "size": 9, "lfs": {"oid": ""}},
            "missing.safetensors": {"type": "file", "size": 1, "lfs": {"oid": ""}},
        }
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(self.kinds(problems), ["MISSING", "SIZE"])

    def test_empty_file_matches_when_oid_blank(self):
        """Edge case: an empty manifest-verified file with no LFS oid is fine."""
        make_file(os.path.join(self.tmp, "empty.json"), 0)
        manifest = {"empty.json": {"type": "file", "size": 0, "lfs": {"oid": ""}}}
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_oid_sha256_prefix_optional(self):
        """Edge case: oid with and without the sha256: prefix both work."""
        make_file(os.path.join(self.tmp, "a.safetensors"), 4, b"abcd")
        make_file(os.path.join(self.tmp, "b.safetensors"), 4, b"efgh")
        manifest = {
            "a.safetensors": {"type": "file", "size": 4,
                              "lfs": {"oid": vw.file_sha256(os.path.join(self.tmp, "a.safetensors"))}},
            "b.safetensors": {"type": "file", "size": 4,
                              "lfs": {"oid": "sha256:" + vw.file_sha256(os.path.join(self.tmp, "b.safetensors"))}},
        }
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_ignores_download_artifacts(self):
        """Edge case: lock files and download artifacts are not treated as missing."""
        make_file(os.path.join(self.tmp, "model.safetensors"), 4, b"abcd")
        make_file(os.path.join(self.tmp, "model.safetensors.lock"), 3)
        make_file(os.path.join(self.tmp, "download", "junk.incomplete"), 3)
        make_file(os.path.join(self.tmp, "refs", "main"), 0)
        manifest = shas(self.tmp, {"model.safetensors": b"abcd"})
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_dangling_symlink_reported(self):
        """Failure case: a snapshot symlink that points nowhere is reported."""
        os.makedirs(os.path.join(self.tmp, "snapshots", "abc123"))
        os.symlink("/nonexistent/target", os.path.join(self.tmp, "snapshots", "abc123", "model.safetensors"))
        manifest = {
            "model.safetensors": {"type": "file", "size": 10, "lfs": {"oid": ""}},
        }
        problems = vw.verify(self.tmp, manifest, workers=2)
        # The broken symlink is unusable, so the manifest file is effectively missing too.
        self.assertEqual(self.kinds(problems), ["BROKEN", "MISSING"])

    def test_unreadable_file_reported(self):
        """Failure case: an unreadable file (permissions) is reported, not crashed."""
        path = os.path.join(self.tmp, "model.safetensors")
        manifest = shas(self.tmp, {"model.safetensors": b"abcd"})
        os.chmod(path, 0o000)
        try:
            problems = vw.verify(self.tmp, manifest, workers=2)
            self.assertEqual(self.kinds(problems), ["UNREADABLE"])
        finally:
            os.chmod(path, 0o644)

    def test_manifest_missing_key_skipped(self):
        """Edge case: manifest entries without 'size' are still checked for hash."""
        make_file(os.path.join(self.tmp, "model.safetensors"), 4, b"abcd")
        manifest = shas(self.tmp, {"model.safetensors": b"abcd"})
        del manifest["model.safetensors"]["size"]
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_symlink_cycle_terminates(self):
        """Edge case: a symlink cycle in the tree does not hang the walk."""
        make_file(os.path.join(self.tmp, "model.safetensors"), 4, b"abcd")
        os.symlink(self.tmp, os.path.join(self.tmp, "loop"))
        manifest = shas(self.tmp, {"model.safetensors": b"abcd"})
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_resolve_snapshot_uses_refs_main(self):
        """Expected use: hub layout resolves the snapshot pointed to by refs/main."""
        snap = os.path.join(self.tmp, "snapshots", "rev2")
        make_file(os.path.join(snap, "model.safetensors"), 4, b"abcd")
        os.makedirs(os.path.join(self.tmp, "snapshots", "rev1"))
        os.makedirs(os.path.join(self.tmp, "refs"))
        with open(os.path.join(self.tmp, "refs", "main"), "w") as f:
            f.write("rev2\n")  # trailing newline like huggingface_hub >= 1.x
        tree_dir, is_snapshot = vw.resolve_snapshot(self.tmp)
        self.assertTrue(is_snapshot)
        self.assertEqual(os.path.basename(tree_dir), "rev2")

    def test_resolve_snapshot_single(self):
        """Edge case: only one snapshot exists; it is used even without refs/main."""
        snap = os.path.join(self.tmp, "snapshots", "rev1")
        make_file(os.path.join(snap, "model.safetensors"), 4, b"abcd")
        tree_dir, is_snapshot = vw.resolve_snapshot(self.tmp)
        self.assertTrue(is_snapshot)
        self.assertEqual(os.path.basename(tree_dir), "rev1")

    def test_resolve_snapshot_multiple_no_refs_raises(self):
        """Failure case: multiple snapshots and no usable refs/main is an error."""
        os.makedirs(os.path.join(self.tmp, "snapshots", "rev1"))
        os.makedirs(os.path.join(self.tmp, "snapshots", "rev2"))
        with self.assertRaises(vw.VerifierError):
            vw.resolve_snapshot(self.tmp)

    def test_resolve_snapshot_refs_points_missing_raises(self):
        """Failure case: refs/main points at a revision that does not exist."""
        os.makedirs(os.path.join(self.tmp, "snapshots", "rev1"))
        os.makedirs(os.path.join(self.tmp, "snapshots", "rev2"))
        os.makedirs(os.path.join(self.tmp, "refs"))
        with open(os.path.join(self.tmp, "refs", "main"), "w") as f:
            f.write("rev3")
        with self.assertRaises(vw.VerifierError):
            vw.resolve_snapshot(self.tmp)

    def test_manifest_save_and_load_roundtrip(self):
        """Expected use: a saved manifest round-trips and drives verification."""
        make_file(os.path.join(self.tmp, "model.safetensors"), 4, b"abcd")
        manifest = shas(self.tmp, {"model.safetensors": b"abcd"})
        manifest_path = os.path.join(self.tmp, "manifest.json")
        vw.save_manifest(manifest_path, manifest)
        loaded = vw.load_manifest(manifest_path)
        self.assertEqual(loaded, manifest)
        problems = vw.verify(self.tmp, loaded, workers=2)
        self.assertEqual(problems, [])

    def test_load_manifest_malformed_raises(self):
        """Failure case: a corrupt manifest file raises VerifierError."""
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w") as f:
            f.write("{not json")
        with self.assertRaises(vw.VerifierError):
            vw.load_manifest(bad)

    def test_verify_with_workers_less_than_one(self):
        """Edge case: workers=0 or negative falls back to 1 without error."""
        make_file(os.path.join(self.tmp, "model.safetensors"), 4, b"abcd")
        manifest = shas(self.tmp, {"model.safetensors": b"abcd"})
        self.assertEqual(vw.verify(self.tmp, manifest, workers=0), [])

    def test_fetch_manifest_http_error(self):
        """Failure case: an HTTP error from the API raises VerifierError with a message."""
        with mock.patch.object(vw.urllib.request, "urlopen",
                               side_effect=HTTPError("url", 404, "Not Found", None, None)):
            with self.assertRaises(vw.VerifierError) as ctx:
                vw.fetch_manifest("org/model", "main")
        self.assertIn("404", str(ctx.exception))

    def test_fetch_manifest_network_error(self):
        """Failure case: a network error (timeout, DNS) raises VerifierError."""
        with mock.patch.object(vw.urllib.request, "urlopen",
                               side_effect=URLError("connection refused")):
            with self.assertRaises(vw.VerifierError):
                vw.fetch_manifest("org/model", "main")

    def test_fetch_manifest_garbage_payload(self):
        """Failure case: a non-list API response raises instead of crashing."""
        with mock.patch.object(vw.urllib.request, "urlopen",
                               side_effect=json_response({"oops": True})):
            with self.assertRaises(vw.VerifierError):
                vw.fetch_manifest("org/model", "main")

    def test_fetch_only_does_not_need_model_dir(self):
        """Expected use: --fetch-only succeeds even when no model dir exists."""
        with mock.patch.object(vw, "fetch_manifest", return_value={}):
            rc = vw.main(["--repo", "org/model", "--fetch-only", "--quiet"])
        self.assertEqual(rc, 0)

    def test_fetch_only_rejects_manifest(self):
        """Failure case: --fetch-only with --manifest is an argument error (exit 2)."""
        rc = vw.main(["--repo", "org/model", "--fetch-only", "--manifest", "/nope.json", "--quiet"])
        self.assertEqual(rc, 2)

    def test_main_missing_model_dir_returns_2(self):
        """Failure case: verify without --path and no model dir gives exit 2."""
        with mock.patch.dict(os.environ, {"HF_HOME": self.tmp}):
            rc = vw.main(["--repo", "org/model", "--quiet", "--manifest", "x"])
        self.assertEqual(rc, 2)

    def test_main_verify_ok(self):
        """Expected use: full CLI verify returns 0 for a valid tree + saved manifest."""
        manifest = shas(self.tmp, {"model.safetensors": b"abcd"})
        manifest_path = os.path.join(self.tmp, "manifest.json")
        vw.save_manifest(manifest_path, manifest)
        rc = vw.main(["--repo", "org/model", "--path", self.tmp,
                      "--manifest", manifest_path, "--quiet"])
        self.assertEqual(rc, 0)

    def test_main_verify_failure_returns_1(self):
        """Failure case: a corrupt file makes the CLI exit 1."""
        manifest = shas(self.tmp, {"model.safetensors": b"abcd"})
        with open(os.path.join(self.tmp, "model.safetensors"), "wb") as f:
            f.write(b"wxyz")  # 4 bytes, different content
        manifest_path = os.path.join(self.tmp, "manifest.json")
        vw.save_manifest(manifest_path, manifest)
        rc = vw.main(["--repo", "org/model", "--path", self.tmp,
                      "--manifest", manifest_path, "--quiet"])
        self.assertEqual(rc, 1)

    def test_gitattributes_is_a_real_file_verified(self):
        """Edge case: .gitattributes is a manifest file and must be verified."""
        manifest = shas(self.tmp, {".gitattributes": b"*.safetensors filter=lfs\n"})
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_manifest_empty_oid_hash_still_checked(self):
        """Edge case: entries with an LFS oid present are hashed even if size matches."""
        make_file(os.path.join(self.tmp, "model.safetensors"), 4, b"abcd")
        manifest = {
            "model.safetensors": {"type": "file", "size": 4,
                                  "lfs": {"oid": "sha256:" + vw.file_sha256(
                                      os.path.join(self.tmp, "model.safetensors"))}},
        }
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_lfs_null_and_absent_not_crash(self):
        """Edge case: lfs null / absent in the manifest must not crash."""
        make_file(os.path.join(self.tmp, "config.json"), 3, b"abc")
        make_file(os.path.join(self.tmp, "model.bin"), 3, b"xyz")
        manifest = {
            "config.json": {"type": "file", "size": 3, "lfs": None},
            "model.bin": {"type": "file", "size": 3},
        }
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_uppercase_oid_prefix(self):
        """Edge case: uppercase SHA256: prefix and hex are normalized."""
        data = b"abcd"
        make_file(os.path.join(self.tmp, "model.safetensors"), 4, data)
        oid = vw.file_sha256(os.path.join(self.tmp, "model.safetensors"))
        manifest = {
            "model.safetensors": {"type": "file", "size": 4,
                                  "lfs": {"oid": "SHA256:" + oid.upper()}},
        }
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(problems, [])

    def test_same_size_rewrite_caught_by_hash(self):
        """Failure case: a same-size rewrite (size ok) is still caught by hash."""
        make_file(os.path.join(self.tmp, "model.safetensors"), 4, b"ABCD")
        manifest = {
            "model.safetensors": {"type": "file", "size": 4,
                                  "lfs": {"oid": "sha256:" + vw.file_sha256(
                                      os.path.join(self.tmp, "model.safetensors"))}},
        }
        with open(os.path.join(self.tmp, "model.safetensors"), "wb") as f:
            f.write(b"WXYZ")  # same size, different content
        problems = vw.verify(self.tmp, manifest, workers=2)
        self.assertEqual(self.kinds(problems), ["HASH"])

    def test_fetch_manifest_pagination(self):
        """Expected use: a paginated API response is followed to completion."""
        pages = [
            ([{"type": "file", "path": "a.txt", "size": 1, "lfs": {"oid": ""}}], '<https://x?cursor=2>; rel="next"'),
            ([{"type": "file", "path": "b.txt", "size": 1, "lfs": {"oid": ""}}], None),
        ]
        with mock.patch.object(
            vw.urllib.request, "urlopen",
            side_effect=[json_response(payload, link) for payload, link in pages]):
            manifest = vw.fetch_manifest("org/model", "main")
        self.assertEqual(set(manifest), {"a.txt", "b.txt"})


class _FakeResp:
    def __init__(self, payload, link=None):
        self._payload = json.dumps(payload).encode()
        self.headers = {}
        if link:
            self.headers["Link"] = link

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def json_response(payload, link=None):
    return _FakeResp(payload, link)


import json  # noqa: E402


if __name__ == "__main__":
    unittest.main()
