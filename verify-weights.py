#!/usr/bin/env python3
"""verify-weights.py — Verify local model files against the Hugging Face tree manifest.

Compares every file under the model directory against the manifest served by the
Hugging Face API: presence, size, and (for LFS files) the content SHA-256. A
size-only check does not catch a shard that was corrupted mid-download while
keeping its apparent size wrong only by a few hundred MB (see issue #30); hashing
the LFS blobs does.

Only the Python 3 standard library is used, so the script runs on the head node
and on worker nodes without installing anything.

Usage:
  python3 verify-weights.py [--repo MODEL_ID] [--path DIR] [--revision REV]
                            [--manifest FILE] [--workers N] [--quiet]

The manifest is fetched from the Hugging Face API unless --manifest points at
a previously saved one. check-weights.sh --verify fetches it once on the head
and ships it to the worker, so the worker validates against the same manifest
without needing model-API access.

Exit status is 0 when every file in the manifest is present, matches the
manifest size, and (for LFS files) matches the manifest SHA-256. Files in the
model directory that are not in the manifest (lock files, download artifacts,
trees metadata, ...) are ignored.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import urllib.request

API_TREE_URL = "https://huggingface.co/api/models/{repo}/tree/{revision}"

# Directory names that are download machinery, not model content. "snapshots"
# is deliberately kept: in the standard hub layout the model files live under
# snapshots/<revision>/ and must be hashed, while the deduplicated blobs/ (which
# the snapshot entries point at) would just be hashed twice if walked.
IGNORED_DIRS = {".cache", ".git", "blobs", "refs"}
IGNORED_SUFFIXES = (".lock", ".metadata", ".incomplete")
IGNORED_FILES = {".gitignore", "CACHEDIR.TAG"}

# Prefix to strip from paths found under a standard hub snapshot directory.
SNAPSHOT_PREFIX = re.compile(r"^snapshots/[^/]+/")


def fetch_manifest(repo: str, revision: str) -> dict[str, dict]:
    """Fetch the full recursive file manifest for a repo revision.

    Returns a mapping of relative path -> entry. Handles API pagination via the
    Link header, following it until every page has been read.
    """
    manifest: dict[str, dict] = {}
    url = API_TREE_URL.format(repo=repo, revision=revision) + "?recursive=true"
    headers = {"User-Agent": "verify-weights"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    while url:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
            link = resp.headers.get("Link", "")
        for entry in payload:
            if entry.get("type") == "file":
                manifest[entry["path"]] = entry
        # Follow the next-page cursor if the server paginated the result.
        url = ""
        if 'rel="next"' in link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part[part.find("<") + 1 : part.find(">")]
                    break
    return manifest


def load_manifest_file(path: str) -> dict[str, dict]:
    """Load a manifest previously saved with --save-manifest."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest_file(path: str, manifest: dict[str, dict]) -> None:
    """Save a manifest for reuse by another node / later run."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def iter_model_files(root: str):
    """Yield relative path and absolute path of every model file under root.

    Under the standard hub layout the content lives in snapshots/<revision>/, so
    that prefix is stripped to match the manifest paths. Direct-download caches
    (files at the repo root) are used as-is.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for filename in filenames:
            if filename in IGNORED_FILES or filename.endswith(IGNORED_SUFFIXES):
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
            rel_path = SNAPSHOT_PREFIX.sub("", rel_path)
            yield rel_path, abs_path


def file_sha256(path: str) -> str:
    """Return the hex SHA-256 of a file, streaming it in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: str, manifest: dict[str, dict], workers: int) -> tuple[list, list, list]:
    """Verify the model directory against the manifest.

    Returns (missing, size_mismatch, hash_mismatch) lists of relative paths.
    """
    local = {rel: path for rel, path in iter_model_files(root)}
    expected = set(manifest)
    missing = sorted(expected - set(local))
    size_mismatch = []

    for rel in sorted(set(local) & expected):
        expected_size = manifest[rel].get("size")
        if expected_size is not None and os.path.getsize(local[rel]) != expected_size:
            size_mismatch.append(rel)

    # Only LFS files carry a content SHA-256 we can compare against.
    hash_targets = [
        (rel, manifest[rel]["lfs"]["oid"].removeprefix("sha256:"))
        for rel in sorted(set(local) & expected)
        if manifest[rel].get("lfs", {}).get("oid")
    ]
    hash_mismatch: list[str] = []
    if hash_targets:
        def check(item):
            rel, expected_oid = item
            actual_oid = file_sha256(local[rel])
            return rel, actual_oid, expected_oid
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for rel, actual_oid, expected_oid in pool.map(check, hash_targets):
                if actual_oid != expected_oid:
                    hash_mismatch.append(rel)

    return missing, size_mismatch, hash_mismatch


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None, help="HF repo id, e.g. RadixArk/Qwen3.8-Flash-Next-NVFP4")
    parser.add_argument("--path", default=None, help="Local model directory (defaults to $HF_HOME/hub/models--ORG--NAME)")
    parser.add_argument("--revision", default="main", help="HF revision (default: main)")
    parser.add_argument("--manifest", default=None, help="Use this saved manifest file instead of fetching the API")
    parser.add_argument("--save-manifest", default=None, help="Save the fetched manifest to this file")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Fetch (and save) the manifest, then exit without verifying")
    parser.add_argument("--workers", type=int, default=8, help="Parallel hash workers (default: 8)")
    parser.add_argument("--quiet", action="store_true", help="Only print problems")
    args = parser.parse_args(argv)

    repo = args.repo or os.environ.get("MODEL_ID", "")
    if not repo:
        print("ERROR: pass --repo or set MODEL_ID", file=sys.stderr)
        return 2

    path = args.path
    if not path:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        org, _, name = repo.partition("/")
        path = os.path.join(hf_home, "hub", f"models--{org}--{name}")

    if args.fetch_only:
        if args.manifest:
            print("ERROR: --fetch-only cannot be combined with --manifest", file=sys.stderr)
            return 2
        try:
            manifest = fetch_manifest(repo, args.revision)
        except Exception as exc:
            print(f"ERROR: could not fetch manifest for {repo}@{args.revision}: {exc}", file=sys.stderr)
            return 2
        if args.save_manifest:
            save_manifest_file(args.save_manifest, manifest)
        if not args.quiet:
            print(f"Manifest: {len(manifest)} files")
        return 0

    if not os.path.isdir(path):
        print(f"ERROR: model directory not found: {path}", file=sys.stderr)
        return 2

    if args.manifest:
        # The launcher needs the checksums without hashing anything locally.
        if args.manifest:
            print("ERROR: --fetch-only cannot be combined with --manifest", file=sys.stderr)
            return 2
        try:
            manifest = fetch_manifest(repo, args.revision)
        except Exception as exc:
            print(f"ERROR: could not fetch manifest for {repo}@{args.revision}: {exc}", file=sys.stderr)
            return 2
        if args.save_manifest:
            save_manifest_file(args.save_manifest, manifest)
        if not args.quiet:
            print(f"Manifest: {len(manifest)} files")
        return 0

    if args.manifest:
        try:
            manifest = load_manifest_file(args.manifest)
        except Exception as exc:  # IOError, malformed JSON
            print(f"ERROR: could not load manifest {args.manifest}: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            manifest = fetch_manifest(repo, args.revision)
        except Exception as exc:  # network errors, HTTP errors, malformed JSON
            print(f"ERROR: could not fetch manifest for {repo}@{args.revision}: {exc}", file=sys.stderr)
            return 2
        if args.save_manifest:
            save_manifest_file(args.save_manifest, manifest)

    expected_bytes = sum(e.get("size", 0) for e in manifest.values())
    if not args.quiet:
        print(f"Manifest: {len(manifest)} files ({expected_bytes / 1e9:.1f} GB)")
        print(f"Verifying {path} ...")
        print("Hashing LFS files; this reads the full checkpoint and takes a while on first run.\n")

    missing, size_mismatch, hash_mismatch = verify(path, manifest, args.workers)

    problems = missing + size_mismatch + hash_mismatch
    if problems:
        for rel in missing:
            print(f"MISSING   {rel}")
        for rel in size_mismatch:
            expected_size = manifest[rel].get("size")
            actual_size = os.path.getsize(path + "/" + rel)
            print(f"SIZE      {rel}: expected {expected_size} bytes, found {actual_size}")
        for rel in hash_mismatch:
            print(f"HASH      {rel}: SHA-256 does not match the manifest")
        print(f"\n{len(problems)} problem(s) in {len(manifest)} expected files.")
        return 1

    if not args.quiet:
        print(f"OK: {len(manifest)} files verified, all sizes and SHA-256 hashes match.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
