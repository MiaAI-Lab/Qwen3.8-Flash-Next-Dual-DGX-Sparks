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
                            [--manifest FILE] [--save-manifest FILE] [--fetch-only]
                            [--workers N] [--quiet]

The manifest is fetched from the Hugging Face API unless --manifest points at
a previously saved one. check-weights.sh --verify fetches it once on the head
and ships it to the worker, so the worker validates against the same manifest
without needing model-API access. HF_API_BASE overrides the API endpoint (used
by tests and mirrors); HF_TOKEN authenticates private repos.

Exit status:
  0  every manifest file is present, matches its size, and (for LFS files) its SHA-256
  1  at least one problem was found (missing / size / hash / unreadable / broken link)
  2  the manifest could not be obtained, the model directory could not be resolved,
     or the snapshot revision could not be determined

Files in the model directory that are not in the manifest (lock files, download
artifacts, trees metadata, stale snapshots, ...) are reported as extras but are
not failures. Under the standard hub layout, the verified tree is the snapshot
that refs/main points to (or the only snapshot, or an explicit --path); the
deduplicated blobs/ and other snapshots are not hashed twice.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

API_BASE = os.environ.get("HF_API_BASE", "https://huggingface.co")
API_TREE_URL = API_BASE + "/api/models/{repo}/tree/{revision}?recursive=true"

# Directory names that are download machinery, not model content. "snapshots"
# is handled explicitly (the verified tree is the selected snapshot); blobs/
# holds the deduplicated blobs and must not be walked.
IGNORED_DIRS = {".cache", ".git", "blobs"}
IGNORED_SUFFIXES = (".lock", ".metadata", ".incomplete")
IGNORED_FILES = {".gitignore", "CACHEDIR.TAG"}

_EMPTY_OID = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class VerifierError(Exception):
    """Raised when the expected manifest or the model tree cannot be resolved."""


def fetch_manifest(repo: str, revision: str) -> dict[str, dict]:
    """Fetch the full recursive file manifest for a repo revision.

    Returns a mapping of relative path -> entry. Handles API pagination via the
    Link header, following it until every page has been read, and raises
    VerifierError with a useful message on any HTTP or JSON failure.
    """
    manifest: dict[str, dict] = {}
    url = API_TREE_URL.format(repo=repo, revision=revision)
    headers = {"User-Agent": "verify-weights"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    while url:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read())
                link = resp.headers.get("Link", "")
        except urllib.error.HTTPError as exc:
            raise VerifierError(
                f"HTTP {exc.code} for {repo}@{revision} "
                f"(repo not found, rate-limited, or auth required: {exc.reason})"
            ) from exc
        except Exception as exc:  # URLError, timeout, malformed JSON, ...
            raise VerifierError(f"could not fetch {url}: {exc}") from exc
        if not isinstance(payload, list):
            raise VerifierError(f"unexpected manifest payload for {repo}@{revision}")
        for entry in payload:
            if isinstance(entry, dict) and entry.get("type") == "file":
                manifest[entry["path"]] = entry
        # Follow the next-page cursor if the server paginated the result.
        url = ""
        if 'rel="next"' in link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part[part.find("<") + 1 : part.find(">")]
                    break
    return manifest


def load_manifest(path: str) -> dict[str, dict]:
    """Load a manifest previously saved with --save-manifest."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # IOError, malformed JSON
        raise VerifierError(f"could not load manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerifierError(f"manifest {path} is not a mapping of path -> entry")
    return payload


def save_manifest(path: str, manifest: dict[str, dict]) -> None:
    """Save a manifest for reuse by another node / later run."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


def resolve_snapshot(root: str) -> tuple[str, bool]:
    """Return (tree_dir, is_snapshot) for the model directory root.

    The standard hub layout stores content under snapshots/<revision>/; a
    direct-download cache stores files at the model root. When snapshots exist,
    the revision pointed to by refs/main is preferred, falling back to the only
    available snapshot. When multiple snapshots exist and the revision cannot
    be resolved, that is an error: hashing the wrong (stale) snapshot would
    produce false failures.
    """
    snapshots = os.path.join(root, "snapshots")
    if not os.path.isdir(snapshots):
        return root, False

    revision = None
    refs_main = os.path.join(root, "refs", "main")
    if os.path.isfile(refs_main):
        # huggingface_hub >= 1.x writes this with a trailing newline; strip it
        # (see README gotcha) before using the value as a directory name.
        with open(refs_main, "r", encoding="utf-8") as handle:
            revision = handle.read().strip()
    if revision:
        candidate = os.path.join(snapshots, revision)
        if os.path.isdir(candidate):
            return candidate, True

    try:
        entries = sorted(
            os.path.join(snapshots, name)
            for name in os.listdir(snapshots)
            if os.path.isdir(os.path.join(snapshots, name))
        )
    except OSError as exc:
        raise VerifierError(f"could not read snapshot dir {snapshots}: {exc}") from exc

    if len(entries) == 1:
        return entries[0], True
    if not entries:
        return root, False
    raise VerifierError(
        f"multiple snapshots under {snapshots} and refs/main is missing or points "
        "at a nonexistent revision; pass --path to a specific snapshot"
    )


def iter_model_files(tree_dir: str) -> tuple[dict[str, str], list[str]]:
    """Walk tree_dir and return (files, broken_links).

    files maps manifest-relative path -> absolute path. Only real files are
    yielded; dangling symlinks are recorded in broken_links. The walk follows
    directory symlinks but tracks visited real paths, so symlink cycles (for
    example a snapshot dir pointing back at the model root) terminate.
    """
    files: dict[str, str] = {}
    broken: list[str] = []
    visited: set[str] = set()
    stack = [tree_dir]
    while stack:
        dirpath = stack.pop()
        real = os.path.realpath(dirpath)
        if real in visited:
            continue
        visited.add(real)
        try:
            names = sorted(os.listdir(dirpath))
        except OSError as exc:
            raise VerifierError(f"could not read directory {dirpath}: {exc}") from exc
        for name in names:
            path = os.path.join(dirpath, name)
            if os.path.isdir(path):  # follows symlinks
                if name not in IGNORED_DIRS:
                    stack.append(path)
                continue
            rel = os.path.relpath(path, tree_dir).replace(os.sep, "/")
            if os.path.islink(path) and not os.path.exists(path):
                broken.append(rel)
                continue
            if not os.path.isfile(path):
                continue  # sockets, fifos, and other oddities are not model files
            if name in IGNORED_FILES or name.endswith(IGNORED_SUFFIXES):
                continue
            files[rel] = path
    return files, broken


def file_sha256(path: str) -> str:
    """Return the hex SHA-256 of a file, streaming it in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_oid(entry: dict) -> str:
    """Return the lowercase hex SHA-256 for a manifest entry ("" if none).

    Handles lfs being absent or null (regular git files, some API responses),
    any-case "sha256:" prefix or none. Only non-empty LFS oids are hash-checked;
    empty files are covered by the size check (their well-known hash is implied).
    """
    lfs = entry.get("lfs") or {}
    oid = lfs.get("oid", "") or ""
    oid = oid.strip().lower()
    # The HF API always uses a lowercase "sha256:" prefix; tolerate any case
    # and a missing prefix so manifests from other sources also verify.
    if oid.startswith("sha256:"):
        oid = oid[len("sha256:"):]
    if oid == _EMPTY_OID:
        return ""  # an empty file has a well-known hash; size already covers it
    return oid


class Problem:
    """A single verification problem, categorized for reporting."""

    def __init__(self, kind: str, rel: str, detail: str = ""):
        self.kind = kind
        self.rel = rel
        self.detail = detail


def verify(tree_dir: str, manifest: dict[str, dict], workers: int) -> list[Problem]:
    """Verify the model tree against the manifest and return all problems.

    Every manifest entry is checked for presence; present files are checked for
    size and, for entries carrying an LFS SHA-256, for content hash. Problems
    are collected instead of raising, so one bad file does not hide another.
    """
    if workers < 1:
        workers = 1

    local, broken = iter_model_files(tree_dir)
    expected = set(manifest)
    problems: list[Problem] = []

    for rel in sorted(expected - set(local)):
        problems.append(Problem("MISSING", rel))
    for rel in broken:
        problems.append(Problem("BROKEN", rel, "dangling symlink"))

    # Per-file size / readability check. A file that cannot be opened or stat'd
    # is reported and excluded from hashing (avoiding a duplicate error).
    hashable: list[str] = []
    for rel in sorted(set(local) & expected):
        path = local[rel]
        entry = manifest[rel]
        expected_size = entry.get("size")
        try:
            actual_size = os.path.getsize(path)
        except OSError as exc:
            problems.append(Problem("UNREADABLE", rel, str(exc)))
            continue
        if isinstance(expected_size, int) and actual_size != expected_size:
            problems.append(
                Problem("SIZE", rel, f"expected {expected_size} bytes, found {actual_size}")
            )
            # The size already differs, but a manifest side-by-side read can
            # race with a writer; still hash it so a same-size rewrite is caught.
        hashable.append(rel)

    hash_targets = [
        (rel, expected_oid(manifest[rel]))
        for rel in hashable
        if expected_oid(manifest[rel])
    ]
    if hash_targets:
        def check(item: tuple[str, str]) -> Problem | None:
            rel, oid = item
            expected_size = manifest[rel].get("size")
            expected = expected_oid(manifest[rel])
            try:
                actual = file_sha256(local[rel])
            except OSError as exc:
                return Problem("UNREADABLE", rel, str(exc))
            if actual != expected:
                return Problem("HASH", rel, f"expected {expected[:12]}…, found {actual[:12]}…")
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for outcome in pool.map(check, hash_targets):
                if outcome is not None:
                    problems.append(outcome)

    return problems


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

    if args.fetch_only:
        if args.manifest:
            print("ERROR: --fetch-only cannot be combined with --manifest", file=sys.stderr)
            return 2
        try:
            manifest = fetch_manifest(repo, args.revision)
        except VerifierError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.save_manifest:
            save_manifest(args.save_manifest, manifest)
        if not args.quiet:
            print(f"Manifest: {len(manifest)} files")
        return 0

    if args.path:
        tree_dir = args.path
        # If --path names the model root under the hub layout with multiple
        # snapshots, resolve which one to hash.
        try:
            tree_dir, _ = resolve_snapshot(args.path)
        except VerifierError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        org, _, name = repo.partition("/")
        model_dir = os.path.join(hf_home, "hub", f"models--{org}--{name}")
        if not os.path.isdir(model_dir):
            print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
            return 2
        try:
            tree_dir, _ = resolve_snapshot(model_dir)
        except VerifierError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    if not os.path.isdir(tree_dir):
        print(f"ERROR: model directory not found: {tree_dir}", file=sys.stderr)
        return 2

    if args.manifest:
        try:
            manifest = load_manifest(args.manifest)
        except VerifierError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            manifest = fetch_manifest(repo, args.revision)
        except VerifierError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.save_manifest:
            save_manifest(args.save_manifest, manifest)

    expected_bytes = sum(e.get("size", 0) for e in manifest.values() if isinstance(e.get("size"), int))
    if not args.quiet:
        print(f"Manifest: {len(manifest)} files ({expected_bytes / 1e9:.1f} GB)")
        print(f"Verifying {tree_dir} ...")
        print("Hashing LFS files; this reads the full checkpoint and takes a while on first run.\n")

    problems = verify(tree_dir, manifest, args.workers)

    if problems:
        for p in problems:
            line = f"{p.kind:<10} {p.rel}"
            if p.detail:
                line = f"{line} ({p.detail})"
            print(line)
        print(f"\n{len(problems)} problem(s) in {len(manifest)} expected files.")
        return 1

    if not args.quiet:
        print(f"OK: {len(manifest)} files verified, all sizes and SHA-256 hashes match.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
