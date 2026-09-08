#!/usr/bin/env python3
"""Print a snapshot hash for a Hugging Face hub repo dir (models--org--name).

Exit codes:
  0  complete — every shard named by model.safetensors.index.json exists
  1  incomplete snapshot found (partial download)
  2  no snapshot directory at all

Prefers refs/main when that snapshot is complete, else the newest complete
snapshot, else refs/main even if incomplete (so a partial tree can resume).

Usage: resolve_snapshot.py <hub-repo-dir>
"""
from __future__ import annotations

import json
import pathlib
import sys


def complete(snapshot: pathlib.Path) -> bool:
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file():
        return False
    weight_map = json.loads(index.read_text()).get("weight_map", {})
    return bool(weight_map) and all(
        (snapshot / name).is_file() for name in set(weight_map.values())
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: resolve_snapshot.py <hub-repo-dir>", file=sys.stderr)
        return 2
    repo = pathlib.Path(argv[1])
    snap_root = repo / "snapshots"
    main_ref = ""
    ref_file = repo / "refs" / "main"
    if ref_file.is_file():
        main_ref = ref_file.read_text().strip()
    if main_ref and complete(snap_root / main_ref):
        print(main_ref)
        return 0
    complete_snaps: list[pathlib.Path] = []
    if snap_root.is_dir():
        complete_snaps = [p for p in snap_root.iterdir() if p.is_dir() and complete(p)]
    if complete_snaps:
        complete_snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(complete_snaps[0].name)
        return 0
    if main_ref and (snap_root / main_ref).is_dir():
        print(main_ref)
        return 1
    if snap_root.is_dir():
        cands = sorted(
            (p for p in snap_root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            print(cands[0].name)
            return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
