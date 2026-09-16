#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Evict a checkpoint's own clean pages from the page cache, without root.

On GB10 the page cache competes with the model for one pool of unified memory.
A checkpoint that was just written — an `hf download`, an rsync to the worker,
an NFS read on a previous launch — leaves its bytes resident as clean pages.
vLLM then loads the same bytes again, and weight loading can die partway with
`CUDA out of memory` on a machine that has nothing else running.

`echo 3 > /proc/sys/vm/drop_caches` fixes it and needs root, which these nodes
do not have passwordless. `posix_fadvise(POSIX_FADV_DONTNEED)` drops the clean
pages of files you can open, needs no privileges, and touches only the files
named here rather than the whole system's cache.

Clean pages only: dirty pages are written back by the kernel on its own
schedule and are not dropped, so this cannot lose data. Read-only opens.

    python3 evict_page_cache.py <dir> [<dir> ...]

Prints one summary line to stderr. Never fails the caller: a file that cannot
be opened or a platform without posix_fadvise is skipped, because evicting the
cache is an optimisation and refusing to launch over it would be worse than
the problem. Reports what it actually managed to do.
"""
from __future__ import annotations

import os
import sys

# Only the big, re-read-by-the-engine files are worth evicting. Configs and
# tokenizers are kilobytes and the engine wants them in cache anyway.
SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")


def evict(path: str) -> int:
    """Drops `path`'s clean pages. Returns bytes evicted, 0 if it could not."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        size = os.fstat(fd).st_size
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return size
    except (OSError, AttributeError):
        # AttributeError: no posix_fadvise on this platform. Not an error here.
        return 0
    finally:
        os.close(fd)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: evict_page_cache.py <dir> [<dir> ...]", file=sys.stderr)
        return 2

    total = files = skipped = 0
    for root in argv:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(SUFFIXES):
                    continue
                n = evict(os.path.join(dirpath, name))
                if n:
                    total += n
                    files += 1
                else:
                    skipped += 1

    if files or skipped:
        msg = f"page cache: released {total / 2**30:.1f} GiB across {files} file(s)"
        if skipped:
            msg += f", {skipped} skipped"
        print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
