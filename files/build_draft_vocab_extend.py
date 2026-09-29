#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Extend a shipped draft vocabulary with a second language (TP-aware repo).

Ported from the single-Spark recipe's files/build_draft_vocab_extend.py
(oscarmenendezgarcia), which exists because rebuilding from scratch with a
dictionary as corpus dropped 90 byte-fallback ids and destroyed Spanish.
Three rules from that failure, kept verbatim:

  1. The shipped 47k goes in WHOLE as a floor. Never lose an id that works.
  2. The byte-fallback range is pinned unconditionally, whatever the
     frequencies say (those ids assemble every multi-byte UTF-8 char:
     accented Latin, Cyrillic, CJK, emoji).
  3. Ids are only ADDED by FREQUENCY over real text. Never a dictionary,
     where every inflected form weighs as much as "que".

Usage:
  python3 files/build_draft_vocab_extend.py \
      --base files/draft_vocab_en_code_47k.txt \
      --corpus ~/.cache/draft-vocab-corpus/es.txt \
      --size 65536 --out files/draft_vocab_es_en_code_65k.txt

Prints the per-shard id spread at --tp-report ranges, because this repo's
lm_head is vocab-parallel: a decode step waits for the slowest rank, so a
vocabulary concentrated in one rank's id range saves bandwidth only there.
Corpus-ranked ids are never dropped or moved to balance.
"""
import argparse, json, os, pathlib, sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="shipped vocabulary (floor)")
ap.add_argument("--corpus", nargs="+", required=True,
                help="text files; .jsonl needs a 'text' field; suffix :N repeats")
ap.add_argument("--size", type=int, default=65536)
ap.add_argument("--model", default=os.environ.get(
    "DRAFT_VOCAB_MODEL", "nvidia/Qwen3.8-Flash-Next-NVFP4"))
ap.add_argument("--byte-fallback-max", type=int, default=400,
                help="every id below this is kept unconditionally")
ap.add_argument("--tp-report", type=int, default=2,
                help="report the id spread over this many equal ranges "
                     "(set to the tensor-parallel size); information only")
ap.add_argument("--out", required=True)
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
vocab_size = len(tok)

base = {int(l) for l in pathlib.Path(a.base).read_text().split() if l.strip()}
print(f"  floor (shipped): {len(base):,} ids")

cnt = Counter()
CH = 1 << 20
for spec in a.corpus:
    path, rep = (spec.rsplit(":", 1) + ["1"])[:2] if ":" in spec and spec.rsplit(":",1)[1].isdigit() else (spec, "1")
    rep = int(rep)
    p = pathlib.Path(path)
    for _ in range(rep):
        if p.suffix == ".jsonl":
            for line in p.open(encoding="utf-8"):
                t = json.loads(line).get("text", "")
                if t:
                    cnt.update(tok(t, add_special_tokens=False)["input_ids"])
        else:
            with p.open(encoding="utf-8") as f:
                while True:
                    chunk = f.read(CH)
                    if not chunk:
                        break
                    cnt.update(tok(chunk, add_special_tokens=False)["input_ids"])
    print(f"  {p.name} x{rep}: {sum(cnt.values()):,} occurrences, "
          f"{len(cnt):,} distinct ids")

# byte-fallback and specials, unconditionally
pinned = {i for i in range(min(a.byte_fallback_max, vocab_size))}
pinned |= set(tok.all_special_ids or [])
print(f"  pinned unconditionally: {len(pinned):,} (byte-fallback + specials)")

vocab = set(base) | pinned
if len(vocab) > a.size:
    sys.exit(f"  floor + pinned ({len(vocab):,}) already exceeds --size {a.size:,}")

slots = a.size - len(vocab)
extra = [i for i, _ in cnt.most_common() if i not in vocab][:slots]
vocab |= set(extra)
print(f"  added by frequency: {len(extra):,}  -> total {len(vocab):,}")

tot = sum(cnt.values())
cub = sum(c for i, c in cnt.items() if i in vocab)
print(f"  coverage over the corpus: {cub / tot * 100:.3f}%")
print(f"  byte-fallback kept (<{a.byte_fallback_max}): "
      f"{len([i for i in vocab if i < a.byte_fallback_max])}")

nsh = max(1, a.tp_report)
width = -(-vocab_size // nsh)
spread = [sum(1 for i in vocab if sh * width <= i < (sh + 1) * width)
          for sh in range(nsh)]
print(f"  shard spread over {nsh} ranges of {width:,} ids: {spread}")

o = pathlib.Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
o.write_text("".join(f"{i}\n" for i in sorted(vocab)))
print(f"  wrote {len(vocab):,} ids -> {o}")
