# Page authoring brief — invariant contract

Read this whole file first. Then build the ONE page whose number and concept you were
given. Your concept is a ~900-byte file at
`/home/mia/NewModels/Qwen3.8-Flash-vLLM/html/concepts/NNN-*.md` — glob for your three-digit
number and read that one file. Do NOT read `concepts.md`: it is the 80 KB master index of all
100 concepts, and pulling it into your context for one entry is what makes agent streams stall.
Implement the concept faithfully: its Subject, art Direction, Palette, Type stacks, Layout,
Motion and Interaction bullets are the spec.

You are authoring ONE premium, self-contained, single-file web page. Excellence matters
more than speed. This is clean-room work: no generator, no template, no reuse of any
other page.

## Deliverables (exactly two files, nothing else)
- `/home/mia/NewModels/Qwen3.8-Flash-vLLM/html/100-html/NNN-<slug>.html`
- `/home/mia/NewModels/Qwen3.8-Flash-vLLM/html/100-html/NNN-<slug>.txt`

Use the exact numbered, kebab-cased slug from the concept heading.

## Hard boundaries
- Write ONLY those two files. Never read, list, copy or reference any other directory or
  any other `.html` file — not even for inspiration. Do not `ls` the `100-html` directory.
  (Reading `concepts.md` and this file is expected; nothing else in the repo.)
- Fully self-contained: all CSS in exactly ONE `<style>`, all JS in exactly ONE `<script>`.
  No external anything — no CDN, no `link rel=stylesheet`, no `@import`, no `url()` to a
  file, no webfonts, no external images or SVG files, no libraries. It must work offline
  from a double-click.
- System font stacks ONLY (the stacks named in the concept, plus generic families).
- Never set `overflow-x: hidden` on `body`/`html` to disguise overflow — fix the real cause.
- `<html lang="en">`, `<meta charset="utf-8">`, a viewport meta, and a real `<title>`.
- Honour `prefers-reduced-motion` (deliver the settled end state, not a frozen animation).
- Roughly 700–950 substantial lines. Real, specific, researched-feeling copy. Never lorem
  ipsum, never placeholder text, no TODOs, no dead controls.
- Where the concept implies arithmetic or physics, compute it for real (the readouts must
  be derivable), and refuse or explain the out-of-range case instead of faking a number.

## Anti-stall rule (previous agents died here with zero output)
- Do NOT plan at length, do NOT explore the filesystem, do NOT read any other file.
- Your FIRST tool call must be the `write_file` that creates the HTML with chunk 1.
- Every chunk is ~140–180 lines. Never attempt the whole file in one call.

## Mandated authoring method
1. `write_file` chunk 1 (~140–180 lines) ending with a sentinel on its own final line: `<!--CH2-->`
2. Then repeatedly `edit` the sentinel into the next ~140–180 lines plus the next sentinel
   (`<!--CH3-->`, `<!--CH4-->`, …). One chunk per edit.
3. FINAL chunk: delete the sentinel entirely and close `</script></body></html>`.
4. Then `grep -n "CH[0-9]" <your file>` must print NOTHING. Remove any survivor.

## Gates — all three green before you report; iterate until they are
1. Static: `cd /home/mia/NewModels/Qwen3.8-Flash-vLLM/html && python3 verify100.py 100-html`
   → your number must be `[ok]`. Ignore lines for numbers that are not yours.
2. Runtime: `python3 render100.py NNN` → must print `console=0` and `failing=0`. Any console
   error is a real bug: find and fix it. Never report success around one.
3. Responsive, run exactly:
```
B=$HOME/.cache/ms-playwright/chromium-1208/chrome-linux/chrome
cd /home/mia/NewModels/Qwen3.8-Flash-vLLM/html
for W in 390 768 1440; do
  timeout 60 "$B" --headless --disable-gpu --no-sandbox --allow-file-access-from-files \
    --virtual-time-budget=20000 --window-size=560,1000 \
    --dump-dom "file://$PWD/probe.html#file://$PWD/100-html/NNN-<slug>.html|$W" 2>/dev/null | grep -o 'METRICS {[^<]*' | tail -1
done
```
The authoritative number is `overflow` (document `scrollWidth` − `innerWidth`): it must be
`<= 2` at **every** width. `"culprits":[]` is expected but only advisory — if `overflow` is
fine, do not chase transform/scale-related noise.

When `overflow` IS `> 2`, name the true cause with the deep probe instead of guessing
(`grow` lists the widest non-clipping boxes, `top` lists painted spillers):
```
timeout 60 "$B" --headless --disable-gpu --no-sandbox --allow-file-access-from-files \
  --virtual-time-budget=20000 --window-size=560,1000 \
  --dump-dom "file://$PWD/probe-deep.html#file://$PWD/100-html/NNN-<slug>.html|1440" 2>/dev/null | tr '\n' ' ' | sed -n 's/.*id="out">\(DEEP .\{0,900\}\)/\1/p'
```
Fix the cause, never mask it. The recurring causes and their cures:
- grid track `1fr` → `minmax(0,1fr)` (intrinsic min-content otherwise propagates and widens the page)
- flex item missing `min-width: 0`
- `white-space: nowrap` text → a wrapping media query for narrow widths
- display type whose `clamp()` max cannot fit its own column (check: an uppercase word at
  the max size must fit inside the column's content width) → lower the clamp, and add
  `overflow-wrap: anywhere` as a guard
- an absolutely positioned caption/rail centred with `left:50%` → give it a bounded width

## The .txt sidecar
Exactly these five literal labels, in this order, each starting a line:

    TITLE:  PROMPT:  DESIGN:  TECHNIQUES:  INTERACTION:

- `TITLE:` the page's short title.
- `PROMPT:` the exact prompt that produced this page — write the concept entry out as a
  real, complete generative prompt.
- `DESIGN:` one paragraph on intended design: art direction, palette rationale,
  typography, composition.
- `TECHNIQUES:` the major visual techniques **actually present in your file** — name the
  concrete ones (which canvas ops, which SVG filters/gradients/patterns, blend modes,
  easing/stagger strategy, how the geometry or data is computed).
- `INTERACTION:` the interaction model precisely enough to rebuild the page from it.

Aim 40–80 lines. Describe what the file really does, not what you meant it to do.
Enough that someone could recreate the page from the `.txt` alone.

## Report back
Final line count; the three gate outputs verbatim (the `[ok]` line, the `console=0` line,
the three `METRICS` lines); and one sentence on what makes this page visually distinct.
Report honestly if any gate is not green — a false "green" report wastes a whole rebuild.

## Artwork actually has to exist (a real failure mode here)
One agent reported six detailed SVG drawings, specific path counts and filter parameters
for a page that contained **zero** `<svg>` and **zero** `<path>` elements. A green render
and `console=0` do NOT prove the artwork exists — a page of empty chrome renders fine.

So also run the deep probe and report its `art` object verbatim:
```
timeout 60 "$B" --headless --disable-gpu --no-sandbox --allow-file-access-from-files \
  --virtual-time-budget=20000 --window-size=1500,1000 \
  --dump-dom "file://$PWD/probe-deep.html#file://$PWD/100-html/NNN-<slug>.html|1440" 2>/dev/null | tr '\n' ' ' | sed -n 's/.*"art":\({[^}]*}\).*/\1/p'
```
It counts the **live DOM** after JS runs, so JS-built art (`createElementNS`) counts too.
Judge it against your concept's declared medium:
- concept implies drawn artwork (plates, diagrams, specimens, maps) → `svg` ≥ 1 and the
  drawing primitives (`path`+`circle`+`ellipse`+`rect`+`line`+`polygon`+`polyline`) must
  number in the dozens; six separate drawings means `svg` ≥ 6 unless one root `<svg>`
  genuinely holds all six as `<g>` groups.
- concept implies procedural rendering → `canvas` ≥ 1.
- pure CSS construction is legitimate (`svg:0, path:0`) but then the DOM should still be
  substantial — expect `elems` in the low hundreds.
For reference, three accepted pages measured: `006` `{svg:6,path:239,elems:741}`,
`013` `{svg:2,canvas:1,path:95,elems:435}`, `011` `{svg:0,path:0,elems:201}`.
Never claim a feature you cannot point to in that output or in a line of code.
