### 049 — Develop: A Raw Image Bench
- Subject: a photo develop bench — histogram, tone curves, zones, exposure latitude — operating on a synthesized scene drawn in SVG/canvas (no external images).
- Direction: darkroom-grey bench UI; the "image" is a procedurally rendered landscape so the whole page stays self-contained.
- Palette: #12141a #1c2027 #e8eaee #ff9f1c #3ba7ff #6fd39a #ff5d5d.
- Type: `"Helvetica Neue", Helvetica, system-ui`; numeric readouts `ui-monospace, Menlo`.
- Layout: canvas viewport left with before/after split handle; right controls (exposure, contrast, blacks, whites, clarity, split tone); bottom histogram with zone overlay.
- Motion: histogram recomputes live during drags; a split-sweep animation shows before/after; clipping warnings pulse.
- Interaction: real per-pixel tone mapping (LUT built from a draggable curve), draggable curve with control points, presets, exposure simulation, histogram hover shows RGB counts.
