### 036 — Contour: A Cartographer's Drafting Table
- Subject: a hand-drafted topographic sheet with contours, hypsometric tints, spot heights, a graticule and scale bar.
- Direction: survey sheet; contours as procedural SVG polylines from a value field with marching squares, hairline pen weights.
- Palette: #f4f1e6 #1b1f1c #6d8f5e #c9a04a #4a7ea1 #8d5a3c.
- Type: `Optima, "Avenir Next", system-ui` (cartographic); coordinates `ui-monospace, Menlo`.
- Layout: full-bleed map sheet; north arrow + scale bar + legend box; margin column with projection notes; index to spot heights.
- Motion: contours draw outward in elevation order (staggered dash reveal); a relief-hillshade layer fades in computed per-pixel on canvas.
- Interaction: contour interval stepper, tint/hillshade/labels toggles, dragger that re-seeds the terrain, click anywhere → elevation + pseudo-coordinate readout.
