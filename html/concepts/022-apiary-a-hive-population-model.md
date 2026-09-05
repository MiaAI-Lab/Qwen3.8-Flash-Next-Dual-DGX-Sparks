### 022 — Apiary: A Hive Population Model
- Subject: a seasonal honeybee colony simulation with varroa pressure and forage availability, visualised as a brood-nest heatmap.
- Direction: naturalist's field guide meets data instrument; honeycomb geometry drives the layout; bees drawn as SVG sprites.
- Palette: #f6f0e2 #2b241a #d99b16 #7b4a12 #3f6b4f #b1362f.
- Type: `"Iowan Old Style", Palatino, Georgia, serif` headers; `ui-monospace, Menlo` data.
- Layout: hex comb canvas (centre, hex grid of cells: brood/pollen/honey/empty), right rail of 4 sparkline series, bottom calendar scrubber.
- Motion: comb cells update per simulated day with staggered colour transitions; forager sprites fly procedural paths between comb and a flower strip.
- Interaction: play/pause/step day, sliders for forage, varroa, queen age; clicking a cell shows its contents and age; presets (strong/waning swarm).
