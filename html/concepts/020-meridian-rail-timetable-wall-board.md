### 020 — Meridian: Rail Timetable Wall Board
- Subject: a split-flap style departure board for 12 fictional lines, plus a journey planner.
- Direction: enamel signage, black board with white/amber flaps, mechanical precision, subtle bevels via gradients only.
- Palette: #0b0c0e #16181c #f2f2f0 #ffb020 #4ea3e0 #d1495b.
- Type: `"Helvetica Neue", Helvetica, system-ui` uppercase; flap glyphs monospaced `ui-monospace, Menlo`.
- Layout: full-width board grid (rows of flaps), a planner strip below, a platform map column.
- Motion: split-flap flip cascade on load and on refresh (CSS rotateX per cell, staggered); rows pulse when a service is delayed.
- Interaction: filter chips (line/destination) re-flip only matching rows; planner selects origin→destination and shows next three departures; click a row → detail drawer with stops.
