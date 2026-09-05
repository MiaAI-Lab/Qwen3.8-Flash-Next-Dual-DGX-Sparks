### 016 — Nocturne: Observatory Night Assistant
- Subject: a dark observing-planner for 14 deep-sky objects, with altitude-over-time curves and moon-phase penalty.
- Direction: dimmed red-light astronomy aesthetic; SVG line charts; no bright whites (a dark-page page must still be legible, not a black void).
- Palette: #06080d #0d1220 #151d2c #e25555 (dim red) #cfd8e3 #f0c674 (amber).
- Type: labels `Optima, "Avenir Next", system-ui`; numerals `ui-monospace, Menlo`.
- Layout: header with site coordinates + local sidereal clock; main SVG altitude chart (multiple curves); left selectable object list; bottom detail card.
- Motion: time scrubber animates the "now" line; curves illuminate progressively on load; twinkle on the star list markers.
- Interaction: click object to toggle its curve; scrub time (drag on chart) → altitudes update, transit/azimuth recompute; moon-phase slider dims targets; keyboard arrows step 15 min.
