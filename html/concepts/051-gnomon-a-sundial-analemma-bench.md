### 051 — Gnomon: A Sundial & Analemma Bench
- Subject: build a horizontal sundial for any latitude — hour lines by real trig, plus an analemma loop and a declination table.
- Direction: brass-on-slate instrument plate; engraved hairlines; a drawn shadow that tracks a computed sun position.
- Palette: #eae4d6 #171a1d #a8862f #3f5d6b #8c4a34.
- Type: `Optima, "Avenir Next", system-ui`; hour labels `ui-monospace, Menlo`.
- Layout: dial face SVG hero; latitude dial control; right column: equation-of-time table, declination chart; bottom analemma plot.
- Motion: sun azimuth/elevation animate across a day, gnomon shadow sweeps the hour lines; analemma draws as a continuous loop.
- Interaction: latitude slider redraws every hour line live, date+time scrubber moves the shadow, solstice/equinox preset buttons, toggle true-sun vs mean-sun time.
