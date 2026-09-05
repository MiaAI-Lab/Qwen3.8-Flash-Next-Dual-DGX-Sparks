### 063 — Astrolabe: A Planispheric Instrument
- Subject: a working astrolabe — rotate the rete, read altitude/azimuth, tell time by star altitude, with a latitude plate swap.
- Direction: engraved brass on dark leather; hand-hatched limb divisions; almucantar and azimuth grids as real stereographic circles.
- Palette: #e8dfc8 #14110d #b08a34 #3b5f74 #8c4a34.
- Type: `Copperplate, Optima, "Hoefler Text", Georgia, serif` engravings; degrees `ui-monospace, Menlo`.
- Layout: layered astrolabe (mother, plate, rete, rule) as stacked SVG groups; latitude plate picker; star almanac list; reading panel.
- Motion: rete rotates with inertia and settles; a selected star's altitude readout updates continuously; sun pointer tracks the date ring.
- Interaction: drag to rotate rete/rule, latitude select redraws the plate (real stereographic math), click a star → altitude/azimuth/time, date+time entry sets the rule.
