### 046 — T-Minus: A Launch Window Board
- Subject: an orbital launch countdown board with staging diagram, telemetry tiles, and a hold/go criteria matrix.
- Direction: range-control board; heavy black panel, amber/white flap text is reserved for 020 — here use backlit LCD segments and a printed procedure sheet.
- Palette: #0a0d10 #161b20 #ff9f1c #e6ebf0 #2f9e6b #d64545.
- Type: `"Helvetica Neue", Helvetica, system-ui`; segment readouts `ui-monospace, SFMono-Regular, Menlo`.
- Layout: countdown hero with segmented digits; vehicle staging SVG (3 stages + fairing) left; telemetry tile grid right; GO/NO-GO matrix bottom.
- Motion: countdown ticks in real time with per-digit flicker; staging events highlight the relevant vehicle section; telemetry values jitter within real tolerances.
- Interaction: start/hold/resume countdown, scrub T-0 events, toggle a nominal/anomaly simulation that trips a NO-GO with an explanation, retry a failed check.
