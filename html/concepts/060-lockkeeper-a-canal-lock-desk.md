### 060 — Lockkeeper: A Canal Lock Desk
- Subject: operate a narrow-lock flight — gates, paddles, water levels, boat passage — with interlocks and a passage log.
- Direction: British Waterways board; enamelled plates, brass windlass, level gauges drawn in SVG with live water fill.
- Palette: #0f1416 #1b2426 #c9a227 #2f6f6b #d1341f #e9e4d6.
- Type: `"Helvetica Neue", Helvetica, system-ui`; levels `ui-monospace, Menlo`.
- Layout: lock cross-section SVG hero (paddles, gates, water, boat); control desk with windlass buttons; flight profile strip (3 locks); event log.
- Motion: water fills/drains with animated surface and bobbing boat; gate swing in 3D-ish rotateY; paddle lift animates and flow ripples appear.
- Interaction: strict interlock state machine (can't open paddles with gates open / can't open gates against head), sequence auto-cycle, flight selector, log accumulates times.
