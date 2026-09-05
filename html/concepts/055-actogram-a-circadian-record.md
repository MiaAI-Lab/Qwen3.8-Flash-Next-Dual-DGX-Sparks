### 055 — Actogram: A Circadian Record
- Subject: double-plotted activity records for 3 conditions (LD, DD, jet-lag), with free-running period fits and light schedule bars.
- Direction: chronobiology plotter; crisp black-on-white raster rows, no decoration; a real scientific instrument look.
- Palette: #f7f5ef #101314 #1f6f8b #c0392b #d6a419 #55606a.
- Type: `"Helvetica Neue", Helvetica, system-ui`; times `ui-monospace, Menlo`.
- Layout: actogram raster (canvas) hero; left condition selector + subject stats; bottom light-schedule strip and period estimate card.
- Motion: rows plot progressively day by day; a regression line fits the onset drift with animated slope; free-run drift ticks.
- Interaction: switch conditions, adjust light-period slider (re-renders schedule and drift), hover a row → date/onset/activity %, toggle rhythm-fit overlay.
