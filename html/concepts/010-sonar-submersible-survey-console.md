### 010 — Sonar: Submersible Survey Console
- Subject: a sonar sweep console rendering a procedurally generated seabed with detected contacts and a scrolling A-scan.
- Direction: dark navy instrument, phosphor-green traces, scanlines, mechanical bezels; absolutely no glassmorphism.
- Palette: #04090e #08161f #0e2a33 #38f0b0 #f3f7f5 #ffb703 (caution).
- Type: labels `"Helvetica Neue", Helvetica, system-ui` uppercase tight; readouts `ui-monospace, SFMono-Regular, Menlo`.
- Layout: 3-column console — B-scope PPI canvas (left, square), seabed profile canvas (right), bottom strip of A-scan + contact table with real nomenclature.
- Motion: rotating sweep with afterglow falloff (canvas trails), contacts bloom as the beam passes, A-scan scrolls right-to-left, depth counter ticks.
- Interaction: range knob (200/500/1000 m) rescales, gain slider changes noise floor, clicking a contact row slews the beam and annotates the PPI, ping button fires a manual pulse.
