### 025 — Seismograph: Reading the Earthquake
- Subject: three live-ish seismogram channels, a magnitude scale explainer, and a hypocentre locator from P/S delay.
- Direction: scientific plotter output on graph paper; ink lines with slight pen wobble; station icons in a small network map.
- Palette: #f2f4ef #10151b #c0392b #1f6f6b #d6a419 #6c7a86.
- Type: `"Helvetica Neue", Helvetica, system-ui`; time labels `ui-monospace, Menlo`.
- Layout: station map top-left, three stacked channel canvases (procedural waveforms), right column magnitude ladder; bottom locator worksheet.
- Motion: continuous waveform scroll with event injections; a picked-event marker propagates to all channels; magnitude ladder fills.
- Interaction: click to pick P and S on any channel → distance circles redraw and hypocentre triangulates; station toggles; a synthetic event generator button; magnitude slider scales amplitudes.
