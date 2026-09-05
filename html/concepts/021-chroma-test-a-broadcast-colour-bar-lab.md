### 021 — Chroma Test: A Broadcast Colour Bar Lab
- Subject: a lab explainer of video test patterns (SMPTE bars, EBU fan, convergence, grey ramps) with live generated signals.
- Direction: engineering-room schematic; dark bezel framing around bright generated canvases; no nostalgic scanline slop — treat it as calibration equipment.
- Palette: #0c0d10 #1a1c21 #ffffff #ff0000 #00ff00 #0000ff #ffd400 #00d8d8 #ff00ff.
- Type: `"Helvetica Neue", Helvetica, system-ui`; instrument readouts `ui-monospace, Menlo`.
- Layout: canvas rack of generated patterns, each with measured values; right column theory notes; bottom vectorscope.
- Motion: live 100 Hz redraw of the vectorscope from the bar signal; a "fault" toggle injects artefacts (crush, chroma shift, moiré).
- Interaction: pattern selector, levels/tilt sliders that truly rescale pixel values and update readouts, fault injector, gamut toggle 709/P3.
