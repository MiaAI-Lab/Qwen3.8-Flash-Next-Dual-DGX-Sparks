### 028 — Signal Garden: A Radio Propagation Map
- Subject: HF skywave propagation across 6 bands with an MUF dial, day/night terminator, and band-condition table.
- Direction: cartographic night side; SVG world-ish grid (stylised, not real geo), glowing ionosphere layers in cross-section.
- Palette: #05070f #0b1224 #16233d #59c2ff #c86bfa #ffd166 #ff6b6b.
- Type: `"Helvetica Neue", Helvetica, system-ui`; band labels `ui-monospace, Menlo`.
- Layout: left ionosphere cross-section (D/E/F1/F2 layers with ray path), right propagation dial; bottom band table.
- Motion: ray path bends through layers with animated dash; terminator rotates with the time scrubber; layer densities pulse with solar flux.
- Interaction: time-of-day scrubber, solar flux (SFI) and SN sliders that recompute band verdicts (open/marginal/closed), band row click highlights its ray path.
