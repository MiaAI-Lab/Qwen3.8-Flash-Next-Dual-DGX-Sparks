### 041 — One-Line: A Substation Switchboard
- Subject: a medium-voltage one-line diagram with breaker states, metered load flow, and a relay trip event you can trigger.
- Direction: control-room HMI per IEC-ish symbols drawn as SVG; dark panel, lamp fields, no fake skeuomorph chrome.
- Palette: #0b0f12 #141a1f #2f6f4a #d6b134 #c0392b #59a7d6.
- Type: `"Helvetica Neue", Helvetica, system-ui`; tags/readouts `ui-monospace, SFMono-Regular, Menlo`.
- Layout: one-line diagram centre (feeders, bus, transformer, breakers), left lamp annunciator field, right metering table (MW/MVar/kV/Hz).
- Motion: closed path energised with an animated dash flow; opening a breaker stops flow downstream and lights the annunciator; fault injection flashes the trip zone.
- Interaction: click breakers to open/close (with interlock rules — no sourceless energisation), trip test button that runs a relay sequence with a timeline log, restore service.
