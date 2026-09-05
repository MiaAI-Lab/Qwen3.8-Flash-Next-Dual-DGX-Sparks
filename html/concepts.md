# 100-html — master concept list (hand-authored, no generator)

Deliverables go **only** in `100-html/` as `NNN-slug.html` + `NNN-slug.txt`.
This file lives outside the deliverable dir on purpose. Never reuse a "Done" concept.

## Done / taken (do not duplicate the visual language)
001 aurora glass hero · 002 clockwork orrery · 003 brutalist type foundry · 004 neumorphic synth ·
007 Bauhaus kinetic · 008 flow-field loom · 011 luxury chrono watch · 013 volcano cross-section ·
014 pendulum wave lab · 017 holographic cards · 018 papercut mountains · 023 kinetic type scramble ·
027 neon sign shop · 005 ferroflux magnetic cabinet (brief below)

### 005 — Ferroflux: A Magnetic Fluid Cabinet (issued 2026-09-02, post-compaction)
- Subject: ferrofluid demonstration cabinet — Rosensweig normal-field instability, spike formation, coil geometry, colloid composition.
- Direction: pale precise laboratory ground (warm graph paper), NOT a dark neon scene; one black mirror pool is the only dark mass; hairline field lines in a single accent; scientific exhibit panel, not sci-fi.
- Palette: #f1eee7 #e3ded2 #1a1815 #0d0c0b (fluid) + one accent (#4a5fc7 or #b5342a) #8b8578.
- Type: display `Didot, "Bodoni MT", "Hoefler Text", Optima, serif`; body `"Helvetica Neue", Helvetica, system-ui`; readouts `ui-monospace, Menlo`.
- Layout: fluid canvas stage hero; right instrument column (field mT, coil current, gap, composition); three lower diagram plates (surfactant-coated nanoparticle colloid, force balance setting spike spacing, coil cross-section); exhibition caption with real numbers.
- Motion: field ramp must genuinely transition flat mirror → ridges → rising spikes with a travelling specular highlight; own metaball/marching-squares + value noise; reduced-motion renders the settled state only.
- Interaction: field slider 0–400 mT driving computed λ (surface tension vs magnetic stress vs gravity), refusing spikes below onset; coil geometry selector; gap slider; field-pulse button; click a spike → height + local field; hover composition plate → layer labels.

## Diversity ledger (keep these families spread out, never 2 in a row)
risograph/editorial print · cyanotype blueprint · lab instrument HMI · letterpress specimen ·
isometric diorama · dark observatory console · Japanese woodblock · retro-future CRT terminal ·
botanical plate · art-deco poster · swiss rail timetable · watercolor · enameled metal signage ·
arcade/pixel · paper engineering · analog synth rack · meteorological chart · stained glass ·
marble/terrazzo · mid-century travel poster · technical patent drawing · ink blottery ·
kiln/pottery · horology · tapestry/weave · stained timber · printmaking registration

---

### 006 — Halide: A Cyanotype Archive
- Subject: six-page archive of cyanotype (blueprint) plates of marine organisms, each with plate number, exposure log, chemical bath notes.
- Direction: Prussian-blue photogram; paper texture drawn in CSS/SVG; white line-art exposures; faint fog/bleed at plate edges.
- Palette: #08213a #0d3a63 #1d5c92 #7fa8c9 #e7ecef (paper) #b53a2b (archive stamp red).
- Type: headline `"Iowan Old Style", Palatino, Georgia, serif` small-caps; body `Optima, "Avenir Next", "Helvetica Neue", system-ui`; data `ui-monospace, Menlo, Consolas`.
- Layout: archive drawer — sticky left index of plate numbers; right = stacked plates, each an SVG organism drawn in white strokes over a blue wash, with a caption rail.
- Motion: plates "develop" on scroll-into-view: a radial exposure wipe reveals line art, opacity from 0.25→1, plus slow grain drift.
- Interaction: click index → smooth scroll + plate lifts and shows exposure log; a "bath" slider (wash time) that re-renders the wash density; hover plate → magnifier ring following the cursor revealing fine hatch strokes.

### 009 — Tessellate: Escher-style Tiling Studio
- Subject: an interactive regular/semiregular tiling studio with 9 tile families and a live colour scheme swap.
- Direction: mathematical print, black ink on bone paper, one accent; everything hand-built with SVG paths and `<pattern>`.
- Palette: #12100e #f4efe6 #c8452f #2f6f4e #d8a53c.
- Type: display `Futura, "Trebuchet MS", "Helvetica Neue", sans-serif`; labels `ui-monospace, Menlo`.
- Layout: full-bleed tiling canvas as the page background; floating control plinth bottom-left; a symmetry-group table right side; a "fundamental domain" inset diagram.
- Motion: tiles animate between symmetry groups with a staggered per-tile rotate/scale morph; hover a tile → its neighbours ripple tint.
- Interaction: 9 tiling buttons (p3/p4/p6/…), two colour schemes, a "show lattice" toggle drawing the underlying grid, a "scramble phase" that reseeds offsets; keyboard 1–9.

### 010 — Sonar: Submersible Survey Console
- Subject: a sonar sweep console rendering a procedurally generated seabed with detected contacts and a scrolling A-scan.
- Direction: dark navy instrument, phosphor-green traces, scanlines, mechanical bezels; absolutely no glassmorphism.
- Palette: #04090e #08161f #0e2a33 #38f0b0 #f3f7f5 #ffb703 (caution).
- Type: labels `"Helvetica Neue", Helvetica, system-ui` uppercase tight; readouts `ui-monospace, SFMono-Regular, Menlo`.
- Layout: 3-column console — B-scope PPI canvas (left, square), seabed profile canvas (right), bottom strip of A-scan + contact table with real nomenclature.
- Motion: rotating sweep with afterglow falloff (canvas trails), contacts bloom as the beam passes, A-scan scrolls right-to-left, depth counter ticks.
- Interaction: range knob (200/500/1000 m) rescales, gain slider changes noise floor, clicking a contact row slews the beam and annotates the PPI, ping button fires a manual pulse.

### 012 — Addendum: A Letterpress Specimen Broadsheet
- Subject: a type-specimen broadsheet for a fictional foundry face, showing glyph sets, pairings, and a printed-sheet mock.
- Direction: hot-metal letterpress; deep ink bite, slight misregistration, visible paper tooth; rules and ornaments as real typographic furniture.
- Palette: #1b1712 #f6f1e4 #b3462f #1f4d40 #8a7a5c.
- Type: hero `Baskerville, "Hoefler Text", Georgia, serif` at fluid clamp sizes; secondary `"Arial Black", Futura, Impact` for contrast blocks; body `Georgia, serif`.
- Layout: broadsheet grid with rules; giant specimen letter bleeding off the grid; columns of body copy at real sizes; a glyph matrix table.
- Motion: "ink bite" toggle animates press impression (scale + shadow + texture shift); the hero letter cycles A→Z on a timer; misregistration slider offsets channel overlays.
- Interaction: type-size stepper that live-resizes a paragraph, a contenteditable specimen line, glyph matrix click → inserts that glyph into the specimen line.

### 015 — Stratified: Isometric Geology Diorama
- Subject: an exploded isometric block diagram of sedimentary strata with extractable layers and a well-log column.
- Direction: technical isometric illustration, flat ink + halftone, no photos; SVG transform matrix for iso projection.
- Palette: #101418 #f2ede3 #c96f4a #d8b26c #6f8f7a #4f6d8c #8d6f9c.
- Type: `Futura, "Trebuchet MS", sans-serif` labels; `ui-monospace, Menlo` depths.
- Layout: central exploded SVG diorama (draggable explode slider), right well-log strip with lithology symbols, bottom glossary.
- Motion: layers separate along the explode axis with staggered easing; a slow "deposition" build animates strata arriving in order.
- Interaction: drag/explode slider, click a layer → highlights, shows age/thickness/permeability, dims others; well-log hover cross-links to the layer.

### 016 — Nocturne: Observatory Night Assistant
- Subject: a dark observing-planner for 14 deep-sky objects, with altitude-over-time curves and moon-phase penalty.
- Direction: dimmed red-light astronomy aesthetic; SVG line charts; no bright whites (a dark-page page must still be legible, not a black void).
- Palette: #06080d #0d1220 #151d2c #e25555 (dim red) #cfd8e3 #f0c674 (amber).
- Type: labels `Optima, "Avenir Next", system-ui`; numerals `ui-monospace, Menlo`.
- Layout: header with site coordinates + local sidereal clock; main SVG altitude chart (multiple curves); left selectable object list; bottom detail card.
- Motion: time scrubber animates the "now" line; curves illuminate progressively on load; twinkle on the star list markers.
- Interaction: click object to toggle its curve; scrub time (drag on chart) → altitudes update, transit/azimuth recompute; moon-phase slider dims targets; keyboard arrows step 15 min.

### 019 — Kilnwork: A Pottery Studio Ledger
- Subject: glaze recipes, cone charts, and a firing schedule for a wood-fired studio, as a hand-kept ledger.
- Direction: warm paper, ruled ledger grid, handwritten-ish annotations via CSS transforms, swatch tiles computed from oxide percentages.
- Palette: #f3ece1 #2a2118 #a0522d #4f6b5b #b98a3c #e8dcc8.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; numerals `ui-monospace, Menlo`.
- Layout: two-page spread; left = recipe cards with ingredient tables; right = glaze swatch grid (procedurally shaded) + firing schedule step chart.
- Motion: a firing-schedule chart draws itself (stroke-dashoffset); swatch tiles cross-fade as atmosphere slider changes oxidation→reduction.
- Interaction: click a recipe → its swatches highlight and the schedule loads; atmosphere + cone sliders recompute swatch colours; ingredient percentage edits re-tint the swatch live.

### 020 — Meridian: Rail Timetable Wall Board
- Subject: a split-flap style departure board for 12 fictional lines, plus a journey planner.
- Direction: enamel signage, black board with white/amber flaps, mechanical precision, subtle bevels via gradients only.
- Palette: #0b0c0e #16181c #f2f2f0 #ffb020 #4ea3e0 #d1495b.
- Type: `"Helvetica Neue", Helvetica, system-ui` uppercase; flap glyphs monospaced `ui-monospace, Menlo`.
- Layout: full-width board grid (rows of flaps), a planner strip below, a platform map column.
- Motion: split-flap flip cascade on load and on refresh (CSS rotateX per cell, staggered); rows pulse when a service is delayed.
- Interaction: filter chips (line/destination) re-flip only matching rows; planner selects origin→destination and shows next three departures; click a row → detail drawer with stops.

### 021 — Chroma Test: A Broadcast Colour Bar Lab
- Subject: a lab explainer of video test patterns (SMPTE bars, EBU fan, convergence, grey ramps) with live generated signals.
- Direction: engineering-room schematic; dark bezel framing around bright generated canvases; no nostalgic scanline slop — treat it as calibration equipment.
- Palette: #0c0d10 #1a1c21 #ffffff #ff0000 #00ff00 #0000ff #ffd400 #00d8d8 #ff00ff.
- Type: `"Helvetica Neue", Helvetica, system-ui`; instrument readouts `ui-monospace, Menlo`.
- Layout: canvas rack of generated patterns, each with measured values; right column theory notes; bottom vectorscope.
- Motion: live 100 Hz redraw of the vectorscope from the bar signal; a "fault" toggle injects artefacts (crush, chroma shift, moiré).
- Interaction: pattern selector, levels/tilt sliders that truly rescale pixel values and update readouts, fault injector, gamut toggle 709/P3.

### 022 — Apiary: A Hive Population Model
- Subject: a seasonal honeybee colony simulation with varroa pressure and forage availability, visualised as a brood-nest heatmap.
- Direction: naturalist's field guide meets data instrument; honeycomb geometry drives the layout; bees drawn as SVG sprites.
- Palette: #f6f0e2 #2b241a #d99b16 #7b4a12 #3f6b4f #b1362f.
- Type: `"Iowan Old Style", Palatino, Georgia, serif` headers; `ui-monospace, Menlo` data.
- Layout: hex comb canvas (centre, hex grid of cells: brood/pollen/honey/empty), right rail of 4 sparkline series, bottom calendar scrubber.
- Motion: comb cells update per simulated day with staggered colour transitions; forager sprites fly procedural paths between comb and a flower strip.
- Interaction: play/pause/step day, sliders for forage, varroa, queen age; clicking a cell shows its contents and age; presets (strong/waning swarm).

### 024 — Typeface Anatomy Poster: The Caslon Leg
- Subject: an anatomy diagram of serif letterforms — terminals, brackets, apertures, counters — annotated like a medical plate.
- Direction: engraving-style plate; hairline leader lines; giant letterform drawn as SVG outlines with numbered callouts.
- Palette: #f7f3ea #16130f #9c3b22 #34504a #a89a7d.
- Type: `Baskerville, "Hoefler Text", Georgia, serif` display; labels small caps serif; callouts `ui-monospace, Menlo`.
- Layout: full-bleed plate; SVG letterform ("R" and "g") centre, 16 numbered leader lines to glossary text in the margins; bottom comparison row of 4 serifs.
- Motion: leader lines draw in sequentially (stroke-dashoffset); the numbered glossary items fade in matching the line.
- Interaction: hover a glossary term → its leader line and anatomical region highlight; a "grid/skeleton/ink" toggle revealing outline vs skeleton vs metrics; letter selector R/g/a/Q.

### 025 — Seismograph: Reading the Earthquake
- Subject: three live-ish seismogram channels, a magnitude scale explainer, and a hypocentre locator from P/S delay.
- Direction: scientific plotter output on graph paper; ink lines with slight pen wobble; station icons in a small network map.
- Palette: #f2f4ef #10151b #c0392b #1f6f6b #d6a419 #6c7a86.
- Type: `"Helvetica Neue", Helvetica, system-ui`; time labels `ui-monospace, Menlo`.
- Layout: station map top-left, three stacked channel canvases (procedural waveforms), right column magnitude ladder; bottom locator worksheet.
- Motion: continuous waveform scroll with event injections; a picked-event marker propagates to all channels; magnitude ladder fills.
- Interaction: click to pick P and S on any channel → distance circles redraw and hypocentre triangulates; station toggles; a synthetic event generator button; magnitude slider scales amplitudes.

### 026 — Foldout: Paper Engineering Poster
- Subject: an explainer of map folds, kirigami and pop-up mechanisms, with CSS 3D folds you operate.
- Direction: clean studio photography-free mock; pure CSS 3D transforms, paper shadows, cut-line overlays.
- Palette: #ffffff #ece7dd #1a1a1a #e2574c #2e6f9e #e8b13a.
- Type: `Futura, "Trebuchet MS", sans-serif`; spec table `ui-monospace, Menlo`.
- Layout: hero stage with a 3-panel folded map in CSS 3D; three mechanism demos below (pop-up, vortex, parallel fold); spec table.
- Motion: fold opens on load with perspective easing; crease highlights animate; hover a mechanism → it articulates.
- Interaction: drag slider drives fold angle live; click a panel → focus and zoom; a "show crease pattern" toggle overlays mountain/valley lines (mountain red, valley blue).

### 028 — Signal Garden: A Radio Propagation Map
- Subject: HF skywave propagation across 6 bands with an MUF dial, day/night terminator, and band-condition table.
- Direction: cartographic night side; SVG world-ish grid (stylised, not real geo), glowing ionosphere layers in cross-section.
- Palette: #05070f #0b1224 #16233d #59c2ff #c86bfa #ffd166 #ff6b6b.
- Type: `"Helvetica Neue", Helvetica, system-ui`; band labels `ui-monospace, Menlo`.
- Layout: left ionosphere cross-section (D/E/F1/F2 layers with ray path), right propagation dial; bottom band table.
- Motion: ray path bends through layers with animated dash; terminator rotates with the time scrubber; layer densities pulse with solar flux.
- Interaction: time-of-day scrubber, solar flux (SFI) and SN sliders that recompute band verdicts (open/marginal/closed), band row click highlights its ray path.

### 029 — Millwork: A Joinery Cabinet Card
- Subject: a woodworker's cut list and joint explainer for a small walnut cabinet, with an exploded view and tolerances.
- Direction: shop-drawing on tan card stock; dimension lines with arrows; wood grain drawn with SVG turbulence-free stroked paths.
- Palette: #efe6d6 #241c13 #8b5a2b #5e3a1d #2f5d50 #b3341f.
- Type: `"Iowan Old Style", Palatino, Georgia, serif` headers; dimensions `ui-monospace, Menlo`.
- Layout: hero exploded SVG assembly (draggable explode), cut-list table with live totals, joint detail plates row.
- Motion: explode/assemble animates parts to their positions; grain strokes slowly shift like a planing pass.
- Interaction: explode slider, part click → highlights cut list row and shows grain direction + tolerance, unit toggle mm/inch recomputes the whole list, "optimize" reorders the cut list by length.

### 030 — Halogen: A Slide Projector Slide Library
- Subject: a carousel of 35mm slides (drawn, not photographed) about a fictional expedition, with projector controls and a slide-log.
- Direction: dark room, projected frame with keystone and light-falloff; slide mounts as drawn cards; dust specks.
- Palette: #0a0a0c #17161a #f3efe6 #d9743f #4b7f9e #cbb26a.
- Type: `"Iowan Old Style", Palatino, Georgia, serif` captions; log `ui-monospace, Menlo`.
- Layout: projected screen area (large) with a drawn scene per slide; below: slide tray strip of mounts; right log panel with frame notes and exposure data.
- Motion: tray advance rotates the carousel (CSS 3D) and the frame flickers with a per-frame random gain; dust specks drift in the beam.
- Interaction: advance/rewind, click a mount to jump, brightness control changes the projected gain, focus knob applies a live blur; keyboard ←/→; auto-show toggle.

### 031 — Ferment: A Kombucha & Kraut Log
- Subject: a fermentation tracker with pH curves, brine percentage calculator and a jar-by-jar shelf.
- Direction: home-lab notebook; hand-ruled charts; jar vessels drawn in SVG with liquid fills and airlocks bubbling.
- Palette: #f4eee3 #23201b #7b2d3a #3f6f4a #c98a2e #5e7fa3.
- Type: `Georgia, "Iowan Old Style", serif`; numbers `ui-monospace, Menlo`.
- Layout: jar shelf (SVG jar cards) top; pH curve chart (SVG) with sampled points; brine calculator card; notes column.
- Motion: airlock bubbles at a rate tied to activity; liquid fill level and colour age over simulated days; curve draws in.
- Interaction: brine % calculator (weight + salt → %, warns out of range), select jar → chart and notes update, add reading form appends a point, day scrubber ages all jars.

### 032 — Foundry Floor: A Type Casting Machine Diagram
- Subject: a hot-metal typesetting machine explained by numbered subsystem, with a working cycle animation.
- Direction: patent-drawing line art on aged paper; hatched section fills; brass accents; exploded callouts.
- Palette: #efe8d9 #1c1a16 #96712c #7d3a2c #3c5a52.
- Type: `Baskerville, "Hoefler Text", Georgia, serif` + `ui-monospace, Menlo` numbers.
- Layout: wide SVG machine elevation as hero, numbered parts legend in two columns, cycle timeline strip at bottom, matrix detail inset.
- Motion: the casting cycle animates (matrix advance → mould close → inject → trim) with a moving highlight along the timeline; a counter of characters/hour ticks.
- Interaction: step/next cycle buttons, speed control, hover legend item → part glows + tooltip; part click pins the callout; a "section view" toggle swaps in the hatched cross-section.

---

### 033 — Loomcard: A Jacquard Weave Studio
- Subject: punch-card driven weave designer — peg plan, lifting diagram, and a woven cloth preview from warp/weft colours.
- Direction: industrial textile archive; cardboard cards, thread sheen rendered with repeating-linear-gradients, no images.
- Palette: #efe7d8 #221d17 #7d3b4b #2f5f5a #c9922f #6b7f9c.
- Type: `Futura, "Trebuchet MS", sans-serif`; card numbers `ui-monospace, Menlo`.
- Layout: left peg-plan grid (clickable cells), centre cloth preview canvas, right lift diagram + thread inventory table.
- Motion: shuttle simulation animates weft passes across the cloth, shed opens (warp threads alternate lift) in sync.
- Interaction: paint the peg plan, choose weave structure (plain/twill/satin/herringbone) which regenerates cards, swap warp/weft palettes, thread counter recomputes yardage.

### 034 — Vellum: An Illuminated Manuscript Shop
- Subject: a scriptorium page builder — rubrication, drop cap, gold leaf, marginalia vines — with a commission order form.
- Direction: medieval folio; parchment mottling in CSS, real illuminated initial built from SVG, gold via layered gradients (not flat yellow).
- Palette: #e9dcc0 #2b2318 #8e2f22 #1f3f6b #b08d2e #3c5a3a.
- Type: `Baskerville, "Hoefler Text", Georgia, serif`; rubrics in small caps; annotations `"Iowan Old Style", Palatino, serif`.
- Layout: folio spread with ruled prickings; left margin annotations; right column liturgy-style text with drop cap; bottom commission table.
- Motion: gold leaf "lays down" with a wipe and specular sweep; vine marginalia draw themselves (stroke-dashoffset); ink appears as if quill-written (clip reveal).
- Interaction: choose drop-cap letter, ink colour, ruling density, and border style; text area is contenteditable and re-flows the initial; hover annotation → translation tooltip.

### 035 — Anemography: A Wind Station Field Log
- Subject: 12 months of wind observations as a wind rose, Beaufort scale cards, and a prevailing-direction almanac.
- Direction: polar scientific plot on gridded field-notebook paper; hand-inked look; instrument bezel for the anemometer readout.
- Palette: #f1ece1 #141a1e #1f6f8b #d9822b #4b7f5b #b1362f.
- Type: `"Helvetica Neue", Helvetica, system-ui`; readings `ui-monospace, Menlo`.
- Layout: large SVG wind rose (16 sectors × 6 speed rings) left; Beaufort ladder (0–12 with sea-state descriptions) right; month strip below.
- Motion: sectors grow from the centre on load; a compass needle drifts; selected month re-animates the rose.
- Interaction: month selector, speed-range filter that dims rings, hover sector → count/frequency/bearing readout, Beaufort row click → highlights matching sectors.

### 036 — Contour: A Cartographer's Drafting Table
- Subject: a hand-drafted topographic sheet with contours, hypsometric tints, spot heights, a graticule and scale bar.
- Direction: survey sheet; contours as procedural SVG polylines from a value field with marching squares, hairline pen weights.
- Palette: #f4f1e6 #1b1f1c #6d8f5e #c9a04a #4a7ea1 #8d5a3c.
- Type: `Optima, "Avenir Next", system-ui` (cartographic); coordinates `ui-monospace, Menlo`.
- Layout: full-bleed map sheet; north arrow + scale bar + legend box; margin column with projection notes; index to spot heights.
- Motion: contours draw outward in elevation order (staggered dash reveal); a relief-hillshade layer fades in computed per-pixel on canvas.
- Interaction: contour interval stepper, tint/hillshade/labels toggles, dragger that re-seeds the terrain, click anywhere → elevation + pseudo-coordinate readout.

### 037 — Mise: A Recipe Set in a Swiss Grid
- Subject: a single recipe (lamb braise) composed as a rigorous editorial grid with a step timeline and unit converter.
- Direction: Helvetica-grid editorial; asymmetric blocks, strict baseline rhythm, one photographic-free hero drawn as a vector pot cutaway.
- Palette: #ffffff #111111 #d6301f #f2c94c #0f4c5c #ececec.
- Type: `"Helvetica Neue", Helvetica, Arial, sans-serif` (weights only, no serif); numerals tabular `ui-monospace, Menlo`.
- Layout: 12-column grid; giant dish name; ingredient table with quantities; timeline of 7 steps with durations as bars; technique notes column.
- Motion: timeline bars fill in sequence; ingredient rows stagger-highlight as the active step changes; a simmer animation inside the pot cutaway.
- Interaction: servings scaler that recomputes every quantity (real math, rounded sensibly), unit toggle g/oz/ml/fl-oz, step player with next/prev driving the highlight.

### 038 — Enamelworks: A Signage Compositor
- Subject: a wall of vitreous enamel signs (wayfinding, hazard, info) you retype and recolour, plus an ISO pictogram set drawn in SVG.
- Direction: industrial signage realism — porcelain sheen via gradients, steel rivets, bevel highlights, no photography.
- Palette: #0f1214 #e8e6e0 #d1341f #1f6f4e #1f4f8b #e8b13a.
- Type: `"Helvetica Neue", Helvetica, system-ui` uppercase (pictogram-adjacent); spec text `ui-monospace, Menlo`.
- Layout: brick-free wall grid of 8 sign panels in mixed sizes; a compositor control bar; bottom spec table (dimensions, colour codes, fixings).
- Motion: specular sweep travels across enamel on hover; rivets catch light with a staggered glint; hazard chevrons animate their dash offset.
- Interaction: edit sign text live, swap pictogram, pick standard colour codes (updates spec table), rotate a sign to show its reverse side with mounting brackets.

### 039 — Lichen: A Slow Colony Field Guide
- Subject: 6 lichen species as radial growth simulations on rock, with a dichotomous key and substrate notes.
- Direction: natural-history plate on stone grey; procedural radial lobes drawn on canvas; specimen labels with real binomials.
- Palette: #e8e6df #1e211f #7f8f5a #b8a24a #6b8c9c #8a5a6b.
- Type: `"Iowan Old Style", Palatino, Georgia, serif` italics for binomials; data `ui-monospace, Menlo`.
- Layout: 6 specimen plates in a staggered grid; left dichotomous key panel; bottom substrate comparison strip (bark/stone/soil).
- Motion: each colony grows over simulated years — lobes advance, apothecia dots appear; a time-lapse year counter per plate.
- Interaction: year scrubber per plate, key answers filter to the matching species (highlight its plate), growth-rate slider, hover → thallus/cortex detail callout.

### 040 — Registration: A Four-Colour Risograph Bench
- Subject: a poster built from 4 spot-colour layers with halftone dots, misregistration drift, and ink coverage notes.
- Direction: riso print bench; each layer a separate canvas/screen; paper stock texture; burn/multiply blending real (mix-blend-mode).
- Palette: #f4efe4 #1b1b1e #d64f2a #1f7a8c #e8b13a #7b4fa0.
- Type: `Impact, "Arial Black", Futura, sans-serif` display; bench notes `ui-monospace, Menlo`.
- Layout: poster stage (blend of 4 layers) left; right stack of 4 separations thumbnails; bottom ink/coverage table and paper picker.
- Motion: layers peel apart into an exploded 3D stack and re-register; per-layer halftone dots re-scale when the line-screen changes.
- Interaction: per-layer offset sliders (X/Y/rotate) that produce real misregistration, screen frequency (45/65/85 lpi), overprint mode multiply vs normal, swap paper stock.

### 041 — One-Line: A Substation Switchboard
- Subject: a medium-voltage one-line diagram with breaker states, metered load flow, and a relay trip event you can trigger.
- Direction: control-room HMI per IEC-ish symbols drawn as SVG; dark panel, lamp fields, no fake skeuomorph chrome.
- Palette: #0b0f12 #141a1f #2f6f4a #d6b134 #c0392b #59a7d6.
- Type: `"Helvetica Neue", Helvetica, system-ui`; tags/readouts `ui-monospace, SFMono-Regular, Menlo`.
- Layout: one-line diagram centre (feeders, bus, transformer, breakers), left lamp annunciator field, right metering table (MW/MVar/kV/Hz).
- Motion: closed path energised with an animated dash flow; opening a breaker stops flow downstream and lights the annunciator; fault injection flashes the trip zone.
- Interaction: click breakers to open/close (with interlock rules — no sourceless energisation), trip test button that runs a relay sequence with a timeline log, restore service.

### 042 — Species: A Counterpoint Workshop
- Subject: five species of strict counterpoint with a live rules checker, showing parallel fifths and dissonance on the downbeat.
- Direction: engraved music plate; SVG staff, noteheads, stems; error annotations in red ink like a professor's marks.
- Palette: #f6f2e8 #17140f #b3341f #2f5d50 #8a7a5c.
- Type: `Baskerville, "Hoefler Text", Georgia, serif` for text; interval labels `ui-monospace, Menlo`.
- Layout: two-staff engraving stage; interval analysis strip under the staff; species selector rail; rules reference card.
- Motion: playback cursor sweeps the staff, sounding noteheads fill; violations ring and their interval cell reddens.
- Interaction: click a note to change its pitch (drag/stepper) → rules checker re-evaluates live; species switch loads a new exercise; note-audible toggle (uses WebAudio oscillator, self-contained).

### 043 — Herbarium: Pressed Specimen Sheets
- Subject: 8 herbarium sheets of wild plants, each with a hand-drawn SVG plate, collection label, locality and phenology.
- Direction: archival specimen sheet; translucent mounting strips; typewritten labels; faint foxing on paper.
- Palette: #eee8d9 #1f1c16 #4f6b45 #7c6a3f #8a4b3b #5b7f96.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; label fields `ui-monospace, Menlo`.
- Layout: wall grid of sheets; selected sheet enlarges to a detail view with a locality micro-map (SVG) and a phenology bar (Jan–Dec).
- Motion: specimen fades/depresses into the sheet (scale + shadow settle); detail view zooms with a cross-fade; herbarium barcode digits tick.
- Interaction: filter by family/flower colour/height class (dimming non-matching sheets), sheet click → detail, hover a label field → highlight, search box filtering deterministically.

### 044 — Snowpit: An Avalanche Layer Profile
- Subject: a ski-slope snowpit with layer crystals, hardness column, stability score, and a weather history strip.
- Direction: field-glaciology chart; layered column with crystal glyphs drawn per layer; cold, clean, high-legibility.
- Palette: #eef2f4 #101a20 #2f6f8b #7f9db3 #c9552e #e8b13a.
- Type: `"Helvetica Neue", Helvetica, system-ui`; depths `ui-monospace, Menlo`.
- Layout: left depth column (0–160 cm) with layer bands + crystal SVG; middle hardness/temperature profiles; right test results (ECT) and hazard rose.
- Motion: profile builds from the surface down; a propagation test animation crack-steps across the column.
- Interaction: select a different pit (3 presets: wind slab / surface hoar / wet loose), hover a layer → grain type, size, hardness and bond; toggle terrain aspect overlay.

### 045 — Graphite: A Pencil Grade Bench
- Subject: the 9H→9B scale as tonal swatches, a value ramp, hatching/stippling demos, and a smudge comparison.
- Direction: pure graphite greys on toothed paper; every mark generated by stroked SVG/canvas paths, no images.
- Palette: #f2efe8 #14151a #2b2e35 #4a4e58 #7a7e88 #b4553a (one warm accent for annotations).
- Type: `Optima, "Avenir Next", "Helvetica Neue", sans-serif`; grades `ui-monospace, Menlo`.
- Layout: hero value ramp full width; 18 grade swatch cards; technique plates (hatch, cross-hatch, stipple, blending stump) below; paper tooth comparator.
- Motion: strokes draw themselves progressively (path reveal) so each plate looks "drawn"; swatch density builds up as you hover.
- Interaction: pick a grade → the live sketch pad uses that density; a pressure slider changes stroke alpha/count; export-free "draw" pad on canvas with pointer events.

### 046 — T-Minus: A Launch Window Board
- Subject: an orbital launch countdown board with staging diagram, telemetry tiles, and a hold/go criteria matrix.
- Direction: range-control board; heavy black panel, amber/white flap text is reserved for 020 — here use backlit LCD segments and a printed procedure sheet.
- Palette: #0a0d10 #161b20 #ff9f1c #e6ebf0 #2f9e6b #d64545.
- Type: `"Helvetica Neue", Helvetica, system-ui`; segment readouts `ui-monospace, SFMono-Regular, Menlo`.
- Layout: countdown hero with segmented digits; vehicle staging SVG (3 stages + fairing) left; telemetry tile grid right; GO/NO-GO matrix bottom.
- Motion: countdown ticks in real time with per-digit flicker; staging events highlight the relevant vehicle section; telemetry values jitter within real tolerances.
- Interaction: start/hold/resume countdown, scrub T-0 events, toggle a nominal/anomaly simulation that trips a NO-GO with an explanation, retry a failed check.

### 047 — Levain: A Bakehouse Formula Sheet
- Subject: baker's-percentages calculator, bulk fermentation timeline, and a crumb structure diagram for a country loaf.
- Direction: bakery formula card; kraft-and-flour palette, butcher-paper texture, tabular rigour with real arithmetic.
- Palette: #efe4cf #241b12 #a5591f #6b4423 #4f6b4a #c9a227.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; formulas `ui-monospace, Menlo`.
- Layout: formula table (flours/water/salt/starter) with live percentages; timeline of folds/bulk/bench/proof as a Gantt; crumb SVG cross-section; oven notes.
- Motion: timeline bars fill with elapsed time; starter levies with a bubbly SVG animation; crumb voids grow subtly.
- Interaction: change flour weight/hydration/starter % → all weights and percentages recompute (real baker's math, including dough temperature target); preset styles (80%/85%/90% hydration).

### 048 — Palimpsest: A Multispectral Reader
- Subject: an erased medieval text under a later writing; a wavelength slider reveals the undertext layer by layer.
- Direction: dark conservation lab; two text layers drawn in SVG (upper script, faded undertext), rendered with blend modes per band.
- Palette: #0d0c0a #1c1a16 #c8b892 #8b6b3f #4aa3c7 #b45c8f #d2d84a.
- Type: `Baskerville, "Hoefler Text", Georgia, serif` for both scripts; band labels `ui-monospace, Menlo`.
- Layout: folio stage centre with band selector ring (UV/vis/blue/green/IR); left band explanation notes; right transcription column that fills as undertext becomes legible.
- Motion: wavelength change cross-fades the blend; reveal uses a moving scan band; transcription words appear as their glyphs gain contrast.
- Interaction: continuous wavelength slider + 6 preset bands, contrast/gamma controls that actually change filter parameters, toggle raking light, click a word → zoom + transcription entry.

### 049 — Develop: A Raw Image Bench
- Subject: a photo develop bench — histogram, tone curves, zones, exposure latitude — operating on a synthesized scene drawn in SVG/canvas (no external images).
- Direction: darkroom-grey bench UI; the "image" is a procedurally rendered landscape so the whole page stays self-contained.
- Palette: #12141a #1c2027 #e8eaee #ff9f1c #3ba7ff #6fd39a #ff5d5d.
- Type: `"Helvetica Neue", Helvetica, system-ui`; numeric readouts `ui-monospace, Menlo`.
- Layout: canvas viewport left with before/after split handle; right controls (exposure, contrast, blacks, whites, clarity, split tone); bottom histogram with zone overlay.
- Motion: histogram recomputes live during drags; a split-sweep animation shows before/after; clipping warnings pulse.
- Interaction: real per-pixel tone mapping (LUT built from a draggable curve), draggable curve with control points, presets, exposure simulation, histogram hover shows RGB counts.

### 050 — Patch Panel: A Telephone Exchange Floor
- Subject: a 1940s operator switchboard with lamp field, patch cords you plug, and a call log with call detail.
- Direction: ebonite panel, brass jacks, glow lamps; cords as real SVG bezier curves with physics-ish sag.
- Palette: #0f0e0c #1c1a16 #c9a227 #d96f2b #c0392b #6f8fa8 #eae2d2.
- Type: `"Iowan Old Style", Palatino, Georgia, serif` numerals; log `ui-monospace, Menlo`.
- Layout: big switchboard SVG (rows of jacks + lamps + keys), cord reel shelf, operator call log table, trunk route map.
- Motion: incoming calls light lamps with a ringing flicker; plugging a cord animates its path; disconnect retracts the cord with easing.
- Interaction: drag a cord from an incoming jack to an outgoing jack to connect (validates busy/free), ringing queue with accept/clear, call log accumulates durations, night-mode toggle dims the field.

---

### 051 — Gnomon: A Sundial & Analemma Bench
- Subject: build a horizontal sundial for any latitude — hour lines by real trig, plus an analemma loop and a declination table.
- Direction: brass-on-slate instrument plate; engraved hairlines; a drawn shadow that tracks a computed sun position.
- Palette: #eae4d6 #171a1d #a8862f #3f5d6b #8c4a34.
- Type: `Optima, "Avenir Next", system-ui`; hour labels `ui-monospace, Menlo`.
- Layout: dial face SVG hero; latitude dial control; right column: equation-of-time table, declination chart; bottom analemma plot.
- Motion: sun azimuth/elevation animate across a day, gnomon shadow sweeps the hour lines; analemma draws as a continuous loop.
- Interaction: latitude slider redraws every hour line live, date+time scrubber moves the shadow, solstice/equinox preset buttons, toggle true-sun vs mean-sun time.

### 052 — Tessera: A Mosaic Pavement Workshop
- Subject: a Byzantine-style mosaic floor with andamento (flow lines), tessera counts, grout and stone palette tables.
- Direction: stone-and-gold craft board; each tessera a slightly rotated/tilted rect with per-piece colour jitter, drawn on canvas.
- Palette: #efe6d3 #1a1712 #b7934a #2f6f6b #8e3b3b #3d5a8a.
- Type: `"Iowan Old Style", Palatino, Georgia, serif` titles; counts `ui-monospace, Menlo`.
- Layout: mosaic panel hero (border pattern + central emblem); right andamento overlay toggle + materials table; bottom border-pattern library.
- Motion: mosaic lays tessera row by row (progressive draw) with a subtle grout-settle; gold pieces glint with a slow specular pass.
- Interaction: reseed, tessera size slider (updates count and cost), andamento toggle showing flow curves, click a region to lift it and show its cut list.

### 053 — Barograph: A Mercury Pressure Diary
- Subject: 30 days of barometric readings, a mercury column instrument, and falling-pressure storm warnings.
- Direction: Victorian scientific instrument; brass bezel, engraved scale, inked drum trace on gridded paper.
- Palette: #f0ead9 #191c1f #9c7a2c #2f5f7a #a8402f.
- Type: `Baskerville, "Hoefler Text", Georgia, serif` headings; readings `ui-monospace, Menlo`.
- Layout: mercury column SVG left with live meniscus; drum trace chart hero; day log table; right storm-criteria card.
- Motion: mercury rises/falls with easing when a day is selected; drum trace draws chronologically; trend arrow re-decides with hysteresis.
- Interaction: day scrubber, 3-day trend selector, unit toggle hPa/inHg/mmHg recomputing the whole table, hover trace → reading + tendency.

### 054 — Giornate: A Fresco Plaster Planner
- Subject: plan a fresco wall by giornata (plaster days), with intonaco drying time, pigment-on-wet-lime notes, and a scaffold view.
- Direction: lime-plaster surfaces, earth pigments as swatch chips with real mineral names, drawn scaffold poles.
- Palette: #ece4d5 #211d17 #a3563a #4f6b4a #2f4f6b #c9a227.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; schedules `ui-monospace, Menlo`.
- Layout: wall elevation hero split into coloured giornate regions; pigment table with a secco/wet-good flags; scaffold toggle view; drying calendar.
- Motion: plaster regions lay on in sequence with a wet-edge sheen that dries out over the simulated timeline; pigment labels fade in per day.
- Interaction: select a giornata → its area, pigments and deadline; toggle scaffold; a "wet edge" timer that shows which joins are still workable; repaint-region tool assigning a day.

### 055 — Actogram: A Circadian Record
- Subject: double-plotted activity records for 3 conditions (LD, DD, jet-lag), with free-running period fits and light schedule bars.
- Direction: chronobiology plotter; crisp black-on-white raster rows, no decoration; a real scientific instrument look.
- Palette: #f7f5ef #101314 #1f6f8b #c0392b #d6a419 #55606a.
- Type: `"Helvetica Neue", Helvetica, system-ui`; times `ui-monospace, Menlo`.
- Layout: actogram raster (canvas) hero; left condition selector + subject stats; bottom light-schedule strip and period estimate card.
- Motion: rows plot progressively day by day; a regression line fits the onset drift with animated slope; free-run drift ticks.
- Interaction: switch conditions, adjust light-period slider (re-renders schedule and drift), hover a row → date/onset/activity %, toggle rhythm-fit overlay.

### 056 — Preserves: A Canning Safety Bench
- Subject: pH-driven process selection (water bath vs pressure canner), 20 foods with real processing times, and a jar labeller.
- Direction: pantry utility; mason-jar SVGs, hand-lettered labels, safety-critical data set apart visually (never decorative-only emphasis).
- Palette: #f2ebdb #231d15 #a6402c #4f6b4a #c98a2e #5b7f96.
- Type: `"Iowan Old Style", Palatino, Georgia, serif` headings; times/pressures `ui-monospace, Menlo`.
- Layout: food table (pH class, pack style, time, pressure, altitude adjustment); jar labeller card; altitude correction card; method decision flow drawn in SVG.
- Motion: jar fills and seals with a lid-ping animation when a valid process is chosen; the decision flow highlights its path.
- Interaction: pick food + jar size + altitude → recompute time/pressure with real altitude correction rules; labeller writes contents/date/plan; unsafe combos refuse with an explanation.

### 057 — Rig: A Sloop Rigging Plan
- Subject: standing/running rigging inventory, points-of-sail diagram, and load estimates per sailing angle.
- Direction: marine survey drawing; thin line work on off-white, sail plan with dashed foot/leech, load bars in a warm scale.
- Palette: #f1eee6 #14181b #1f4f6b #c0392b #4f6b4a #c9a227.
- Type: `Optima, "Avenir Next", system-ui`; rig dimensions `ui-monospace, Menlo`.
- Layout: side elevation sail plan hero + top-down points-of-sail compass; rig inventory table with wire sizes; load bar chart.
- Motion: boat heels and sails trim as pointing angle changes; load bars ease to new values; a telltale streamer animates.
- Interaction: pointing-angle dial (no-go zones greyed), reef toggle reduces sail area and loads, inventory row hover highlights the wire on the plan, unit toggle kN/kg.

### 058 — Formant: A Vowel Tract Explorer
- Subject: the 8 cardinal vowels as vocal-tract shapes and spectrograms, with formant values and a vowel-space quadrilateral.
- Direction: phonetics lab; spectrogram canvas with inked formant tracks, midsagittal tract drawn as a bezier silhouette.
- Palette: #f4f3ef #121417 #2f6f8b #b8452f #d6a419 #6b5b95.
- Type: `"Helvetica Neue", Helvetica, system-ui`; IPA glyphs serif `Baskerville, Georgia, serif`; values `ui-monospace, Menlo`.
- Layout: vowel quadrilateral hero; per-vowel card with tract SVG + spectrogram; F1×F4 table; articulation notes column.
- Motion: tract morphs between vowels (path interpolation); spectrogram re-renders with formant bands sliding.
- Interaction: draggable articulator dot inside the quadrilateral → tract and spectrogram update continuously; IPA buttons; toggle formant labels/centres.

### 059 — Blazon: A Heraldry Composer
- Subject: compose a coat of arms from a blazon grammar; validate the rule of tincture; explain each ordinary and charge.
- Direction: heraldic scroll; shield SVG with real division lines (per pale/chevron/fess), ermine and vair patterns as SVG `<pattern>`.
- Palette: #efe7d2 #1c1710 #b02a2a #1f4f8b #e2c14e #2f6b4a (metals tincture rule enforced).
- Type: `Baskerville, "Hoefler Text", Georgia, serif`; blazon code `ui-monospace, Menlo`.
- Layout: shield hero + helm/mantling; left field/ordinary/charge pickers; right generated blazon text with glossary tooltips; rule-violation banner.
- Motion: mantling unfurls (path draw); charges arrive with a scale-settle; tincture violation shakes the offending division and strikes it.
- Interaction: build via selects (field → ordinary → charge → tinctures), live blazon string generation in correct grammar, rule-of-tincture validator with plain-English fix, randomise-a-coat button.

### 060 — Lockkeeper: A Canal Lock Desk
- Subject: operate a narrow-lock flight — gates, paddles, water levels, boat passage — with interlocks and a passage log.
- Direction: British Waterways board; enamelled plates, brass windlass, level gauges drawn in SVG with live water fill.
- Palette: #0f1416 #1b2426 #c9a227 #2f6f6b #d1341f #e9e4d6.
- Type: `"Helvetica Neue", Helvetica, system-ui`; levels `ui-monospace, Menlo`.
- Layout: lock cross-section SVG hero (paddles, gates, water, boat); control desk with windlass buttons; flight profile strip (3 locks); event log.
- Motion: water fills/drains with animated surface and bobbing boat; gate swing in 3D-ish rotateY; paddle lift animates and flow ripples appear.
- Interaction: strict interlock state machine (can't open paddles with gates open / can't open gates against head), sequence auto-cycle, flight selector, log accumulates times.

### 061 — Cabinet: A Mineral Collection
- Subject: 12 species with crystal habit (3D wireframe), Mohs hardness test, cleavage planes, and a streak plate.
- Direction: drawer-label mineralogy; glass-front drawer aesthetic; crystals as real 3D wireframes via CSS transforms or projected SVG.
- Palette: #eee9dc #191713 #6b4f8a #2f7a6b #b8863b #4a6b8a.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; data fields `ui-monospace, Menlo`.
- Layout: drawer grid of specimen cards; detail panel with rotating crystal, hardness scratch row (1–10), cleavage diagram, occurrence list.
- Motion: crystal rotates continuously (slow), hardness tester scratches across and reports; specimen card lifts on hover revealing habit name.
- Interaction: rotate crystal by drag, hardness slider → which minerals scratch it, cleavage toggle overlays planes, filter by system/hardness/lustre.

### 062 — Endgame: A Study Table
- Subject: an endgame study viewer with SVG board, move list, tablebase-style eval bars, and key-square highlighter.
- Direction: chess periodical; checkered board in wood-tone flat colours, algebraic notation column, no clip-art pieces (vector pieces only).
- Palette: #f3efe4 #1b1b1b #b0803f #4a4f55 #2f6b4a #b8402f.
- Type: `Baskerville, "Hoefler Text", Georgia, serif`; notation `ui-monospace, Menlo`.
- Layout: board hero left; notation + eval right; study list bottom; key-square legend.
- Motion: pieces glide along legal paths; eval bar eases; mate-in-N line animates as a pulsing path for the mating square.
- Interaction: step through the study (first/next/back/auto), drag a piece to try a move and get "not the study line" feedback, flip board, study selector with 4 studies.

### 063 — Astrolabe: A Planispheric Instrument
- Subject: a working astrolabe — rotate the rete, read altitude/azimuth, tell time by star altitude, with a latitude plate swap.
- Direction: engraved brass on dark leather; hand-hatched limb divisions; almucantar and azimuth grids as real stereographic circles.
- Palette: #e8dfc8 #14110d #b08a34 #3b5f74 #8c4a34.
- Type: `Copperplate, Optima, "Hoefler Text", Georgia, serif` engravings; degrees `ui-monospace, Menlo`.
- Layout: layered astrolabe (mother, plate, rete, rule) as stacked SVG groups; latitude plate picker; star almanac list; reading panel.
- Motion: rete rotates with inertia and settles; a selected star's altitude readout updates continuously; sun pointer tracks the date ring.
- Interaction: drag to rotate rete/rule, latitude select redraws the plate (real stereographic math), click a star → altitude/azimuth/time, date+time entry sets the rule.

### 064 — Mordant: A Natural Dye Lab
- Subject: fibre × dye × pH × exhaust steps producing real fabric swatches; mordant chemistry notes and a recipe card.
- Direction: dyer's swatch book; fabric texture via repeating gradients; stepwise colour blending (subtractive-ish) computed in JS.
- Palette: #f3ece0 #241f1a #7b3f52 #4f6b4a #b8863b #4a5f8a (plus computed dye outputs).
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; recipes `ui-monospace, Menlo`.
- Layout: swatch grid hero (12 fabrics); left dye/vat selector with bath colour; right recipe card + pH ladder; bottom exhaust sequence strip.
- Motion: swatches dip into the bath and take colour progressively (multiple dips deepen); pH change retints with a cross-fade.
- Interaction: choose dye (madder/weld/indigo/cochineal/oak), fibre (wool/cotton/silk/linen with uptake factors), pH slider, dip count → swatch colours recompute; save recipe card.

### 065 — Signature: A Bookbinding Structure
- Subject: sew-on-tape structure — signatures, sewing stations, kettle stitches, boards and spine — with a working sewing animation.
- Direction: binder's bench drawing; cut-paper sheets, linen thread, bone folder; exploded structure diagram.
- Palette: #efe8d9 #201b16 #8a4a34 #2f5d50 #b8963b #6b7a8a.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; specs `ui-monospace, Menlo`.
- Layout: exploded spine diagram hero; sewing station diagram with numbered stations; materials table (thread, tape, board); tools list.
- Motion: thread path sews through stations in sequence (dash animation along a real path); boards and spine assemble on a toggle.
- Interaction: explode/assemble slider, station click → explains that stitch, signature-count stepper recalculates spine thickness (real taper math), board material select updates the spec table.

### 066 — Carillon: A Bell Foundry Chime
- Subject: 8 bells with real pitch/weight/diameter relationships, a strike-pattern sequencer, and change-ringing notation.
- Direction: bell-metal browns with engraved lettering; tower cross-section with the frame; sound as WebAudio (self-contained).
- Palette: #e9e2d2 #17140f #8a6a2f #4f5f6b #a8402f.
- Type: `Copperplate, Baskerville, "Hoefler Text", serif`; counts `ui-monospace, Menlo`.
- Layout: bell row hero (scaled SVG bells sized by real diameter), sequencer grid (steps × bells), tower section card, change-ringing table.
- Motion: struck bells swing and their clapper strikes; the sequencer playhead steps; sound rings with a decaying partial stack.
- Interaction: program a pattern by clicking the grid, tempo control, presets (Westminster, changes on 8, tingsha), toggle sound, hover bell → hum/nominal partial frequencies.

### 067 — Reaction: A Morphogenesis Studio
- Subject: Gray-Scott reaction-diffusion producing animal-coat patterns (spots, stripes, loops, coral) with parameter maps.
- Direction: scientific generative board; the pattern is the hero; a small parameter-space map showing the region you're in.
- Palette: #12151a #1d222a #e8eaee #4fd1c5 #f6ad55 #b794f4.
- Type: `"Helvetica Neue", Helvetica, system-ui`; params `ui-monospace, SFMono-Regular, Menlo`.
- Layout: full-bleed simulation canvas; floating parameter console; presets row with thumbnails (drawn, not images); bifurcation map inset.
- Motion: continuous simulation stepping; preset morph animates feed/kill rates along a path so patterns transform into each other.
- Interaction: feed/kill/diffusion sliders with named regions, preset morph, brush to seed chemical, pause/step, resolution toggle, reset with a click-drag seed.

### 068 — Core: A Tree-Ring Laboratory
- Subject: a cross-dated tree-core sequence with ring widths, fire scars, a master chronology and a floating-sequence match.
- Direction: lab bench on white; core samples as thin SVG strips with early/latewood banding; a pointer-year marker.
- Palette: #f4f1e8 #1a1713 #8a5a2b #4f6b4a #2f6f8b #b8402f.
- Type: `"Helvetica Neue", Helvetica, system-ui`; years `ui-monospace, Menlo`.
- Layout: core strip hero (two cores, one floating); ring-width series chart below; master chronology overlay; match score card.
- Motion: cores slide horizontally as you cross-date; matching years lock with a click and a snap highlight; the pointer years flash.
- Interaction: drag the floating core to align, auto-match button computing correlation across offsets, click a scar → year and cause, zoom ring ruler.

### 069 — Truss: A Bridge Load Bench
- Subject: a Warren/Pratt truss with method-of-joints member forces, live deflection shape, and a load mover.
- Direction: structural calc sheet; member diagram with tension/compression in two colours, force table, influence-line mini-charts.
- Palette: #f1f2ee #14181c #2f6f8b #c0392b #4f6b4a #c9a227.
- Type: `"Helvetica Neue", Helvetica, system-ui`; forces `ui-monospace, Menlo`.
- Layout: truss SVG hero with supports and a movable load; member force table; deflection plot; utilisation bars.
- Motion: forces recolour/scale with load position; deflection shape animates smoothly; failing member flashes and the analysis refuses past capacity.
- Interaction: drag the load along the deck, switch truss type, add/remove panel points, toggle zero-force members, unit toggle kN/kip, view reaction readouts.

### 070 — Terrarium: A Closed System Jar
- Subject: a sealed jar ecosystem — water cycle, O2/CO2 exchange, condensation — modelled over months with species balance.
- Direction: glass-and-botanical; condensation droplets as SVG, plant silhouettes, gas exchange as a labelled cycle diagram.
- Palette: #eef2ec #16211a #3f6b4a #7fa86b #4a7ea1 #c9a227.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; series `ui-monospace, Menlo`.
- Layout: jar hero (SVG with layers: pebbles/charcoal/soil/plants/glass), cycle diagram right, 4 time-series small charts, species picker.
- Motion: water evaporates, condenses and drips on a loop; plants grow subtly; gas curves undulate with the light cycle.
- Interaction: light-level slider, species mix chooser affecting transpiration, day scrubber, seal/break-seal toggle, warnings when the balance collapses.

### 071 — Spine: An Arrow Building Bench
- Subject: shaft spine matching (static vs dynamic), point weight effects, fletching choices, and a build spec sheet.
- Direction: bowyer's bench; wood and carbon textures as gradients; arrows drawn at true scale; deflection boards illustrated.
- Palette: #efe6d4 #201a13 #8a5a2b #2f5d50 #a8402f #b8963b.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; specs `ui-monospace, Menlo`.
- Layout: arrow assembly hero with callouts (point, shaft, nock, feather), spine calculator card, deflection board SVG, spec sheet output.
- Motion: arrow flexes (a sine-bend path animation) at the computed frequency when "loosed"; fletching helical spins on hover.
- Interaction: draw weight / draw length / point weight / bow type inputs → recommended spine and point weight with real lookup logic; deflection-board simulation showing the arrow's measured deflection.

### 072 — Contrast: A Colour Interaction Auditor
- Subject: simultaneous contrast illusions, colour-contrast failures, and a WCAG contrast auditor over a live palette.
- Direction: optical-illusion gallery; precise geometric fields; measurement callouts; a strict, evidence-first tone.
- Palette: #101014 #f4f4f2 #ff3b30 #2f8bff #ffd60a #30d158 (plus computed pairs).
- Type: `"Helvetica Neue", Helvetica, system-ui`; ratios `ui-monospace, SFMono-Regular, Menlo`.
- Layout: illusion gallery grid (4 real illusions drawn in SVG/CSS: simultaneous contrast, Bezold, von Bezold spreading, Munker); auditor panel with computed ratios; palette table.
- Motion: illusion reveal overlays animate (guides appear/disappear proving equal luminance); ratio bars ease when colours change.
- Interaction: pick base/accent colours → all pairs audited with pass/fail per WCAG level; illusion toggles with an explanation; generate an accessible alternative palette (real algorithm).

### 073 — Coppersmith: Raising a Vessel
- Objective: explain raising sheet copper over stakes — anneal, quench, stakes, planishing — with a stage-by-stage vessel.
- Direction: metalsmith's bench; hammered surface via radial gradients and tiny highlight dots; anneal colour temper strip.
- Palette: #eae3d6 #191512 #a9663a #6f4523 #4f6b5a #9aa3aa (steel).
- Type: `Optima, "Avenir Next", "Helvetica Neue", sans-serif`; stage data `ui-monospace, Menlo`.
- Layout: stage strip (5 SVG vessel profiles), stake library row with profiles, anneal temperature strip, tool notes column.
- Motion: vessel profile morphs between stages; heat glow travels across the copper with temper colours; hammer marks pulse in rhythm.
- Interaction: stage scrubber morphing the profile, stake selector explaining contact area, hammer-size slider affecting stage count, anneal temperature picker showing colour temper.

### 074 — Half-Life: A Decay Laboratory
- Subject: decay chains (U-238 → Pb-206), half-life maths, a cloud-chamber canvas and a Geiger counter with audio clicks.
- Direction: radiation lab on slate; chain diagram as a real chart of N vs Z; cloud-chamber tracks drawn on canvas.
- Palette: #101317 #1a2028 #d6e02a #ff6b35 #4fd1c5 #e8eaee.
- Type: `"Helvetica Neue", Helvetica, system-ui`; nuclides `ui-monospace, SFMono-Regular, Menlo`.
- Layout: decay chain grid (isotope tiles with t½) hero, activity calculator, cloud chamber canvas, detector readout panel.
- Motion: random decays animate along the chain; cloud chamber draws curved alpha tracks and long thin beta tracks continuously; counter clicks (WebAudio).
- Interaction: choose source (U-238, Co-60, C-14, Ra-226), mass/time calculator with real exponential math, time-scrub to see remaining activity, shield toggle changing detected count.

### 075 — Glazing: A Leaded Window Studio
- Subject: design a stained-glass window — cartoon, cut line, came, solder, cement, grisaille — with a glass texture library.
- Direction: cathedral light; window as SVG with hand-drawn came paths, glass painted with multi-stop gradients plus streaky noise, light bloom behind.
- Palette: #0d0b10 #e6ddc8 #2f5f8b #8e2f3b #d8a53c #2f6b4a #6b4f8a.
- Type: `Copperplate, "Hoefler Text", Baskerville, serif`; shop notes `ui-monospace, Menlo`.
- Layout: window hero (lancet + rose), tool/step timeline right, glass palette swatch grid, light-angle control.
- Motion: sun-angle change moves the projected colour spill across the floor; glass streaks shimmer slowly; leading solder beads glint.
- Interaction: paint pieces from the glass palette, choose a cartoon template (3), toggle cartoon/cut-line/came/x-ray, sun-angle slider, lead came width slider.

---

### 076 — Range Chart: A Fossil Assemblage Board
- Subject: 14 index fossils plotted on a geologic range chart (first/last appearance), with an assemblage→age identifier.
- Direction: museum cabinet board; inked range bars on a strat column; fossil silhouettes as SVG line drawings.
- Palette: #ece6d7 #1b1813 #7d5a34 #4f6b5a #2f5f7a #a8402f.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; ages `ui-monospace, Menlo`.
- Layout: range chart hero (x = time, y = taxa with range bars); strat column left; fossil drawer grid; assemblage tester panel.
- Motion: range bars extend from first appearance with staggered easing; selecting an age sweeps a vertical band and dims out-of-range taxa.
- Interaction: click taxa to build an assemblage → intersecting ranges produce a constrained age window with reasoning; scrub the time band; hover fossil → ecology note.

### 077 — Drawdown: A Wells & Aquifer Desk
- Subject: pumping wells, cones of depression, capture zones, and a water-table cross-section that responds to pumping.
- Direction: hydrogeology section drawing; blue water table line, laminated aquifer bands, flowing particle streaks.
- Palette: #eef1ec #141a1d #2f6f8b #6b8f5a #b8863b #a8402f.
- Type: `"Helvetica Neue", Helvetica, system-ui`; heads `ui-monospace, Menlo`.
- Layout: cross-section hero (layers, wells, water table, stream), plan view with capture-zone polygons, pumping schedule table.
- Motion: water table deforms live into a cone when pumping starts; particles advect toward wells; recovery rebounds when pumping stops.
- Interaction: start/stop each well, rate sliders (Theis-ish drawdown math, superposed), toggle capture zones, dry-well warning when the cone hits the well screen.

### 078 — Frame: A Signal Box Lever Desk
- Subject: an interlocking lever frame with a track diagram, block instruments and route setting for a small junction.
- Direction: railway signalling; lacquered levers with coloured heads, tape-drawn track diagram, glass block instruments.
- Palette: #12100e #1e1a16 #c9a227 #2f6f4a #b8402f #4a7ea1.
- Type: `"Helvetica Neue", Helvetica, system-ui`; lever numbers `ui-monospace, Menlo`.
- Layout: track diagram hero (SVG, switches and signals), lever row below with tints per function, block instrument case, interlocking rule panel.
- Motion: setting a route animates points throwing and signals clearing in dependency order; a train occupies sections and the block instrument changes.
- Interaction: lever pull sequence enforcing an interlocking table (refuses with the violated rule), route release, simulate a train movement section by section.

### 079 — Varve: An Ice-Core Log
- Subject: annual layers down 2 000 years — dust peaks, volcanic sulphate spikes, isotope temperature curve, depth-age model.
- Direction: glaciology core rack; translucent ice bands, dark dust laminae, drilled-core segment breaks.
- Palette: #eef3f5 #101a1e #2f6f8b #7f9db3 #c9552e #e8b13a.
- Type: `"Helvetica Neue", Helvetica, system-ui`; depths/ages `ui-monospace, Menlo`.
- Layout: vertical core strip hero (scrollable depth), isotope curve chart with tie-lines, event table (eruptions), accumulation-rate plot.
- Motion: scrolling the depth reveals layers progressively; a selected event draws a tie-line across both charts simultaneously.
- Interaction: depth scrubber with readout (age, accumulation, dust), event row click jumps and annotates, toggle layer-count markers, zoom the isotope curve.

### 080 — Dynamometer: A DC Motor Bench
- Subject: a brushed motor on a brake — torque/RPM/power/efficiency curves, operating point under load, heatsink warning.
- Direction: test-bench instrumentation; analog gauges drawn in SVG plus a digital logger table; no stock "dashboard" clichés.
- Palette: #101317 #1a1f25 #e8eaee #ffb020 #4fd1c5 #ff5d5d.
- Type: `"Helvetica Neue", Helvetica, system-ui`; readings `ui-monospace, SFMono-Regular, Menlo`.
- Layout: curves chart hero with the live operating point; 4 gauge dials; brake-load control; logger table appending rows.
- Motion: needle dials move with slight overshoot; the operating point slides along the curve as load changes; logged rows highlight on add.
- Interaction: load and voltage sliders, run/stop, export-free CSV view in a table, simulate armature heating changing constants over time.

### 081 — Spore Print: An Agaric Key
- Subject: 12 species with spore-print colour swatches, gill attachment diagrams, cap profiles and a habitat column.
- Direction: mycology monograph; radial spore-print gradients, hairline dissection drawings, careful italic binomials.
- Palette: #efe9dc #1d1a15 #7a5c46 #4f6b4a #8e5a7a #b8863b.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; measurements `ui-monospace, Menlo`.
- Layout: spore-print wheel hero (12 radial prints), gill attachment plate (4 diagrams), key table, toxicity banner where relevant.
- Motion: spores fall from the cap onto the card in an animation, resolving into the print colour; radial gradient settles.
- Interaction: filter key (gill attachment × spore colour × ring presence) narrowing the set, species card → detail with spore size, toxicity toggle highlighting edible/caution.

### 082 — Harmonics: A Tide Prediction Desk
- Subject: predict a tide curve from constituents (M2, S2, N2, K1, O1), with a tide clock and a shallow-water overtone.
- Direction: hydrographer's chart; ruled time grid, inked curve, brass clock face, almanac-style tables.
- Palette: #f0ece0 #141a1d #2f5f7a #7f9db3 #b8863b #a8402f.
- Type: `Baskerville, "Hoefler Text", Georgia, serif` headings; times/heights `ui-monospace, Menlo`.
- Layout: 48-hour tide curve hero with HW/LW markers; constituent table with amplitude/phase; analogue tide clock; harbour card.
- Motion: clock hands advance the prediction; curve draws forward; the composite curve visibly resolves from its constituent sines toggled on/off.
- Interaction: toggle constituents to see their contribution, phase/amplitude sliders per constituent, day scrubber, HW/LW table updates live.

### 083 — Tracer: A Gothic Arch Template
- Subject: compass-and-straightedge arch geometry — semicircular, equilateral, lancet, four-centred — with radii, centres and centres-of-curvature arcs.
- Direction: mason's tracing floor; construction lines in faint blue chalk, finished curves in ink, compass arcs animate.
- Palette: #efe9db #191510 #7a5a34 #2f5d50 #3d6b8a #a8402f.
- Type: `Copperplate, Baskerville, "Hoefler Text", serif`; dimensions `ui-monospace, Menlo`.
- Layout: tracing floor hero (large arch with construction geometry), style selector, template gallery, voussoir layout strip with lengths.
- Motion: construction animates step by step (compass swings arcs, centres appear, then ink line follows); voussoirs divide with a stagger.
- Interaction: span/rise sliders live-redraw the geometry (real construction math), style buttons, step-back/forward through construction, toggle chalk lines.

### 084 — Harmonograph: A Pendulum Drawer
- Subject: damped pendulum harmonographs — 2 and 3 pendulum builds with length, damping, and pen-arm ratios — plotting Lissajous decay.
- Direction: Viktor-Olifant-adjacent but original: dark plinth, ink-on-paper plot canvas, brass rigging diagram.
- Palette: #0f1114 #191d22 #efe9dc #c9a227 #4fd1c5 #d1341f.
- Type: `"Helvetica Neue", Helvetica, system-ui`; ratios `ui-monospace, Menlo`.
- Layout: plot canvas hero (progressively drawn), pendulum rig SVG, parameter panel (lengths/damping/coupling), plot gallery thumbnails.
- Motion: the plot draws continuously with the pen tracing; pendulum bobs swing in phase with the plot; decay visibly thins the line.
- Interaction: length ratio steppers (rational ratios give closed figures), damping slider, paper rotation, stop/plot, save-figure-free re-seed, 4 preset builds.

### 085 — Wireless: A Morse Operating Bench
- Subject: key, sounder, and sidetone — dot/dash timing rules, prosigns, a sending-accuracy trainer and a decoder log.
- Direction: telegraphy table; ebonite key, brass sounder drawn in SVG, telegraph-printer tape strip for the log.
- Palette: #0f0d0b #1c1814 #c9a227 #d96f2b #6f8fa8 #e8e0cf.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; tape text `ui-monospace, Menlo`.
- Layout: key+sounder SVG hero with a timing diagram; code table (A–Z, 0–9, prosigns) as a dense grid; printer-tape log strip.
- Motion: key lever depresses and the sounder's armature clicks in sync; tape advances printing decoded characters; timing bars animate.
- Interaction: click-to-send on the code grid, WPM slider changing element duration, type-in-a-word trainer scoring timing accuracy (WebAudio sidetone), speed-reader log.

### 086 — Specus: A Roman Aqueduct Survey
- Subject: a 30 km aqueduct alignment with gradient in permils, inversion siphons, arcades, and a castelum divisorum.
- Direction: surveyor's sheet; long section profile with ground line and channel line, cross-sections as technical plates.
- Palette: #eee7d6 #1c1712 #9c7a4a #4f6b5a #2f5f7a #a8402f.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; gradients `ui-monospace, Menlo`.
- Layout: long section hero (distance × elevation with channel slope), terrain profile, siphon inline detail, specus cross-section plate, distribution table.
- Motion: water flows along the channel line (dash flow); the siphon section animates rising mains; arcade stones build in order.
- Interaction: slope slider recomputes total drop and required arcade heights, toggle siphon vs arcade crossing, click a structure → its detail plate and capacity, unit toggle.

### 087 — Kaleidos: A Mirror Wedge Renderer
- Subject: kaleidoscope builds — wedge count, mirror offset, tileset and drift — rendering true mirror-group symmetry on canvas.
- Direction: black plinth and light; the render is the whole page; controls as a thin instrument strip.
- Palette: #0b0b0e #14161b #e8eaee #ff6b6b #ffd166 #06d6a0 #4cc9f0.
- Type: `"Helvetica Neue", Helvetica, system-ui`; params `ui-monospace, Menlo`.
- Layout: full-bleed render with a wedge seam overlay toggle; bottom instrument strip; right tileset chooser with drawn swatches.
- Motion: tiles drift and rotate continuously; wedge symmetry keeps it coherent; a slow hue cycle option.
- Interaction: wedge count (2–16), mirror phase offset, tile density, drift speed, click to reseed, symmetry-seam overlay for an educational view.

### 088 — Trace: A Star Fort Designer
- Subject: design a trace italienne — bastion angles, curtain length, ditch, ravelin — with enfilade fire zones and blind-spot analysis.
- Direction: military-engineering plate; engraved hatch fills, star polygon geometry, fire-zone translucent overlays.
- Direction note: no modern cartography, no maps of real places.
- Palette: #ece5d4 #1a1611 #7d6a3f #6b5b8a #2f5f7a #a8402f.
- Type: `Copperplate, Baskerville, "Hoefler Text", serif`; angles `ui-monospace, Menlo`.
- Layout: plan hero (star polygon with dimensions), parameter panel (side count, bastion angle, curtain length), fire-zone overlay toggle, glossary.
- Motion: the trace constructs itself (curtain → bastion faces → ditch → ravelin) with dash reveals; enfilade zones sweep along faces.
- Interaction: geometry sliders live-redraw (real polygon math), toggle enfilade/blind spots, ravelin/cavalier toggles, click a bastion → its flank coverage in degrees.

### 089 — Dripstone: A Cave Formation Timeline
- Subject: speleothem growth over 200 000 years — layers, growth rates, palaeoclimate proxies, and cross-section banding (U-series dated).
- Direction: cave-dark with lamp-lit calcite; banding as concentric SVG rings with per-band isotope colour mapping.
- Palette: #0d0e10 #191b1e #e8dcc8 #c9a227 #4a7ea1 #8e5a7a.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; dates `ui-monospace, Menlo`.
- Layout: stalactite/stalagmite column hero with banding cross-section, growth-rate chart, sample table with U-series ages, climate legend.
- Motion: bands accrete outward over the timeline; a lamp glow moves along the column; the climate curve draws with the timeline.
- Interaction: time scrubber growing the formation, band click → age, δ18O, growth rate mm/yr; toggle hiatus (unconformity) surfaces; sample-to-band linking.

### 090 — Scaling: An Organ Pipe Shop
- Subject: flue pipe scaling tables, cutaway of foot/lingo/mouth/languid/flue, and pitch vs length vs mouth-width maths.
- Direction: organ-builder's shop drawing; wood and metal materials as gradients, precise cross-section plate, tonal notes.
- Palette: #ece5d5 #1b1611 #8a5a2b #b8963b #4f6b5a #3d6b8a.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; scales `ui-monospace, Menlo`.
- Layout: pipe cutaway hero with labelled parts, scaling table (8′/4′/2′ and diameter/mouth ratios), harmonics chart, chest section.
- Motion: air column animates through the foot and mouth; the harmonic series chart lights up partials; a cutter line marks the tuning flue.
- Interaction: note/pitch selector recomputes length (real f·L relation with end correction), cut-up slider affecting tone description, toggle mute/corner-board variants, harmonic toggles.

### 091 — Running Gear: A Locomotive Motion
- Subject: Walschaerts valve gear with correct phasing, tractive-effort calculation, and a cut-off/adhesion explainer.
- Direction: engineering works drawing; machined parts as layered SVG with hatch section fills, dimensioned.
- Palette: #ece7d9 #17140f #8c4a34 #4f6b5a #9aa3aa #b8963b.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; dimensions `ui-monospace, Menlo`.
- Layout: side elevation of one wheel + gear hero (animated linkage), piston travel diagram, tractive effort card with real formula, parts legend.
- Motion: the whole gear articulates at speed with correct phase (eccentric crank 90° lead); reverse lever flips the lap; coupling rod rotates.
- Interaction: speed slider, cut-off %, forward/reverse toggle, freeze-frame at any crank angle with a scrubber, hover a part → name and function.

### 092 — Succession: A Pollinator Calendar
- Subject: nectar/pollen availability Jan–Dec for 18 plants, showing forage gaps, with a garden plan and hive-strength curve.
- Direction: horticultural calendar chart; botanical mini-plates; a heatmap of weeks × plants with gap warnings.
- Palette: #f0ecdd #1e211a #4f6b4a #c9a227 #8e5a7a #4a7ea1.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; weeks `ui-monospace, Menlo`.
- Layout: calendar heatmap hero, plant cards row with drawn flowers, forage-gap bar, hive-strength curve below, planting list.
- Motion: heatmap cells bloom in week order; the gap band pulses red; suggested plantings fade in to fill it.
- Interaction: click a week → what's in flower and the deficit, toggle plants on/off recomputing the curve live, add-to-plan building a planting list, zone/hemisphere toggle.

### 093 — Spectral Type: A Stellar Classifier
- Subject: O B A F G K M with absorption-line evolution, colour/B–V, and an HR diagram you can plot onto.
- Direction: observatory dark plates; spectra as generated gradient bars with absorption notches; the HR diagram as the hero.
- Palette: #080a10 #101522 #e8eaee #ffd7a8 #a8c8ff #ff8f6b #ffe9b0.
- Type: `"Helvetica Neue", Helvetica, system-ui`; magnitudes `ui-monospace, Menlo`.
- Layout: HR diagram hero (log scale, main sequence, giants, WD), spectrum strip with named lines (H, Ca II K, Na D, TiO), class table.
- Motion: selecting a class animates the spectrum (lines deepen/fade) and moves a marker on the HR diagram; the temperature gradient shifts.
- Interaction: temperature slider → continuous spectral typing with line strengths from real trends, plot-a-star by entering temp+luminosity → gets classified, toggle spectral line labels.

### 094 — Nephology: A Cloud Atlas Plate
- Subject: the 10 genera plus species/variants on a sky-height diagram, with a fair-weather vs unsettled index.
- Direction: 19th-century atlas plate; clouds drawn as layered SVG blobs with soft opacity, height bands labelled in km and feet.
- Palette: #eaf0f3 #12202a #6f8fa8 #ffffff #b8963b #a8402f.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; altitudes `ui-monospace, Menlo`.
- Layout: height-band diagram hero (high/mid/low/vertical development) with genera placed, genus plate grid, sky-observation worksheet.
- Motion: clouds drift laterally at speeds tied to their level; growing cumulogenensis builds vertically when you select instability.
- Interaction: select a genus → its plate, altitude range, precipitation hint; an "observe the sky" worksheet that composes a scene from your selections; toggles for viraga/mamma/incus detail.

### 095 — Dry Stone: A Walling Bench
- Subject: build a drystone wall — throughstones, hearting, batter, coping — with a centre-of-mass stability check per course.
- Direction: dry-stone craft drawing; stones as irregular SVG polygons with true shading, section view with hearting.
- Palette: #e9e6dc #1c1b17 #7d7466 #9a8f7a #4f6b5a #a8402f.
- Type: `Optima, "Avenir Next", "Helvetica Neue", sans-serif`; measurements `ui-monospace, Menlo`.
- Layout: wall elevation + section hero (interactive courses), stone library, stability panel (tipping/overhang), coping styles row.
- Motion: courses lay one by one; a plumb/batter guide animates; a stability test tilts the wall and flags stones that would rotate out.
- Interaction: place stones per course (click to add/remove), throughstone toggle tying both faces, batter slider, "load the wall" test reporting the first failure course with the physics reason.

### 096 — Bioluminescence: A Deep Sea Field Guide
- Subject: 10 organisms with light-organ diagrams, emission spectra (peak nm), and depth stratification from 0 to 4 000 m.
- Direction: vertical descent aesthetic; a scroll-driven depth gradient from teal to black; organisms as SVG with glowing photophore dots.
- Palette: #04141c #071018 #0a2a33 #38f0b0 #4cc9f0 #b794f4 #ffd166.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; depths/nm `ui-monospace, Menlo`.
- Layout: depth column hero with organisms placed at their range, spectral peak bars, light-organ cutaways, a luminescence dimmer.
- Motion: photophores pulse in species-specific patterns; a slow particle drift; the background gradient responds to the depth scrubber.
- Interaction: depth scrubber revealing only that zone's fauna, species click → organ diagram + peak wavelength + flash pattern animation, toggle "your eye adapts" luminance model.

### 097 — Overshot: A Waterwheel Works
- Subject: overshot/breastshot/undershot wheels with bucket loading, torque through the revolution, and efficiency comparisons.
- Direction: millwright's drawing; timber wheel as SVG spokes/paddles, water as animated fills, torque polar plot.
- Palette: #eae4d4 #1a1712 #7a5a34 #4f6b5a #2f6f8b #b8863b.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; figures `ui-monospace, Menlo`.
- Layout: rotating wheel hero with per-bucket weight, torque-vs-angle polar plot, head/flow inputs, three-type comparison table.
- Motion: wheel turns at the computed RPM; buckets fill at the top and dump at the bottom; the torque plot traces live.
- Interaction: type selector, head and flow sliders recomputing RPM and power, mill-load slider showing the speed droop, toggle showing bucket forces as arrows.

### 098 — Crazy Quilt: A Patchwork Studio
- Subject: irregular patch blocks with fabric fills, rotation, and an embroidery-stitch library (feather, chain, herringbone) drawn as SVG paths.
- Direction: textile studio; fabric textures from layered gradients, thread as stroked paths with dash patterns per stitch.
- Palette: #f0e9db #241d18 #8e3b52 #2f6b5a #b8863b #4a5f8a #7b4fa0.
- Type: `"Iowan Old Style", Palatino, Georgia, serif`; stitch codes `ui-monospace, Menlo`.
- Layout: quilt grid hero (blocks of irregular patches), block editor, stitch library selector, fabric palette, thread count table.
- Motion: embroidery stitches sew themselves along patch seams (dash-offset reveal); blocks rotate on click with a spring; fabric sheen shifts.
- Interaction: generate irregular patches (seeded), rotate/flip blocks, pick stitch per seam and colour, fabric swaps, "sew all" play button that stitches the whole quilt sequentially.

### 099 — Forge: A Hinge & Scroll Plan
- Subject: forge a hinge — draw, fullers, scroll winding sequence, punch work, planishing — with heat-colour temperatures.
- Direction: smithy drawing; hot-metal gradient from straw to orange on dark slag, anvil and tongs as SVG plates.
- Palette: #14110e #1f1a15 #b8863b #d96f2b #ffb020 #6f7a80 #e8e0cf.
- Type: `Optima, "Avenir Next", "Helvetica Neue", sans-serif`; temps `ui-monospace, Menlo`.
- Layout: sequence strip of 8 SVG steps hero, current step detail plate, heat colour temperature scale, tool table with strikes.
- Motion: the workpiece morphs through steps; heat colour animates with a burner pass and cools over time; hammer strikes flash scale.
- Interaction: step scrubber, temperature slider showing forge-weld/anneal windows and what fails outside them, strike-count selector affecting drawn length, toggle jigs/templates.

### 100 — Centennial: A Century of Interfaces
- Subject: 12 interfaces from punchcard to voice, each rendered in its own era's visual language, on one continuous timeline.
- Direction: the finale page. Each era is a distinct micro-art-direction (card stock, CRT phosphor, beige GUI, flat web, glass, ambient/voice), yet the page holds one spine: a single vertical rail and shared grid. Must not look like 12 unrelated pages.
- Palette: era-specific but always from one master set — #f2efe6 #1b1b1e #0c0f14 #38f0b0 #c0c8d0 #ff6b35 #4cc9f0 #b794f4.
- Type: era-appropriate system stacks only (no external fonts): Courier-esque for cards, Helvetica for GUI, system-ui for modern, mono for terminal.
- Layout: single timeline spine; 12 era panels alternating alignment; each panel has a device drawing (SVG/CSS), a 2-line critique, and a metric (adoption years, input bandwidth).
- Motion: scroll-linked era transitions where each device "powers on" once; a persistent progress tick; an input-latency visual that gets shorter over the decades.
- Interaction: era rail navigation with keyboard, each device responds in its own era idiom (type on the terminal, flip the GUI switches, drag the glass slider, press-and-hold for voice), a "compare 1968 vs 2024" split mode.



