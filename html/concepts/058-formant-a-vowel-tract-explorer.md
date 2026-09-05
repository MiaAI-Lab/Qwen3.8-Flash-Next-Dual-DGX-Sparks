### 058 — Formant: A Vowel Tract Explorer
- Subject: the 8 cardinal vowels as vocal-tract shapes and spectrograms, with formant values and a vowel-space quadrilateral.
- Direction: phonetics lab; spectrogram canvas with inked formant tracks, midsagittal tract drawn as a bezier silhouette.
- Palette: #f4f3ef #121417 #2f6f8b #b8452f #d6a419 #6b5b95.
- Type: `"Helvetica Neue", Helvetica, system-ui`; IPA glyphs serif `Baskerville, Georgia, serif`; values `ui-monospace, Menlo`.
- Layout: vowel quadrilateral hero; per-vowel card with tract SVG + spectrogram; F1×F4 table; articulation notes column.
- Motion: tract morphs between vowels (path interpolation); spectrogram re-renders with formant bands sliding.
- Interaction: draggable articulator dot inside the quadrilateral → tract and spectrogram update continuously; IPA buttons; toggle formant labels/centres.
