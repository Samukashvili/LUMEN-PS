# leafscan — scanner-based photometric stereo for leaf normal maps

Reconstruct a tangent-space **normal map** (+ albedo + height) of a thin flat
subject (grape leaf) using a **flatbed scanner** as a photometric-stereo rig.
A scanner's lamp is fixed relative to its sensor, so one scan = one light
direction. Rotate the subject 90° between scans → four light directions → a
proper photometric solve. Full design: `scanner_photometric_stereo_spec.md`.

## Install

```
pip install -r requirements.txt
```

## Hardware notes (this unit, verified)

- HP LaserJet Professional M1130 MFP (a.k.a. M1132), WIA 2.0, USB.
- The Windows WIA driver **caps at 8-bit** (`BitsPerPixel ∈ {1,8,24}`) — no
  16-bit. So capture 24-bit sRGB colour and linearize in software.
- Resolutions: 75/100/150/200/300/600/**1200** dpi.

## Capture (4 scans, rotate 90° between each)

```
python -m leafscan.cli capture --preview                 # quick look
python -m leafscan.cli capture --out scans/leaf/k0.png --dpi 600 --color
#   rotate the leaf/card 90°, then k1, k2, k3 ...
```

Non-negotiables (spec §13): auto-exposure/auto-colour/sharpening all OFF
(the WIA helper enforces this), identical brightness/contrast every scan,
lossless PNG/TIFF — never JPEG.

Optional extra scans:
- `flat.png` — a blank sheet of the card stock (flat-field). If absent, the
  pipeline fits the lamp falloff from the leaf's white surround.
- `calib0.png`, `calib90.png` — single-face corrugated cardboard at 0°/90° to
  fit light elevation. If absent, elevation is self-calibrated from the leaf.

## Run the pipeline

```
python -m leafscan.cli run --scans scans/leaf --out out
python -m leafscan.cli run --scans scans/leaf --out out --flat scans/leaf/flat.png
python -m leafscan.cli run --scans scans/leaf --out out --calib scans/calib0.png,scans/calib90.png
python -m leafscan.cli run --scans scans/leaf --out out --scale 0.25   # fast dev
```

### Outputs
`out/normal_gl.png`, `out/normal_dx.png`, `out/albedo.png`, `out/albedo_srgb.png`,
`out/height.png`, and `out/qa/` (re-render residuals — the main QA signal —
mask agreement, light-vector dump in `report.txt`).

## Read the QA first (spec §8.4)
- Low, noise-like `qa/residual_scan*.png` → trust the solve.
- Bright along veins → speculars leaking; tighten `solve.rejection`.
- Large-scale gradient → flat-field or `el` wrong.
- Correlated with the outline → non-rigid registration failed.

## Tests
```
python -m pytest tests -q          # or: python -m leafscan.cli selftest
```

## Layout
`leafscan/`: `io` (load/linearize/flat), `lights` (light bookkeeping + tests),
`solve` (robust vectorized PS), `align` (fiducials/rigid/non-rigid/masks),
`calibrate` (az0/el fit), `integrate` (Frankot–Chellappa), `outputs`, `qa`,
`cli`, `capture_wia` (optional WIA helper). Every magic number lives in
`config.yaml`.
