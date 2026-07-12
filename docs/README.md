# Reference material — worked example

LUMEN-PS is a general shallow-surface scanner for mostly diffuse subjects. This
folder uses a leaf because foliage motivated the project and makes fine relief
easy to judge; the reconstruction is not limited to leaves. Paper, fabric,
cardboard, bark, and other flat matte materials are also candidates. Avoid very
shiny subjects, and use caution with rigid objects: raised, heavy, or sharp
items can scratch or crack the scanner glass or damage its lid/mechanism.

A complete filtered run of the pipeline on a Kiwi leaf, captured on the
HP LaserJet M1130/M1132 flatbed at 1200 dpi. These are **compressed, downscaled
images for documentation only** — photos/diagnostics are JPEG, normal maps and
anything with alpha are 8-bit PNG. The originals were the full-res lossless
outputs; regenerate them by re-capturing and running the pipeline.

## `scans/cropped/`
`k0..k3.jpg` — the four scans, rotated 90° between each (petiole steps
left → down → right → up), cropped to a common scanner-space ROI around the leaf.
8-bit sRGB (this scanner's WIA driver caps at 8-bit). *(Lossy previews here; a
real run needs lossless PNG/TIFF captures — the CLI ignores JPEGs on purpose.)*

## `results/`
Final deliverables (downscaled to ≤1500 px):
- `normal_gl.png` / `normal_dx.png` — tangent-space normal maps
- `albedo.jpg` (linear — looks dark) / `albedo_srgb.jpg` — lighting-free base colour
- `alpha.png` — subject silhouette / opacity
- `albedo_srgb_rgba.png`, `normal_gl_rgba.png` — RGBA copies for a transparent plane
- `height.jpg` — Frankot–Chellappa integrated relief

These result previews include residual-guided bad-normal repair. The original
full-resolution filtered sources are from
`sessions/kiwi-leaf-20260712-184906/out`.

## `qa/`
The diagnostic set (spec §8.4 — the re-render residual is the main trust signal):
- `residual_scan0..3.jpg` — |observed − re-rendered| per scan. Low and noise-like
  here → trustworthy solve. Final residual means ≈ 0.009 across all four scans.
- `observed_scan*` / `predicted_scan*` — the two sides of that comparison.
- `mask_agreement.jpg` — per-pixel count of valid samples (4/4 across the leaf).
- `normal_preview.png` — 8-bit preview of the normal map.
- `subsurface_hint.jpg` — R−B channel difference (vein/subsurface hint, §5.2).
- `rejection_coverage.jpg` — how often each pixel survived outlier rejection.
- `misreg_repair.png` — gray pixels were re-solved after rejecting one locally
  inconsistent observation; white pixels required selective inpainting.
- `report.txt` — fitted light vectors + residual stats for this run
  (self-calibrated: az0 = 90°, el = 45°).
