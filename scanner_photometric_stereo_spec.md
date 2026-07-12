# Scanner-Based Photometric Stereo — Shallow-Surface Normal Map Pipeline

## Build Spec

---

## 1. Project Goal

Reconstruct a **tangent-space normal map** (plus albedo and height) of shallow,
mostly diffuse subjects using a **flatbed scanner** as a photometric stereo rig.
Leaves—especially grape and Kiwi leaves—are the primary test cases and the
reason the project exists, but the method is not foliage-specific. Paper,
fabric, cardboard, bark, pressed flowers, and similar matte materials can satisfy
the same model. Highly specular subjects such as polished coins do not.

**Hardware safety:** use caution with any solid object on the platen. Raised,
heavy, sharp, dirty, or oversized items can scratch or crack the scanner glass,
prevent the lid from moving correctly, or damage the scanner mechanism. Never
force an object flat with the lid.

The core insight: a scanner's lamp is fixed relative to its sensor, so a single scan gives exactly one lighting direction. But by **physically rotating the subject on the glass** between scans, we change the light's azimuth *relative to the subject*. Four scans at 0°/90°/180°/270° therefore yield four lighting directions — enough for a proper photometric stereo solve.

**Deliverables:**
- `normal_gl.png` — tangent-space normal map, OpenGL convention (+Y up)
- `normal_dx.png` — same, DirectX convention (-Y / green flipped)
- `albedo.png` — lighting-free base color
- `height.png` — integrated height field (optional, from normals)
- `thickness.png` — optional, from a backlit transmission scan (see §10)
- `qa/` — diagnostic renders and residual maps

---

## 2. Hardware Facts (already established — do not re-derive)

| Property | Value |
|---|---|
| Device | HP LaserJet M1132 MFP, flatbed |
| Sensor type | CIS (contact image sensor) |
| Max optical resolution | **1200 dpi** (1200×1200; do not use interpolated modes above this) |
| Max scan area | 216 × 297 mm (A4) |
| Projection | **Orthographic** — telecentric line scan |
| Connection | USB only |
| **Host OS** | **Windows** (see §4.2 — HPLIP/SANE is Linux-only and is NOT available) |

### Why the scanner beats a camera-in-a-box rig (design rationale — keep these properties intact)

- **Orthographic projection.** Every pixel is viewed straight down. The view vector is a **constant `V = (0, 0, 1)` across the entire image.** A perspective camera's view vector varies from center to edge, which contaminates a photometric solve unless modeled per-pixel. We get a constant view vector for free — the pipeline may assume it.
- **No lens distortion.** No undistortion step required.
- **The lid flattens the subject.** No pre-pressing/ironing of leaves needed.
- **Exposure, transport, and file transfer are automatic.** The only manual step is rotating the subject.

### Confirmed by experiment (do not assume otherwise)

The subject was scanned at 0° and again at 90°, then overlaid. **Highlights clearly come from different directions in the two scans.** The M1132's CIS illumination is therefore **meaningfully directional**, not coaxial/flat. Photometric stereo is viable on this hardware. This gate has already been passed.

---

## 3. Physical & Optical Model

### 3.1 Coordinate frame (scanner space)

- **X** — along the sensor line (across the page, perpendicular to head travel)
- **Y** — direction of scan-head travel
- **Z** — up, out of the glass, toward the subject

View vector `V = (0, 0, 1)` everywhere.

### 3.2 Light model

Model the lamp as a **distant directional light**:

```
L(az, el) = ( cos(el)·cos(az),  cos(el)·sin(az),  sin(el) )
```

- `el` — **elevation** above the glass plane. **Fixed by the hardware. Identical for all four scans.** This is the single most important unknown; see §7.
- `az` — **azimuth**. Fixed in *scanner* space, but the subject rotates beneath it, so in *subject* space it takes four different values (see §6.3).

**Consequence to be aware of:** because elevation is constant, all four light vectors lie on a **single cone** around the Z axis. This is a legitimate and solvable photometric stereo configuration, but it is *not* a general one. Getting `el` wrong does not make the map look broken — it makes the normals **systematically over- or under-tilted** (relief too strong or too weak) while still looking plausibly leaf-like. So `el` must be calibrated, not guessed. See §7.

### 3.3 Known limitation of the model (document it, don't try to fix it)

The real CIS lamp is an **extended line source parallel to the sensor line**, not a point. This means:
- The light direction is well-defined mainly in the plane **perpendicular to the line** — i.e. tilted along the **Y (travel) axis**.
- Along the **X (line) axis**, illumination is comparatively diffuse and symmetric.

**Practical prior:** expect the fitted azimuth to land near the travel axis (`az ≈ 90°` or `≈ 270°`). Use this only as a sanity check / initialization, never as a hardcoded value.

The distant-point-light approximation is imperfect here. **Do not attempt to model the line source analytically.** Instead, fit `az` and `el` empirically against a known target (§7) so the calibration absorbs the model error.

---

## 4. Capture Protocol (human steps — implement a checklist/validator, not automation)

### 4.1 The registration card (strongly recommended)

Place the leaf on a **matte white card** printed with **fiducial marks** (e.g. four high-contrast crosshairs or ArUco markers near the corners, well outside the leaf's footprint).

The card **rotates together with the leaf**. This gives us:
- Exact recovery of the rigid transform between scans — we never have to *assume* the rotation was exactly 90°.
- A consistent, uniform, matte background for masking.

The pipeline should **detect the fiducials and solve the rigid transform from them.** Fall back to a nominal k·90° rotation only if fiducials are absent.

### 4.2 Capture on Windows — the acquisition stack

⚠️ **The host is Windows.** HPLIP / SANE / the `hpaio` backend are **Linux-only and unavailable.** Ignore any SANE-based instructions found elsewhere.

#### Recommended: VueScan (Hamrick)

VueScan **explicitly supports the HP LaserJet M1132 on Windows** — you must install HP's own driver first, then VueScan drives the scanner through it. (HP sold this same unit as the M1136/M1137/M1138/M1139; all VueScan pages for those models apply.)

Critically, VueScan reads **raw sensor data** from the scanner and can write it to a **raw TIFF** file, with the RAW file type selectable as **16-bit gray**. That is exactly the linear, high-bit-depth, un-processed data §5.1 requires — it's the whole reason to use it over the stock HP software.

Settings to lock:
- **Output tab → RAW file type:** `16 bit Gray` (or `48 bit RGB` if you want color for the albedo — see §5.2). Enable saving the RAW file.
- **Filter tab:** everything **off** — no infrared clean, no sharpening, no grain reduction, no restore-fading.
- **Color tab:** no auto-levels, no white balance, no curve. If a gamma/curve setting exists, set it linear (1.0).
- **Input tab:** lock exposure/brightness — **must be identical across all four scans.** If VueScan offers per-scan auto-exposure, disable it.
- Resolution: **1200 dpi** (600 dpi acceptable; config value).
- **Never** JPEG.

**Verify, don't assume:** VueScan can only surface what HP's driver exposes. It's possible the HP WIA/TWAIN driver caps at 8-bit, in which case VueScan's "16-bit" output is 8-bit data padded to 16. **Test this early** — scan a smooth gradient and check the histogram for only-256-distinct-levels. The pipeline must handle both cases; log which one it detected.

VueScan is paid software (the trial watermarks output). Budget for it, or use the fallback below.

#### Free fallback: NAPS2 or direct WIA/TWAIN

- **NAPS2** — free, drives WIA/TWAIN scanners, can save lossless PNG/TIFF. Less control over gamma/linearity; you will almost certainly get 8-bit sRGB output and must linearize in software (§5.1).
- **Direct WIA automation from Python** — `pywin32` + the WIA COM interface lets you script the four scans without touching a GUI. This is the nicest option *if* the HP WIA driver exposes the properties you need. Enumerate the WIA device's properties and dump them before committing.
- **TWAIN from Python** — `pytwain` / the `twain` module, if the WIA path proves too limited.

**Recommendation:** build the pipeline to consume a **folder of TIFF/PNG files** and treat capture as a separate, manual-ish step. Do not couple the solver to any particular acquisition API. Ship a `capture_wia.py` helper as a convenience, but the pipeline's contract is "give me 4 lossless images + a flat-field + a calibration scan," however they were produced.

#### Power-user option: WSL2 + USB passthrough

If the Windows drivers turn out to be too restrictive, you can run the real Linux stack: **WSL2 + `usbipd-win`** to pass the printer's USB device into the Linux VM, then install HPLIP/SANE and use `scanimage` with the `hpaio` backend as originally planned. This gives the most direct control over depth and gamma. Treat this as a fallback if VueScan's data proves non-linear or 8-bit-capped, not as the default.

### 4.2b Scan settings — non-negotiable (regardless of acquisition tool)

- **4 scans**, rotating the card 90° between each: `k = 0, 1, 2, 3`.
- Native optical resolution: **1200 dpi** (600 dpi is an acceptable lower-noise / smaller-file option; make it a config value).
- **Disable every "enhancement":** auto-exposure, auto-color, auto-contrast, brightness, sharpening, descreening, dust removal. All of them. Any adaptive processing breaks the assumption that brightness is comparable across scans. **This is the single most important capture requirement** — an auto-exposure that adapts per scan silently destroys photometric stereo.
- **Highest available bit depth.** Prefer 16-bit; detect and handle 8-bit gracefully.
- Save **lossless** (TIFF or PNG). **Never JPEG.**
- Color or grayscale both fine (see §5.2).

### 4.3 Calibration scans (one-time, per resolution setting)

1. **Flat-field scan** — a blank sheet of the same matte white card stock, no subject. Captures the lamp's spatial illumination profile.
2. **Dark/black-level** — if the backend supports it; otherwise estimate the black level from the darkest percentile of the flat-field's unlit border, or accept an offset of 0.
3. **Light-calibration target** — a piece of **single-face corrugated cardboard** (parallel ridges of known, regular profile). Scan it at 0° and 90°. This is the target used to fit `az` and `el` in §7.

### 4.4 Subject handling

- Press flat under the lid. Optionally lay a **clean sheet of glass or acrylic** on top of the leaf as a weight to force it into the same plane each time. **No adhesive** — adhesive adds its own specular sheen and contaminates the photometric assumption.
- Fresh, turgid leaves deform less than dried/curling ones.
- **Expect residual deformation anyway.** Thin flimsy leaves (grape especially) settle into a *slightly different shape* on every rotation. This is not avoidable mechanically and **must be corrected in software** — see §6.4. This is a hard requirement, not a nice-to-have.

---

## 5. Preprocessing

### 5.1 Linearization (mandatory)

Photometric stereo requires **intensity ∝ incident light**. Scanner output is typically **gamma-encoded (sRGB)**.

- If the backend can emit linear data or accept a linear gamma table, use it.
- Otherwise, **undo the sRGB transfer function** in software to get linear values.
- Make this a config switch (`input_is_srgb: true/false`) and default to `true`, but **verify** empirically (scan a stepwedge / grayscale ramp if available).

Work internally in **float32, linear, [0,1]**.

### 5.2 Channel handling

Scan RGB and convert to a **linear luminance** channel for the photometric solve. Keep the RGB for the albedo output.

**Bonus signal (cheap, take it):** red light penetrates leaf tissue deeper than blue. The **red channel carries more subsurface/vein information; blue stays more surface-bound.** A `R − B` difference image is a crude but free depth/subsurface hint. Compute and save it as `qa/subsurface_hint.png`; optionally expose it as a weak input to the height map. Do not let it drive the normals.

### 5.3 Dark & flat-field correction (mandatory)

```
corrected = (raw − dark) / (flat − dark)
```

CIS lamps have real falloff along the sensor line (LEDs are typically injected at the ends of a light guide). **If this spatial nonuniformity is not divided out, it gets baked directly into the normals as a large-scale phantom gradient.**

Note: flat-fielding removes spatial *intensity* nonuniformity. It does **not** remove the light's *directionality* — which is the signal we want. Good.

---

## 6. Alignment Pipeline

This is the part with the most failure modes. Order matters.

### 6.1 Choose a reference

Scan `k = 0` is the reference frame. All others are brought into its coordinate system.

### 6.2 Rigid alignment

- Detect fiducials on the registration card in each scan.
- Solve the similarity/rigid transform mapping scan `k` → scan `0`.
- Warp scan `k` accordingly (high-quality interpolation — bicubic or Lanczos; the data is float, don't clamp).
- If no fiducials: fall back to de-rotating by nominal `−k·90°` about the image center, then refine with ECC / phase correlation on the leaf mask.

After this step, the leaf should overlap to within a few pixels. That's not good enough — hence §6.4.

### 6.3 Light-direction bookkeeping ⚠️ **THE CRITICAL STEP**

**This is the entire trick. Get it wrong and everything downstream is garbage that still looks superficially plausible.**

The lamp is **fixed in scanner space**. When we de-rotate the *image* of scan `k` by `−k·90°` to align it with the reference, the light direction **rotates with it**. So in the aligned (subject) frame:

```
az_subject[k] = az_scanner − k · 90°
el_subject[k] = el_scanner          # unchanged — always
```

So the four light vectors are:
```
L[k] = ( cos(el)·cos(az₀ − k·90°),
         cos(el)·sin(az₀ − k·90°),
         sin(el) )                    for k = 0,1,2,3
```

- **Sign convention:** if the fiducial-derived rigid transform gives an actual rotation `θ_k` (not exactly 90°), use `az_scanner − θ_k` rather than `az_scanner − k·90°`. **Prefer the measured angle over the nominal one.**
- Write a unit test that asserts the four `L[k]` are unit vectors, share an identical Z component, and are ~90° apart in azimuth.
- Log the four vectors to the console on every run. This is the number one thing a human will want to eyeball when the output looks wrong.

### 6.4 Non-rigid registration (mandatory — the leaf deforms)

Thin leaves settle differently on each rotation. After rigid alignment there will be **residual elastic deformation of several to tens of pixels**, especially near the leaf's lobes and margins. This must be corrected.

**Warp scans 1, 2, 3 onto scan 0** with a dense non-rigid field.

#### ⚠️ Do NOT run optical flow on the raw images.

Optical flow assumes **brightness constancy**. Our images *deliberately violate it* — the brightness differences between scans **are the signal we are trying to measure.** Running flow on raw images will cause it to "helpfully" warp away the very shading differences we need, silently destroying the reconstruction.

**Instead, compute the flow on a lighting-invariant proxy, then apply the resulting field to the original full-detail images.**

Acceptable proxies (implement at least the first, ideally allow choosing):
1. **Leaf silhouette mask, distance-transformed.** Robust, lighting-independent by construction. Captures the outline deformation, which is the dominant error. **Start here.**
2. **High-pass / vein structure.** Vein *positions* are lighting-invariant even though their shading isn't. Extract via high-pass or ridge filter, then flow on that. Adds interior (non-outline) correspondence.
3. **Heavily blurred + locally normalized image** (divide by local mean) — suppresses low-frequency shading differences.

Best practice: build a proxy that **combines** (1) and (2) — outline plus vein skeleton — for both boundary and interior correspondence.

**Implementation options** (easiest first):
- `cv2.optflow.DISOpticalFlow` / `cv2.calcOpticalFlowFarneback` — dense field, ~10 lines. Start here.
- Feature-based **thin-plate spline**: detect corresponding vein junctions (they are excellent, distinctive landmarks), fit a smooth TPS warp. More controllable.
- `SimpleITK` **BSpline / Demons** registration — robust, well-tested, more setup.

**Then:** `warped_k = remap(original_k, flow_field)`. Apply to the **original linear full-detail image**, never to the proxy.

### 6.5 Masking & validity

- Segment leaf from the matte white background (high contrast — easy; Otsu on the linear luminance, then morphological cleanup, largest connected component).
- Build a mask per scan, warp them all into the reference frame.
- **Valid pixel = inside the leaf mask in the reference AND has a valid warped sample from at least 3 of the 4 scans.**
- Pixels failing this are excluded from the solve and inpainted/flood-filled at the end (§9.2).
- Emit `qa/mask_agreement.png` showing how many valid samples each pixel has. **A human should look at this before trusting anything.**

---

## 7. Light Calibration (fitting `az₀` and `el`)

`el` cannot be measured with a ruler and **must not be hardcoded**. Fit it.

### Method A — corrugated cardboard (preferred)

Single-face corrugated card has ridges of approximately known, regular, near-sinusoidal profile.

1. Scan at 0° and 90°.
2. Extract the ridge profile (average many ridges to kill noise).
3. The **shading asymmetry across each ridge** — how much brighter the light-facing flank is than the away-facing flank — is a direct function of `el`. Low `el` (grazing) → strong asymmetry and possible self-shadowing. High `el` (near-coaxial) → weak asymmetry.
4. Fit `(az₀, el)` by nonlinear least squares: forward-render the known ridge profile under `L(az, el)` with a Lambertian model and minimize residual vs. the observed scan.
5. The **0° vs 90° pair also directly disambiguates `az₀`**: whichever axis shows the stronger ridge asymmetry is the axis the light is tilted along.

### Method B — self-calibration fallback

If no calibration target is available, treat `el` as a **single global scalar parameter** and either:
- Let the user tune it interactively until the relief strength looks right, or
- Fit it by minimizing the **re-rendering residual** (§8.4) across the whole leaf.

Expose `el` and `az0` as config values with the fitted results as defaults. **Always log which method produced them.**

---

## 8. The Photometric Stereo Solve

### 8.1 The model

Lambertian reflectance, per pixel:

```
I_k = ρ · (N · L_k)          for k = 0..3
```
- `I_k` — observed linear intensity in aligned scan `k`
- `ρ` — albedo (unknown, **spatially varying** — critical for leaves)
- `N` — unit surface normal (unknown, 2 DOF)
- `L_k` — known light direction from §6.3

Substituting `g = ρ · N` (3 unknowns, unconstrained) linearizes it:

```
I = L · g        where L is 4×3, I is 4×1, g is 3×1
```

Solve per pixel by least squares: `g = (LᵀL)⁻¹ Lᵀ I`

Then:
```
ρ = ‖g‖
N = g / ‖g‖
```

### 8.2 Why 4 scans and not 2 — enforce this

**Do not add a "2-scan mode."** With 2 images you have 2 equations and 3 unknowns: **underdetermined.** Closing it requires assuming uniform albedo — which is catastrophically wrong for a leaf, where dark veins and pale interveinal tissue differ enormously. The solver would convert **albedo changes into fake geometry**, producing a normal map that tracks pigment rather than shape. 3 is the minimum; 4 is what we use.

### 8.3 Robust solve — outlier rejection (required)

Leaf veins are **glossy** and will throw **specular highlights** that violate the Lambertian assumption. Cast shadows do the same from the other direction.

With 4 samples we can afford to reject:
- **Drop the brightest sample** per pixel → kills speculars. This alone justifies the 4th scan.
- Optionally also **drop the darkest** → kills shadowed samples. (But then you're back to 2 — so only do this where ≥4 valid samples exist, and prefer dropping only the brightest by default.)
- Require **≥3 surviving samples**; otherwise mark the pixel invalid.

Implement as a config: `rejection: none | drop_brightest | drop_brightest_and_darkest`. **Default: `drop_brightest`.**

Vectorize this — a per-pixel Python loop over a 4700×4700 image will be unusably slow. Use NumPy: sort the 4 samples along the sample axis, mask, and batch-solve with `np.linalg.lstsq` on reshaped arrays, or build the 3×3 normal equations directly and invert analytically (they're tiny — closed-form 3×3 inverse is fastest).

### 8.4 Validation by re-rendering ⚠️ **Ship this — it's the main QA signal**

Once `N` and `ρ` are recovered, **re-render** each of the four lighting conditions:
```
I_pred_k = ρ · (N · L_k)
```
Compare against the observed `I_k`. Output per-scan **residual maps** to `qa/`.

- Low, noise-like residual → the solve is trustworthy.
- Structured residual (e.g. bright along veins) → speculars leaking through; tighten rejection.
- Large-scale gradient residual → **flat-field correction or `el` is wrong.**
- Residual correlated with the leaf outline → **non-rigid registration failed.**

This single diagnostic will explain almost every failure mode. Make it a first-class output, not an afterthought.

---

## 9. Outputs & Post-Processing

### 9.1 Height map (optional)

Integrate the normal field into a height field via **Frankot–Chellappa** (FFT-based Poisson integration) — fast, robust, handles the non-integrable noise gracefully. `scipy.fft` is sufficient; no external dep needed.

Expect low-frequency drift/doming; high-pass the result if it's only being used for detail.

### 9.2 Cleanup

- Inpaint invalid pixels (from §6.5) — use `cv2.inpaint` or simple flood-fill from valid neighbors.
- Optional mild bilateral / edge-preserving smoothing on the normals. **Expose strength as config; default low.** Over-smoothing destroys the vein detail that is the entire point.
- Renormalize `N` to unit length after any smoothing.

### 9.3 Encoding

Tangent-space normal map, standard encoding:
```
R = N.x · 0.5 + 0.5
G = N.y · 0.5 + 0.5
B = N.z · 0.5 + 0.5
```
- Export **`normal_gl.png`** (OpenGL, +Y up) and **`normal_dx.png`** (DirectX, green channel inverted). Cheap to do both — do both.
- Export **16-bit PNG** to avoid banding on the low-slope interveinal regions.
- Albedo: export as-is (linear) **and** sRGB-encoded, clearly named.

---

## 10. Stretch Goal — Backlit Transmission / Thickness Map

Independent of the photometric solve, and a strong bonus for leaf shaders.

**Capture:** place the leaf on the glass, **lid open, room dark**, and shine a lamp **down through the leaf** from above while scanning. The scanner records **transmitted** light.

**What it gives you:**
- A genuine **thickness / translucency map** — exactly what a good leaf subsurface-scattering shader wants.
- A dramatic, high-contrast rendering of the **vein network** — which is the dominant real relief on a leaf.

**Pipeline:** register it to the reference frame (same fiducials/mask machinery), invert (thick = dark in transmission), normalize → `thickness.png`.

Optionally: blend the transmission-derived vein structure into the height map as a **detail layer**, since it captures vein topology more cleanly than shading alone. Keep it as an optional, weighted, off-by-default input — **the photometric normals remain the source of truth.**

---

## 11. Suggested Stack & Structure

- **Windows host.** Python 3.11+, NumPy, OpenCV (`opencv-contrib-python` for `optflow`), SciPy, `tifffile`, `imageio`. All are pip-installable on Windows with prebuilt wheels — no build toolchain needed.
- Optionally `SimpleITK` for the higher-quality non-rigid registration path (also has Windows wheels).
- Optionally `pywin32` for the WIA capture helper.
- **Everything downstream of capture is platform-agnostic.** Only §4.2 is Windows-specific. Use `pathlib` throughout; no shell-outs, no POSIX path assumptions.
- CLI-driven, config via YAML/TOML. Every magic number in §3–§9 must be a config value, not a literal.

```
leafscan/
  capture_wia.py          # optional Windows WIA helper; pipeline must not depend on it
  config.yaml
  io.py                   # load, linearize, dark/flat-field
  calibrate.py            # fit az0, el from corrugated target
  align.py                # fiducials, rigid, non-rigid warp, masks
  lights.py               # light-vector bookkeeping (+ unit tests)
  solve.py                # robust photometric stereo, vectorized
  integrate.py            # Frankot-Chellappa
  outputs.py              # encoding, GL/DX export
  qa.py                   # re-render residuals, mask agreement, light-vector dump
  cli.py
```

**Process the pipeline at full 1200 dpi but develop against a downsampled copy** — a 4700×4700 float32 stack of 4 images is ~350 MB before intermediates. Support a `--scale` flag.

---

## 12. Priority Order for Implementation

0. **Verify the capture stack first.** Get VueScan (or WIA) producing a lossless scan with auto-exposure off, and **check whether you're actually getting 16-bit or a padded 8-bit**. This determines the linearization path in §5.1 and it's cheap to find out now vs. after you've built everything.
1. **`io.py` + `lights.py` + `solve.py`** with *synthetic* test data (render a known bump surface under 4 known cone lights, solve, assert you recover it). **Prove the math before touching a real scan.**
2. `qa.py` re-rendering residual — you need this diagnostic before real data, or you'll be debugging blind.
3. `io.py` real path: linearization + flat-field.
4. `align.py` rigid (fiducials).
5. `calibrate.py` — fit `az0`, `el` on corrugated card.
6. **First real end-to-end run on the grape leaf.** Expect the outline to be wrong.
7. `align.py` non-rigid — the leaf-deformation fix.
8. Outputs, integration, polish.
9. Stretch: transmission map.

---

## 13. Non-Negotiables (a summary of the traps)

- ❌ **Never** run optical flow on the raw (differently-lit) images.
- ❌ **Never** hardcode the light elevation — fit it.
- ❌ **Never** offer a 2-scan mode.
- ❌ **Never** skip flat-field correction.
- ❌ **Never** use JPEG or leave auto-exposure/auto-color enabled.
- ✅ **Always** log the four light vectors.
- ✅ **Always** emit the re-rendering residual.
- ✅ **Always** apply the warp to the original image, not the proxy.
