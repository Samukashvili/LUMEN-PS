<div align="center">

# LUMEN-PS

### Turn a flatbed scanner into a photometric-stereo material scanner.

Recover **normal maps, albedo, height, and alpha** from four ordinary scans—no camera rig, synchronized lights, or special optics.

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Platform Windows](https://img.shields.io/badge/platform-Windows-0078D4?logo=windows)
![Capture WIA 2.0](https://img.shields.io/badge/capture-WIA%202.0-f0a64a)
![Photometric stereo](https://img.shields.io/badge/reconstruction-photometric%20stereo-65a96b)

<img src="docs/assets/kiwi-relight.gif" width="640" alt="Recovered Kiwi leaf relit through a full 360 degree orbit">

*A real 1200 dpi Kiwi leaf reconstruction, relit through 360°. The animation uses the same normal-map lighting equation as the interactive viewer.*

</div>

> [!IMPORTANT]
> LUMEN-PS began as a foliage scanner, but the method is not leaf-specific. It can recover shallow surface relief from paper, fabric, bark, pressed flowers, prints, cardboard, and other mostly diffuse subjects. Very shiny or mirror-like materials—such as polished coins—break the lighting model. Use caution with any rigid object: a raised, heavy, sharp, or oversized object can scratch or crack the scanner glass or damage the lid/mechanism.

## The idea in one minute

A flatbed scanner already contains a stable moving light and a calibrated line sensor. They sit **extremely close together**, so the lighting difference is genuinely small—easy to miss when two scans are viewed at different rotations. Crucially, however, they are **not perfectly coaxial**. That finite baseline gives the incident light a slight sideways component: a microscopic slope facing the lamp returns a little more light than the same slope facing away. With locked capture settings, registration, linearization, and four observations, that subtle but repeatable signal is enough.

![Cross-section of the scanner lamp and sensor showing their useful offset](docs/assets/scanner-parallax.svg)

The lamp remains fixed in scanner coordinates. Rotate the subject by 90° between scans and the light appears to orbit it in subject coordinates. After the four images are aligned, each pixel has four measured intensities under four known light directions. That is enough to separate surface orientation from base color.

![Four rotations convert the scanner's fixed lamp into four subject-relative lighting directions](docs/assets/four-directions.svg)

<p align="center"><img src="docs/assets/kiwi-four-scans.webp" width="100%" alt="Four real Kiwi leaf scanner captures at 0, 90, 180, and 270 degrees"></p>

## The four orientations, matched pixel-for-pixel

Rotation makes the raw captures easy to compare as objects but hard to compare as lighting measurements. LUMEN-PS de-rotates every view into the first scan's coordinate frame, then performs non-rigid registration to correct the small way a leaf or flexible subject settles after handling. In the registered images below, the tip, veins, and boundary occupy the same pixels; only illumination and capture outliers should change.

<p align="center"><img src="docs/assets/kiwi-aligned-lighting.webp" width="900" alt="Four registered Kiwi leaf luminance observations with fitted light azimuths"></p>

Because the CIS lamp and sensor are so close, the useful difference is much smaller than the shared albedo and overall brightness. The visualization below subtracts each pixel's four-view mean and amplifies the remaining deviation **7×**. Orange means brighter than that pixel's mean; blue means darker. Opposing directional patterns across the four views are the photometric signal the solver uses—not fabricated depth or an edge filter.

<p align="center"><img src="docs/assets/kiwi-lighting-difference.webp" width="900" alt="Lighting-only deviations from the four-view mean amplified seven times"></p>

## From scans to a relightable material

| Stage | What happens | Why it matters |
|:--|:--|:--|
| **1 · Capture** | Scan at 0°, 90°, 180°, and 270° with identical exposure and color settings. | Produces four observations with different subject-relative light azimuths. |
| **2 · Linearize** | Undo sRGB gamma and optionally divide by a blank-card flat field. | Photometric stereo requires pixel values proportional to received light. |
| **3 · Register** | De-rotate using fiducials, refine rigid alignment, then correct small elastic changes. | The same output pixel must represent the same physical point in all four scans. |
| **4 · Calibrate** | Fit lamp azimuth and elevation from a calibration card or from re-render error. | The light elevation controls how strongly recovered normals tilt. |
| **5 · Solve** | Robustly solve `I = ρ(N · L)` per pixel, dropping highlight/shadow outliers. | Separates lighting-free albedo `ρ` from surface normal `N`. |
| **6 · Integrate + verify** | Integrate the normal field into height, then re-render all four input views. | Residual images show where the model explains—or fails to explain—the measurements. |

### What comes out

| Kiwi albedo (lighting removed) | Kiwi OpenGL normal map | Kiwi integrated height |
|:--:|:--:|:--:|
| ![Recovered Kiwi sRGB albedo](docs/assets/kiwi-albedo.webp) | ![Recovered Kiwi OpenGL normal map](docs/assets/kiwi-normal.webp) | ![Integrated Kiwi height field](docs/assets/kiwi-height.webp) |

These three previews come from the same completed `Kiwi Leaf` session used by the relighting animation. The height panel is contrast-mapped from its actual 16-bit `height.png`; it is not the obsolete flat output from an earlier integration failure. LUMEN-PS exports `normal_gl.png`, `normal_dx.png`, linear and sRGB albedo, `height.png`, `alpha.png`, and ready-to-use RGBA albedo/normal maps. Full outputs remain 16-bit where useful; README images are compressed display copies only.

## Why it works

For a mostly matte (Lambertian) point, brightness under light `k` is approximately:

```text
Iₖ = ρ max(N · Lₖ, 0)
```

`Iₖ` is measured intensity, `ρ` is lighting-independent albedo, `N` is the unknown surface normal, and `Lₖ` is the calibrated light vector. Four rotations give four equations. The solver estimates the three components of `ρN`, normalizes that vector to obtain `N`, and keeps its length as `ρ`.

Two practical details make the result far better than a textbook least-squares solve:

- **Registration is both rigid and non-rigid.** Thin subjects can settle differently after rotation; silhouette distance fields and vein/texture structure guide the correction.
- **The solve is robust.** The brightest observation can be rejected to suppress specular glints, while pixels without at least three valid samples are excluded.

### Details hidden inside the simple idea

- **Four scans are deliberate.** Two intensity measurements cannot determine a three-component scaled normal. Three is the mathematical minimum; the fourth gives the solver room to reject one highlight or shadow and still remain determined.
- **The four lights lie on one cone.** Rotation changes azimuth, not elevation. A wrong elevation can still produce plausible-looking normals with systematically exaggerated or flattened relief, so LUMEN-PS fits elevation instead of guessing it.
- **The scanner is repeatable in ways a hand-held camera is not.** Focus, sensor path, working distance, and lamp-to-sensor geometry are mechanically fixed on every pass.
- **Color contains an extra botanical clue.** For leaves, red light penetrates tissue more deeply than blue. LUMEN-PS writes an `R − B` subsurface hint for QA or shading experiments, but deliberately does not let it drive the normal solve. This leaf-specific bonus is optional; the core reconstruction works on other diffuse subjects.
- **Normals and height answer different questions.** The normal map is the direct photometric result and preserves fine local slope. Height is obtained by integrating that slope field, so it is useful for relief but more sensitive to low-frequency drift and boundary conditions.
- **Sixteen-bit normals matter here.** The lamp–sensor baseline is small, so many recovered slopes differ by fine increments that would band more readily in an 8-bit deliverable.

### The reconstruction checks its own work

The recovered albedo and normals are rendered back under each fitted scanner light. If prediction and observation agree, the residual is dark and noise-like. Structured bright areas reveal misregistration, gloss, shadows, bad flat-fielding, or an incorrect light elevation.

| Observed scan | Predicted from recovered maps | Absolute residual |
|:--:|:--:|:--:|
| ![Observed scan](docs/qa/observed_scan0.jpg) | ![Predicted scan](docs/qa/predicted_scan0.jpg) | ![Absolute residual](docs/qa/residual_scan0.jpg) |

The included worked run has mean residuals of **0.0086–0.0092** on normalized linear intensity. See [`docs/`](docs/) for all four observed, predicted, and residual views plus mask agreement and rejection coverage.

## Will my printer/scanner work?

The printer portion is irrelevant; LUMEN-PS needs a compatible **flatbed scanner**. An all-in-one printer is suitable only if its flatbed meets these requirements.

| Requirement | Needed | Notes |
|:--|:--:|:--|
| Windows driver exposed through **WIA 2.0** | **Yes** | Current automatic capture is Windows/WIA. CLI processing can operate on suitable lossless scans captured another way. |
| Flatbed platen | **Yes** | Sheet-fed/ADF scanners cannot keep and rotate the subject on a common plane. |
| Directional, repeatable moving lamp | **Yes** | CIS units commonly have the useful lamp–sensor offset; verify by comparing highlights in 0° and 90° scans. |
| Manual or lockable brightness/contrast/color | **Strongly recommended** | All four scans need identical processing. Disable auto exposure/color, sharpening, and enhancement where the driver allows it. |
| 24-bit color, lossless PNG/TIFF | **Yes** | Never feed JPEG captures to the reconstruction. The tested HP WIA driver is 8-bit per channel, so LUMEN-PS linearizes sRGB in software. |
| 600 dpi or higher optical resolution | **Recommended** | 1200 dpi is supported and captures fine relief; 600 dpi is faster and often sufficient. |
| Enough platen clearance | **Yes** | The subject and fiducial card must lie flat without forcing the lid or loading the glass. |

Verified hardware: **HP LaserJet Professional M1130/M1132 MFP**, USB, WIA 2.0, at 75–1200 dpi. Other devices are expected to work when they meet the physical and capture requirements, but should pass the directional-light test before a full run.

## What scans well?

| Good candidates | Challenging / unsupported |
|:--|:--|
| Leaves and pressed flowers | Polished coins, foil, glossy plastic, wet surfaces |
| Paper, prints, cardboard, embossed stock | Transparent or translucent objects without a diffuse surface |
| Fabric, leather, thin bark, flat natural textures | Deep objects that cast strong shadows or cannot remain registered |
| Shallow relief and mostly matte craft materials | Anything sharp, heavy, hot, dirty, or likely to damage the platen |

The mathematical model assumes mostly diffuse reflection and shallow relief. Mild highlights are handled by outlier rejection; dominant reflections are not. When in doubt, protect the scanner and use a disposable matte carrier card—never press a rigid object down with the lid.

## Quick start

### Windows app

1. Install [Python 3.11 or newer](https://www.python.org/downloads/).
2. Connect the scanner and install its WIA driver.
3. Double-click `run.bat`.

The launcher creates an isolated `.venv`, installs dependencies, starts the local LUMEN-PS bench, and opens `http://127.0.0.1:8756`. In the app: create a session, capture four rotations, process, then drag the light around the interactive result.

### Command line

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m leafscan.cli capture --preview
python -m leafscan.cli capture --out scans\sample\k0.png --dpi 600 --color
# Rotate the subject and repeat for k1.png, k2.png, and k3.png.

python -m leafscan.cli run --scans scans\sample --out out
```

Optional references:

- `flat.png`: blank scan of the same matte carrier card for flat-field correction.
- `calib0.png`, `calib90.png`: single-face corrugated cardboard at 0°/90° for light-elevation fitting. Without them, elevation is self-calibrated from re-rendering residuals.

## Reading QA

- **Low, noise-like residual:** the model explains the scan; trust is high.
- **Bright veins or isolated sparkles:** specular leakage; tighten outlier rejection.
- **Large smooth gradient:** flat-field correction or light elevation is wrong.
- **Residual tracing the outline:** rigid/non-rigid registration failed.
- **Fewer than 3 valid views:** that pixel is underdetermined and excluded.

## Development

```powershell
python -m pytest tests -q
# or
python -m leafscan.cli selftest
```

The implementation is organized into `io`, `lights`, `align`, `calibrate`, `solve`, `integrate`, `outputs`, and `qa`, with the WIA capture bridge and FastAPI/WebGL UI alongside them. Tunable values live in [`leafscan/config.yaml`](leafscan/config.yaml). The detailed derivation and design rationale live in [`scanner_photometric_stereo_spec.md`](scanner_photometric_stereo_spec.md).

---

<div align="center"><sub>Built from the observation that a scanner is already a remarkably repeatable moving light stage—it only needed four turns and the right inverse problem.</sub></div>
