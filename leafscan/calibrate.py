"""Light calibration — fit (az0, el). Spec §7.

`el` (elevation) MUST be fitted, never hardcoded (§13). Getting it wrong doesn't
break the look — it systematically over/under-tilts the relief. Two methods:

  A. corrugated cardboard (preferred): the shading asymmetry across ridges of
     known profile is a direct function of `el`; the 0deg-vs-90deg pair fixes az0.
  B. self-calibration (fallback): treat (az0, el) as globals and minimize the
     photometric re-rendering residual over the leaf itself.
"""
from __future__ import annotations

import numpy as np

from .lights import light_directions
from .solve import photometric_solve, rerender

__all__ = [
    "self_calibrate",
    "calibrate_from_corrugated",
    "azimuth_for_light_side",
    "detect_light_side",
    "light_side_warning",
]


LIGHT_SIDES = ("auto", "left", "right")


def azimuth_for_light_side(side: str, fitted_azimuth: float = 90.0) -> float:
    """Select the requested member of the fitted 180-degree azimuth pair.

    Photometric self-calibration cannot distinguish ``az`` from ``az + 180``:
    negating both lateral light components negates the recovered lateral normal
    components and leaves the re-render exactly unchanged.  ``left`` names the
    branch centred on 90 degrees and ``right`` the branch centred on 270.
    """
    value = str(side).strip().lower()
    if value not in LIGHT_SIDES:
        raise ValueError(f"light.side must be one of {LIGHT_SIDES}, got {side!r}")
    azimuth = float(fitted_azimuth) % 360.0
    if value == "auto":
        return azimuth
    target = 90.0 if value == "left" else 270.0
    opposite = (azimuth + 180.0) % 360.0

    def angular_distance(a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)

    return float(min((azimuth, opposite), key=lambda a: angular_distance(a, target)))


def _edge_penumbra_ensemble(luma_stack, mask_stack, cfg):
    """Measure native-frame platen shadows at several bands in every scan.

    This is deliberately measured in each *native scanner frame*, before the
    rotated observations are aligned.  Surface texture therefore changes
    orientation between captures while a scanner-side penumbra stays fixed.
    Values are normalized by the local platen brightness.  Multiple edge-band
    widths and a per-scan vote make the result less sensitive to resolution,
    segmentation jitter, one unusually curled edge, or a single dirty capture.
    """
    import cv2

    max_dimension = max(128, int(cfg.get("max_dimension", 512)))
    minimum_columns = max(16, int(cfg.get("minimum_edge_columns", 32)))
    band_scales = cfg.get("edge_band_scales", [0.75, 1.0, 1.5])
    if not isinstance(band_scales, (list, tuple)) or not band_scales:
        band_scales = [1.0]
    measurements = []

    for scan_index, (luma, mask) in enumerate(zip(luma_stack, mask_stack)):
        image = np.asarray(luma, dtype=np.float32)
        subject = np.asarray(mask, dtype=bool)
        factor = min(1.0, max_dimension / max(image.shape))
        if factor < 0.999:
            size = (max(1, int(round(image.shape[1] * factor))),
                    max(1, int(round(image.shape[0] * factor))))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            subject = cv2.resize(subject.astype(np.uint8), size,
                                 interpolation=cv2.INTER_NEAREST).astype(bool)

        height, width = subject.shape
        for band_scale in band_scales:
            scale_value = max(0.5, float(band_scale))
            near_start = max(1, int(round(
                float(cfg.get("edge_near_start_px", 2)) * scale_value)))
            near_stop = max(near_start + 1, int(round(
                float(cfg.get("edge_near_stop_px", 6)) * scale_value)))
            far_start = max(near_stop + 1, int(round(
                float(cfg.get("edge_far_start_px", 10)) * scale_value)))
            far_stop = max(far_start + 1, int(round(
                float(cfg.get("edge_far_stop_px", 20)) * scale_value)))
            top_samples, bottom_samples = [], []
            for x in range(width):
                ys = np.flatnonzero(subject[:, x])
                if ys.size < near_stop:
                    continue
                top, bottom = int(ys[0]), int(ys[-1])
                if top >= far_stop:
                    near = float(np.median(
                        image[top - near_stop:top - near_start, x]))
                    far = float(np.median(
                        image[top - far_stop:top - far_start, x]))
                    top_samples.append((far - near) / max(abs(far), 0.05))
                if bottom + far_stop < height:
                    near = float(np.median(
                        image[bottom + near_start:bottom + near_stop, x]))
                    far = float(np.median(
                        image[bottom + far_start:bottom + far_stop, x]))
                    bottom_samples.append((far - near) / max(abs(far), 0.05))

            if (len(top_samples) >= minimum_columns
                    and len(bottom_samples) >= minimum_columns):
                top_value = float(np.median(top_samples))
                bottom_value = float(np.median(bottom_samples))
                measurements.append({
                    "scan": scan_index,
                    "scale": scale_value,
                    "top": top_value,
                    "bottom": bottom_value,
                    "ratio": bottom_value / max(abs(top_value), 1e-4),
                })

    top = float(np.median([m["top"] for m in measurements])) \
        if measurements else 0.0
    bottom = float(np.median([m["bottom"] for m in measurements])) \
        if measurements else 0.0
    ratio = bottom / max(abs(top), 1e-4)
    minimum_shadow = float(cfg.get("right_min_bottom_shadow", 0.012))
    minimum_ratio = float(cfg.get("right_min_shadow_ratio", 0.25))
    votes = [
        m["bottom"] >= minimum_shadow and m["ratio"] >= minimum_ratio
        for m in measurements
    ]
    scan_votes = []
    for scan_index in sorted({m["scan"] for m in measurements}):
        group = [vote for vote, measurement in zip(votes, measurements)
                 if measurement["scan"] == scan_index]
        scan_votes.append(float(np.mean(group)) >= 0.5)
    return {
        "top": top,
        "bottom": bottom,
        "ratio": ratio,
        "scan_count": len(scan_votes),
        "measurement_count": len(measurements),
        "vote_fraction": float(np.mean(votes)) if votes else 0.0,
        "scan_vote_fraction": float(np.mean(scan_votes)) if scan_votes else 0.0,
    }


def _edge_penumbra_cue(luma_stack, mask_stack, cfg):
    """Compatibility wrapper returning the original four-value cue tuple."""
    cue = _edge_penumbra_ensemble(luma_stack, mask_stack, cfg)
    return cue["top"], cue["bottom"], cue["ratio"], cue["scan_count"]


def _relief_shape_ensemble(normal, valid, cfg):
    """Measure reconstructed relief polarity with robust multi-scale methods.

    The binary ambiguity changes the sign of this value.  A positive skew is
    the conservative prior for scanned shallow-relief material: sparse ridges
    and raised details over a broader carrier.  Alongside skew, quantile-tail
    asymmetry measures whether troughs are deeper than peaks without allowing
    a few extreme pixels to dominate.  Neither is trusted without independent
    native-frame platen-shadow evidence.
    """
    import cv2
    from .integrate import frankot_chellappa

    field = np.asarray(normal, dtype=np.float32)
    region = np.asarray(valid, dtype=bool)
    max_dimension = max(128, int(cfg.get("max_dimension", 512)))
    factor = min(1.0, max_dimension / max(region.shape))
    if factor < 0.999:
        size = (max(8, int(round(region.shape[1] * factor))),
                max(8, int(round(region.shape[0] * factor))))
        field = cv2.resize(field, size, interpolation=cv2.INTER_AREA)
        region = cv2.resize(region.astype(np.float32), size,
                            interpolation=cv2.INTER_AREA) > 0.98
    magnitude = np.linalg.norm(field, axis=-1, keepdims=True)
    field = np.where(magnitude > 1e-8, field / np.maximum(magnitude, 1e-8), field)

    sigma = max(0.5, float(cfg.get("relief_highpass_sigma", 2.0)))
    radius = max(1, int(np.ceil(sigma)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    interior = cv2.erode(region.astype(np.uint8), kernel).astype(bool)
    minimum_pixels = max(256, int(cfg.get("minimum_relief_pixels", 1024)))
    if int(interior.sum()) < minimum_pixels:
        return {
            "skew": 0.0, "tail_asymmetry": 0.0,
            "scale_vote_fraction": 0.0, "pixels": int(interior.sum()),
        }

    height = frankot_chellappa(field, region, backend="cpu")
    sigma_scales = cfg.get("relief_sigma_scales", [0.75, 1.0, 2.0])
    if not isinstance(sigma_scales, (list, tuple)) or not sigma_scales:
        sigma_scales = [1.0]
    maximum_skew = float(cfg.get("right_max_relief_skew", -0.15))
    maximum_tail = float(cfg.get("right_max_relief_tail_asymmetry", -0.08))
    skews, tails, votes = [], [], []
    pixel_count = int(interior.sum())
    for sigma_scale in sigma_scales:
        level_sigma = max(0.5, sigma * max(0.5, float(sigma_scale)))
        highpass = height - cv2.GaussianBlur(height, (0, 0), level_sigma)
        values = highpass[interior].astype(np.float64)
        values -= np.median(values)
        deviation = float(values.std())
        if not np.isfinite(deviation) or deviation < 1e-10:
            continue
        standardized = values / deviation
        skew = float(np.mean(standardized ** 3))
        low, high = np.percentile(standardized, [2.0, 98.0])
        tail = float((high + low) / max(high - low, 1e-8))
        if not (np.isfinite(skew) and np.isfinite(tail)):
            continue
        skews.append(skew)
        tails.append(tail)
        votes.append(skew <= maximum_skew or tail <= maximum_tail)
    return {
        "skew": float(np.median(skews)) if skews else 0.0,
        "tail_asymmetry": float(np.median(tails)) if tails else 0.0,
        "scale_vote_fraction": float(np.mean(votes)) if votes else 0.0,
        "pixels": pixel_count,
    }


def _relief_convexity_cue(normal, valid, cfg):
    """Compatibility wrapper returning the original skew/pixel pair."""
    cue = _relief_shape_ensemble(normal, valid, cfg)
    return cue["skew"], cue["pixels"]


def detect_light_side(luma_stack, mask_stack, provisional_normal, valid,
                      config=None):
    """Resolve the remaining left/right branch with an evidence ensemble.

    Pure Lambertian observations cannot resolve the global concave/convex
    ambiguity.  Auto therefore changes the backward-compatible left prior only
    when native scanner-frame penumbra strength and cross-scan consistency
    support the opposite side, while robust skew, quantile-tail, and
    multi-scale statistics on reconstructed relief independently agree.
    If either cue is absent or they disagree, the result remains ``left`` and
    the explicit processing setting is available for unusual recessed objects.

    Returns ``(side, confidence, info)``.
    """
    cfg = dict(config or {})
    edge = _edge_penumbra_ensemble(luma_stack, mask_stack, cfg)
    relief = _relief_shape_ensemble(provisional_normal, valid, cfg)
    minimum_shadow = float(cfg.get("right_min_bottom_shadow", 0.012))
    minimum_ratio = float(cfg.get("right_min_shadow_ratio", 0.25))
    maximum_left_shadow = float(cfg.get(
        "left_max_bottom_shadow", minimum_shadow * 0.5))
    maximum_left_ratio = float(cfg.get(
        "left_max_shadow_ratio", minimum_ratio * 0.5))
    maximum_skew = float(cfg.get("right_max_relief_skew", -0.15))
    maximum_tail = float(cfg.get("right_max_relief_tail_asymmetry", -0.08))
    minimum_edge_votes = float(cfg.get("right_min_edge_vote_fraction", 0.25))
    minimum_edge_scan_votes = float(
        cfg.get("right_min_edge_scan_vote_fraction", 0.25))
    minimum_relief_votes = float(cfg.get("right_min_relief_vote_fraction", 0.50))

    penumbra_strength = (
        edge["scan_count"] >= 2
        and edge["bottom"] >= minimum_shadow
        and edge["ratio"] >= minimum_ratio
    )
    penumbra_consistency = (
        edge["vote_fraction"] >= minimum_edge_votes
        and edge["scan_vote_fraction"] >= minimum_edge_scan_votes
    )
    scanner_supports_right = penumbra_strength and penumbra_consistency
    # A complete absence of the opposite-edge penumbra in every scale and scan
    # is positive evidence for the original left-side rig, not merely a failed
    # right-side test.  Keep this deliberately stricter than the right branch:
    # a marginal right failure remains inconclusive when relief disagrees.
    scanner_decisively_supports_left = (
        edge["scan_count"] >= 2
        and edge["top"] >= minimum_shadow
        and edge["bottom"] <= maximum_left_shadow
        and edge["ratio"] <= maximum_left_ratio
        and edge["vote_fraction"] == 0.0
        and edge["scan_vote_fraction"] == 0.0
    )
    relief_skew = relief["skew"] <= maximum_skew
    relief_tail = relief["tail_asymmetry"] <= maximum_tail
    relief_multiscale = relief["scale_vote_fraction"] >= minimum_relief_votes
    relief_method_count = sum((relief_skew, relief_tail, relief_multiscale))
    relief_supports_right = relief_method_count >= 2
    side = "right" if scanner_supports_right and relief_supports_right else "left"

    shadow_margin = min(
        edge["bottom"] / max(minimum_shadow, 1e-8),
        edge["ratio"] / max(minimum_ratio, 1e-8),
    )
    edge_agreement = min(
        edge["vote_fraction"] / max(minimum_edge_votes, 1e-8),
        edge["scan_vote_fraction"] / max(minimum_edge_scan_votes, 1e-8),
    )
    relief_margin = max(
        (-relief["skew"]) / max(-maximum_skew, 1e-8),
        (-relief["tail_asymmetry"]) / max(-maximum_tail, 1e-8),
    )
    relief_agreement = relief["scale_vote_fraction"] / max(
        minimum_relief_votes, 1e-8)
    if side == "right":
        weakest_group = min(
            shadow_margin, edge_agreement, relief_margin, relief_agreement)
        confidence = float(np.clip(0.5 + 0.25 * (weakest_group - 1.0), 0.5, 1.0))
        reason = "ensemble-right"
    else:
        # Left is the compatibility fallback, so disagreement is reported as
        # deliberately low confidence instead of pretending the sign was seen.
        cues_agree_left = not scanner_supports_right and not relief_supports_right
        if scanner_decisively_supports_left:
            confidence = 0.80
            reason = "ensemble-left-edge-decisive"
        else:
            confidence = 0.80 if cues_agree_left else 0.35
            reason = ("ensemble-left" if cues_agree_left
                      else "ensemble-inconclusive-left-fallback")
    info = {
        "reason": reason,
        "top_shadow": edge["top"],
        "bottom_shadow": edge["bottom"],
        "shadow_ratio": edge["ratio"],
        "edge_scan_count": edge["scan_count"],
        "edge_measurement_count": edge["measurement_count"],
        "edge_vote_fraction": edge["vote_fraction"],
        "edge_scan_vote_fraction": edge["scan_vote_fraction"],
        "relief_skew": relief["skew"],
        "relief_tail_asymmetry": relief["tail_asymmetry"],
        "relief_scale_vote_fraction": relief["scale_vote_fraction"],
        "relief_pixels": relief["pixels"],
        "method_votes": {
            "penumbra_strength": bool(penumbra_strength),
            "penumbra_consistency": bool(penumbra_consistency),
            "penumbra_decisive_left": bool(scanner_decisively_supports_left),
            "relief_skew": bool(relief_skew),
            "relief_tail": bool(relief_tail),
            "relief_multiscale": bool(relief_multiscale),
        },
    }
    return side, float(confidence), info


def light_side_warning(mode, selected_side, suggested_side, confidence, info=None):
    """Return a non-blocking user warning for conflicting post-solve evidence."""
    mode = str(mode).strip().lower()
    selected = str(selected_side).strip().lower()
    suggested = str(suggested_side).strip().lower()
    confidence = float(confidence)
    if mode == "auto" and confidence < 0.5:
        return (
            "Automatic scanner light-side detection was inconclusive because "
            "the platen-edge and reconstructed-relief methods disagreed. Compare "
            "the Left and Right overrides if the normal or height map looks inverted."
        )
    if mode != "auto" and suggested != selected and confidence >= 0.50:
        return (
            f"The selected {selected}-side mode conflicts with automatic output "
            f"evidence favoring {suggested} (confidence {confidence:.0%}). "
            "The normal and height maps may be inverted; try the other light-side mode."
        )
    return None


# --------------------------------------------------------------------------- #
# Method B — self-calibration by re-render residual (spec §7 Method B)
# --------------------------------------------------------------------------- #
def _residual_for(az0, el, I_stack, thetas, valid_stack, rejection, min_surviving):
    L = light_directions(az0, el, thetas)
    out = photometric_solve(I_stack, L, valid_stack=valid_stack,
                            rejection=rejection, min_surviving=min_surviving)
    pred = rerender(out["normal"], out["albedo"], L)
    m = out["valid"]
    if not m.any():
        return np.inf
    return float(np.abs(pred - I_stack)[:, m].mean())


def self_calibrate(I_stack, thetas, valid_stack=None,
                   az0_seed=90.0, el_seed=30.0,
                   rejection="drop_brightest", min_surviving=3,
                   el_bounds=(10.0, 70.0), refine=True, verbose=True):
    """Fit (az0, el) by minimizing the re-render residual. Coarse grid + refine.

    Runs best on a downscaled stack (caller's responsibility). Returns
    (az0, el, residual, info).
    """
    az_grid = (az0_seed + np.arange(0, 360, 30)) % 360
    el_grid = np.linspace(el_bounds[0], el_bounds[1], 13)

    best = (az0_seed, el_seed, np.inf)
    for az0 in az_grid:
        for el in el_grid:
            r = _residual_for(az0, el, I_stack, thetas, valid_stack,
                              rejection, min_surviving)
            if r < best[2]:
                best = (float(az0), float(el), r)
    az0, el, r = best
    if verbose:
        print(f"[calib B] grid best: az0={az0:.1f} el={el:.1f} residual={r:.5f}")

    if refine:
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda p: _residual_for(p[0], p[1], I_stack, thetas, valid_stack,
                                        rejection, min_surviving),
                x0=[az0, el], method="Nelder-Mead",
                options={"xatol": 0.25, "fatol": 1e-5, "maxiter": 200},
            )
            if res.fun < r:
                az0, el, r = float(res.x[0]) % 360, float(res.x[1]), float(res.fun)
        except Exception as e:  # scipy optional at this step
            if verbose:
                print(f"[calib B] refine skipped: {e}")
    el = float(np.clip(el, *el_bounds))
    if verbose:
        print(f"[calib B] final: az0={az0:.2f} el={el:.2f} residual={r:.5f}")
    return az0, el, r, {"method": "selfcal", "residual": r}


# --------------------------------------------------------------------------- #
# Method A — corrugated cardboard (spec §7 Method A)
# --------------------------------------------------------------------------- #
def _ridge_profile(luma_masked, axis):
    """Average intensity profile perpendicular to ridges assumed along ``axis``.

    axis='x' => ridges run along X, profile varies along Y (collapse over X).
    """
    x = luma_masked.astype(np.float64)
    if axis == "x":
        prof = np.nanmean(np.where(x > 0, x, np.nan), axis=1)
    else:
        prof = np.nanmean(np.where(x > 0, x, np.nan), axis=0)
    prof = prof[np.isfinite(prof)]
    return prof


def _fit_el_from_profile(prof, az0, ridge_axis, el_bounds=(10, 70)):
    """Fit el by forward-rendering a sinusoidal ridge under L(az0,el)."""
    prof = prof - prof.mean()
    amp = np.abs(prof).max() + 1e-8
    prof = prof / amp
    n = len(prof)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # surface slope perpendicular to ridge ~ cos; normal tilts in that plane
    slope = np.cos(t)  # dz along the across-ridge axis
    from scipy.optimize import minimize_scalar

    def model(el):
        L = light_directions(az0, el, [0.0])[0]
        # normal of a ridge whose across-axis slope is `slope`
        if ridge_axis == "x":  # varies along Y -> Ny = -slope
            N = np.stack([np.zeros_like(slope), -slope, np.ones_like(slope)], 1)
        else:                  # varies along X -> Nx = -slope
            N = np.stack([-slope, np.zeros_like(slope), np.ones_like(slope)], 1)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        shade = np.clip(N @ L, 0, None)
        shade = shade - shade.mean()
        s = shade / (np.abs(shade).max() + 1e-8)
        # align phase by correlation
        return np.mean((s - prof) ** 2)

    res = minimize_scalar(model, bounds=el_bounds, method="bounded")
    return float(res.x), float(res.fun)


def calibrate_from_corrugated(luma0, mask0, luma90, mask90, az0_prior=90.0,
                              verbose=True):
    """Fit (az0, el) from 0deg/90deg corrugated-card scans.

    The axis with the stronger ridge asymmetry is the one the light tilts along
    (fixes az0); the asymmetry magnitude fixes el.
    """
    p0x = _ridge_profile(luma0 * mask0, "x")
    p0y = _ridge_profile(luma0 * mask0, "y")
    a_x = np.abs(p0x - p0x.mean()).max() if p0x.size else 0
    a_y = np.abs(p0y - p0y.mean()).max() if p0y.size else 0
    ridge_axis = "x" if a_x >= a_y else "y"
    # light tilts along the axis showing stronger asymmetry:
    az0 = 90.0 if ridge_axis == "x" else 0.0
    prof = p0x if ridge_axis == "x" else p0y
    el, err = _fit_el_from_profile(prof, az0, ridge_axis)
    if verbose:
        print(f"[calib A] ridge_axis={ridge_axis} az0={az0:.1f} el={el:.2f} err={err:.4f}")
    return float(az0), float(el), err, {"method": "cardboard", "ridge_axis": ridge_axis}
