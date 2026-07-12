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

__all__ = ["self_calibrate", "calibrate_from_corrugated"]


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
