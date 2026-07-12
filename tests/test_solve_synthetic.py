"""Spec priority 1: render a known bump under 4 known cone lights, solve, recover it.

Proves the math before any real scan. Uses *spatially varying* albedo (veins),
because that is exactly what a 2-scan solver would turn into fake geometry.
"""
import numpy as np
import pytest

from leafscan.lights import light_directions, nominal_thetas
from leafscan.solve import photometric_solve, rerender


def _synthetic_surface(H=96, W=128, seed=0):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    xs = (xs - W / 2) / (W / 4)
    ys = (ys - H / 2) / (H / 4)
    # gentle bump + ripples => slopes stay small so nothing self-shadows
    z = 0.6 * np.exp(-(xs**2 + ys**2)) + 0.15 * np.sin(1.5 * xs) * np.cos(1.7 * ys)
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    N = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    # leaf-like albedo: bright interveinal tissue with dark stripes (veins)
    rho = 0.7 + 0.25 * (np.sin(3.0 * xs) > 0.4).astype(np.float64)
    rho -= 0.3 * (np.abs(np.sin(6.0 * ys)) > 0.95)
    rho = np.clip(rho, 0.15, 1.0)
    return N.astype(np.float32), rho.astype(np.float32)


def _angular_error_deg(Na, Nb, mask):
    d = np.clip(np.sum(Na * Nb, axis=-1), -1, 1)
    return np.rad2deg(np.arccos(d))[mask]


def test_recovers_known_surface():
    N_true, rho_true = _synthetic_surface()
    L = light_directions(90.0, 35.0, nominal_thetas())
    I = np.clip(np.einsum("hwc,nc->nhw", N_true, L), 0, None) * rho_true[None]
    out = photometric_solve(I, L, rejection="none", min_surviving=3)
    m = out["valid"]
    ang = _angular_error_deg(out["normal"], N_true, m)
    assert m.all()
    assert ang.mean() < 0.5 and ang.max() < 3.0
    rel = np.abs(out["albedo"] - rho_true)[m] / rho_true[m]
    assert rel.mean() < 0.01


def test_drop_brightest_kills_specular_outlier():
    N_true, rho_true = _synthetic_surface()
    L = light_directions(90.0, 35.0, nominal_thetas())
    I = np.clip(np.einsum("hwc,nc->nhw", N_true, L), 0, None) * rho_true[None]
    # inject a big specular spike into scan 2 over a patch (glossy vein glint)
    I_bad = I.copy()
    I_bad[2, 30:50, 40:70] += 1.5

    ang_none = _angular_error_deg(
        photometric_solve(I_bad, L, rejection="none")["normal"], N_true,
        np.ones(N_true.shape[:2], bool),
    )
    ang_drop = _angular_error_deg(
        photometric_solve(I_bad, L, rejection="drop_brightest")["normal"], N_true,
        np.ones(N_true.shape[:2], bool),
    )
    # rejection must substantially reduce the damage from the outlier
    assert ang_drop.max() < ang_none.max()
    assert ang_drop.mean() < 0.6


def test_rerender_residual_is_small_on_clean_data():
    N_true, rho_true = _synthetic_surface()
    L = light_directions(90.0, 35.0, nominal_thetas())
    I = np.clip(np.einsum("hwc,nc->nhw", N_true, L), 0, None) * rho_true[None]
    out = photometric_solve(I, L, rejection="none")
    pred = rerender(out["normal"], out["albedo"], L)
    assert np.abs(pred - I).mean() < 1e-3


def test_min_surviving_marks_invalid():
    N_true, rho_true = _synthetic_surface(H=16, W=16)
    L = light_directions(90.0, 35.0, nominal_thetas())
    I = np.clip(np.einsum("hwc,nc->nhw", N_true, L), 0, None) * rho_true[None]
    valid = np.ones((4, 16, 16), bool)
    valid[:3, 0, 0] = False  # only 1 valid sample at (0,0)
    out = photometric_solve(I, L, valid_stack=valid, min_surviving=3)
    assert not out["valid"][0, 0]
    assert out["valid"][8, 8]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_residual_repair_fixes_local_misregistration():
    """Emulate a locally misaligned scan (manual re-flattening mismatch): shift a
    patch of one scan by a few px. Repair must recover the true normals there."""
    from leafscan.cleanup import residual_repair

    N_true, rho_true = _synthetic_surface()
    L = light_directions(90.0, 35.0, nominal_thetas())
    I = np.clip(np.einsum("hwc,nc->nhw", N_true, L), 0, None) * rho_true[None]
    I_bad = I.copy()
    # scan 1: patch shifted 3 px right (misregistration), where albedo varies
    I_bad[1, 20:60, 30:80] = I[1, 20:60, 27:77]

    first = photometric_solve(I_bad, L, rejection="drop_brightest")
    # synthetic data is noise-free and perfectly Lambertian, so the gates can be
    # much tighter than the real-scan defaults
    out, repaired, fill = residual_repair(I_bad, L, first, flush_deg=4.0,
                                          hard_rel=0.10, improve_deg=1.0,
                                          fill_flush_deg=8.0, fill_rel=0.10)

    patch = np.zeros(N_true.shape[:2], bool)
    patch[20:60, 30:80] = True
    ang_first = _angular_error_deg(first["normal"], N_true, patch)
    ang_fixed = _angular_error_deg(out["normal"], N_true, patch & out["valid"] & ~fill)

    # detection: flagged pixels live (almost) only inside the shifted patch
    flagged = repaired | fill
    assert flagged[patch].sum() > 0
    assert flagged[~patch].sum() <= 0.001 * (~patch).sum()
    # repair: error inside the patch drops dramatically for surviving pixels
    assert ang_first.max() > 10.0
    assert ang_fixed.max() < 2.0
    # untouched pixels keep the original solution
    away = ~patch.copy()
    away[15:65, 22:85] = False
    d = _angular_error_deg(out["normal"], first["normal"], away)
    assert d.max() < 0.05  # float32 arccos noise on identical vectors
