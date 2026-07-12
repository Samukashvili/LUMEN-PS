"""Robust, vectorized photometric stereo — spec §8.

Lambertian model per pixel:  I_k = rho * (N . L_k),  k = 0..N-1.
Substitute g = rho * N (3 unknowns) => linear:  I = L g.
Solve per pixel by (weighted) least squares; rho = |g|, N = g/|g|.

Outlier rejection (spec §8.3): drop the brightest sample to kill glossy-vein
speculars; optionally also the darkest to kill shadows. Never drop below
``min_surviving`` samples.  Fully vectorized — no per-pixel Python loop.
"""
from __future__ import annotations

import numpy as np

__all__ = ["compute_weights", "photometric_solve", "rerender"]


def compute_weights(
    I: np.ndarray,
    valid: np.ndarray,
    rejection: str = "drop_brightest",
    min_surviving: int = 3,
) -> np.ndarray:
    """Return float weights in {0,1}, shape (N, P), after outlier rejection.

    I, valid: (N, P). ``valid`` marks which samples exist (in-mask, warped-in).
    """
    N = I.shape[0]
    w = valid.astype(np.float32).copy()
    n_valid = valid.sum(axis=0)

    if rejection in ("drop_brightest", "drop_brightest_and_darkest"):
        # brightest valid sample per pixel; drop only if enough will survive
        masked_hi = np.where(valid, I, -np.inf)
        b_idx = np.argmax(masked_hi, axis=0)
        can_drop = n_valid - 1 >= min_surviving
        pix = np.flatnonzero(can_drop)
        w[b_idx[pix], pix] = 0.0

    if rejection == "drop_brightest_and_darkest":
        surviving = w.sum(axis=0)
        masked_lo = np.where(w > 0, I, np.inf)
        d_idx = np.argmin(masked_lo, axis=0)
        can_drop = surviving - 1 >= min_surviving
        pix = np.flatnonzero(can_drop)
        w[d_idx[pix], pix] = 0.0

    return w


def photometric_solve(
    I_stack: np.ndarray,
    L: np.ndarray,
    valid_stack: np.ndarray | None = None,
    rejection: str = "drop_brightest",
    min_surviving: int = 3,
    ridge_lambda: float = 1e-6,
):
    """Solve for normals + albedo.

    Parameters
    ----------
    I_stack : (N, H, W) float32 linear luminance, aligned to the reference frame.
    L       : (N, 3) light directions from :mod:`lights`.
    valid_stack : (N, H, W) bool or None. None => all samples valid.

    Returns
    -------
    dict with:
      normal : (H, W, 3) float32, unit where valid (Nz >= 0), zero elsewhere.
      albedo : (H, W) float32.
      valid  : (H, W) bool — pixels actually solved.
      nsamples : (H, W) int8 — surviving sample count per pixel.
      weights  : (N, H, W) float32 — rejection weights (for QA).
    """
    N, H, W = I_stack.shape
    P = H * W
    I = I_stack.reshape(N, P).astype(np.float32)
    if valid_stack is None:
        valid = np.ones((N, P), dtype=bool)
    else:
        valid = valid_stack.reshape(N, P).astype(bool)

    w = compute_weights(I, valid, rejection, min_surviving)
    surviving = w.sum(axis=0)
    solvable = surviving >= min_surviving
    pix = np.flatnonzero(solvable)

    normal = np.zeros((P, 3), dtype=np.float32)
    albedo = np.zeros(P, dtype=np.float32)

    if pix.size:
        Ip = I[:, pix]                       # (N, M)
        wp = w[:, pix]                        # (N, M)
        LL = np.einsum("ni,nj->nij", L, L)   # (N, 3, 3)
        A = np.einsum("nm,nij->mij", wp, LL) # (M, 3, 3)
        b = np.einsum("nm,ni->mi", wp * Ip, L)  # (M, 3)
        A = A + ridge_lambda * np.eye(3, dtype=A.dtype)[None]
        g = np.linalg.solve(A, b[..., None])[..., 0]  # (M, 3)

        rho = np.linalg.norm(g, axis=1)
        good = rho > 1e-8
        n = np.zeros_like(g)
        n[good] = g[good] / rho[good, None]
        # Enforce a viewer-facing normal (+Z out of the glass).
        flip = n[:, 2] < 0
        n[flip] *= -1.0

        normal[pix] = n
        albedo[pix] = rho

    return {
        "normal": normal.reshape(H, W, 3),
        "albedo": albedo.reshape(H, W),
        "valid": solvable.reshape(H, W),
        "nsamples": surviving.reshape(H, W).astype(np.int8),
        "weights": w.reshape(N, H, W),
    }


def rerender(normal: np.ndarray, albedo: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Re-render the N lighting conditions: I_pred_k = rho * max(N.L_k, 0)."""
    ndotl = np.einsum("hwc,nc->nhw", normal, L)
    return np.clip(ndotl, 0.0, None) * albedo[None]
