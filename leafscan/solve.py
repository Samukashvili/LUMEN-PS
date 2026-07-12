"""Robust, vectorized photometric stereo — spec §8.

Lambertian model per pixel:  I_k = rho * (N . L_k),  k = 0..N-1.
Substitute g = rho * N (3 unknowns) => linear:  I = L g.
Solve per pixel by (weighted) least squares; rho = |g|, N = g/|g|.

Outlier rejection (spec §8.3): drop the brightest sample to kill glossy-vein
speculars; optionally also the darkest to kill shadows. Never drop below
``min_surviving`` samples.  Fully vectorized — no per-pixel Python loop.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

__all__ = ["compute_weights", "photometric_solve", "rerender"]


def _array_backend(requested: str | None):
    """Return NumPy or optional CuPy without making CUDA a hard dependency."""
    name = (requested or os.environ.get("LEAFSCAN_COMPUTE", "auto")).lower()
    if name not in ("auto", "cpu", "gpu"):
        raise ValueError("compute backend must be 'auto', 'cpu', or 'gpu'")
    # The binary-pattern CPU path solves at most 16 tiny systems and is faster
    # than CUDA transfer/kernel startup even at 25 MP. GPU remains forceable.
    if name in ("auto", "cpu"):
        return np, False
    from .compute import cupy_backend
    cp = cupy_backend(name)
    if cp is not None:
        return cp, True
    if name == "gpu":
        warnings.warn("GPU solve unavailable; using optimized CPU solve")
    return np, False


def _solve_binary_patterns(I, L, w, solvable, ridge_lambda, xp):
    """Solve binary-weight systems once per distinct sample pattern.

    Four captures produce at most 16 systems. Grouping pixels by pattern avoids
    constructing and factorizing one 3x3 matrix per pixel, which dominated
    full-resolution runs after the cleanup pass added another solve.
    """
    N, P = I.shape
    codes = xp.zeros(P, dtype=xp.uint16)
    for k in range(N):
        codes |= (w[k] > 0).astype(xp.uint16) << k
    g = xp.zeros((P, 3), dtype=xp.float32)
    eye = xp.eye(3, dtype=xp.float32)
    for code in range(1, 1 << N):
        pix = xp.flatnonzero(solvable & (codes == code))
        if not pix.size:
            continue
        keep = xp.asarray([(code >> k) & 1 for k in range(N)], dtype=xp.bool_)
        Lk = L[keep]
        inverse = xp.linalg.inv(Lk.T @ Lk + ridge_lambda * eye)
        # (M,N) @ (N,3): no per-pixel matrices and bounded temporary memory.
        b = (w[:, pix] * I[:, pix]).T @ L
        g[pix] = b @ inverse.T
    return g


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
    weights: np.ndarray | None = None,
    backend: str | None = None,
):
    """Solve for normals + albedo.

    Parameters
    ----------
    I_stack : (N, H, W) float32 linear luminance, aligned to the reference frame.
    L       : (N, 3) light directions from :mod:`lights`.
    valid_stack : (N, H, W) bool or None. None => all samples valid.
    weights : (N, H, W) float or None. If given, use these per-sample weights
        directly and skip the ``rejection`` heuristic (caller-driven rejection,
        e.g. residual-based misregistration cleanup).

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
    xp, on_gpu = _array_backend(backend)
    L = xp.asarray(L, dtype=xp.float32)
    I = xp.asarray(I_stack.reshape(N, P), dtype=xp.float32)
    if valid_stack is None:
        valid = xp.ones((N, P), dtype=xp.bool_)
    else:
        valid = xp.asarray(valid_stack.reshape(N, P), dtype=xp.bool_)

    if weights is not None:
        w = xp.asarray(weights.reshape(N, P), dtype=xp.float32) * valid
    else:
        # Weight selection is tiny (four rows) and compute_weights is also used
        # independently by callers, so retain one canonical NumPy implementation.
        w = xp.asarray(compute_weights(np.asarray(I_stack).reshape(N, P),
                                       np.asarray(valid_stack).reshape(N, P)
                                       if valid_stack is not None else np.ones((N, P), bool),
                                       rejection, min_surviving))
    surviving = w.sum(axis=0)
    solvable = surviving >= min_surviving
    normal = xp.zeros((P, 3), dtype=xp.float32)
    albedo = xp.zeros(P, dtype=xp.float32)

    if bool(solvable.any()):
        binary = bool(xp.all((w == 0) | (w == 1)))
        if binary:
            g = _solve_binary_patterns(I, L, w, solvable, ridge_lambda, xp)
        else:
            pix = xp.flatnonzero(solvable)
            LL = xp.einsum("ni,nj->nij", L, L)
            A = xp.einsum("nm,nij->mij", w[:, pix], LL)
            b = xp.einsum("nm,ni->mi", w[:, pix] * I[:, pix], L)
            A += ridge_lambda * xp.eye(3, dtype=xp.float32)[None]
            g = xp.zeros((P, 3), dtype=xp.float32)
            g[pix] = xp.linalg.solve(A, b[..., None])[..., 0]

        rho = xp.linalg.norm(g, axis=1)
        good = rho > 1e-8
        n = xp.zeros_like(g)
        n[good] = g[good] / rho[good, None]
        # Enforce a viewer-facing normal (+Z out of the glass).
        flip = n[:, 2] < 0
        n[flip] *= -1.0

        normal[solvable] = n[solvable]
        albedo[solvable] = rho[solvable]

    if on_gpu:
        normal, albedo, solvable, surviving, w = [xp.asnumpy(x) for x in
                                                  (normal, albedo, solvable, surviving, w)]
        # CuPy caches allocations by default. Returning them here prevents the
        # 4 GB laptop GPU from staying full while integration/viewer starts.
        from .compute import release_gpu_memory
        release_gpu_memory(xp)

    return {
        "normal": normal.reshape(H, W, 3),
        "albedo": albedo.reshape(H, W),
        "valid": solvable.reshape(H, W),
        "nsamples": surviving.reshape(H, W).astype(np.int8),
        "weights": w.reshape(N, H, W),
    }


def rerender(normal: np.ndarray, albedo: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Re-render the N lighting conditions: I_pred_k = rho * max(N.L_k, 0)."""
    ndotl = np.einsum("hwc,nc->nhw", normal, np.asarray(L, dtype=np.float32))
    return np.clip(ndotl, 0.0, None) * albedo[None]
