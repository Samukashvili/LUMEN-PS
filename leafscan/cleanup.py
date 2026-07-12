"""Misregistration repair — detect and fix pixels whose samples violate the
Lambertian consensus.

Manual re-flattening between rotations can never reproduce the exact same leaf
pose, so after rigid + non-rigid alignment small patches (mostly along veins and
creases, where image gradients are strong) remain locally misaligned by a few
pixels in one or two of the scans. Those samples drag the solved normal
sideways — speckle clusters that don't sit "flush" with the surrounding field.

Why the obvious detectors fail on real data (verified on the karalioki-leaf
session):

* Per-pixel residual thresholds alone can't work: leaves are not Lambertian
  (gloss, subsurface), so the all-samples residual has a huge floor (median
  leave-one-out rel ~ 0.3). And with 4 symmetric cone lights the residual
  space is ONE-dimensional (null vector ~ (1,-1,1,-1)): any single-sample
  error yields equal-magnitude residuals on every scan, so WHICH scan is bad
  is unidentifiable from one pixel alone.
* Normal-space flushness alone misses coherent artifact blobs (up to ~100 px
  wide) that agree with their own local median.

So this module combines them:

1. DETECT: suspect = not flush with the 5-px median normal field OR extreme
   leave-one-out residual (catches coherent blobs).
2. IDENTIFY + FIX: solve every leave-one-out candidate and keep the one whose
   normal agrees best with a trusted-neighborhood reference (Gaussian blur of
   non-suspect normals) — the offending scan is identified spatially, which is
   the only sound way (see above). Genuine sharp features keep their original
   solution because no candidate improves them.
3. FILL: pixels still far from the reference AND physically inconsistent
   (>= 2 bad samples of 4 — unrecoverable) are returned in ``fill`` for
   inpainting. Consistent-but-sharp pixels are never filled.
"""
from __future__ import annotations

import cv2
import numpy as np

from .solve import photometric_solve, rerender

__all__ = ["relative_residual", "residual_repair", "inpaint_field"]


def relative_residual(I_stack, normal, albedo, L, albedo_floor=0.02):
    """Per-sample residual normalized by albedo: (N,H,W) float32.

    Misregistration error scales with local image contrast, which scales with
    albedo — normalizing by albedo makes one threshold work across the bright
    lamina and dark veins without blowing up noise in near-black pixels
    (``albedo_floor``).
    """
    pred = rerender(normal, albedo, L)
    pred -= I_stack
    np.abs(pred, out=pred)
    pred /= np.maximum(albedo, albedo_floor)[None]
    return pred.astype(np.float32, copy=False)


def _unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-6)


def _median_angle(normal):
    """Angle (deg) of each normal to the 5-px component-median field."""
    med = np.stack([cv2.medianBlur(normal[..., c], 5) for c in range(3)], axis=2)
    med = _unit(med)
    return np.degrees(np.arccos(np.clip((normal * med).sum(axis=2), -1, 1)))


def _trusted_reference(normal, good, sigma):
    """Smooth consensus normal field interpolated from trusted pixels only."""
    H, W = good.shape
    g = good.astype(np.float32)
    ng = normal * g[..., None]
    num = cv2.GaussianBlur(ng, (0, 0), sigma)
    den = cv2.GaussianBlur(g, (0, 0), sigma)
    hole = den < 1e-4
    if hole.any():
        # Fill uncovered support on a small copy. The old implementation kept
        # doubling Gaussian sigma up to 4096; on 1200-dpi scans that could spend
        # tens of minutes convolving millions of pixels. A nearest-supported
        # fill is linear-time and is subsequently normalized/interpolated.
        from scipy.ndimage import distance_transform_edt
        sw, sh = max(1, W // 4), max(1, H // 4)
        ns = cv2.resize(ng, (sw, sh), interpolation=cv2.INTER_AREA)
        gs = cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA)
        s = max(sigma / 4.0, 1.0)
        nums = cv2.GaussianBlur(ns, (0, 0), s)
        dens = cv2.GaussianBlur(gs, (0, 0), s)
        supported = dens >= 1e-5
        if supported.any():
            iy, ix = distance_transform_edt(~supported, return_distances=False,
                                            return_indices=True)
            wide = nums / np.maximum(dens, 1e-6)[..., None]
            wide[~supported] = wide[iy[~supported], ix[~supported]]
        else:
            wide = cv2.resize(normal, (sw, sh), interpolation=cv2.INTER_AREA)
        wide = cv2.resize(wide, (W, H), interpolation=cv2.INTER_LINEAR)
        num[hole] = wide[hole]
        den[hole] = 1.0
    return _unit(num / np.maximum(den, 1e-6)[..., None])


def inpaint_field(arr, mask, radius=5):
    """Telea-inpaint ``mask`` pixels of a float field (per channel, via 8-bit).

    8-bit is plenty here: filled regions are smooth interpolations of
    surroundings, and 1/255 on a normal component is < 0.5 deg.
    """
    m8 = mask.astype(np.uint8) * 255
    out = arr.copy()
    chans = arr.shape[2] if arr.ndim == 3 else 1
    for c in range(chans):
        x = arr[..., c] if arr.ndim == 3 else arr
        lo, hi = float(x.min()), float(x.max())
        scale = max(hi - lo, 1e-6)
        u8 = np.clip((x - lo) / scale * 255, 0, 255).astype(np.uint8)
        f = cv2.inpaint(u8, m8, radius, cv2.INPAINT_TELEA).astype(np.float32)
        f = f / 255 * scale + lo
        if arr.ndim == 3:
            out[..., c] = np.where(mask, f, arr[..., c])
        else:
            out = np.where(mask, f, arr)
    return out


def residual_repair(
    I_stack,
    L,
    first,
    valid_stack=None,
    flush_deg=12.0,
    hard_rel=0.7,
    improve_deg=5.0,
    fill_flush_deg=20.0,
    fill_rel=0.5,
    ref_sigma=6.0,
    albedo_floor=0.02,
    min_surviving=3,
    ridge_lambda=1e-6,
    backend=None,
):
    """Detect + repair misregistered pixels. Returns (out, repaired, fill).

    Parameters
    ----------
    first : dict from :func:`photometric_solve` — the production solution
        (e.g. drop_brightest); kept as-is wherever nothing better is found.
    flush_deg : suspect gate — angle to the 5-px median normal field.
    hard_rel : suspect gate — leave-one-out relative residual that flags a
        pixel regardless of flushness (coherent artifact blobs).
    improve_deg : a leave-one-out candidate must beat the current normal's
        agreement with the trusted reference by this margin to be accepted.
    fill_flush_deg / fill_rel : a pixel lands in ``fill`` only if it is BOTH
        still this far from the trusted reference AND this inconsistent —
        sharp-but-consistent geometry is never filled.

    Returns
    -------
    out : solve dict (same keys as :func:`photometric_solve`); equals ``first``
        except at repaired pixels.
    repaired : (H,W) bool — offender dropped, re-solved from consistent scans.
    fill : (H,W) bool — unrecoverable; caller should inpaint (keep valid!).
    """
    N, H, W = I_stack.shape
    P = H * W
    if valid_stack is None:
        valid = np.ones((N, H, W), dtype=bool)
    else:
        valid = valid_stack.astype(bool)
    zeros = np.zeros((H, W), dtype=bool)
    normal0 = first["normal"]

    # ---- 1. detect ----
    base = photometric_solve(I_stack, L, valid_stack=valid, rejection="none",
                             min_surviving=min_surviving, ridge_lambda=ridge_lambda,
                             backend=backend)
    rel = relative_residual(I_stack, base["normal"], base["albedo"], L, albedo_floor)
    base_valid = base["valid"]
    del base  # keep peak memory down: only the residual + validity are needed
    n_valid = valid.sum(axis=0)
    # rescale the raw residual to leave-one-out units: an LS outlier keeps only
    # (n-3)/n of its error in its own residual (outlier masking)
    loo_gain = (np.maximum(n_valid, 1) / np.maximum(n_valid - 3, 1e-6)).astype(np.float32)
    rel *= valid
    worst = rel.max(axis=0)
    del rel
    worst *= np.where(n_valid > 3, loo_gain, 0.0).astype(np.float32)

    ang = _median_angle(normal0)
    solvable = first["valid"] & base_valid & (n_valid - 1 >= min_surviving)
    suspect = solvable & ((ang > flush_deg) | (worst > hard_rel))
    if not suspect.any():
        return first, zeros, zeros

    # ---- 2. identify the offending scan spatially ----
    ref = _trusted_reference(normal0, first["valid"] & ~suspect, ref_sigma)

    flat = np.flatnonzero(suspect.ravel())
    M = flat.size
    I_sub = I_stack.reshape(N, P)[:, flat][..., None]          # (N, M, 1)
    v_sub = valid.reshape(N, P)[:, flat][..., None]
    ref_sub = ref.reshape(P, 3)[flat]

    first_dot = (normal0.reshape(P, 3)[flat] * ref_sub).sum(axis=1)
    best_dot = first_dot.copy()
    best_k = np.full(M, -1, dtype=np.int64)
    best_n = normal0.reshape(P, 3)[flat].copy()
    best_a = first["albedo"].reshape(P)[flat].copy()
    for k in range(N):
        v_k = v_sub.copy()
        v_k[k] = False
        cand = photometric_solve(I_sub, L, valid_stack=v_k, rejection="none",
                                 min_surviving=min_surviving,
                                 ridge_lambda=ridge_lambda, backend=backend)
        nk = cand["normal"][:, 0, :]
        ok = cand["valid"][:, 0] & v_sub[k, :, 0]              # dropped sample existed
        d = np.where(ok, (nk * ref_sub).sum(axis=1), -2.0)
        better = d > best_dot
        best_dot = np.where(better, d, best_dot)
        best_k = np.where(better, k, best_k)
        best_n[better] = nk[better]
        best_a[better] = cand["albedo"][:, 0][better]

    ang_first = np.degrees(np.arccos(np.clip(first_dot, -1, 1)))
    ang_best = np.degrees(np.arccos(np.clip(best_dot, -1, 1)))
    accept = (best_k >= 0) & (ang_first - ang_best > improve_deg)

    # ---- 3. merge + decide what is still unrecoverable ----
    out = {k: v.copy() for k, v in first.items()}
    rp = flat[accept]
    out["normal"].reshape(P, 3)[rp] = best_n[accept]
    out["albedo"].reshape(P)[rp] = best_a[accept]
    w = out["weights"].reshape(N, P)
    w[:, rp] = valid.reshape(N, P)[:, rp]
    w[best_k[accept], rp] = 0.0
    out["nsamples"].reshape(P)[rp] = w[:, rp].sum(axis=0).astype(np.int8)

    final_dot = (out["normal"].reshape(P, 3)[flat] * ref_sub).sum(axis=1)
    bad = (final_dot < np.cos(np.deg2rad(fill_flush_deg))) & \
          (worst.reshape(P)[flat] > fill_rel)
    fill = np.zeros(P, dtype=bool)
    fill[flat[bad]] = True
    fill = fill.reshape(H, W)
    repaired = np.zeros(P, dtype=bool)
    repaired[rp] = True
    repaired = repaired.reshape(H, W) & ~fill
    return out, repaired, fill
