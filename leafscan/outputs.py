"""Encoding & export — spec §9.3. Tangent-space normal maps + albedo + height."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import linear_to_srgb

__all__ = ["encode_normal", "save_png", "write_outputs"]


def encode_normal(normal: np.ndarray, directx: bool = False) -> np.ndarray:
    """Encode a unit normal (H,W,3) to [0,1] RGB. DirectX flips green (-Y)."""
    n = normal.astype(np.float32).copy()
    if directx:
        n[..., 1] = -n[..., 1]
    return n * 0.5 + 0.5


def _to_uint(img01, bits):
    maxv = (1 << bits) - 1
    a = np.clip(img01, 0.0, 1.0) * maxv
    return a.astype(np.uint16 if bits > 8 else np.uint8)


def save_png(path, img01, bits=8):
    """Save a [0,1] float image as an 8- or 16-bit PNG (OpenCV backend).

    Handles grayscale, RGB, and RGBA. OpenCV reliably writes 16-bit multi-channel
    PNGs (Pillow/imageio do not).
    """
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_uint(img01, bits)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[..., ::-1]                 # RGB -> BGR
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., [2, 1, 0, 3]]         # RGBA -> BGRA
    cv2.imwrite(str(path), arr)
    return path


def write_outputs(out_dir, normal, albedo_rgb, valid, height=None, alpha=None,
                  normal_bits=16, albedo_linear=True, albedo_srgb=True):
    """Write the full deliverable set (spec §1, §9.3). Returns list of paths.

    ``alpha`` (H,W) in [0,1] is the leaf opacity/silhouette. When provided it is
    saved as ``alpha.png`` and baked into RGBA copies for direct use on a
    transparent plane.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    # zero-out invalid regions in the encoded normal to flat (0,0,1)
    flat = np.zeros_like(normal)
    flat[..., 2] = 1.0
    n = np.where(valid[..., None], normal, flat)

    written.append(save_png(out / "normal_gl.png", encode_normal(n, directx=False), normal_bits))
    written.append(save_png(out / "normal_dx.png", encode_normal(n, directx=True), normal_bits))

    alb_lin = np.clip(albedo_rgb, 0, 1)
    alb_srgb = linear_to_srgb(alb_lin)
    if albedo_linear:
        written.append(save_png(out / "albedo.png", alb_lin, 16))
    if albedo_srgb:
        written.append(save_png(out / "albedo_srgb.png", alb_srgb, 8))

    if alpha is not None:
        a = np.clip(alpha, 0, 1).astype(np.float32)
        written.append(save_png(out / "alpha.png", a, 8))
        # RGBA convenience copies (premultiply not applied — straight alpha)
        written.append(save_png(out / "albedo_srgb_rgba.png",
                                np.dstack([alb_srgb, a]), 8))
        written.append(save_png(out / "normal_gl_rgba.png",
                                np.dstack([encode_normal(n, directx=False), a]), 8))

    if height is not None:
        h = height.copy()
        m = valid
        if m.any():
            lo, hi = np.percentile(h[m], [1, 99])
        else:
            lo, hi = h.min(), h.max()
        hn = np.clip((h - lo) / (hi - lo + 1e-8), 0, 1)
        written.append(save_png(out / "height.png", hn * valid, 16))
    return written
