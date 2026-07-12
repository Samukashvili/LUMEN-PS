"""Normal-field -> height integration via Frankot-Chellappa (spec §9.1).

FFT-based least-squares Poisson solve; robust to non-integrable noise. Only
scipy.fft needed. Expect low-frequency doming — optionally high-pass the result.
"""
from __future__ import annotations

import numpy as np
from scipy import fft

__all__ = ["frankot_chellappa", "integrate_height"]


def frankot_chellappa(normal: np.ndarray, mask=None) -> np.ndarray:
    """Integrate a normal map (H,W,3) into a height field (H,W).

    Slopes p = -Nx/Nz, q = -Ny/Nz. Returns height with zero mean over ``mask``.
    """
    N = normal.astype(np.float64)
    nz = N[..., 2]
    nz = np.where(np.abs(nz) < 1e-6, 1e-6, nz)
    p = -N[..., 0] / nz
    q = -N[..., 1] / nz
    if mask is not None:
        p = p * mask
        q = q * mask

    H, W = p.shape
    wx = 2 * np.pi * fft.fftfreq(W)
    wy = 2 * np.pi * fft.fftfreq(H)
    u, v = np.meshgrid(wx, wy)
    denom = u**2 + v**2
    denom[0, 0] = 1.0

    P = fft.fft2(p)
    Q = fft.fft2(q)
    Z = (-1j * u * P - 1j * v * Q) / denom
    Z[0, 0] = 0.0
    z = np.real(fft.ifft2(Z))

    if mask is not None and mask.any():
        z = z - z[mask].mean()
    else:
        z = z - z.mean()
    return z.astype(np.float32)


def integrate_height(normal, mask=None, highpass_sigma=0.0):
    """Frankot-Chellappa + optional high-pass to remove low-frequency drift."""
    z = frankot_chellappa(normal, mask)
    if highpass_sigma and highpass_sigma > 0:
        import cv2
        lp = cv2.GaussianBlur(z, (0, 0), highpass_sigma)
        z = z - lp
    return z
