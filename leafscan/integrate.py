"""Normal-field -> height integration via Frankot-Chellappa (spec §9.1).

FFT-based least-squares Poisson solve; robust to non-integrable noise. Only
scipy.fft needed. Expect low-frequency doming — optionally high-pass the result.
"""
from __future__ import annotations

import numpy as np
from scipy import fft

__all__ = ["frankot_chellappa", "integrate_height"]


def frankot_chellappa(normal: np.ndarray, mask=None, backend="auto") -> np.ndarray:
    """Integrate a normal map (H,W,3) into a height field (H,W).

    Slopes p = -Nx/Nz, q = -Ny/Nz. Returns height with zero mean over ``mask``.
    """
    from .compute import cupy_backend, release_gpu_memory
    cp = cupy_backend(backend)
    if cp is not None:
        try:
            N = cp.asarray(normal, dtype=cp.float32)
            nz = cp.where(cp.abs(N[..., 2]) < 1e-6, 1e-6, N[..., 2])
            p, q = -N[..., 0] / nz, -N[..., 1] / nz
            if mask is not None:
                m = cp.asarray(mask)
                p *= m
                q *= m
            H, W = p.shape
            wx = 2 * cp.pi * cp.fft.fftfreq(W)
            wy = 2 * cp.pi * cp.fft.fftfreq(H)
            u, v = cp.meshgrid(wx, wy)
            denom = u * u + v * v
            denom[0, 0] = 1.0
            Z = (-1j * u * cp.fft.fft2(p) - 1j * v * cp.fft.fft2(q)) / denom
            Z[0, 0] = 0.0
            z = cp.fft.ifft2(Z).real.astype(cp.float32)
            z -= z[m].mean() if mask is not None and bool(m.any()) else z.mean()
            return cp.asnumpy(z)
        except cp.cuda.memory.OutOfMemoryError:
            if str(backend).lower() == "gpu":
                raise
            # Continue into the lower-memory float32 CPU implementation.
            pass
        finally:
            release_gpu_memory(cp)

    # CPU fallback is float32 to halve peak RAM versus the old float64 path.
    N = normal.astype(np.float32)
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


def integrate_height(normal, mask=None, highpass_sigma=0.0, backend="auto"):
    """Frankot-Chellappa + optional high-pass to remove low-frequency drift."""
    z = frankot_chellappa(normal, mask, backend=backend)
    if highpass_sigma and highpass_sigma > 0:
        import cv2
        lp = cv2.GaussianBlur(z, (0, 0), highpass_sigma)
        z = z - lp
    return z
