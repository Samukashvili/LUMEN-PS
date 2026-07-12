"""Load, linearize, dark/flat-field — spec §5.

Everything internal is float32, linear, [0, 1]. Scanner output is 8-bit sRGB on
this hardware (verified), so we undo the sRGB transfer function to recover
intensity-proportional-to-light, which photometric stereo requires.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "load_image_linear",
    "to_luminance",
    "subsurface_hint",
    "flat_field_correct",
    "estimate_black_level",
    "downscale",
]


def _read_raw(path):
    """Read an image file to a numpy array in its native dtype/shape."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(str(path))
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Undo the sRGB EOTF. Input/output in [0,1] float."""
    x = np.asarray(x, dtype=np.float32)
    a = 0.055
    lin = np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)
    return lin.astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Apply the sRGB OETF. Input/output in [0,1] float."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    a = 0.055
    srgb = np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1 / 2.4) - a)
    return srgb.astype(np.float32)


def load_image_linear(path, input_is_srgb: bool = True, scale: float = 1.0):
    """Load an image as float32 linear RGB in [0,1].

    Returns (rgb_linear[H,W,3], meta). Grayscale is broadcast to 3 channels.
    ``meta`` records native depth and distinct-level count so the caller can log
    whether 16-bit data is genuine or padded 8-bit (spec §4.2 / §12.0).
    """
    raw = _read_raw(path)
    native_dtype = raw.dtype
    if raw.dtype == np.uint8:
        maxv = 255.0
    elif raw.dtype == np.uint16:
        maxv = 65535.0
    else:
        maxv = float(raw.max()) if raw.max() > 0 else 1.0

    distinct = int(np.unique(raw).size)
    x = raw.astype(np.float32) / maxv

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.shape[-1] == 4:
        x = x[..., :3]

    lin = srgb_to_linear(x) if input_is_srgb else x
    if scale != 1.0:
        lin = downscale(lin, scale)
    meta = {
        "native_dtype": str(native_dtype),
        "distinct_levels": distinct,
        "genuine_high_bit": distinct > 256,
        "shape": lin.shape,
    }
    return lin.astype(np.float32), meta


def to_luminance(rgb_linear: np.ndarray, weights=(0.2126, 0.7152, 0.0722)) -> np.ndarray:
    """Linear luminance channel used for the photometric solve (spec §5.2)."""
    w = np.asarray(weights, dtype=np.float32)
    return (rgb_linear * w).sum(axis=-1).astype(np.float32)


def subsurface_hint(rgb_linear: np.ndarray) -> np.ndarray:
    """R - B difference (spec §5.2): red penetrates leaf tissue, blue stays surface.

    Returned normalized to [0,1] for saving as qa/subsurface_hint. Never drives
    the normals.
    """
    d = rgb_linear[..., 0] - rgb_linear[..., 2]
    lo, hi = np.percentile(d, [1, 99])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((d - lo) / (hi - lo), 0, 1).astype(np.float32)


def estimate_black_level(flat_linear: np.ndarray, percentile: float = 1.0) -> float:
    """Dark level from the darkest percentile of the flat-field (spec §4.3.2)."""
    return float(np.percentile(flat_linear, percentile))


def flat_field_correct(
    img_linear: np.ndarray,
    flat_linear: np.ndarray | None,
    dark: float | np.ndarray = 0.0,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """corrected = (raw - dark) / (flat - dark)  — spec §5.3 (mandatory).

    Removes CIS lamp spatial falloff. Does NOT remove light *directionality*
    (the signal we want). If ``flat_linear`` is None, only dark subtraction is
    applied. Works on 2-D (luminance) or 3-D (RGB) images.
    """
    img = img_linear.astype(np.float32)
    if flat_linear is None:
        return np.clip(img - dark, 0.0, None)

    flat = flat_linear.astype(np.float32)
    if flat.ndim == 3 and img.ndim == 2:
        flat = to_luminance(flat)
    if blur_sigma and blur_sigma > 0:
        import cv2
        k = int(max(3, round(blur_sigma * 4) | 1))
        flat = cv2.GaussianBlur(flat, (k, k), blur_sigma)
    if img.ndim == 3 and flat.ndim == 2:
        flat = flat[..., None]  # broadcast illumination profile across channels

    denom = flat - dark
    denom = np.where(np.abs(denom) < 1e-4, 1e-4, denom)
    out = (img - dark) / denom
    return np.clip(out, 0.0, None).astype(np.float32)


def downscale(img: np.ndarray, scale: float) -> np.ndarray:
    """High-quality area downscale (scale in (0,1]); passthrough at 1.0."""
    if scale >= 0.999:
        return img
    import cv2
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
