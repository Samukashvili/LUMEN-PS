"""Optional CUDA helpers with conservative VRAM-aware tiling.

The RTX-class GPUs commonly used with LUMEN-PS have much less VRAM than system
RAM. Operators therefore accept NumPy arrays, process bounded row tiles on the
GPU, and return NumPy arrays so the rest of the pipeline has one canonical data
representation and identical exports on CPU and CUDA.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

_warned = False

# Keep generated CUDA kernels with other project runtime state. This avoids
# profile-folder permission problems and makes cleanup/local installs predictable.
_cache = Path(__file__).resolve().parent.parent / ".lumen-ps" / "cupy-cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUPY_CACHE_DIR", str(_cache))


def cupy_backend(requested="auto"):
    global _warned
    name = str(requested or "auto").lower()
    if name == "cpu":
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="CUDA path could not be detected.*")
            import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except Exception as exc:
        if name == "gpu" and not _warned:
            warnings.warn(f"CUDA unavailable ({exc}); using CPU")
            _warned = True
    return None


def backend_description(requested="auto"):
    cp = cupy_backend(requested)
    if cp is None:
        return "optimized CPU"
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    free, total = cp.cuda.runtime.memGetInfo()
    return f"CUDA {name} ({free / 2**30:.1f}/{total / 2**30:.1f} GiB free)"


def release_gpu_memory(cp):
    """Return cached blocks so the viewer and desktop regain VRAM between stages."""
    if cp is None:
        return
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def tile_rows(width, arrays_per_pixel=32, fraction=0.22, minimum=32):
    """Choose rows that use only a conservative fraction of currently free VRAM."""
    cp = cupy_backend("auto")
    if cp is None:
        return minimum
    free, _ = cp.cuda.runtime.memGetInfo()
    budget = max(64 << 20, int(free * fraction))
    return max(minimum, budget // max(1, width * arrays_per_pixel * 4))


def color_albedo(rgb_stack, normal, L, weights, backend="auto"):
    """Lighting-free RGB albedo, CUDA-tiled when available."""
    cp = cupy_backend(backend)
    if cp is None:
        ndotl = np.clip(np.einsum("hwc,nc->nhw", normal, L), 0, None)
        w = weights * (ndotl > 1e-3)
        est = rgb_stack / np.clip(ndotl[..., None], 1e-3, None)
        return np.clip((est * w[..., None]).sum(0) /
                       np.clip(w.sum(0), 1e-6, None)[..., None], 0, 1).astype(np.float32)

    _, H, W, _ = rgb_stack.shape
    out = np.empty((H, W, 3), np.float32)
    light = cp.asarray(L, dtype=cp.float32)
    rows = min(H, tile_rows(W, arrays_per_pixel=30))
    try:
        for y in range(0, H, rows):
            sl = slice(y, min(H, y + rows))
            n = cp.asarray(normal[sl], dtype=cp.float32)
            rgb = cp.asarray(rgb_stack[:, sl], dtype=cp.float32)
            wt = cp.asarray(weights[:, sl], dtype=cp.float32)
            shade = cp.clip(cp.einsum("hwc,nc->nhw", n, light), 0, None)
            wt *= shade > 1e-3
            num = (rgb / cp.maximum(shade[..., None], 1e-3) * wt[..., None]).sum(0)
            result = cp.clip(num / cp.maximum(wt.sum(0), 1e-6)[..., None], 0, 1)
            out[sl] = cp.asnumpy(result)
    except cp.cuda.memory.OutOfMemoryError:
        if str(backend).lower() == "gpu":
            raise
        release_gpu_memory(cp)
        return color_albedo(rgb_stack, normal, L, weights, backend="cpu")
    finally:
        release_gpu_memory(cp)
    return out


def residual_stack(I_stack, normal, albedo, L, backend="auto"):
    """Re-render and absolute residual in bounded CUDA tiles."""
    cp = cupy_backend(backend)
    if cp is None:
        pred = np.clip(np.einsum("hwc,nc->nhw", normal, L), 0, None) * albedo[None]
        return np.abs(pred - I_stack), pred
    N, H, W = I_stack.shape
    pred = np.empty((N, H, W), np.float32)
    resid = np.empty_like(pred)
    light = cp.asarray(L, dtype=cp.float32)
    rows = min(H, tile_rows(W, arrays_per_pixel=18))
    try:
        for y in range(0, H, rows):
            sl = slice(y, min(H, y + rows))
            n = cp.asarray(normal[sl], dtype=cp.float32)
            a = cp.asarray(albedo[sl], dtype=cp.float32)
            obs = cp.asarray(I_stack[:, sl], dtype=cp.float32)
            p = cp.clip(cp.einsum("hwc,nc->nhw", n, light), 0, None) * a[None]
            pred[:, sl] = cp.asnumpy(p)
            resid[:, sl] = cp.asnumpy(cp.abs(p - obs))
    except cp.cuda.memory.OutOfMemoryError:
        if str(backend).lower() == "gpu":
            raise
        release_gpu_memory(cp)
        return residual_stack(I_stack, normal, albedo, L, backend="cpu")
    finally:
        release_gpu_memory(cp)
    return resid, pred
