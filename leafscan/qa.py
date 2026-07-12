"""QA outputs — spec §8.4 (re-render residual is the MAIN diagnostic) + §6.5."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .solve import rerender

__all__ = ["write_qa", "colorize", "residual_report"]


def colorize(x, mask=None, gamma=1.0, percentile=(1, 99)):
    """Normalize a scalar field to an 8-bit heat-ish grayscale for inspection."""
    x = x.astype(np.float32)
    ref = x[mask] if (mask is not None and mask.any()) else x
    lo, hi = np.percentile(ref, percentile)
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    y = np.clip((x - lo) / (hi - lo), 0, 1) ** gamma
    if mask is not None:
        y = y * mask
    return (y * 255).astype(np.uint8)


def residual_report(I_stack, normal, albedo, L, valid):
    """Per-scan residual maps + scalar stats (spec §8.4)."""
    pred = rerender(normal, albedo, L)
    resid = np.abs(pred - I_stack)  # (N,H,W)
    stats = []
    m = valid
    for k in range(resid.shape[0]):
        rk = resid[k][m] if m.any() else resid[k].ravel()
        stats.append({
            "scan": k,
            "mean": float(rk.mean()),
            "p95": float(np.percentile(rk, 95)),
            "max": float(rk.max()),
        })
    return resid, pred, stats


def write_qa(qa_dir, *, I_stack, normal, albedo, L, valid, nsamples,
             thetas=None, az0=None, el=None, subsurface=None, weights=None,
             extra_text=None):
    """Emit all diagnostic renders + a residual/light-vector text report."""
    from PIL import Image
    qa = Path(qa_dir)
    qa.mkdir(parents=True, exist_ok=True)

    # 1. re-render residuals (the money diagnostic)
    resid, pred, stats = residual_report(I_stack, normal, albedo, L, valid)
    for k in range(resid.shape[0]):
        Image.fromarray(colorize(resid[k], valid, percentile=(1, 99))).save(
            qa / f"residual_scan{k}.png")
        Image.fromarray(colorize(I_stack[k], valid)).save(qa / f"observed_scan{k}.png")
        Image.fromarray(colorize(pred[k], valid)).save(qa / f"predicted_scan{k}.png")

    # 2. mask agreement (§6.5)
    n = nsamples.astype(np.float32)
    Image.fromarray((n / max(1, n.max()) * 255).astype(np.uint8)).save(
        qa / "mask_agreement.png")

    # 3. normal preview (GL encoding, valid only)
    from .outputs import encode_normal
    flat = np.zeros_like(normal); flat[..., 2] = 1.0
    nprev = np.where(valid[..., None], normal, flat)
    Image.fromarray((encode_normal(nprev) * 255).astype(np.uint8)).save(
        qa / "normal_preview.png")

    # 4. subsurface hint (§5.2)
    if subsurface is not None:
        Image.fromarray((np.clip(subsurface, 0, 1) * 255).astype(np.uint8)).save(
            qa / "subsurface_hint.png")

    # 5. rejection weight coverage (how often each sample survived)
    if weights is not None:
        cov = weights.mean(axis=0)
        Image.fromarray(colorize(cov, valid, percentile=(0, 100))).save(
            qa / "rejection_coverage.png")

    # 6. text report — light vectors + residual stats
    lines = []
    if az0 is not None and el is not None:
        lines.append(f"az0 = {az0:.3f} deg   el = {el:.3f} deg")
    if thetas is not None:
        lines.append(f"thetas = {np.round(np.asarray(thetas), 3).tolist()}")
    lines.append("Light vectors L[k]:")
    for k, v in enumerate(L):
        az = np.rad2deg(np.arctan2(v[1], v[0])) % 360
        lines.append(f"  L[{k}] = ({v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f}) az={az:6.2f} "
                     f"|L|={np.linalg.norm(v):.4f}")
    lines.append("Re-render residual (lower = more trustworthy):")
    for s in stats:
        lines.append(f"  scan{s['scan']}: mean={s['mean']:.4f} p95={s['p95']:.4f} "
                     f"max={s['max']:.4f}")
    lines.append(f"valid pixels: {int(valid.sum())} / {valid.size}")
    if extra_text:
        lines.append(extra_text)
    (qa / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    return stats
