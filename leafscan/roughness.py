"""Roughness recovery from registered photometric-stereo observations.

The scanner supplies only four lighting directions, so absolute specular
calibration is under-constrained.  We fit the *shape* of three established
isotropic reflectance lobes while solving a separate non-negative diffuse and
specular amplitude at every pixel:

* GGX / Trowbridge-Reitz
* Beckmann
* Ward

Each fit returns perceptual PBR roughness (the microfacet alpha is r**2), plus a
confidence map.  ``retarget_baseline`` then translates the recovered values so
their robust centre lands on a user supplied roughness.  It deliberately never
min/max-normalizes the map: a recovered 0.14-wide span remains 0.14 wide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MODELS",
    "RoughnessEstimate",
    "consensus_roughness",
    "estimate_microfacet_roughness",
    "microfacet_basis",
    "retarget_baseline",
]


MODELS = ("ggx", "beckmann", "ward")
_EPS = np.float32(1e-8)


@dataclass
class RoughnessEstimate:
    raw: np.ndarray
    confidence: np.ndarray
    fit_rmse: np.ndarray


def consensus_roughness(
    maps,
    baseline: float,
    *,
    agreement_scale: float = 0.06,
    max_deviation: float | None = None,
    min_roughness: float = 0.02,
    max_roughness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine BRDF models while suppressing model-specific excursions."""
    stack = np.stack([np.asarray(value, dtype=np.float32) for value in maps])
    if stack.shape[0] < 2:
        raise ValueError("consensus needs at least two roughness maps")
    median = np.median(stack, axis=0)
    spread = np.ptp(stack, axis=0)
    scale = max(float(agreement_scale), 1e-6)
    agreement = np.exp(-np.square(spread / scale)).astype(np.float32)
    base = float(np.clip(baseline, min_roughness, max_roughness))
    result = base + agreement * (median - base)
    lower, upper = float(min_roughness), float(max_roughness)
    if max_deviation is not None:
        deviation = max(0.0, float(max_deviation))
        lower = max(lower, base - deviation)
        upper = min(upper, base + deviation)
    return np.clip(result, lower, upper).astype(np.float32), agreement


def _smith_ggx(no_x: np.ndarray, alpha2) -> np.ndarray:
    no_x = np.clip(no_x, 1e-5, 1.0)
    return 2.0 * no_x / (
        no_x + np.sqrt(alpha2 + (1.0 - alpha2) * no_x * no_x)
    )


def _smith_beckmann(no_x: np.ndarray, alpha) -> np.ndarray:
    """Walter et al.'s stable rational approximation to Beckmann G1."""
    no_x = np.clip(no_x, 1e-5, 1.0)
    tan_theta = np.sqrt(np.maximum(1.0 - no_x * no_x, 0.0)) / no_x
    a = 1.0 / np.maximum(alpha * tan_theta, 1e-6)
    a2 = a * a
    approx = (3.535 * a + 2.181 * a2) / (
        1.0 + 2.276 * a + 2.577 * a2
    )
    return np.where(a >= 1.6, 1.0, approx)


def microfacet_basis(
    model: str,
    roughness,
    no_l: np.ndarray,
    no_v: np.ndarray,
    no_h: np.ndarray,
) -> np.ndarray:
    """Return a direct-light specular basis, excluding unknown Fresnel scale.

    ``roughness`` is perceptual PBR roughness.  GGX/Beckmann/Ward slope width is
    alpha = roughness**2.  Constant factors are omitted because the fit solves a
    free non-negative specular amplitude for every pixel.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown roughness model {model!r}; use one of {MODELS}")

    no_l = np.clip(np.asarray(no_l, dtype=np.float32), 0.0, 1.0)
    no_v = np.clip(np.asarray(no_v, dtype=np.float32), 1e-5, 1.0)
    no_h = np.clip(np.asarray(no_h, dtype=np.float32), 1e-5, 1.0)
    r = np.clip(np.asarray(roughness, dtype=np.float32), 0.02, 1.0)
    alpha = np.maximum(r * r, 4e-4)
    if alpha.ndim and alpha.shape == no_v.shape:
        alpha = alpha[None, :]
    alpha2 = alpha * alpha

    no_h2 = no_h * no_h
    tan_h2 = np.maximum(1.0 - no_h2, 0.0) / np.maximum(no_h2, 1e-8)

    if model == "ggx":
        denom = no_h2 * (alpha2 - 1.0) + 1.0
        distribution = alpha2 / np.maximum(denom * denom, 1e-12)
        geometry = _smith_ggx(no_l, alpha2) * _smith_ggx(
            no_v[None, :], alpha2
        )
        basis = distribution * geometry / (4.0 * no_v[None, :])
    elif model == "beckmann":
        distribution = np.exp(
            np.clip(-tan_h2 / np.maximum(alpha2, 1e-8), -80.0, 0.0)
        ) / np.maximum(alpha2 * no_h2 * no_h2, 1e-12)
        geometry = _smith_beckmann(no_l, alpha) * _smith_beckmann(
            no_v[None, :], alpha
        )
        basis = distribution * geometry / (4.0 * no_v[None, :])
    else:  # isotropic Ward direct-light profile
        distribution = np.exp(
            np.clip(-tan_h2 / np.maximum(alpha2, 1e-8), -80.0, 0.0)
        )
        basis = distribution * np.sqrt(
            no_l / np.maximum(no_v[None, :], 1e-5)
        )

    return np.where(no_l > 0.0, basis, 0.0).astype(np.float32, copy=False)


def _fit_nonnegative_two_term(y, diffuse_basis, spec_basis, weights):
    """Fit y = diffuse*x + specular*b with both coefficients >= 0."""
    a = np.sum(weights * diffuse_basis * diffuse_basis, axis=0)
    b = np.sum(weights * diffuse_basis * spec_basis, axis=0)
    c = np.sum(weights * spec_basis * spec_basis, axis=0)
    yx = np.sum(weights * y * diffuse_basis, axis=0)
    yb = np.sum(weights * y * spec_basis, axis=0)
    det = a * c - b * b

    diffuse_u = (yx * c - yb * b) / np.maximum(det, 1e-10)
    spec_u = (yb * a - yx * b) / np.maximum(det, 1e-10)
    residual_u = y - diffuse_u[None, :] * diffuse_basis - spec_u[None, :] * spec_basis
    sse_u = np.sum(weights * residual_u * residual_u, axis=0)
    sse_u[(det <= 1e-10) | (diffuse_u < 0.0) | (spec_u < 0.0)] = np.inf

    # Boundary of the non-negative quadrant: diffuse-only.
    diffuse_d = np.maximum(yx / np.maximum(a, 1e-10), 0.0)
    residual_d = y - diffuse_d[None, :] * diffuse_basis
    sse_d = np.sum(weights * residual_d * residual_d, axis=0)

    # Other boundary: specular-only.
    spec_s = np.maximum(yb / np.maximum(c, 1e-10), 0.0)
    residual_s = y - spec_s[None, :] * spec_basis
    sse_s = np.sum(weights * residual_s * residual_s, axis=0)

    use_u = (sse_u <= sse_d) & (sse_u <= sse_s)
    use_d = ~use_u & (sse_d <= sse_s)
    diffuse = np.where(use_u, diffuse_u, np.where(use_d, diffuse_d, 0.0))
    specular = np.where(use_u, spec_u, np.where(use_d, 0.0, spec_s))
    sse = np.where(use_u, sse_u, np.where(use_d, sse_d, sse_s))
    return diffuse, specular, sse


def estimate_microfacet_roughness(
    intensity_stack: np.ndarray,
    normal: np.ndarray,
    lights: np.ndarray,
    valid: np.ndarray,
    *,
    sample_valid: np.ndarray | None = None,
    model: str = "ggx",
    min_roughness: float = 0.04,
    max_roughness: float = 1.0,
    candidates: int = 24,
    saturation: float = 0.995,
    min_n_dot_l: float = 0.03,
    noise_floor: float = 0.002,
    tile_rows: int = 64,
) -> RoughnessEstimate:
    """Fit one microfacet model to a registered four-light observation stack.

    Absolute specular colour/Fresnel is intentionally a fitted nuisance
    parameter.  Roughness is selected from lobe *shape*, which makes the result
    insensitive to unknown scanner gain and the user's baseline calibration.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown roughness model {model!r}; use one of {MODELS}")
    y_all = np.asarray(intensity_stack, dtype=np.float32)
    n_all = np.asarray(normal, dtype=np.float32)
    l_all = np.asarray(lights, dtype=np.float32)
    pixel_valid = np.asarray(valid, dtype=bool)
    if y_all.ndim != 3:
        raise ValueError("intensity_stack must have shape (K,H,W)")
    k_count, height, width = y_all.shape
    if n_all.shape != (height, width, 3):
        raise ValueError("normal must have shape (H,W,3)")
    if l_all.shape != (k_count, 3):
        raise ValueError("lights must have shape (K,3)")
    if pixel_valid.shape != (height, width):
        raise ValueError("valid must have shape (H,W)")
    if sample_valid is None:
        sample_valid = np.ones_like(y_all, dtype=bool)
    else:
        sample_valid = np.asarray(sample_valid, dtype=bool)
        if sample_valid.shape != y_all.shape:
            raise ValueError("sample_valid must match intensity_stack")

    lo = float(np.clip(min_roughness, 0.02, 1.0))
    hi = float(np.clip(max_roughness, lo, 1.0))
    grid = np.linspace(lo, hi, max(8, int(candidates)), dtype=np.float32)
    l_norm = l_all / np.maximum(np.linalg.norm(l_all, axis=1, keepdims=True), 1e-8)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half = l_norm + view[None, :]
    half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-8)

    raw = np.full((height, width), np.nan, dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)
    fit_rmse = np.full((height, width), np.nan, dtype=np.float32)

    for y0 in range(0, height, max(1, int(tile_rows))):
        y1 = min(height, y0 + max(1, int(tile_rows)))
        pixels = (y1 - y0) * width
        obs = y_all[:, y0:y1].reshape(k_count, pixels)
        normals = n_all[y0:y1].reshape(pixels, 3)
        no_l = (normals @ l_norm.T).T.astype(np.float32, copy=False)
        no_h = (normals @ half.T).T.astype(np.float32, copy=False)
        no_v = normals[:, 2].astype(np.float32, copy=False)

        weights_bool = sample_valid[:, y0:y1].reshape(k_count, pixels).copy()
        weights_bool &= np.isfinite(obs)
        weights_bool &= no_l > float(min_n_dot_l)
        if saturation < 1.0:
            weights_bool &= obs < float(saturation)
        active = pixel_valid[y0:y1].reshape(pixels)
        active &= weights_bool.sum(axis=0) >= 3
        active &= no_v > 0.0
        weights = weights_bool.astype(np.float32)
        obs = np.where(weights_bool, obs, 0.0)
        diffuse_basis = np.clip(no_l, 0.0, 1.0)
        count = np.maximum(weights.sum(axis=0), 1.0)

        a = np.sum(weights * diffuse_basis * diffuse_basis, axis=0)
        yx = np.sum(weights * obs * diffuse_basis, axis=0)
        diffuse_only = np.maximum(yx / np.maximum(a, 1e-10), 0.0)
        diffuse_residual = obs - diffuse_only[None, :] * diffuse_basis
        diffuse_sse = np.sum(weights * diffuse_residual * diffuse_residual, axis=0) / count
        signal_rms = np.sqrt(np.sum(weights * obs * obs, axis=0) / count)

        best_sse = np.full(pixels, np.inf, dtype=np.float32)
        best_r = np.full(pixels, (lo + hi) * 0.5, dtype=np.float32)
        best_spec = np.zeros(pixels, dtype=np.float32)
        best_variation = np.zeros(pixels, dtype=np.float32)

        for roughness in grid:
            spec_basis = microfacet_basis(
                model, roughness, no_l, no_v, no_h
            )
            # Candidate-dependent scale is unidentifiable from Fresnel/gain.
            # RMS-normalization keeps only the lobe shape and stabilizes NNLS.
            basis_rms = np.sqrt(
                np.sum(weights * spec_basis * spec_basis, axis=0) / count
            )
            usable_basis = basis_rms > 1e-12
            spec_basis = spec_basis / np.maximum(basis_rms[None, :], 1e-12)
            _, specular, sse = _fit_nonnegative_two_term(
                obs, diffuse_basis, spec_basis, weights
            )
            sse = sse / count
            mean_basis = np.sum(weights * spec_basis, axis=0) / count
            variation = np.sqrt(np.maximum(1.0 - mean_basis * mean_basis, 0.0))
            better = (sse < best_sse) & usable_basis
            best_sse[better] = sse[better]
            best_r[better] = roughness
            best_spec[better] = specular[better]
            best_variation[better] = variation[better]

        improvement = np.clip(
            (diffuse_sse - best_sse - float(noise_floor) ** 2)
            / np.maximum(diffuse_sse + float(noise_floor) ** 2, 1e-10),
            0.0,
            1.0,
        )
        spec_fraction = np.clip(
            best_spec / np.maximum(signal_rms, 1e-6), 0.0, 1.0
        )
        geometry = np.clip(best_variation / 0.15, 0.0, 1.0)
        # Three samples can exactly determine diffuse, specular amplitude, and
        # roughness; there is then no residual degree of freedom with which to
        # trust the estimate. Four valid observations provide the one available
        # cross-check in this capture design.
        degrees_of_freedom = np.clip(count - 3.0, 0.0, 1.0)
        conf = np.sqrt(
            improvement
            * np.clip(spec_fraction / 0.10, 0.0, 1.0)
            * geometry
            * degrees_of_freedom
        )
        good = active & np.isfinite(best_sse)

        raw_tile = raw[y0:y1].reshape(pixels)
        conf_tile = confidence[y0:y1].reshape(pixels)
        rmse_tile = fit_rmse[y0:y1].reshape(pixels)
        raw_tile[good] = best_r[good]
        conf_tile[good] = conf[good]
        rmse_tile[good] = np.sqrt(np.maximum(best_sse[good], 0.0))

    return RoughnessEstimate(raw=raw, confidence=confidence, fit_rmse=fit_rmse)


def _centre(values: np.ndarray, weights: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(np.sum(values * weights) / np.maximum(weights.sum(), 1e-8))
    if statistic == "mode":
        hist, edges = np.histogram(values, bins=128, range=(0.0, 1.0), weights=weights)
        index = int(np.argmax(hist))
        in_bin = (values >= edges[index]) & (values <= edges[index + 1])
        if in_bin.any():
            return float(
                np.sum(values[in_bin] * weights[in_bin])
                / np.maximum(weights[in_bin].sum(), 1e-8)
            )
    if statistic not in ("median", "mode"):
        raise ValueError("baseline statistic must be median, mean, or mode")
    return float(np.median(values))


def retarget_baseline(
    raw: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    baseline: float,
    *,
    statistic: str = "median",
    detail_strength: float = 1.0,
    max_deviation: float | None = None,
    spatial_sigma: float = 2.0,
    geometry_guide: np.ndarray | None = None,
    geometry_sigma: float = 0.08,
    min_confidence: float = 0.04,
    min_support_fraction: float = 0.02,
    min_roughness: float = 0.02,
    max_roughness: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Translate a roughness field to ``baseline`` without range normalization."""
    import cv2

    raw = np.asarray(raw, dtype=np.float32)
    confidence = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    valid = np.asarray(valid, dtype=bool)
    if raw.shape != confidence.shape or raw.shape != valid.shape:
        raise ValueError("raw, confidence, and valid must have identical shapes")

    base = float(np.clip(baseline, min_roughness, max_roughness))
    trusted = valid & np.isfinite(raw) & (confidence >= float(min_confidence))
    trusted_fraction = float(trusted.sum() / max(1, valid.sum()))
    result = np.full(raw.shape, base, dtype=np.float32)
    if not trusted.any() or trusted_fraction < float(min_support_fraction):
        return result, {
            "source_center": None,
            "trusted_fraction": trusted_fraction,
            "output_min": base,
            "output_p01": base,
            "output_median": base,
            "output_p99": base,
            "output_max": base,
        }

    values = raw[trusted]
    weights_1d = np.maximum(confidence[trusted], 1e-4)
    global_center = _centre(values, weights_1d, statistic)
    support = np.where(trusted, np.maximum(confidence, 1e-4), 0.0).astype(np.float32)
    source = np.where(trusted, raw, 0.0).astype(np.float32)

    if spatial_sigma > 0.0:
        numerator = cv2.GaussianBlur(
            source * support, (0, 0), float(spatial_sigma),
            borderType=cv2.BORDER_REPLICATE,
        )
        denominator = cv2.GaussianBlur(
            support, (0, 0), float(spatial_sigma),
            borderType=cv2.BORDER_REPLICATE,
        )
        local = np.where(
            denominator > 1e-8, numerator / np.maximum(denominator, 1e-8), global_center
        )
        if geometry_guide is not None:
            guide = np.asarray(geometry_guide, dtype=np.float32)
            if guide.shape[:2] != raw.shape:
                raise ValueError("geometry_guide must match the roughness image")
            if guide.ndim == 2:
                guide = guide[..., None]
            # Seed unsupported pixels from confidence-weighted neighbours, then
            # filter along the normal field. Raised vein boundaries stop the
            # smoothing kernel, but geometry never chooses the sign/magnitude
            # of roughness contrast; the photometric evidence still does that.
            seed = np.where(trusted, raw, local).astype(np.float32)
            if hasattr(cv2, "ximgproc") and hasattr(
                cv2.ximgproc, "jointBilateralFilter"
            ):
                regularized = cv2.ximgproc.jointBilateralFilter(
                    guide, seed, -1, float(geometry_sigma),
                    max(float(spatial_sigma), 0.5),
                    borderType=cv2.BORDER_REPLICATE,
                )
            else:
                regularized = local
        else:
            # Four scans leave only one residual degree of freedom after fitting
            # diffuse level, specular amplitude, and roughness. The defensible
            # estimate is therefore the confidence-weighted local field, not the
            # visibly quantized/noisy single-pixel grid choice.
            regularized = local
    else:
        regularized = np.where(trusted, raw, global_center)

    centre = _centre(regularized[trusted], weights_1d, statistic)
    translated = base + float(detail_strength) * (regularized - centre)
    output_min = float(min_roughness)
    output_max = float(max_roughness)
    if max_deviation is not None:
        deviation = max(0.0, float(max_deviation))
        output_min = max(output_min, base - deviation)
        output_max = min(output_max, base + deviation)
    translated = np.clip(translated, output_min, output_max)
    result[valid] = translated[valid]

    output_values = result[valid]
    q01, q50, q99 = np.percentile(output_values, [1, 50, 99])
    return result, {
        "source_center": centre,
        "trusted_fraction": trusted_fraction,
        "output_min": float(output_values.min()),
        "output_p01": float(q01),
        "output_median": float(q50),
        "output_p99": float(q99),
        "output_max": float(output_values.max()),
    }
