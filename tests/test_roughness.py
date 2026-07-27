import numpy as np

from leafscan.lights import light_directions, nominal_thetas
from leafscan.roughness import (
    MODELS,
    consensus_roughness,
    estimate_microfacet_roughness,
    microfacet_basis,
    retarget_baseline,
)


def _roughness_scene(model="ggx", height=48, width=72):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = 0.20 * np.sin(xx / 8.0) + 0.08 * np.cos(yy / 7.0)
    ny = 0.17 * np.cos(xx / 11.0) - 0.07 * np.sin(yy / 6.0)
    normal = np.stack([nx, ny, np.ones_like(nx)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    roughness = 0.56 + 0.14 * xx / max(1, width - 1)
    diffuse = 0.38 + 0.08 * np.sin(yy / 9.0)
    specular = 0.16 + 0.04 * np.cos(xx / 10.0)

    lights = light_directions(90.0, 35.0, nominal_thetas()).astype(np.float32)
    view = np.array([0.0, 0.0, 1.0], np.float32)
    half = lights + view[None, :]
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    no_l = np.clip(np.einsum("hwc,kc->khw", normal, lights), 0.0, 1.0)
    no_h = np.clip(np.einsum("hwc,kc->khw", normal, half), 1e-5, 1.0)
    no_v = normal[..., 2]
    basis = microfacet_basis(
        model,
        roughness.reshape(-1),
        no_l.reshape(4, -1),
        no_v.reshape(-1),
        no_h.reshape(4, -1),
    ).reshape(4, height, width)
    basis /= np.sqrt(np.mean(basis * basis, axis=0, keepdims=True) + 1e-12)
    intensity = diffuse[None] * no_l + specular[None] * basis
    valid = np.ones((height, width), bool)
    return intensity.astype(np.float32), normal, lights, valid, roughness


def test_all_microfacet_methods_return_finite_physical_maps():
    for model in MODELS:
        intensity, normal, lights, valid, _ = _roughness_scene(model)
        estimate = estimate_microfacet_roughness(
            intensity, normal, lights, valid, model=model, candidates=32,
            saturation=2.0, noise_floor=0.0, tile_rows=13,
        )
        assert np.isfinite(estimate.raw[valid]).all()
        assert ((estimate.raw[valid] >= 0.04) & (estimate.raw[valid] <= 1.0)).all()
        assert ((estimate.confidence >= 0.0) & (estimate.confidence <= 1.0)).all()


def test_ggx_fit_recovers_narrow_non_normalized_span():
    intensity, normal, lights, valid, truth = _roughness_scene("ggx")
    estimate = estimate_microfacet_roughness(
        intensity, normal, lights, valid, model="ggx", candidates=48,
        saturation=2.0, noise_floor=0.0,
    )
    trusted = valid & (estimate.confidence > 0.08)
    assert trusted.mean() > 0.35
    error = np.abs(estimate.raw[trusted] - truth[trusted])
    assert np.median(error) < 0.04
    assert np.corrcoef(estimate.raw[trusted], truth[trusted])[0, 1] > 0.65


def test_baseline_retarget_translates_instead_of_normalizing():
    yy, xx = np.mgrid[0:32, 0:64].astype(np.float32)
    raw = 0.31 + 0.14 * xx / 63.0
    confidence = np.ones_like(raw)
    valid = np.ones_like(raw, dtype=bool)

    result, stats = retarget_baseline(
        raw, confidence, valid, 0.63, spatial_sigma=0.0,
        min_confidence=0.0,
    )

    assert abs(float(np.median(result)) - 0.63) < 1e-5
    assert abs(float(np.ptp(result)) - float(np.ptp(raw))) < 1e-5
    assert 0.55 < result.min() < 0.57
    assert 0.69 < result.max() < 0.71
    assert abs(stats["source_center"] - float(np.median(raw))) < 1e-5


def test_no_specular_evidence_falls_back_to_baseline():
    height, width = 12, 16
    lights = light_directions(90.0, 35.0, nominal_thetas()).astype(np.float32)
    normal = np.zeros((height, width, 3), np.float32)
    normal[..., 2] = 1.0
    no_l = np.einsum("hwc,kc->khw", normal, lights)
    intensity = 0.5 * no_l
    valid = np.ones((height, width), bool)

    estimate = estimate_microfacet_roughness(
        intensity, normal, lights, valid, model="ggx", saturation=2.0,
    )
    result, stats = retarget_baseline(
        estimate.raw, estimate.confidence, valid, 0.67,
    )

    assert np.allclose(result, 0.67)
    assert stats["trusted_fraction"] == 0.0


def test_max_deviation_is_a_baseline_prior_not_a_range_stretch():
    raw = np.linspace(0.05, 0.95, 200, dtype=np.float32).reshape(10, 20)
    confidence = np.ones_like(raw)
    valid = np.ones_like(raw, dtype=bool)

    result, _ = retarget_baseline(
        raw, confidence, valid, 0.72, spatial_sigma=0.0,
        min_confidence=0.0, max_deviation=0.08,
    )

    assert result.min() >= 0.64 - 1e-6
    assert result.max() <= 0.80 + 1e-6
    assert abs(float(np.median(result)) - 0.72) < 1e-5


def test_consensus_shrinks_model_disagreement_to_baseline():
    a = np.array([[0.66, 0.66]], np.float32)
    b = np.array([[0.67, 0.67]], np.float32)
    c = np.array([[0.68, 0.92]], np.float32)

    result, agreement = consensus_roughness(
        [a, b, c], 0.70, agreement_scale=0.06, max_deviation=0.08,
    )

    assert agreement[0, 0] > 0.8
    assert agreement[0, 1] < 1e-4
    assert result[0, 0] < 0.68
    assert abs(float(result[0, 1]) - 0.70) < 1e-4


def test_normal_guidance_preserves_supported_material_boundary():
    raw = np.full((24, 32), 0.58, np.float32)
    raw[:, 16:] = 0.78
    confidence = np.ones_like(raw)
    valid = np.ones_like(raw, dtype=bool)
    normal = np.zeros((24, 32, 3), np.float32)
    normal[..., 2] = 1.0
    normal[:, 16:, 0] = 0.35
    normal[:, 16:, 2] = np.sqrt(1.0 - 0.35**2)

    plain, _ = retarget_baseline(
        raw, confidence, valid, 0.68, spatial_sigma=4.0,
        min_confidence=0.0,
    )
    guided, _ = retarget_baseline(
        raw, confidence, valid, 0.68, spatial_sigma=4.0,
        geometry_guide=normal, geometry_sigma=0.08,
        min_confidence=0.0,
    )

    plain_edge_contrast = float(plain[:, 16].mean() - plain[:, 15].mean())
    guided_edge_contrast = float(guided[:, 16].mean() - guided[:, 15].mean())
    assert guided_edge_contrast > plain_edge_contrast * 3
