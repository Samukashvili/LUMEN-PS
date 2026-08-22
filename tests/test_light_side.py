import numpy as np

from leafscan import calibrate
from leafscan.calibrate import (
    azimuth_for_light_side,
    detect_light_side,
    light_side_warning,
)
from leafscan.cli import load_config
from leafscan.lights import light_directions, nominal_thetas
from leafscan.solve import photometric_solve


def _ridge_normal(height=160, width=200, inverted=False):
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    z = (
        0.18 * np.exp(-(xx / 0.055) ** 2)
        + 0.08 * np.exp(-((xx + 0.42) / 0.08) ** 2 - ((yy - 0.15) / 0.55) ** 2)
    )
    if inverted:
        z = -z
    gy, gx = np.gradient(z)
    normal = np.stack([-gx, -gy, np.ones_like(z)], axis=-1).astype(np.float32)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal


def _scanner_frames(side):
    height, width = 160, 200
    mask = np.zeros((height, width), bool)
    mask[20:140, 20:180] = True
    frames = []
    for _ in range(4):
        image = np.ones((height, width), np.float32)
        image[mask] = 0.48
        if side == "right":
            image[141:145, 20:180] = 0.92
            image[14:18, 20:180] = 0.995
        else:
            image[14:18, 20:180] = 0.92
            image[141:145, 20:180] = 0.995
        frames.append(image)
    return frames, [mask] * 4, mask


def test_light_side_selects_the_requested_180_degree_branch():
    assert azimuth_for_light_side("left", 90.0) == 90.0
    assert azimuth_for_light_side("right", 90.0) == 270.0
    assert azimuth_for_light_side("left", 271.5) == 91.5
    assert azimuth_for_light_side("right", 91.5) == 271.5


def test_auto_detects_right_when_penumbra_and_relief_agree():
    frames, masks, valid = _scanner_frames("right")
    side, confidence, info = detect_light_side(
        frames, masks, _ridge_normal(inverted=True), valid
    )

    assert side == "right"
    assert confidence >= 0.5
    assert info["bottom_shadow"] > info["top_shadow"]
    assert info["relief_skew"] < 0


def test_auto_keeps_left_when_cues_support_left():
    frames, masks, valid = _scanner_frames("left")
    side, confidence, info = detect_light_side(
        frames, masks, _ridge_normal(inverted=False), valid
    )

    assert side == "left"
    assert confidence >= 0.5
    assert info["top_shadow"] > info["bottom_shadow"]


def test_decisive_left_penumbra_overrules_conflicting_relief_prior():
    frames, masks, valid = _scanner_frames("left")
    side, confidence, info = detect_light_side(
        frames, masks, _ridge_normal(inverted=True), valid
    )

    assert side == "left"
    assert confidence >= 0.5
    assert info["reason"] == "ensemble-left-edge-decisive"
    assert info["method_votes"]["penumbra_decisive_left"] is True


def test_auto_falls_back_to_left_when_cues_genuinely_disagree(monkeypatch):
    monkeypatch.setattr(
        calibrate, "_edge_penumbra_ensemble",
        lambda *_args, **_kwargs: {
            "top": 0.040, "bottom": 0.009, "ratio": 0.225,
            "scan_count": 4, "measurement_count": 12,
            "vote_fraction": 0.0, "scan_vote_fraction": 0.0,
        },
    )
    monkeypatch.setattr(
        calibrate, "_relief_shape_ensemble",
        lambda *_args, **_kwargs: {
            "skew": -0.25, "tail_asymmetry": -0.10,
            "scale_vote_fraction": 1.0, "pixels": 4096,
        },
    )
    side, confidence, info = detect_light_side(
        np.zeros((4, 2, 2), np.float32),
        np.ones((4, 2, 2), bool),
        np.dstack([np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2))]),
        np.ones((2, 2), bool),
    )

    assert side == "left"
    assert confidence < 0.5
    assert info["reason"] == "ensemble-inconclusive-left-fallback"


def test_rescanned_kiwi_edge_margin_is_decisive_left(monkeypatch):
    """Regression for light-direction-test-20260821-232323."""
    monkeypatch.setattr(
        calibrate, "_edge_penumbra_ensemble",
        lambda *_args, **_kwargs: {
            "top": 0.0503526, "bottom": -0.0008884, "ratio": -0.0176427,
            "scan_count": 4, "measurement_count": 12,
            "vote_fraction": 0.0, "scan_vote_fraction": 0.0,
        },
    )
    monkeypatch.setattr(
        calibrate, "_relief_shape_ensemble",
        lambda *_args, **_kwargs: {
            "skew": -0.208076, "tail_asymmetry": -0.040799,
            "scale_vote_fraction": 1.0, "pixels": 103205,
        },
    )
    cfg = load_config()["light"]["auto_side"]
    side, confidence, info = detect_light_side(
        np.zeros((4, 2, 2), np.float32),
        np.ones((4, 2, 2), bool),
        np.dstack([np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2))]),
        np.ones((2, 2), bool),
        cfg,
    )

    assert side == "left"
    assert confidence >= 0.5
    assert info["reason"] == "ensemble-left-edge-decisive"
    assert light_side_warning("auto", side, side, confidence, info) is None


def test_auto_accepts_native_resolution_issue_penumbra(monkeypatch):
    """Regression for issue #12/#13's native-resolution aligned measurements."""
    monkeypatch.setattr(
        calibrate, "_edge_penumbra_ensemble",
        lambda *_args, **_kwargs: {
            "top": 0.04423, "bottom": 0.01230, "ratio": 0.2782,
            "scan_count": 4, "measurement_count": 12,
            "vote_fraction": 0.50, "scan_vote_fraction": 0.50,
        },
    )
    monkeypatch.setattr(
        calibrate, "_relief_shape_ensemble",
        lambda *_args, **_kwargs: {
            "skew": -1.7679, "tail_asymmetry": -0.31,
            "scale_vote_fraction": 1.0, "pixels": 98549,
        },
    )
    cfg = load_config()["light"]["auto_side"]
    side, confidence, info = detect_light_side(
        np.zeros((4, 2, 2), np.float32),
        np.ones((4, 2, 2), bool),
        np.dstack([np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2))]),
        np.ones((2, 2), bool),
        cfg,
    )

    assert side == "right"
    assert confidence >= 0.5
    assert info["reason"] == "ensemble-right"


def test_warning_flags_manual_conflict_and_auto_uncertainty():
    conflict = light_side_warning("left", "left", "right", 0.54)
    uncertain = light_side_warning("auto", "left", "left", 0.35)

    assert "normal and height maps may be inverted" in conflict
    assert "inconclusive" in uncertain
    assert light_side_warning("left", "left", "right", 0.35) is None


def test_manual_right_recovers_right_side_synthetic_normals():
    truth = _ridge_normal(height=48, width=64)
    mask = np.ones(truth.shape[:2], bool)
    thetas = nominal_thetas()
    right = light_directions(
        azimuth_for_light_side("right", 90.0), 35.0, thetas
    )
    observations = np.einsum("hwc,kc->khw", truth, right).astype(np.float32)
    observations = np.clip(observations, 0.0, None)

    recovered_right = photometric_solve(
        observations, right, rejection="none", min_surviving=3
    )["normal"]
    left = light_directions(
        azimuth_for_light_side("left", 90.0), 35.0, thetas
    )
    recovered_left = photometric_solve(
        observations, left, rejection="none", min_surviving=3
    )["normal"]

    assert np.max(np.abs(recovered_right[mask] - truth[mask])) < 2e-4
    assert np.max(np.abs(recovered_left[..., :2] + truth[..., :2])) < 2e-4
    assert np.max(np.abs(recovered_left[..., 2] - truth[..., 2])) < 2e-4


def test_default_processing_mode_is_auto():
    assert load_config()["light"]["side"] == "auto"
