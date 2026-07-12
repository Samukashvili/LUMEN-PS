"""Spec §6.3 requires: four L[k] are unit vectors, share an identical Z, ~90 apart."""
import numpy as np
import pytest

from leafscan.lights import light_directions, nominal_thetas, light_direction


def test_lights_are_unit_vectors():
    L = light_directions(90.0, 30.0, nominal_thetas())
    norms = np.linalg.norm(L, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_identical_elevation_single_cone():
    L = light_directions(37.0, 42.0, nominal_thetas())
    assert np.allclose(L[:, 2], L[0, 2], atol=1e-12)
    assert np.isclose(L[0, 2], np.sin(np.deg2rad(42.0)))


def test_ninety_degrees_apart_in_azimuth():
    L = light_directions(0.0, 20.0, nominal_thetas())
    az = np.sort(np.rad2deg(np.arctan2(L[:, 1], L[:, 0])) % 360.0)
    diffs = np.diff(np.r_[az, az[0] + 360.0])
    assert np.allclose(diffs, 90.0, atol=1e-9)


def test_measured_angle_preferred_over_nominal():
    thetas = np.array([0.0, 88.5, 181.2, 269.0])  # imperfect physical rotations
    L = light_directions(95.0, 25.0, thetas)
    az = np.rad2deg(np.arctan2(L[:, 1], L[:, 0])) % 360.0
    assert np.allclose(az, (95.0 - thetas) % 360.0, atol=1e-9)


def test_single_matches_stack():
    v = light_direction(60.0, 15.0)
    L = light_directions(60.0, 15.0, [0.0])
    assert np.allclose(v, L[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
