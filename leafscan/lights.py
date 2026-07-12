"""Light-direction bookkeeping — spec §6.3, THE critical step.

The lamp is fixed in *scanner* space. To align scan ``k`` to the reference we
de-rotate its image by the card's physical rotation ``theta_k``; the light
rotates with the image, so in the aligned (subject) frame::

    az_subject[k] = az0 - theta_k        # azimuth turns with the card
    el_subject[k] = el                   # elevation NEVER changes (single cone)

    L[k] = ( cos(el)*cos(az_k),
             cos(el)*sin(az_k),
             sin(el) )

``theta_k`` is the *measured* card rotation from fiducials when available, else
the nominal ``k * 90 deg`` (spec: prefer the measured angle over the nominal).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "light_direction",
    "light_directions",
    "nominal_thetas",
    "format_light_table",
]


def light_direction(az_deg: float, el_deg: float) -> np.ndarray:
    """Single distant-directional light vector on the elevation cone."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    c = np.cos(el)
    return np.array([c * np.cos(az), c * np.sin(az), np.sin(el)], dtype=np.float64)


def nominal_thetas(n: int = 4, step_deg: float = 90.0, sign: float = 1.0) -> np.ndarray:
    """Fallback card rotations when fiducials are unavailable: sign*step*[0..n-1].

    ``sign`` encodes the physical rotation direction (CW vs CCW). It is unknown
    until calibration disambiguates it; fiducials override this entirely.
    """
    return sign * step_deg * np.arange(n, dtype=np.float64)


def light_directions(az0_deg: float, el_deg: float, thetas_deg) -> np.ndarray:
    """Return the (N, 3) stack of light vectors for card rotations ``thetas_deg``.

    ``thetas_deg[k]`` is the card's physical rotation between scan 0 and scan k.
    """
    thetas = np.asarray(thetas_deg, dtype=np.float64)
    az = az0_deg - thetas
    el = np.deg2rad(el_deg)
    c = np.cos(el)
    L = np.stack(
        [c * np.cos(np.deg2rad(az)), c * np.sin(np.deg2rad(az)), np.full_like(az, np.sin(el))],
        axis=1,
    )
    return L


def format_light_table(L: np.ndarray, thetas_deg=None, az0_deg=None, el_deg=None) -> str:
    """Human-readable dump of the light vectors — logged on every run (spec §6.3)."""
    lines = ["Light vectors L[k] (subject frame):"]
    if az0_deg is not None and el_deg is not None:
        lines.append(f"  az0 = {az0_deg:.3f} deg   el = {el_deg:.3f} deg")
    for k, v in enumerate(L):
        az = np.rad2deg(np.arctan2(v[1], v[0])) % 360.0
        th = "" if thetas_deg is None else f"  theta={np.asarray(thetas_deg)[k]:+.2f}"
        lines.append(
            f"  L[{k}] = ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})"
            f"  |L|={np.linalg.norm(v):.4f}  az={az:6.2f}{th}"
        )
    return "\n".join(lines)
