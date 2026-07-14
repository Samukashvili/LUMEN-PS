"""Rigid alignment regression tests.

The PCB case: a rigid rectangular subject sitting off-centre, physically
rotated 90deg between scans. findTransformECC estimates the backward
(ref -> mov) warp; feeding it the forward matrix "converges" on the nearly
symmetric mask distance transform but lands the subject far off. The refine
step must therefore use the right convention and never make the nominal
initialisation worse.
"""
import numpy as np

from leafscan import align


def _board_scene(H=360, W=440, speckle=False):
    """Off-centre rectangular 'board' with an asymmetric notch (so orientation
    is observable) on a bright background."""
    mask = np.zeros((H, W), bool)
    mask[60:240, 90:350] = True
    mask[60:100, 90:150] = False          # corner notch breaks symmetry
    luma = np.full((H, W), 0.9, np.float32)
    luma[mask] = 0.15
    yy, xx = np.mgrid[0:H, 0:W]
    luma += 0.02 * np.sin(xx / 7.0) * mask  # some surface texture
    if speckle:                            # feature-rich albedo (pads/print)
        rng = np.random.default_rng(7)
        for x, y in zip(rng.integers(95, 345, 250), rng.integers(105, 235, 250)):
            luma[y - 2:y + 3, x - 2:x + 3] = 0.7
    return luma.astype(np.float32), mask


def _rotated_scan(luma, mask, k, shift=(0, 0)):
    """Rotate the whole frame by k*90deg and shift: emulates re-placing the
    subject on the glass with a small positioning error."""
    l = np.rot90(luma, k).copy()
    m = np.rot90(mask, k).copy()
    dy, dx = shift
    l = np.roll(np.roll(l, dy, axis=0), dx, axis=1)
    m = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    # square the frame so rot90 shapes match the reference canvas
    H = max(l.shape)
    canvas_l = np.full((H, H), 0.9, np.float32)
    canvas_m = np.zeros((H, H), bool)
    canvas_l[: l.shape[0], : l.shape[1]] = l
    canvas_m[: m.shape[0], : m.shape[1]] = m
    return canvas_l, canvas_m


def test_rigid_align_rect_board_all_rotations():
    luma, mask = _board_scene()
    ref_luma, ref_mask = _rotated_scan(luma, mask, 0)
    cfg = {"fiducials": {"enabled": False},
           "rigid": {"nominal_step_deg": 90.0, "ecc_refine": True}}
    for k in range(1, 4):
        mov_luma, mov_mask = _rotated_scan(luma, mask, k, shift=(5 * k, -3 * k))
        _, (wm,), theta, method = align.rigid_align(
            ref_luma, mov_luma, [mov_mask], k,
            ref_mask=ref_mask, mov_mask=mov_mask, cfg=cfg)
        iou = (ref_mask & wm).sum() / (ref_mask | wm).sum()
        assert iou > 0.97, f"scan{k}: IoU {iou:.3f} via {method}"
        err = abs(((theta - 90 * k + 180) % 360) - 180)
        assert err < 2.0, f"scan{k}: theta {theta:.2f} via {method}"


def test_feature_refine_fixes_shadow_biased_masks():
    """When the segmentation mask carries a one-sided shadow fringe, the
    centroid/outline stages misplace the subject by a few px; the feature
    stage must recover sub-2px accuracy from interior texture."""
    luma, mask = _board_scene(speckle=True)
    ref_luma, ref_mask = _rotated_scan(luma, mask, 0)
    mov_luma, mov_mask = _rotated_scan(luma, mask, 1, shift=(7, -4))
    # one-sided shadow fringe: fattens the mask and biases its centroid
    biased = mov_mask | np.roll(mov_mask, 10, axis=1)

    def run(feature_refine):
        cfg = {"fiducials": {"enabled": False},
               "rigid": {"nominal_step_deg": 90.0, "ecc_refine": False,
                         "feature_refine": feature_refine}}
        warped, _, _, method = align.rigid_align(
            ref_luma, mov_luma, [mov_mask], 1,
            ref_mask=ref_mask, mov_mask=biased, cfg=cfg)
        m = ref_mask & (warped > 0)
        return float(np.abs(warped - ref_luma)[m].mean()), method

    e_off, _ = run(False)
    e_on, method = run(True)
    assert "feat" in method, method
    assert e_on < e_off * 0.6, (e_on, e_off)


def test_ecc_refine_never_degrades_nominal():
    """Whatever ECC returns, the accepted transform must overlap at least as
    well as the nominal initialisation it started from."""
    luma, mask = _board_scene()
    _, ref_mask = _rotated_scan(luma, mask, 0)
    _, mov_mask = _rotated_scan(luma, mask, 3, shift=(12, -8))

    def centroid(m):
        ys, xs = np.nonzero(m)
        return np.array([xs.mean(), ys.mean()])

    import cv2
    c_mov, c_ref = centroid(mov_mask), centroid(ref_mask)
    M = cv2.getRotationMatrix2D(tuple(c_mov), -270.0, 1.0)
    M[0, 2] += c_ref[0] - c_mov[0]
    M[1, 2] += c_ref[1] - c_mov[1]
    M = M.astype(np.float32)

    refined = align._ecc_refine_on_dt(ref_mask, mov_mask, M)
    if refined is not None:
        iou_init = align._warped_mask_iou(ref_mask, mov_mask, M)
        iou_ref = align._warped_mask_iou(ref_mask, mov_mask, refined)
        assert iou_ref >= iou_init


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
