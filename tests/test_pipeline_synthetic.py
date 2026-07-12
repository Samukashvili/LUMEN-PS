"""End-to-end integration test on synthetic *rotated* scans.

Simulates the real capture: one fixed scanner light, subject physically rotated
k*90deg on the glass. Renders 4 sRGB PNGs, runs the whole pipeline, and checks
the re-render residual is small and the normals are recovered. This exercises
load/linearize/flat/mask/rigid/nonrigid/solve/integrate/outputs/qa together.
"""
import numpy as np
from PIL import Image

from leafscan.cli import (_assemble_bed_canvas, _centered_frame,
                          _pad_stack_to_common_canvas, load_config, run_pipeline)
from leafscan.io import linear_to_srgb
from leafscan.lights import light_direction


def _leaf_scene(S=200):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    cx = cy = S / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = r < S * 0.35
    z = 0.5 * np.exp(-(r / (S * 0.22)) ** 2)             # gentle central bump
    z += 0.05 * np.sin(xx / 6.0) * (r < S * 0.35)         # fine ripples (veins)
    rho = 0.55 + 0.3 * (np.sin(yy / 9.0) > 0.3)           # vein-like albedo bands
    rho = np.clip(rho, 0.2, 0.9)
    return z, rho, mask


def _render_rotated(z, rho, mask, k, L_scanner):
    zk = np.rot90(z, k); rk = np.rot90(rho, k); mk = np.rot90(mask, k)
    dzdx = np.gradient(zk, axis=1); dzdy = np.gradient(zk, axis=0)
    N = np.stack([-dzdx, -dzdy, np.ones_like(zk)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    shade = np.clip(N @ L_scanner, 0, None) * rk
    img = np.where(mk[..., None], shade[..., None], 1.0)   # white background
    img = np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img
    return np.clip(img, 0, 1).astype(np.float32)


def test_pipeline_end_to_end(tmp_path):
    S = 200
    az0, el = 90.0, 35.0
    z, rho, mask = _leaf_scene(S)
    L_scanner = light_direction(az0, el)

    scan_dir = tmp_path / "scans"
    scan_dir.mkdir()
    for k in range(4):
        lin = _render_rotated(z, rho, mask, k, L_scanner)
        srgb8 = (linear_to_srgb(lin) * 255).astype(np.uint8)
        Image.fromarray(srgb8, "RGB").save(scan_dir / f"k{k}.png")

    cfg = load_config()
    cfg["light"]["source"] = "config"     # skip self-cal; use the known truth
    cfg["light"]["az0_deg"] = az0
    cfg["light"]["el_deg"] = el
    cfg["align"]["nonrigid"]["enabled"] = True

    res = run_pipeline(cfg, sorted(scan_dir.glob("k*.png")), tmp_path / "out",
                       verbose=False)

    # residual must be small (clean synthetic data)
    means = [s["mean"] for s in res["residual"]]
    assert max(means) < 0.05, f"residual too high: {means}"
    assert res["valid_px"] > 0.5 * mask.sum()

    # deliverables exist
    for name in ("normal_gl.png", "normal_dx.png", "albedo.png", "height.png"):
        assert (res["out_dir"] / name).exists(), name
    assert (res["out_dir"] / "qa" / "report.txt").exists()
    # thetas recovered near k*90
    for k, th in enumerate(res["thetas"]):
        assert abs(((th - 90 * k + 180) % 360) - 180) < 5, res["thetas"]


def test_pipeline_roi_captures_end_to_end(tmp_path):
    """Smart-ROI simulation: each rotated scan cropped to a different bed rect.

    With capture_rois + capture_dpi the pipeline must rebuild bed geometry,
    align cleanly, and deliver a centred leaf.
    """
    S = 200
    az0, el = 90.0, 35.0
    z, rho, mask = _leaf_scene(S)
    L_scanner = light_direction(az0, el)

    # dpi=25.4 => exactly 1 px per mm, so px offsets double as mm offsets
    dpi = 25.4
    offsets = [(10, 20), (25, 5), (0, 15), (18, 0)]   # (x, y) px == mm
    scan_dir = tmp_path / "scans"
    scan_dir.mkdir()
    rois = []
    for k, (ox, oy) in enumerate(offsets):
        lin = _render_rotated(z, rho, mask, k, L_scanner)
        w, h = S - 22 - ox // 2, S - 24 - oy // 2      # per-scan sizes differ
        sub = lin[oy:oy + h, ox:ox + w]
        srgb8 = (linear_to_srgb(sub) * 255).astype(np.uint8)
        Image.fromarray(srgb8, "RGB").save(scan_dir / f"k{k}.png")
        rois.append((float(ox), float(oy), float(sub.shape[1]), float(sub.shape[0])))

    cfg = load_config()
    cfg["light"]["source"] = "config"
    cfg["light"]["az0_deg"] = az0
    cfg["light"]["el_deg"] = el

    res = run_pipeline(cfg, sorted(scan_dir.glob("k*.png")), tmp_path / "out",
                       verbose=False, auto_crop=True,
                       capture_rois=rois, capture_dpi=dpi)

    means = [s["mean"] for s in res["residual"]]
    assert max(means) < 0.05, f"residual too high: {means}"
    assert res["valid_px"] > 0.5 * mask.sum()
    for k, th in enumerate(res["thetas"]):
        assert abs(((th - 90 * k + 180) % 360) - 180) < 5, res["thetas"]

    # the delivered leaf must be centred in the output frame
    alpha = np.asarray(Image.open(res["out_dir"] / "alpha.png"), dtype=np.float32)
    ys, xs = np.nonzero(alpha > 127)
    H, W = alpha.shape[:2]
    cx, cy = xs.mean(), ys.mean()
    tol = 0.05 * max(H, W) + 2
    assert abs(cx - W / 2) < tol and abs(cy - H / 2) < tol, (cx, cy, W, H)


def test_assemble_bed_canvas_places_scans_at_offsets():
    a = np.ones((4, 5), np.float32)          # roi at (2mm, 1mm)
    b = np.full((6, 3), 2, np.float32)       # roi at (0mm, 0mm)
    dpi = 25.4                                # 1 px per mm
    stack, rect = _assemble_bed_canvas([a, b], [(2, 1, 5, 4), (0, 0, 3, 6)],
                                       dpi, scale=1.0)
    assert rect == (0, 0, 7, 6)
    assert all(s.shape == (6, 7) for s in stack)
    assert np.all(stack[0][1:5, 2:7] == 1)   # a's payload at its bed offset
    assert np.all(stack[1][0:6, 0:3] == 2)   # b's payload at the origin


def test_centered_frame_centers_mask():
    mask = np.zeros((100, 120), bool)
    mask[10:30, 80:110] = True               # off-centre blob
    frame = _centered_frame(mask)
    assert frame is not None
    sy, sx = frame
    assert abs((sy.start + sy.stop) / 2 - (10 + 30) / 2) <= 1
    assert abs((sx.start + sx.stop) / 2 - (80 + 110) / 2) <= 1


def test_variable_roi_stack_is_edge_padded():
    stack = [np.ones((4, 6), dtype=np.float32), np.full((7, 6), 2, dtype=np.float32)]
    padded = _pad_stack_to_common_canvas(stack)
    assert [a.shape for a in padded] == [(7, 6), (7, 6)]
    assert np.all(padded[0][1:5] == 1)
    assert np.all(padded[0][0] == 1) and np.all(padded[0][-1] == 1)


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
