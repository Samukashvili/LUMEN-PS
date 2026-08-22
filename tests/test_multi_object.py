import json

import cv2
import numpy as np
from PIL import Image

from leafscan.cli import load_config, run_pipeline
from leafscan.io import linear_to_srgb
from leafscan.lights import light_direction
from leafscan.multi_object import associate_objects, detect_scan_objects


def _subject(size, variant):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2
    rx = size * (0.26 if variant == 0 else 0.33)
    ry = size * (0.36 if variant == 0 else 0.24)
    r = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = r < 1
    z = (0.22 + variant * 0.07) * np.exp(-2.5 * r)
    rho = 0.35 + 0.18 * variant + 0.1 * np.sin((xx + variant * yy) / 7)
    return z, np.clip(rho, 0.2, 0.85), mask


def _render(z, rho, mask, rotation, light):
    z = np.rot90(z, rotation)
    rho = np.rot90(rho, rotation)
    mask = np.rot90(mask, rotation)
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(z)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    intensity = np.clip(normal @ light, 0, None) * rho
    return np.where(mask, intensity, 1.0)


def _write_two_object_scans(root, size=120, largest_not_first=False):
    scan_dir = root / "scans"
    scan_dir.mkdir()
    light = light_direction(90, 35)
    subjects = [_subject(size, 0), _subject(size, 1)]
    # Swap the left/right placement in alternating scans. Identity association
    # must follow appearance, not the object's bed position.
    for k in range(4):
        canvas = np.ones((size + 30, size * 2 + 60), np.float32)
        order = ((1, 0) if k % 2 == 0 else (0, 1)) if largest_not_first \
            else ((0, 1) if k % 2 == 0 else (1, 0))
        for slot, subject_index in enumerate(order):
            image = _render(*subjects[subject_index], k, light)
            x = 15 + slot * (size + 30)
            canvas[15:15 + size, x:x + size] = image
        rgb = np.repeat(canvas[..., None], 3, axis=-1)
        srgb = (linear_to_srgb(rgb) * 255).astype(np.uint8)
        Image.fromarray(srgb, "RGB").save(scan_dir / f"k{k}.png")
    return sorted(scan_dir.glob("k*.png"))


def test_association_survives_position_swaps_and_rotation(tmp_path):
    scans = _write_two_object_scans(tmp_path)
    cfg = load_config()
    detections = [detect_scan_objects(path, cfg) for path in scans]

    assert [len(item["components"]) for item in detections] == [2, 2, 2, 2]
    tracks, costs = associate_objects(detections)
    assert len(tracks) == 2
    assert max(max(scan_costs) for scan_costs in costs) < 1.0

    # The two subjects have opposite aspect ratios after each 90-degree turn.
    # The matched track must alternate aspect rather than following its slot.
    for track in tracks:
        aspects = [component["descriptor"]["aspect"] for component in track]
        assert max(aspects) - min(aspects) < 0.08


def test_multi_object_pipeline_exports_one_square_shared_atlas(tmp_path):
    # Put the largest subject second in scan-0 atlas order.  Auto reconstructs
    # it first, so returned side metadata must follow reconstruction order,
    # while the output placements must remain in atlas order.
    scans = _write_two_object_scans(tmp_path, largest_not_first=True)
    cfg = load_config()
    cfg["multi_object"]["enabled"] = True
    cfg["multi_object"]["atlas_size"] = 512
    cfg["light"].update({"source": "config", "az0_deg": 90, "el_deg": 35})
    cfg["align"]["nonrigid"]["enabled"] = False
    cfg["solve"]["misreg"]["enabled"] = False
    cfg["roughness"]["enabled"] = False
    cfg["integrate"]["enabled"] = False

    result = run_pipeline(cfg, scans, tmp_path / "out", verbose=False)

    assert result["object_count"] == 2
    assert result["atlas_size"] == 512
    assert result["light_side_mode"] == "auto"
    assert result["light_side_info"]["reason"] != "manual-override"
    for filename in ("normal_gl.png", "normal_dx.png", "albedo.png",
                     "albedo_srgb.png", "alpha.png",
                     "albedo_srgb_rgba.png", "normal_gl_rgba.png"):
        image = cv2.imread(str(result["out_dir"] / filename), cv2.IMREAD_UNCHANGED)
        assert image.shape[:2] == (512, 512), filename

    with open(result["out_dir"] / "multi_object_manifest.json",
              encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["object_count"] == 2
    assert len(manifest["atlas"]["placements"]) == 2
    assert (result["out_dir"] / "qa" / "object_detection.png").exists()
