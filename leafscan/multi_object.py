"""Multi-object scan tracking and square texture-atlas reconstruction.

Each physical object is segmented and associated independently across the four
rotations. The established single-object pipeline then receives one tight crop
per rotation, so its rigid and non-rigid registration are local to that object.
All delivered maps are packed with one shared layout.
"""
from __future__ import annotations

import json
import math
import tempfile
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import align, io


def _read_rgb(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not read scan image: {path}")
    if raw.ndim == 2:
        return np.repeat(raw[..., None], 3, axis=-1)
    if raw.shape[-1] == 4:
        raw = raw[..., :3]
    return raw[..., ::-1]  # BGR -> RGB


def _write_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rgb[..., ::-1] if rgb.ndim == 3 else rgb
    if not cv2.imwrite(str(path), data):
        raise OSError(f"Could not write temporary object crop: {path}")


def _to_linear(rgb, input_is_srgb):
    if rgb.dtype == np.uint8:
        x = rgb.astype(np.float32) / 255.0
    elif rgb.dtype == np.uint16:
        x = rgb.astype(np.float32) / 65535.0
    else:
        maximum = float(np.nanmax(rgb))
        x = rgb.astype(np.float32) / (maximum if maximum > 0 else 1.0)
    return io.srgb_to_linear(x) if input_is_srgb else x


def _component_descriptor(rgb, component):
    mask = component["mask"]
    u8 = np.clip(rgb.astype(np.float32) / max(float(np.max(rgb)), 1.0) * 255, 0, 255)
    u8 = u8.astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [12, 6],
                        [0, 180, 0, 256]).reshape(-1)
    hist /= max(float(hist.sum()), 1.0)

    moments = cv2.moments(mask.astype(np.uint8))
    hu = cv2.HuMoments(moments).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30))
    x, y, w, h = component["bbox"]
    component["descriptor"] = {
        "hist": hist,
        "hu": hu,
        "area_fraction": component["area"] / mask.size,
        "aspect": min(w, h) / max(w, h),
        "position": np.asarray(component["centroid"], np.float32)
        / np.asarray([mask.shape[1], mask.shape[0]], np.float32),
    }
    return component


def detect_scan_objects(path, cfg):
    """Return low-resolution components plus the preview used to detect them."""
    raw = _read_rgb(path)
    mcfg = cfg.get("multi_object", {})
    max_dim = max(256, int(mcfg.get("detection_max_dimension", 1800)))
    factor = min(1.0, max_dim / max(raw.shape[:2]))
    if factor < 0.999:
        small = cv2.resize(raw, None, fx=factor, fy=factor,
                           interpolation=cv2.INTER_AREA)
    else:
        small = raw
    linear = _to_linear(small, cfg["io"]["input_is_srgb"])
    luma = io.to_luminance(linear, tuple(cfg["io"]["luma_weights"]))
    mask_cfg = cfg["align"]["mask"]
    components = align.segment_objects(
        luma,
        close_radius=int(mask_cfg.get("close_radius", 5)),
        open_radius=int(mask_cfg.get("open_radius", 3)),
        detect_interior_holes=bool(mask_cfg.get("detect_interior_holes", False)),
        min_component_fraction=float(mcfg.get("min_component_fraction", 0.001)),
        max_objects=int(mcfg.get("max_objects", 32)),
    )
    for component in components:
        _component_descriptor(small, component)
    return {
        "path": Path(path),
        "preview": small,
        "components": components,
        "native_shape": raw.shape[:2],
        "scale_x": raw.shape[1] / small.shape[1],
        "scale_y": raw.shape[0] / small.shape[0],
    }


def _match_cost(reference, candidate):
    a, b = reference["descriptor"], candidate["descriptor"]
    area = abs(math.log(max(a["area_fraction"], 1e-8)
                        / max(b["area_fraction"], 1e-8)))
    aspect = abs(a["aspect"] - b["aspect"])
    hu = float(np.mean(np.minimum(np.abs(a["hu"] - b["hu"]), 4.0)))
    color = float(np.abs(a["hist"] - b["hist"]).sum())
    position = float(np.linalg.norm(a["position"] - b["position"]))
    # Hu moments and colour carry most of the identity signal. Position is a
    # soft tie-breaker only: users are explicitly allowed to move every object.
    return 0.85 * area + 0.8 * aspect + 0.18 * hu + 0.9 * color + 0.2 * position


def associate_objects(detections, max_match_cost=3.0):
    """Associate every scan's components to scan 0 using global assignment."""
    if not detections or not detections[0]["components"]:
        raise ValueError("No separate objects were detected in the first scan")
    reference = detections[0]["components"]
    tracks = [[component] for component in reference]
    costs_by_scan = [[0.0] * len(reference)]
    for scan_index, detection in enumerate(detections[1:], start=1):
        candidates = detection["components"]
        if len(candidates) < len(reference):
            raise ValueError(
                f"Scan {scan_index} contains {len(candidates)} detected objects; "
                f"scan 0 contains {len(reference)}. Keep every object separated "
                "and visible in all four rotations."
            )
        matrix = np.asarray([
            [_match_cost(ref, candidate) for candidate in candidates]
            for ref in reference
        ], dtype=np.float32)
        rows, cols = linear_sum_assignment(matrix)
        assignment = {int(row): int(col) for row, col in zip(rows, cols)}
        scan_costs = []
        for ref_index, track in enumerate(tracks):
            candidate_index = assignment[ref_index]
            cost = float(matrix[ref_index, candidate_index])
            if cost > float(max_match_cost):
                raise ValueError(
                    f"Object {ref_index + 1} could not be identified reliably "
                    f"in scan {scan_index} (match cost {cost:.2f})."
                )
            track.append(candidates[candidate_index])
            scan_costs.append(cost)
        costs_by_scan.append(scan_costs)
    return tracks, costs_by_scan


def _native_crop_box(detection, component, margin_fraction):
    x, y, w, h = component["bbox"]
    x0 = x * detection["scale_x"]
    x1 = (x + w) * detection["scale_x"]
    y0 = y * detection["scale_y"]
    y1 = (y + h) * detection["scale_y"]
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * float(margin_fraction))
    native_h, native_w = detection["native_shape"]
    left = max(0, int(math.floor(cx - side * 0.5)))
    top = max(0, int(math.floor(cy - side * 0.5)))
    right = min(native_w, int(math.ceil(cx + side * 0.5)))
    bottom = min(native_h, int(math.ceil(cy + side * 0.5)))
    return left, top, right, bottom


def _write_detection_preview(path, detection):
    preview = detection["preview"].copy()
    for index, component in enumerate(detection["components"], start=1):
        x, y, w, h = component["bbox"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (241, 166, 74), 3)
        cv2.putText(preview, str(index), (x + 8, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (241, 166, 74), 2,
                    cv2.LINE_AA)
    _write_rgb(path, preview)


def _background_for(filename, shape, dtype, roughness_baseline):
    maximum = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    if len(shape) == 2:
        value = roughness_baseline if filename.startswith("roughness") else 0.0
        return np.full(shape, round(maximum * value), dtype=dtype)
    canvas = np.zeros(shape, dtype=dtype)
    if filename.startswith("normal_"):
        # OpenCV arrays are BGR(A): flat tangent normal RGB=(.5,.5,1).
        canvas[..., 0] = maximum
        canvas[..., 1] = round(maximum * 0.5)
        canvas[..., 2] = round(maximum * 0.5)
        if shape[-1] == 4:
            canvas[..., 3] = 0
    return canvas


def pack_atlas(object_dirs, out_dir, cfg):
    """Pack all common material maps into the same square cell layout."""
    roots = [Path(path) for path in object_dirs]
    common = None
    for root in roots:
        names = {path.name for path in root.glob("*.png")}
        common = names if common is None else common & names
    filenames = sorted(common or ())
    if not filenames:
        raise RuntimeError("Object reconstruction did not produce material maps")
    # A re-run may disable height, roughness, or alpha. Remove only files owned
    # by the reconstruction so the result viewport cannot pick up a stale map
    # from the previous mode/configuration.
    known_material_maps = {
        "normal_gl.png", "normal_dx.png", "albedo.png", "albedo_srgb.png",
        "alpha.png", "albedo_srgb_rgba.png", "normal_gl_rgba.png",
        "height.png", "roughness.png", "roughness_consensus.png",
        "roughness_ggx.png", "roughness_beckmann.png", "roughness_ward.png",
    }
    out_dir = Path(out_dir)
    for stale in known_material_maps - set(filenames):
        (out_dir / stale).unlink(missing_ok=True)

    count = len(roots)
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    size = max(512, int(cfg.get("multi_object", {}).get("atlas_size", 4096)))
    cell_w, cell_h = size // columns, size // columns
    gutter = int(round(min(cell_w, cell_h)
                       * float(cfg.get("multi_object", {}).get("gutter_fraction", 0.04))))

    sample_shapes = []
    for root in roots:
        sample = cv2.imread(str(root / filenames[0]), cv2.IMREAD_UNCHANGED)
        sample_shapes.append(sample.shape[:2])
    usable_w = max(1, cell_w - 2 * gutter)
    usable_h = max(1, cell_h - 2 * gutter)
    global_scale = min(
        usable_w / max(width for _, width in sample_shapes),
        usable_h / max(height for height, _ in sample_shapes),
    )
    placements = []
    for index, (height, width) in enumerate(sample_shapes):
        row, column = divmod(index, columns)
        placed_w = max(1, int(round(width * global_scale)))
        placed_h = max(1, int(round(height * global_scale)))
        x = column * cell_w + (cell_w - placed_w) // 2
        y = row * cell_h + (cell_h - placed_h) // 2
        placements.append((x, y, placed_w, placed_h))

    written = []
    baseline = float(cfg.get("roughness", {}).get("baseline", 0.7))

    def pack_one(filename, source_roots, destination):
        first = cv2.imread(str(source_roots[0] / filename), cv2.IMREAD_UNCHANGED)
        atlas_shape = (size, size) if first.ndim == 2 else (size, size, first.shape[-1])
        atlas = _background_for(filename, atlas_shape, first.dtype, baseline)
        for root, (x, y, width, height) in zip(source_roots, placements):
            source = cv2.imread(str(root / filename), cv2.IMREAD_UNCHANGED)
            if source is None:
                raise RuntimeError(f"Missing object map {root / filename}")
            interpolation = cv2.INTER_AREA if global_scale < 1 else cv2.INTER_LANCZOS4
            resized = cv2.resize(source, (width, height), interpolation=interpolation)
            atlas[y:y + height, x:x + width] = resized
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), atlas):
            raise OSError(f"Could not write atlas map: {destination}")
        return destination

    for filename in filenames:
        destination = pack_one(filename, roots, out_dir / filename)
        written.append(destination)

    qa_roots = [root / "qa" for root in roots]
    common_qa = None
    for root in qa_roots:
        names = {path.name for path in root.glob("*.png")}
        common_qa = names if common_qa is None else common_qa & names
    for filename in sorted(common_qa or ()):
        pack_one(filename, qa_roots, out_dir / "qa" / filename)
    return {
        "size": size,
        "columns": columns,
        "rows": rows,
        "placements": placements,
        "files": [path.name for path in written],
    }


def run_multi_object_pipeline(
    cfg, scan_paths, out_dir, flat_path=None, calib_paths=None, scale=None,
    verbose=True, log_fn=None, auto_crop=False, cancel_check=None,
    capture_rois=None, capture_dpi=None,
):
    """Detect, track, reconstruct, and atlas all subjects in the scan set."""
    del flat_path, capture_rois, capture_dpi  # object crops estimate their own local flat
    out_dir = Path(out_dir)
    qa_dir = out_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    log_fn = log_fn or print

    def log(message):
        if verbose:
            log_fn(str(message))

    def check_cancelled():
        if cancel_check:
            cancel_check()

    log("[objects] detecting separate subjects in all four scans...")
    detections = []
    for index, path in enumerate(scan_paths):
        check_cancelled()
        detection = detect_scan_objects(path, cfg)
        detections.append(detection)
        log(f"[objects] scan{index}: {len(detection['components'])} objects detected")
    tracks, costs = associate_objects(
        detections,
        max_match_cost=float(cfg.get("multi_object", {}).get("max_match_cost", 3.0)),
    )
    _write_detection_preview(qa_dir / "object_detection.png", detections[0])
    log(f"[objects] associated {len(tracks)} objects across every rotation")

    crop_boxes = [[None] * len(scan_paths) for _ in tracks]
    object_results = []
    object_output_dirs = []
    margin = float(cfg.get("multi_object", {}).get("crop_margin_fraction", 0.12))
    with tempfile.TemporaryDirectory(prefix="lumen-multi-") as temp_name:
        temp = Path(temp_name)
        # Read one large source at a time, then emit all small tracked crops.
        for scan_index, detection in enumerate(detections):
            check_cancelled()
            raw = _read_rgb(detection["path"])
            for object_index, track in enumerate(tracks):
                box = _native_crop_box(detection, track[scan_index], margin)
                crop_boxes[object_index][scan_index] = list(box)
                left, top, right, bottom = box
                _write_rgb(
                    temp / "scans" / f"object_{object_index + 1:02d}"
                    / f"k{scan_index}.png",
                    raw[top:bottom, left:right],
                )
            del raw

        from .cli import _run_single_pipeline
        object_cfg = deepcopy(cfg)
        object_cfg.setdefault("multi_object", {})["enabled"] = False
        object_cfg["align"]["mask"]["keep_largest"] = True
        object_cfg["runtime"]["auto_crop"] = True
        for object_index in range(len(tracks)):
            check_cancelled()
            number = object_index + 1
            log(f"[objects] reconstructing object {number}/{len(tracks)}")
            scan_root = temp / "scans" / f"object_{number:02d}"
            object_out = temp / "results" / f"object_{number:02d}"
            result = _run_single_pipeline(
                object_cfg,
                [scan_root / f"k{k}.png" for k in range(len(scan_paths))],
                object_out,
                flat_path=None,
                calib_paths=calib_paths,
                scale=scale,
                verbose=verbose,
                log_fn=lambda line, n=number: log_fn(f"[object {n:02d}] {line}"),
                auto_crop=True,
                cancel_check=cancel_check,
            )
            object_results.append(result)
            object_output_dirs.append(object_out)

        check_cancelled()
        atlas = pack_atlas(object_output_dirs, out_dir, cfg)

    manifest = {
        "version": 1,
        "object_count": len(tracks),
        "atlas": {
            **atlas,
            "placements": [
                {"object": index + 1, "x": x, "y": y, "width": w, "height": h}
                for index, (x, y, w, h) in enumerate(atlas["placements"])
            ],
        },
        "tracks": [
            {
                "object": index + 1,
                "crop_boxes": crop_boxes[index],
                "match_costs": [round(costs[scan][index], 4)
                                for scan in range(len(detections))],
            }
            for index in range(len(tracks))
        ],
    }
    with open(out_dir / "multi_object_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    report_lines = [
        f"multi-object atlas: {len(tracks)} objects",
        f"atlas: {atlas['size']}x{atlas['size']}  "
        f"grid={atlas['columns']}x{atlas['rows']}",
        "association cost by object (scan0, scan1, scan2, scan3):",
    ]
    for track in manifest["tracks"]:
        report_lines.append(
            f"  object {track['object']:02d}: {track['match_costs']}"
        )
    (qa_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    log(f"[atlas] packed {len(tracks)} objects into "
        f"{atlas['size']}x{atlas['size']} maps ({atlas['columns']} columns)")
    residual = []
    scan_count = len(object_results[0].get("residual", []))
    for scan_index in range(scan_count):
        samples = [result["residual"][scan_index] for result in object_results]
        residual.append({
            "scan": scan_index,
            "mean": float(np.mean([sample["mean"] for sample in samples])),
            "p95": float(np.mean([sample["p95"] for sample in samples])),
            "max": float(np.max([sample["max"] for sample in samples])),
        })
    return {
        "out_dir": out_dir,
        "az0": float(np.mean([result["az0"] for result in object_results])),
        "el": float(np.mean([result["el"] for result in object_results])),
        "thetas": object_results[0]["thetas"],
        "residual": residual,
        "valid_px": sum(result["valid_px"] for result in object_results),
        "roughness": {"objects": len(object_results)},
        "object_count": len(object_results),
        "atlas_size": atlas["size"],
    }
