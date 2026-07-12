"""leafscan CLI — orchestrates the full pipeline (spec §11, §12).

    python -m leafscan.cli run   --scans scans/leaf --out out
    python -m leafscan.cli capture --out scans/leaf/k0.png --dpi 600 --color
    python -m leafscan.cli selftest

The pipeline consumes a folder of lossless images; it does not depend on the
capture helper (spec §4.2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from . import align, calibrate as calib, integrate, io, outputs, qa
from .lights import light_directions, nominal_thetas, format_light_table
from .solve import photometric_solve

DEFAULT_CFG = Path(__file__).with_name("config.yaml")


# --------------------------------------------------------------------------- #
def load_config(path=None, overrides=None):
    with open(path or DEFAULT_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        cfg.setdefault(k, {})
        if isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _find_scans(scan_arg, n=4):
    """Resolve 4 scan paths from a folder (sorted) or an explicit comma list."""
    p = Path(scan_arg)
    if p.is_dir():
        exts = (".tif", ".tiff", ".png")
        skip = ("flat", "calib", "thumb", "preview", "small", "trans")
        files = sorted([f for f in p.iterdir()
                        if f.suffix.lower() in exts
                        and not any(s in f.stem.lower() for s in skip)])
        if len(files) < n:
            raise SystemExit(f"Need {n} scans in {p}, found {len(files)}: {files}")
        return files[:n]
    files = [Path(x) for x in str(scan_arg).split(",")]
    if len(files) != n:
        raise SystemExit(f"Expected {n} scan paths, got {len(files)}")
    return files


def _color_albedo(rgb_stack, normal, L, weights, backend="auto"):
    """Lighting-free per-channel base colour: mean over surviving samples of
    rgb_k / max(N.L_k, 0). Spec §5.2 (keep RGB for albedo)."""
    from .compute import color_albedo
    return color_albedo(rgb_stack, normal, L, weights, backend=backend)


def _apply_box(arr, box_full, scale):
    """Crop a loaded (already `scale`d) array to a full-res ROI box (X0,X1,Y0,Y1)."""
    X0, X1, Y0, Y1 = box_full
    x0, x1 = int(round(X0 * scale)), int(round(X1 * scale))
    y0, y1 = int(round(Y0 * scale)), int(round(Y1 * scale))
    h, w = arr.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    return arr[y0:y1, x0:x1]


def _fill_invalid(arr, valid, region):
    """Nearest-valid fill of invalid pixels within ``region`` (spec §9.2)."""
    from scipy.ndimage import distance_transform_edt
    invalid = region & ~valid
    if not invalid.any():
        return arr
    idx = distance_transform_edt(~valid, return_indices=True, return_distances=False)
    filled = arr.copy()
    iy, ix = idx
    if arr.ndim == 3:
        filled[invalid] = arr[iy[invalid], ix[invalid]]
    else:
        filled[invalid] = arr[iy[invalid], ix[invalid]]
    return filled


# --------------------------------------------------------------------------- #
def run_pipeline(cfg, scan_paths, out_dir, flat_path=None, calib_paths=None,
                 scale=None, verbose=True, log_fn=None, auto_crop=False,
                 cancel_check=None, capture_rois=None, capture_dpi=None):
    """Run the full pipeline.

    ``log_fn`` — optional callback(str) for streaming progress (the web UI passes
    one; defaults to print). ``auto_crop`` — crop the 4 scans to a common leaf
    ROI before the full-res solve so large scans fit in memory.
    ``capture_rois`` — optional per-scan (x, y, w, h) bed rectangles in mm (None
    entries = full bed) from the smart-ROI capture; with ``capture_dpi`` they let
    the pipeline place each ROI capture at its true bed offset so alignment and
    flat-fielding see consistent geometry.
    """
    out_dir = Path(out_dir)
    qa_dir = out_dir / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = scale if scale is not None else cfg["runtime"]["scale"]
    is_srgb = cfg["io"]["input_is_srgb"]
    lw = tuple(cfg["io"]["luma_weights"])

    if log_fn is None:
        log_fn = print

    def log(*a):
        if verbose:
            log_fn(" ".join(str(x) for x in a))

    def check_cancelled():
        if cancel_check:
            cancel_check()

    compute_backend = cfg.get("runtime", {}).get("compute", "auto")
    from .compute import backend_description
    log(f"[compute] {backend_description(compute_backend)}; tiled CUDA arrays enabled")
    # Avoid monopolizing every CPU core during OpenCV-only alignment stages.
    import cv2
    cv2.setNumThreads(max(1, int(cfg.get("runtime", {}).get("cpu_threads", 4))))

    # ---- 0. optional auto-ROI crop (keeps full-res stacks within memory) ----
    crop_box = None          # full-bed-space box (equal-size full captures only)
    canvas_box = None        # canvas-space box (variable/ROI captures)
    rois = list(capture_rois) if capture_rois else None
    have_geometry = bool(rois) and len(rois) == len(scan_paths) \
        and any(r is not None for r in rois) and bool(capture_dpi)
    if auto_crop and not have_geometry:
        from PIL import Image
        source_shapes = []
        for path in scan_paths:
            check_cancelled()
            with Image.open(path) as image:
                source_shapes.append((image.height, image.width))
        if len(set(source_shapes)) == 1:
            crop_box = align.union_leaf_bbox(scan_paths, cfg)
            log(f"[crop] auto-ROI x[{crop_box[0]},{crop_box[1]}] y[{crop_box[2]},{crop_box[3]}]")
        else:
            log("[crop] variable-size captures without ROI geometry; deferring common crop until load")

    # ---- 1. load + linearize ----
    log(f"[load] {len(scan_paths)} scans, scale={scale}, srgb={is_srgb}")
    rgb, metas = [], []
    for i, sp in enumerate(scan_paths):
        check_cancelled()
        r, meta = io.load_image_linear(sp, is_srgb, scale)
        if crop_box is not None:
            r = _apply_box(r, crop_box, scale)
        rgb.append(r); metas.append((sp, meta))
    native_shapes = [r.shape for r in rgb]
    bed_rect = None
    if have_geometry:
        # Rebuild one bed-coordinate canvas from the per-scan ROI captures so
        # alignment and flat-fielding see the same geometry as full-bed scans.
        rgb, bed_rect = _assemble_bed_canvas(rgb, rois, capture_dpi, scale)
        log(f"[crop] placed ROI captures at bed offsets; canvas {rgb[0].shape[:2]}")
    elif len({r.shape[:2] for r in rgb}) > 1:
        rgb = _pad_stack_to_common_canvas(rgb)
        log(f"[crop] normalized variable ROI scans to common canvas {rgb[0].shape[:2]}")
    luma = []
    for i, (sp, meta) in enumerate(metas):
        check_cancelled()
        luma.append(io.to_luminance(rgb[i], lw))
        log(f"  scan{i}: {sp.name} {native_shapes[i]} -> {rgb[i].shape} "
            f"native={meta['native_dtype']} distinct={meta['distinct_levels']} "
            f"genuine_high_bit={meta['genuine_high_bit']}")
    variable_roi = have_geometry or len({s[:2] for s in native_shapes}) > 1
    if auto_crop and variable_roi:
        canvas_box = _common_content_bbox(luma, cfg)
        log(f"[crop] common ROI x[{canvas_box[0]},{canvas_box[1]}] "
            f"y[{canvas_box[2]},{canvas_box[3]}]")
        rgb = [_apply_box(r, canvas_box, 1.0) for r in rgb]
        luma = [_apply_box(l, canvas_box, 1.0) for l in luma]
    H, W = luma[0].shape

    # ---- 2. flat-field + dark (mandatory, §5.3) ----
    ref_mask0 = align.segment_leaf(luma[0], **_mask_kw(cfg))
    flat = None
    if flat_path:
        flat_rgb, _ = io.load_image_linear(flat_path, is_srgb, scale)
        if bed_rect is not None:
            flat_rgb = _crop_bed_rect(flat_rgb, bed_rect)   # full-bed -> canvas
        elif crop_box is not None:
            flat_rgb = _apply_box(flat_rgb, crop_box, scale)
        if canvas_box is not None:
            flat_rgb = _apply_box(flat_rgb, canvas_box, 1.0)
        if flat_rgb.shape[:2] == luma[0].shape:
            flat = io.to_luminance(flat_rgb, lw)
            flat_src = f"scan:{Path(flat_path).name}"
        else:
            log(f"[flat] blank scan {flat_rgb.shape[:2]} does not match the capture "
                f"frame {luma[0].shape}; falling back to background fit")
    if flat is not None:
        flats = [flat] * len(luma)
    else:
        # Fit the lamp falloff from each scan's OWN background: with ROI
        # captures the scans cover different bed regions, so scan 0's falloff
        # is not valid for the others.
        flat_src = "background-fit (no usable blank scan)"
        pre_masks = [ref_mask0] + [align.segment_leaf(l, **_mask_kw(cfg))
                                   for l in luma[1:]]
        flats = [align.estimate_flat_field_from_background(l, m)
                 for l, m in zip(luma, pre_masks)]
    bl = cfg["io"]["black_level"]
    dark = io.estimate_black_level(flats[0], cfg["io"]["black_percentile"]) if bl is None else bl
    log(f"[flat] source={flat_src}  dark={dark:.4f}")

    blur = cfg["io"]["flat_field_blur_sigma"]
    luma = [io.flat_field_correct(l, f, dark, blur) for l, f in zip(luma, flats)]
    rgb = [io.flat_field_correct(r, f, dark, blur) for r, f in zip(rgb, flats)]

    # ---- 3. masks ----
    masks = [align.segment_leaf(l, **_mask_kw(cfg)) for l in luma]
    ref_luma, ref_rgb, ref_mask = luma[0], rgb[0], masks[0]

    # ---- 4. rigid align each k -> 0 ----
    acfg = cfg["align"]
    thetas = [0.0]
    al_luma = [ref_luma]; al_rgb = [ref_rgb]; al_mask = [ref_mask]
    for k in range(1, len(luma)):
        check_cancelled()
        wl, (wr, wm), th, method = align.rigid_align(
            ref_luma, luma[k], [rgb[k], masks[k]], k,
            ref_mask=ref_mask, mov_mask=masks[k], cfg=acfg)
        log(f"[rigid] scan{k}: theta={th:+.2f} deg via {method}")
        thetas.append(th)
        al_luma.append(wl); al_rgb.append(wr); al_mask.append(wm)

    # ---- 5. non-rigid warp each k -> 0 (proxy flow, applied to originals) ----
    if acfg["nonrigid"]["enabled"]:
        for k in range(1, len(al_luma)):
            check_cancelled()
            wl, (wr, wm), flow, md = align.nonrigid_warp(
                ref_luma, ref_mask, al_luma[k], al_mask[k],
                [al_rgb[k], al_mask[k]], acfg["nonrigid"])
            log(f"[nonrigid] scan{k}: max_disp={md:.1f}px "
                f"(clamp {acfg['nonrigid']['max_warp_px']})")
            al_luma[k], al_rgb[k], al_mask[k] = wl, wr, wm

    I_stack = np.stack(al_luma, axis=0).astype(np.float32)
    rgb_stack = np.stack(al_rgb, axis=0).astype(np.float32)

    # ---- 6. validity / agreement (§6.5) ----
    nsamples, valid, per_sample = align.mask_agreement(
        ref_mask, al_mask, acfg["mask"]["min_valid_samples"])
    valid_stack = per_sample
    log(f"[valid] {int(valid.sum())} pixels valid "
        f"({100*valid.sum()/max(1,ref_mask.sum()):.1f}% of leaf)")

    # ---- 7. calibrate lights (§7) ----
    az0, el, source = _calibrate(cfg, calib_paths, I_stack, thetas, valid_stack,
                                 is_srgb, lw, log)
    check_cancelled()

    L = light_directions(az0, el, thetas)
    log(format_light_table(L, thetas, az0, el))

    # ---- 8. photometric solve (§8) ----
    scfg = cfg["solve"]
    out = photometric_solve(I_stack, L, valid_stack=valid_stack,
                            rejection=scfg["rejection"],
                            min_surviving=scfg["min_surviving"],
                            ridge_lambda=float(scfg["ridge_lambda"]),
                            backend=cfg.get("runtime", {}).get("compute", "auto"))
    check_cancelled()
    normal, albedo_scalar, valid = out["normal"], out["albedo"], out["valid"]
    log(f"[solve] rejection={scfg['rejection']} -> {int(valid.sum())} solved pixels")

    # ---- 8.5 misregistration repair (§8.5) ----
    # Manual re-flattening between rotations leaves small locally-misaligned
    # patches in single scans; their samples break Lambertian consensus and
    # produce speckled off-normal pixels. Drop the worst-residual sample and
    # re-solve where possible; invalidate (-> inpaint at step 9) the rest.
    repair_map = None
    misreg_fill = None
    mcfg = scfg.get("misreg", {})
    if mcfg.get("enabled", False):
        from . import cleanup
        out, repaired, misreg_fill = cleanup.residual_repair(
            I_stack, L, out, valid_stack=valid_stack,
            flush_deg=float(mcfg.get("flush_deg", 12.0)),
            hard_rel=float(mcfg.get("hard_rel", 0.7)),
            improve_deg=float(mcfg.get("improve_deg", 5.0)),
            fill_flush_deg=float(mcfg.get("fill_flush_deg", 20.0)),
            fill_rel=float(mcfg.get("fill_rel", 0.5)),
            ref_sigma=float(mcfg.get("ref_sigma", 6.0)),
            albedo_floor=float(mcfg.get("albedo_floor", 0.02)),
            min_surviving=scfg["min_surviving"],
            ridge_lambda=float(scfg["ridge_lambda"]),
            backend=cfg.get("runtime", {}).get("compute", "auto"))
        if misreg_fill.any():
            # inpaint unrecoverable pixels IN PLACE — they stay valid so the
            # alpha/core silhouette is not punched full of holes
            dil = int(mcfg.get("fill_dilate_px", 1))
            if dil > 0:
                import cv2
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil * 2 + 1,) * 2)
                misreg_fill = cv2.dilate(misreg_fill.astype(np.uint8), k).astype(bool) \
                    & out["valid"]
            n = cleanup.inpaint_field(out["normal"], misreg_fill)
            nrm = np.linalg.norm(n, axis=-1, keepdims=True)
            out["normal"] = n / np.clip(nrm, 1e-8, None)
        normal, albedo_scalar, valid = out["normal"], out["albedo"], out["valid"]
        repair_map = np.zeros(valid.shape, np.uint8)
        repair_map[repaired] = 128
        repair_map[misreg_fill] = 255
        log(f"[misreg] repaired={int(repaired.sum())}px "
            f"inpainted={int(misreg_fill.sum())}px "
            f"({100 * (repaired.sum() + misreg_fill.sum()) / max(1, valid.sum()):.2f}% of leaf)")

    # ---- 9. colour albedo + cleanup (§9.2) ----
    albedo_rgb = _color_albedo(rgb_stack, normal, L, out["weights"],
                               backend=compute_backend)
    check_cancelled()
    if misreg_fill is not None and misreg_fill.any():
        from . import cleanup
        albedo_rgb = cleanup.inpaint_field(albedo_rgb, misreg_fill)
    ocfg = cfg["output"]
    if ocfg["smooth"]["normals_bilateral"]:
        import cv2
        s = ocfg["smooth"]
        normal = cv2.bilateralFilter(normal, s["bilateral_d"],
                                     s["bilateral_sigma_color"], s["bilateral_sigma_space"])
        nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = normal / np.clip(nrm, 1e-8, None)

    # Edge trim + clean padding (fixes background-bleed halo and warp-stretch at
    # the boundary — same idea as UV padding). The leaf/background boundary is a
    # ring of mixed leaf+white pixels, and the warp's bilinear remap smears white
    # inward there. So: keep only the clean interior (erode), then re-pad outward
    # from that interior (nearest-valid) instead of leaving the stretched rim.
    import cv2
    ecfg = ocfg.get("edge", {})
    trim_px = int(ecfg.get("trim_px", 6))
    pad_px = int(ecfg.get("pad_px", 0))
    solved = ref_mask & valid
    core = solved                                 # clean, background-free silhouette
    if trim_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (trim_px * 2 + 1,) * 2)
        core = cv2.erode(solved.astype(np.uint8), k).astype(bool)

    # Cut the alpha at `core` and DO NOT extend data past it: nearest-valid fill
    # produces radial streaks at the silhouette (the "stretch" artifact), so we
    # only fill genuine interior holes (hidden inside the leaf). pad_px optionally
    # adds mip-bleed padding OUTSIDE the leaf — beyond the alpha, never seen on a
    # cutout.
    fill_region = core
    if pad_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_px * 2 + 1,) * 2)
        fill_region = cv2.dilate(core.astype(np.uint8), k).astype(bool)

    normal = _fill_invalid(normal, solved, fill_region)      # interior holes (+ pad)
    albedo_rgb = _fill_invalid(albedo_rgb, solved, fill_region)
    nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.where(nrm > 1e-8, normal / np.clip(nrm, 1e-8, None), normal)
    out_valid = fill_region       # clean core (+ optional padding)
    qa_valid = solved             # evaluate residual only where we truly solved

    # Alpha = the clean core silhouette, so the visible cutout is exactly the
    # background-free data — no white ring, no fill stretch.
    acfg = ocfg.get("alpha", {})
    alpha = None
    if acfg.get("enabled", True):
        alpha = core.astype(np.float32)
        feather = float(acfg.get("feather_px", 0.0))
        if feather > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    log(f"[edge] trim={trim_px}px pad={pad_px}px  core={int(core.sum())}px "
        f"solved={int(solved.sum())}px  alpha={'on' if alpha is not None else 'off'}")

    # ---- 9.5 final framing: centre the leaf in the delivered maps ----
    # The pre-solve crop is a memory optimisation computed from the UNALIGNED
    # union of leaf positions, so the reference leaf can sit off-centre in it
    # (especially with per-scan ROI captures). Re-crop symmetrically around the
    # final silhouette so outputs and QA share consistent, centred framing.
    if auto_crop:
        frame = _centered_frame(core if core.any() else solved)
        if frame is not None:
            sy, sx = frame
            normal = normal[sy, sx]
            albedo_rgb = albedo_rgb[sy, sx]
            albedo_scalar = albedo_scalar[sy, sx]
            core = core[sy, sx]
            out_valid = out_valid[sy, sx]
            qa_valid = qa_valid[sy, sx]
            nsamples = nsamples[sy, sx]
            I_stack = I_stack[:, sy, sx]
            rgb_stack = rgb_stack[:, sy, sx]
            out["weights"] = out["weights"][:, sy, sx]
            if alpha is not None:
                alpha = alpha[sy, sx]
            if repair_map is not None:
                repair_map = repair_map[sy, sx]
            log(f"[frame] centered crop -> {normal.shape[1]}x{normal.shape[0]} "
                f"(x[{sx.start},{sx.stop}] y[{sy.start},{sy.stop}])")

    # ---- 10. height integration (§9.1) ----
    height = None
    if cfg["integrate"]["enabled"]:
        # integrate over the clean core so padded pixels don't inject fake slopes
        height = integrate.integrate_height(
            normal, core, cfg["integrate"]["highpass_sigma"],
            backend=compute_backend)
        log("[integrate] Frankot-Chellappa height computed")
    check_cancelled()

    # ---- 11. outputs + QA ----
    written = outputs.write_outputs(
        out_dir, normal, albedo_rgb, out_valid, height, alpha=alpha,
        normal_bits=ocfg["normal_bits"],
        albedo_linear=ocfg["albedo_linear"], albedo_srgb=ocfg["albedo_srgb"])
    check_cancelled()
    for p in written:
        log(f"[out] {p}")

    sub = io.subsurface_hint(rgb_stack[0])
    stats = qa.write_qa(qa_dir, I_stack=I_stack, normal=normal, albedo=albedo_scalar,
                        L=L, valid=qa_valid, nsamples=nsamples, thetas=thetas,
                        az0=az0, el=el, subsurface=sub, weights=out["weights"],
                        repair=repair_map,
                        backend=compute_backend,
                        extra_text=f"light_source={source}  flat={flat_src}")
    log(f"[qa] residual means: {[round(s['mean'],4) for s in stats]}  -> {qa_dir}")
    return {"out_dir": out_dir, "az0": az0, "el": el, "thetas": thetas,
            "residual": stats, "valid_px": int(out_valid.sum())}


def _mask_kw(cfg):
    m = cfg["align"]["mask"]
    return dict(close_radius=m["close_radius"], open_radius=m["open_radius"],
                keep_largest=m["keep_largest"])


def _pad_stack_to_common_canvas(stack):
    """Edge-pad variable-size ROI captures so the solver has one image canvas."""
    if not stack:
        return stack
    h = max(a.shape[0] for a in stack)
    w = max(a.shape[1] for a in stack)
    out = []
    for arr in stack:
        top = (h - arr.shape[0]) // 2
        left = (w - arr.shape[1]) // 2
        pad = ((top, h - arr.shape[0] - top), (left, w - arr.shape[1] - left))
        if arr.ndim == 3:
            pad += ((0, 0),)
        out.append(np.pad(arr, pad, mode="edge"))
    return out


def _assemble_bed_canvas(stack, rois_mm, dpi, scale):
    """Place ROI captures at their true bed offsets on one shared canvas.

    ``rois_mm[i]`` is the (x, y, w, h) glass rectangle scan i was captured from
    (None = full bed, offset 0). Returns ``(new_stack, bed_rect)`` where
    ``bed_rect = (x0, y0, x1, y1)`` is the canvas extent in *scaled* full-bed
    pixels — usable to crop a full-bed flat scan onto the same canvas.
    """
    px_per_mm = float(dpi) / 25.4 * float(scale)
    offs = []
    for roi in rois_mm:
        x = float(roi[0]) if roi else 0.0
        y = float(roi[1]) if roi else 0.0
        offs.append((int(round(x * px_per_mm)), int(round(y * px_per_mm))))
    x0 = min(o[0] for o in offs)
    y0 = min(o[1] for o in offs)
    x1 = max(o[0] + a.shape[1] for o, a in zip(offs, stack))
    y1 = max(o[1] + a.shape[0] for o, a in zip(offs, stack))
    out = []
    for arr, (ox, oy) in zip(stack, offs):
        top, left = oy - y0, ox - x0
        pad = ((top, (y1 - y0) - arr.shape[0] - top),
               (left, (x1 - x0) - arr.shape[1] - left))
        if arr.ndim == 3:
            pad += ((0, 0),)
        out.append(np.pad(arr, pad, mode="edge"))
    return out, (x0, y0, x1, y1)


def _crop_bed_rect(arr, rect):
    """Crop a full-bed image to a (x0, y0, x1, y1) pixel rect, edge-padding any
    overhang so the result always matches the rect size."""
    x0, y0, x1, y1 = rect
    h, w = arr.shape[:2]
    sub = arr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    ph, pw = (y1 - y0) - sub.shape[0], (x1 - x0) - sub.shape[1]
    if ph or pw:
        top, left = max(0, -y0), max(0, -x0)
        pad = ((top, ph - top), (left, pw - left))
        if arr.ndim == 3:
            pad += ((0, 0),)
        sub = np.pad(sub, pad, mode="edge")
    return sub


def _centered_frame(mask, margin_frac=0.04):
    """Symmetric crop window (y-slice, x-slice) centring ``mask`` in the frame.

    Returns None when there is nothing to centre or nothing to crop.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    H, W = mask.shape
    bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    m = int(round(margin_frac * max(bw, bh)))
    cx = int(round((xs.min() + xs.max()) / 2.0))
    cy = int(round((ys.min() + ys.max()) / 2.0))
    hw, hh = bw // 2 + m + 1, bh // 2 + m + 1
    x0, x1 = max(0, cx - hw), min(W, cx + hw)
    y0, y1 = max(0, cy - hh), min(H, cy + hh)
    if x0 == 0 and y0 == 0 and x1 == W and y1 == H:
        return None
    return slice(y0, y1), slice(x0, x1)


def _common_content_bbox(luma, cfg, margin_frac=0.04):
    """Return a common current-resolution bbox around all segmented subjects."""
    kw = _mask_kw(cfg)
    x0 = y0 = 1e18
    x1 = y1 = -1e18
    full_h, full_w = luma[0].shape
    for image in luma:
        mask = align.segment_leaf(image, **kw)
        ys, xs = np.nonzero(mask)
        if xs.size:
            x0, x1 = min(x0, xs.min()), max(x1, xs.max())
            y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    if x1 < x0:
        return (0, full_w, 0, full_h)
    margin = int(round(margin_frac * max(full_h, full_w)))
    return (max(0, int(x0) - margin), min(full_w, int(x1) + margin),
            max(0, int(y0) - margin), min(full_h, int(y1) + margin))


def _calibrate(cfg, calib_paths, I_stack, thetas, valid_stack, is_srgb, lw, log):
    lc = cfg["light"]
    if calib_paths:  # Method A
        c0, c90 = calib_paths
        r0, _ = io.load_image_linear(c0, is_srgb, cfg["runtime"]["scale"])
        r90, _ = io.load_image_linear(c90, is_srgb, cfg["runtime"]["scale"])
        l0, l90 = io.to_luminance(r0, lw), io.to_luminance(r90, lw)
        m0 = align.segment_leaf(l0, **_mask_kw(cfg))
        m90 = align.segment_leaf(l90, **_mask_kw(cfg))
        # corrugated fills most of the frame; if mask is tiny, use full frame
        if m0.mean() < 0.05:
            m0 = np.ones_like(m0); m90 = np.ones_like(m90)
        az0, el, err, info = calib.calibrate_from_corrugated(
            l0, m0, l90, m90, az0_prior=lc["az0_deg"])
        return az0, el, f"cardboard(err={err:.4f})"

    if lc["source"] == "config":
        log(f"[calib] using config az0={lc['az0_deg']} el={lc['el_deg']}")
        return lc["az0_deg"], lc["el_deg"], "config"

    # Method B — self-cal on a downscaled copy for speed.
    # Force rejection='none': with 4 samples and 3 unknowns the fit is
    # overdetermined, so the residual genuinely constrains (az0, el). Under
    # drop_brightest the per-pixel solve is exactly determined and the residual
    # would instead be dominated by the (specular) dropped sample.
    log("[calib] self-calibrating (Method B, all-samples) on downscaled stack...")
    ds = _downscale_stack(I_stack, 0.25)
    dv = _downscale_stack(valid_stack.astype(np.float32), 0.25) > 0.5
    az0, el, r, info = calib.self_calibrate(
        ds, thetas, valid_stack=dv, az0_seed=lc["az0_deg"], el_seed=lc["el_deg"],
        rejection="none", min_surviving=cfg["solve"]["min_surviving"])
    return az0, el, f"selfcal(res={r:.4f})"


def _downscale_stack(stack, scale):
    import cv2
    out = []
    for i in range(stack.shape[0]):
        a = stack[i]
        h, w = a.shape[:2]
        out.append(cv2.resize(a, (max(1, int(w*scale)), max(1, int(h*scale))),
                              interpolation=cv2.INTER_AREA))
    return np.stack(out, axis=0)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(prog="leafscan")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full pipeline")
    r.add_argument("--scans", required=True, help="folder of 4 scans or comma list")
    r.add_argument("--out", required=True)
    r.add_argument("--flat", default=None, help="optional flat-field scan")
    r.add_argument("--calib", default=None, help="corrugated 0deg,90deg comma pair")
    r.add_argument("--config", default=None)
    r.add_argument("--scale", type=float, default=None)
    r.add_argument("--auto-crop", dest="auto_crop", action="store_true",
                   help="crop scans to a common leaf ROI (fits full-res in memory)")
    r.add_argument("--quiet", action="store_true")

    c = sub.add_parser("capture", help="WIA capture helper")
    c.add_argument("--out", default=None)
    c.add_argument("--dpi", type=int, default=600)
    c.add_argument("--color", action="store_true", default=True)
    c.add_argument("--gray", dest="color", action="store_false")
    c.add_argument("--preview", action="store_true")

    sub.add_parser("selftest", help="run synthetic math tests")

    args = ap.parse_args(argv)

    if args.cmd == "capture":
        from . import capture_wia
        if args.preview:
            return capture_wia.scan_to_file(args.out or "scans/preview.png",
                                            dpi=150, color=True)
        return capture_wia.scan_to_file(args.out, dpi=args.dpi, color=args.color)

    if args.cmd == "selftest":
        import pytest
        root = Path(__file__).resolve().parent.parent
        raise SystemExit(pytest.main([str(root / "tests"), "-q"]))

    if args.cmd == "run":
        cfg = load_config(args.config)
        scans = _find_scans(args.scans)
        calib_paths = args.calib.split(",") if args.calib else None
        res = run_pipeline(cfg, scans, args.out, flat_path=args.flat,
                           calib_paths=calib_paths, scale=args.scale,
                           verbose=not args.quiet, auto_crop=args.auto_crop)
        print(f"\nDONE  az0={res['az0']:.2f} el={res['el']:.2f} "
              f"valid_px={res['valid_px']}  -> {res['out_dir']}")
        return res


if __name__ == "__main__":
    main()
