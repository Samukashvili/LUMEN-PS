"""Alignment pipeline — spec §6. Order matters; this has the most failure modes.

  1. segment the leaf (masking, §6.5)
  2. rigid align scan k -> scan 0 (fiducials if present, else nominal 90*k + ECC
     on a *lighting-invariant* proxy, §6.2)
  3. non-rigid warp on a lighting-invariant proxy, applied to the ORIGINAL image
     (§6.4) — NEVER optical flow on the raw differently-lit images
  4. validity / mask-agreement (§6.5)
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "segment_leaf",
    "detect_aruco",
    "rigid_align",
    "build_proxy",
    "nonrigid_warp",
    "mask_agreement",
    "estimate_flat_field_from_background",
    "union_leaf_bbox",
]


def union_leaf_bbox(scan_paths, cfg, coarse=0.25, margin_frac=0.04):
    """Common leaf ROI across all scans, in FULL-res pixel coords (X0,X1,Y0,Y1).

    Segments each scan at a coarse scale (cheap, one at a time), unions the leaf
    bounding boxes, and adds a margin. Used to crop the stack before a full-res
    solve so it fits in memory (spec §11 memory note).
    """
    from . import io  # local import; io has no align dependency

    is_srgb = cfg["io"]["input_is_srgb"]
    lw = tuple(cfg["io"]["luma_weights"])
    m = cfg["align"]["mask"]
    kw = dict(close_radius=m["close_radius"], open_radius=m["open_radius"],
              keep_largest=m["keep_largest"])
    x0 = y0 = 1e18
    x1 = y1 = -1e18
    full_w = full_h = 0
    for sp in scan_paths:
        r, _ = io.load_image_linear(sp, is_srgb, coarse)
        l = io.to_luminance(r, lw)
        mask = segment_leaf(l, **kw)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        s = 1.0 / coarse
        x0 = min(x0, xs.min() * s); x1 = max(x1, xs.max() * s)
        y0 = min(y0, ys.min() * s); y1 = max(y1, ys.max() * s)
        full_w = max(full_w, int(round(l.shape[1] * s)))
        full_h = max(full_h, int(round(l.shape[0] * s)))
    if x1 < x0:  # no leaf found anywhere -> whole frame
        return (0, full_w, 0, full_h)
    mx = int(round(margin_frac * max(full_w, full_h)))
    return (max(0, int(x0) - mx), min(full_w, int(x1) + mx),
            max(0, int(y0) - mx), min(full_h, int(y1) + mx))


# --------------------------------------------------------------------------- #
# Masking (§6.5)
# --------------------------------------------------------------------------- #
def segment_leaf(luma: np.ndarray, close_radius=5, open_radius=3, keep_largest=True):
    """Otsu on linear luminance -> morphology -> largest component. Leaf=True."""
    x = luma.astype(np.float32)
    x = x / (x.max() + 1e-8)
    u8 = np.clip(x * 255, 0, 255).astype(np.uint8)
    # leaf is DARKER than the bright white background -> invert so leaf is FG
    _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = th > 0
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1,) * 2)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    if open_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius * 2 + 1,) * 2)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    if keep_largest:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = lbl == biggest
    return mask


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        h, w = mask.shape
        return np.array([w / 2.0, h / 2.0])
    return np.array([xs.mean(), ys.mean()])


# --------------------------------------------------------------------------- #
# Fiducials (§6.1) — used when a registration card is present
# --------------------------------------------------------------------------- #
def detect_aruco(img_u8, dict_name="DICT_4X4_50"):
    """Return {marker_id: (cx, cy)} centres, or {} if none / no aruco module."""
    if not hasattr(cv2, "aruco"):
        return {}
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    try:
        det = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(img_u8)
    except AttributeError:  # older API
        corners, ids, _ = cv2.aruco.detectMarkers(img_u8, adict)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = tuple(c.reshape(4, 2).mean(axis=0))
    return out


def _to_u8(luma):
    x = luma / (luma.max() + 1e-8)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Rigid alignment (§6.2)
# --------------------------------------------------------------------------- #
def rigid_align(
    ref_luma,
    mov_luma,
    mov_extra,          # list of extra arrays to warp with the SAME transform (rgb, mask...)
    k: int,
    ref_mask=None,
    mov_mask=None,
    cfg=None,
):
    """Align moving scan k to the reference. Returns (warped_mov_luma, warped_extra,
    theta_deg, method).

    Prefers fiducials; else de-rotate by nominal -k*90 about the leaf centroid and
    refine with ECC on the mask distance transform (lighting-invariant).
    """
    cfg = cfg or {}
    H, W = ref_luma.shape
    fcfg = cfg.get("fiducials", {})
    method = "none"
    M = None

    # ---- fiducial path ----
    if fcfg.get("enabled", True) and fcfg.get("method", "aruco") == "aruco":
        ref_f = detect_aruco(_to_u8(ref_luma), fcfg.get("aruco_dict", "DICT_4X4_50"))
        mov_f = detect_aruco(_to_u8(mov_luma), fcfg.get("aruco_dict", "DICT_4X4_50"))
        common = sorted(set(ref_f) & set(mov_f))
        if len(common) >= fcfg.get("min_markers", 3):
            src = np.array([mov_f[i] for i in common], np.float32)
            dst = np.array([ref_f[i] for i in common], np.float32)
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            method = f"fiducials({len(common)})"

    # ---- fallback: nominal 90*k about centroid + ECC refine ----
    if M is None:
        rcfg = cfg.get("rigid", {})
        step = rcfg.get("nominal_step_deg", 90.0)
        angle = -step * k  # de-rotate to bring scan k back to the reference frame
        c_mov = _centroid(mov_mask) if mov_mask is not None else np.array([W / 2, H / 2])
        c_ref = _centroid(ref_mask) if ref_mask is not None else np.array([W / 2, H / 2])
        R = cv2.getRotationMatrix2D(tuple(c_mov), angle, 1.0)
        R[0, 2] += c_ref[0] - c_mov[0]
        R[1, 2] += c_ref[1] - c_mov[1]
        M = R.astype(np.float32)
        method = f"nominal({-angle:.0f})"

        if rcfg.get("ecc_refine", True) and ref_mask is not None and mov_mask is not None:
            refined = _ecc_refine_on_dt(ref_mask, mov_mask, M)
            if refined is not None:
                M = refined
                method += "+ecc"

    theta = float(np.rad2deg(np.arctan2(M[1, 0], M[0, 0])))
    interp = cv2.INTER_LANCZOS4 if cfg.get("rigid", {}).get(
        "interpolation", "lanczos4") == "lanczos4" else cv2.INTER_CUBIC
    warped_luma = cv2.warpAffine(mov_luma, M, (W, H), flags=interp,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_extra = []
    for arr in mov_extra:
        flg = cv2.INTER_NEAREST if arr.dtype == bool or arr.dtype == np.uint8 else interp
        a = arr.astype(np.uint8) if arr.dtype == bool else arr
        w = cv2.warpAffine(a, M, (W, H), flags=flg, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
        warped_extra.append(w.astype(bool) if arr.dtype == bool else w)
    return warped_luma, warped_extra, theta, method


def _warped_mask_iou(ref_mask, mov_mask, M):
    """Overlap of mov_mask warped by the forward transform M against ref_mask."""
    H, W = ref_mask.shape
    wm = cv2.warpAffine(mov_mask.astype(np.uint8), M, (W, H),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0) > 0
    union = (ref_mask | wm).sum()
    return (ref_mask & wm).sum() / union if union else 0.0


def _ecc_refine_on_dt(ref_mask, mov_mask, M_init, iters=200, eps=1e-5):
    """Refine a Euclidean warp using ECC on mask distance transforms (lighting-free).

    ``findTransformECC`` estimates the BACKWARD map (template -> input, i.e.
    ref -> mov, the matrix warpAffine would take with WARP_INVERSE_MAP), while
    the pipeline applies transforms FORWARD (mov -> ref). Seed it with the
    inverse of the forward init and invert its result back; mixing the two
    conventions "converges" on near-symmetric masks but lands the subject far
    off. The refinement is only accepted when it does not reduce mask overlap.
    """
    def dt(m):
        d = cv2.distanceTransform((m.astype(np.uint8) * 255), cv2.DIST_L2, 3)
        return (d / (d.max() + 1e-8)).astype(np.float32)

    tref, tmov = dt(ref_mask), dt(mov_mask)
    init = cv2.invertAffineTransform(M_init.astype(np.float32)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, warp = cv2.findTransformECC(tref, tmov, init, cv2.MOTION_EUCLIDEAN,
                                       crit, None, 5)
    except cv2.error:
        return None
    M = cv2.invertAffineTransform(warp).astype(np.float32)
    if _warped_mask_iou(ref_mask, mov_mask, M) < \
            _warped_mask_iou(ref_mask, mov_mask, M_init):
        return None
    return M


# --------------------------------------------------------------------------- #
# Non-rigid warp (§6.4) — flow on a lighting-invariant proxy ONLY
# --------------------------------------------------------------------------- #
def build_proxy(luma, mask, cfg):
    """Lighting-invariant proxy for flow: mask distance-transform + vein high-pass.

    Both components are invariant to the shading differences we are measuring.
    """
    kinds = cfg.get("proxy", "mask_dt+veins")
    H, W = luma.shape
    parts = []
    if "mask_dt" in kinds:
        d = cv2.distanceTransform(mask.astype(np.uint8) * 255, cv2.DIST_L2, 3)
        parts.append((d / (d.max() + 1e-8)).astype(np.float32))
    if "veins" in kinds:
        sig = float(cfg.get("vein_highpass_sigma", 6.0))
        lp = cv2.GaussianBlur(luma, (0, 0), sig)
        hp = np.abs(luma - lp)
        hp *= mask.astype(np.float32)
        hi = np.percentile(hp[mask], 99) if mask.any() else 1.0
        parts.append(np.clip(hp / (hi + 1e-8), 0, 1).astype(np.float32))
    if not parts:
        return (luma * mask).astype(np.float32)
    return np.mean(parts, axis=0).astype(np.float32)


def _dis_flow(ref_proxy, mov_proxy, method="dis"):
    a = _to_u8(ref_proxy)
    b = _to_u8(mov_proxy)
    if method == "farneback" or not hasattr(cv2, "DISOpticalFlow_create"):
        return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseSpatialPropagation(True)
    return dis.calc(a, b, None)


def nonrigid_warp(ref_luma, ref_mask, mov_luma, mov_mask, mov_extra, cfg):
    """Compute dense flow ref->mov on the proxy; remap ORIGINAL mov onto ref.

    The flow is computed on a DOWNSCALED proxy (spec §6.4): thin leaves deform by
    hundreds of pixels at 600 dpi, which is beyond DIS/Farneback's search range at
    full res. Estimating on a small copy keeps displacements tractable, then the
    flow is upscaled (magnitudes scaled with it) and applied to the full-detail
    original. Resolution-independent and faster.

    Returns (warped_luma, warped_extra, flow, max_disp_px).
    """
    ncfg = cfg
    H, W = ref_luma.shape
    ref_proxy = build_proxy(ref_luma, ref_mask, ncfg)
    mov_proxy = build_proxy(mov_luma, mov_mask, ncfg)

    ds = float(ncfg.get("flow_downscale", 0.25))
    ds = min(1.0, max(0.05, ds))
    if ds < 0.999:
        sw, sh = max(1, int(W * ds)), max(1, int(H * ds))
        rp = cv2.resize(ref_proxy, (sw, sh), interpolation=cv2.INTER_AREA)
        mp = cv2.resize(mov_proxy, (sw, sh), interpolation=cv2.INTER_AREA)
        flow_s = _dis_flow(rp, mp, ncfg.get("method", "dis"))
        flow = cv2.resize(flow_s, (W, H), interpolation=cv2.INTER_LINEAR) / ds
    else:
        flow = _dis_flow(ref_proxy, mov_proxy, ncfg.get("method", "dis"))

    smooth = float(ncfg.get("flow_smooth_sigma", 2.0))
    if smooth > 0:
        flow = cv2.GaussianBlur(flow, (0, 0), smooth)

    # sanity clamp (full-res pixels)
    maxw = float(ncfg.get("max_warp_px", 60))
    mag = np.linalg.norm(flow, axis=-1)
    max_disp = float(mag.max()) if mag.size else 0.0
    flow = np.clip(flow, -maxw, maxw)

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]

    def remap(arr):
        if arr.dtype == bool:
            r = cv2.remap(arr.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return r.astype(bool)
        return cv2.remap(arr.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    warped_luma = remap(mov_luma)
    warped_extra = [remap(a) for a in mov_extra]
    return warped_luma, warped_extra, flow, max_disp


# --------------------------------------------------------------------------- #
# Validity (§6.5)
# --------------------------------------------------------------------------- #
def mask_agreement(ref_mask, warped_masks, min_valid_samples=3):
    """Per-pixel valid-sample count and the overall valid region.

    valid pixel = inside ref mask AND >= min_valid_samples warped samples present.
    Returns (nsamples[H,W] int, valid[H,W] bool, per_sample_valid[N,H,W] bool).
    """
    stack = np.stack(warped_masks, axis=0)
    nsamples = stack.sum(axis=0).astype(np.int32)
    valid = ref_mask & (nsamples >= min_valid_samples)
    return nsamples, valid, stack


# --------------------------------------------------------------------------- #
# Flat-field fallback: fit a smooth illumination surface from the background
# --------------------------------------------------------------------------- #
def estimate_flat_field_from_background(luma, leaf_mask, order=3):
    """When no blank scan exists, model the lamp falloff from the bright surround.

    Fits a low-order 2-D polynomial to the non-leaf (background) pixels — the
    matte white card/lid — giving a smooth flat-field to divide out (§5.3).
    """
    H, W = luma.shape
    bg = ~leaf_mask
    ys, xs = np.nonzero(bg)
    z = luma[ys, xs]
    xn = xs / W - 0.5
    yn = ys / H - 0.5
    terms = [xn**i * yn**j for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack(terms, axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    gx, gy = np.meshgrid(np.arange(W) / W - 0.5, np.arange(H) / H - 0.5)
    terms_full = [gx**i * gy**j for i in range(order + 1) for j in range(order + 1 - i)]
    flat = sum(c * t for c, t in zip(coef, terms_full)).astype(np.float32)
    return np.clip(flat, 1e-4, None)
