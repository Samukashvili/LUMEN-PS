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


def _color_albedo(rgb_stack, normal, L, weights):
    """Lighting-free per-channel base colour: mean over surviving samples of
    rgb_k / max(N.L_k, 0). Spec §5.2 (keep RGB for albedo)."""
    N, H, W, _ = rgb_stack.shape
    ndotl = np.clip(np.einsum("hwc,nc->nhw", normal, L), 0, None)  # (N,H,W)
    w = weights * (ndotl > 1e-3)
    est = rgb_stack / np.clip(ndotl[..., None], 1e-3, None)        # (N,H,W,3)
    num = (est * w[..., None]).sum(axis=0)
    den = np.clip(w.sum(axis=0), 1e-6, None)[..., None]
    return np.clip(num / den, 0, 1).astype(np.float32)


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
                 scale=None, verbose=True):
    out_dir = Path(out_dir)
    qa_dir = out_dir / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = scale if scale is not None else cfg["runtime"]["scale"]
    is_srgb = cfg["io"]["input_is_srgb"]
    lw = tuple(cfg["io"]["luma_weights"])

    def log(*a):
        if verbose:
            print(*a)

    # ---- 1. load + linearize ----
    log(f"[load] {len(scan_paths)} scans, scale={scale}, srgb={is_srgb}")
    rgb, luma = [], []
    for i, sp in enumerate(scan_paths):
        r, meta = io.load_image_linear(sp, is_srgb, scale)
        rgb.append(r)
        luma.append(io.to_luminance(r, lw))
        log(f"  scan{i}: {sp.name} {meta['shape']} native={meta['native_dtype']} "
            f"distinct={meta['distinct_levels']} genuine_high_bit={meta['genuine_high_bit']}")
    H, W = luma[0].shape

    # ---- 2. flat-field + dark (mandatory, §5.3) ----
    ref_mask0 = align.segment_leaf(luma[0], **_mask_kw(cfg))
    if flat_path:
        flat_rgb, _ = io.load_image_linear(flat_path, is_srgb, scale)
        flat = io.to_luminance(flat_rgb, lw)
        flat_src = f"scan:{Path(flat_path).name}"
    else:
        flat = align.estimate_flat_field_from_background(luma[0], ref_mask0)
        flat_src = "background-fit (no blank scan)"
    bl = cfg["io"]["black_level"]
    dark = io.estimate_black_level(flat, cfg["io"]["black_percentile"]) if bl is None else bl
    log(f"[flat] source={flat_src}  dark={dark:.4f}")

    blur = cfg["io"]["flat_field_blur_sigma"]
    luma = [io.flat_field_correct(l, flat, dark, blur) for l in luma]
    rgb = [io.flat_field_correct(r, flat, dark, blur) for r in rgb]

    # ---- 3. masks ----
    masks = [align.segment_leaf(l, **_mask_kw(cfg)) for l in luma]
    ref_luma, ref_rgb, ref_mask = luma[0], rgb[0], masks[0]

    # ---- 4. rigid align each k -> 0 ----
    acfg = cfg["align"]
    thetas = [0.0]
    al_luma = [ref_luma]; al_rgb = [ref_rgb]; al_mask = [ref_mask]
    for k in range(1, len(luma)):
        wl, (wr, wm), th, method = align.rigid_align(
            ref_luma, luma[k], [rgb[k], masks[k]], k,
            ref_mask=ref_mask, mov_mask=masks[k], cfg=acfg)
        log(f"[rigid] scan{k}: theta={th:+.2f} deg via {method}")
        thetas.append(th)
        al_luma.append(wl); al_rgb.append(wr); al_mask.append(wm)

    # ---- 5. non-rigid warp each k -> 0 (proxy flow, applied to originals) ----
    if acfg["nonrigid"]["enabled"]:
        for k in range(1, len(al_luma)):
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

    L = light_directions(az0, el, thetas)
    log(format_light_table(L, thetas, az0, el))

    # ---- 8. photometric solve (§8) ----
    scfg = cfg["solve"]
    out = photometric_solve(I_stack, L, valid_stack=valid_stack,
                            rejection=scfg["rejection"],
                            min_surviving=scfg["min_surviving"],
                            ridge_lambda=float(scfg["ridge_lambda"]))
    normal, albedo_scalar, valid = out["normal"], out["albedo"], out["valid"]
    log(f"[solve] rejection={scfg['rejection']} -> {int(valid.sum())} solved pixels")

    # ---- 9. colour albedo + cleanup (§9.2) ----
    albedo_rgb = _color_albedo(rgb_stack, normal, L, out["weights"])
    ocfg = cfg["output"]
    if ocfg["smooth"]["normals_bilateral"]:
        import cv2
        s = ocfg["smooth"]
        normal = cv2.bilateralFilter(normal, s["bilateral_d"],
                                     s["bilateral_sigma_color"], s["bilateral_sigma_space"])
        nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = normal / np.clip(nrm, 1e-8, None)

    normal = _fill_invalid(normal, valid, ref_mask)
    albedo_rgb = _fill_invalid(albedo_rgb, valid, ref_mask)
    nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.where(nrm > 1e-8, normal / np.clip(nrm, 1e-8, None), normal)
    out_valid = ref_mask  # solved+filled over the whole leaf

    # ---- 10. height integration (§9.1) ----
    height = None
    if cfg["integrate"]["enabled"]:
        height = integrate.integrate_height(
            normal, out_valid, cfg["integrate"]["highpass_sigma"])
        log("[integrate] Frankot-Chellappa height computed")

    # ---- 11. outputs + QA ----
    written = outputs.write_outputs(
        out_dir, normal, albedo_rgb, out_valid, height,
        normal_bits=ocfg["normal_bits"],
        albedo_linear=ocfg["albedo_linear"], albedo_srgb=ocfg["albedo_srgb"])
    for p in written:
        log(f"[out] {p}")

    sub = io.subsurface_hint(rgb_stack[0])
    stats = qa.write_qa(qa_dir, I_stack=I_stack, normal=normal, albedo=albedo_scalar,
                        L=L, valid=out_valid, nsamples=nsamples, thetas=thetas,
                        az0=az0, el=el, subsurface=sub, weights=out["weights"],
                        extra_text=f"light_source={source}  flat={flat_src}")
    log(f"[qa] residual means: {[round(s['mean'],4) for s in stats]}  -> {qa_dir}")
    return {"out_dir": out_dir, "az0": az0, "el": el, "thetas": thetas,
            "residual": stats, "valid_px": int(out_valid.sum())}


def _mask_kw(cfg):
    m = cfg["align"]["mask"]
    return dict(close_radius=m["close_radius"], open_radius=m["open_radius"],
                keep_largest=m["keep_largest"])


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
                           verbose=not args.quiet)
        print(f"\nDONE  az0={res['az0']:.2f} el={res['el']:.2f} "
              f"valid_px={res['valid_px']}  -> {res['out_dir']}")
        return res


if __name__ == "__main__":
    main()
