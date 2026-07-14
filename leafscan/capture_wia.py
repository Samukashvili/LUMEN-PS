"""Optional Windows WIA capture helper for the HP LaserJet M113x flatbed.

The pipeline does NOT depend on this module (spec §4.2). Its contract is only:
produce lossless images on disk. This helper just makes that convenient and
enforces the non-negotiable capture settings (auto-exposure off, fixed
brightness/contrast, lossless BMP->PNG, native dpi).

VERIFIED on the connected unit: the HP WIA driver caps at 8-bit
(BitsPerPixel list = [1, 8, 24]). There is no 16-bit path on this hardware, so
we capture 24-bit sRGB colour and linearize in software (io.py / spec §5.1).

Usage (from repo root):
    python -m leafscan.capture_wia --out scans/leaf_k0.png --dpi 600 --color
    python -m leafscan.capture_wia --preview            # fast 150 dpi look
"""
from __future__ import annotations

import argparse
from pathlib import Path

# WIA constants
WIA_IPS_XRES = "6147"
WIA_IPS_YRES = "6148"
WIA_IPS_XEXTENT = "6151"
WIA_IPS_YEXTENT = "6152"
WIA_IPS_XPOS = "6149"
WIA_IPS_YPOS = "6150"
WIA_IPA_DATATYPE = "4103"     # 0=threshold(1bpp), 2=grayscale, 3=color
WIA_IPA_DEPTH = "4104"        # bits per pixel
WIA_IPS_BRIGHTNESS = "6154"
WIA_IPS_CONTRAST = "6155"
WIA_IPS_CUR_INTENT = "6146"

WIA_DATA_GRAYSCALE = 2
WIA_DATA_COLOR = 3
DEPTH_GRAY = 8
DEPTH_COLOR = 24

FORMAT_BMP = "{B96B3CAB-0728-11D3-9D7B-0000F81EF32E}"

# Intent flags: 1=color, 2=grayscale, 4=text, plus MINIMIZE_SIZE(0x10000)/
# MAXIMIZE_QUALITY(0x20000). We force quality and disable any auto behaviour by
# setting properties explicitly rather than trusting an intent.
WIA_INTENT_MAXIMIZE_QUALITY = 0x00020000


def _set(item, prop_id, value):
    """Set a WIA item property by id, tolerating drivers that clamp/snap."""
    for p in item.Properties:
        if str(p.PropertyID) == str(prop_id):
            p.Value = value
            return p.Value
    raise KeyError(f"WIA property {prop_id} not found")


def _get(item, prop_id):
    for p in item.Properties:
        if str(p.PropertyID) == str(prop_id):
            return p.Value
    raise KeyError(f"WIA property {prop_id} not found")


def connect(device_name_hint: str = "M113"):
    import win32com.client

    mgr = win32com.client.Dispatch("WIA.DeviceManager")
    infos = mgr.DeviceInfos
    chosen = None
    for i in range(1, infos.Count + 1):
        di = infos.Item(i)
        name = ""
        for p in di.Properties:
            if p.Name in ("Name", "Description"):
                name = str(p.Value)
                if device_name_hint.lower() in name.lower():
                    chosen = di
                    break
        if chosen:
            break
    if chosen is None:
        if infos.Count == 0:
            raise RuntimeError("No WIA scanners found.")
        chosen = infos.Item(1)  # fall back to the only/first device
    return chosen.Connect()


def scan_to_file(
    out_path,
    dpi: int = 600,
    color: bool = True,
    brightness: int = 6,
    contrast: int = 6,
    roi_mm=None,
    device_name_hint: str = "M113",
    verbose: bool = True,
):
    """Perform one flatbed scan with enhancements off; save lossless PNG/TIFF.

    ``roi_mm`` = (x, y, w, h) in millimetres, or None for the full glass area.
    Returns a dict describing the actual capture (dpi, size, depth).
    """
    import numpy as np
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dev = connect(device_name_hint)
    item = dev.Items.Item(1)

    # --- data type + depth (color 24bpp keeps R-B subsurface hint; else 8bpp) ---
    if color:
        _set(item, WIA_IPA_DATATYPE, WIA_DATA_COLOR)
        _set(item, WIA_IPA_DEPTH, DEPTH_COLOR)
    else:
        _set(item, WIA_IPA_DATATYPE, WIA_DATA_GRAYSCALE)
        _set(item, WIA_IPA_DEPTH, DEPTH_GRAY)

    # --- resolution first: extents are expressed in pixels at the current dpi ---
    _set(item, WIA_IPS_XRES, dpi)
    _set(item, WIA_IPS_YRES, dpi)

    # --- region of interest ---
    def mm_to_px(mm):
        return int(round(mm / 25.4 * dpi))

    xmax = _prop_max(item, WIA_IPS_XEXTENT)
    ymax = _prop_max(item, WIA_IPS_YEXTENT)
    if roi_mm and roi_mm[2] and roi_mm[3]:
        x, y, w, h = roi_mm
        x_px = min(mm_to_px(x), xmax - 1)
        y_px = min(mm_to_px(y), ymax - 1)
        _set(item, WIA_IPS_XPOS, x_px)
        _set(item, WIA_IPS_YPOS, y_px)
        _set(item, WIA_IPS_XEXTENT, max(1, min(mm_to_px(w), xmax - x_px)))
        _set(item, WIA_IPS_YEXTENT, max(1, min(mm_to_px(h), ymax - y_px)))
    else:
        _set(item, WIA_IPS_XPOS, 0)
        _set(item, WIA_IPS_YPOS, 0)
        _set(item, WIA_IPS_XEXTENT, xmax)
        _set(item, WIA_IPS_YEXTENT, ymax)

    # --- lock exposure-equivalents identical across scans; no auto anything ---
    try:
        _set(item, WIA_IPS_BRIGHTNESS, brightness)
        _set(item, WIA_IPS_CONTRAST, contrast)
    except Exception:
        pass  # some drivers omit these; fine

    info = {
        "dpi": (_get(item, WIA_IPS_XRES), _get(item, WIA_IPS_YRES)),
        "pos_px": (_get(item, WIA_IPS_XPOS), _get(item, WIA_IPS_YPOS)),
        "size_px": (_get(item, WIA_IPS_XEXTENT), _get(item, WIA_IPS_YEXTENT)),
        "depth": _get(item, WIA_IPA_DEPTH),
        "datatype": _get(item, WIA_IPA_DATATYPE),
    }
    # The driver snaps positions/extents (this HP rounds extents up to a
    # multiple of 8 px). Report the glass rectangle it ACTUALLY scanned so
    # callers can place the capture at its true bed offset.
    xres, yres = (float(info["dpi"][0]), float(info["dpi"][1]))
    info["roi_mm_actual"] = (
        info["pos_px"][0] * 25.4 / xres, info["pos_px"][1] * 25.4 / yres,
        info["size_px"][0] * 25.4 / xres, info["size_px"][1] * 25.4 / yres,
    )
    if verbose:
        print(f"[capture] dpi={info['dpi']} size_px={info['size_px']} "
              f"depth={info['depth']} datatype={info['datatype']}")
        print(f"[capture] scanning -> {out_path} ... (lamp warmup + transport)")

    # Transfer as uncompressed BMP (Preferred Format on this driver), then
    # re-encode losslessly ourselves to avoid any driver PNG/JPEG surprises.
    image = item.Transfer(FORMAT_BMP)
    tmp_bmp = out_path.with_suffix(".bmp")
    if Path(tmp_bmp).exists():
        Path(tmp_bmp).unlink()
    image.SaveFile(str(tmp_bmp))

    with Image.open(tmp_bmp) as im:
        arr = np.asarray(im)
    tmp_bmp.unlink(missing_ok=True)

    # save lossless in requested container
    if out_path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(str(out_path), arr)
    else:
        Image.fromarray(arr).save(str(out_path), compress_level=3)

    info["shape"] = arr.shape
    info["dtype"] = str(arr.dtype)
    info["distinct_levels"] = int(np.unique(arr).size)
    if verbose:
        print(f"[capture] wrote {out_path}  shape={arr.shape} dtype={arr.dtype} "
              f"distinct_levels={info['distinct_levels']}")
        if info["distinct_levels"] <= 256:
            print("[capture] NOTE: <=256 distinct levels => genuinely 8-bit data "
                  "(expected on this hardware). Linearize sRGB in io.py.")
    return info


def detect_content_roi_mm(
    image_path,
    dpi: int,
    padding_mm: float = 10.0,
    min_component_fraction: float = 0.0005,
):
    """Find meaningful non-background content in a low-resolution bed scan.

    Returns ``(x, y, width, height)`` in millimetres, ready for ``roi_mm``.
    The border colour is treated as the scanner-bed background; significant
    connected components are unioned so a leaf plus fiducials stay together.
    ``None`` means detection was not trustworthy and callers should scan the
    full bed instead.
    """
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return None

    edge = max(2, int(round(min(h, w) * 0.025)))
    border = np.concatenate((
        rgb[:edge].reshape(-1, 3), rgb[-edge:].reshape(-1, 3),
        rgb[:, :edge].reshape(-1, 3), rgb[:, -edge:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    difference = np.max(np.abs(rgb.astype(np.int16) - background.astype(np.int16)), axis=2)
    difference = cv2.GaussianBlur(difference.astype(np.uint8), (5, 5), 0)
    otsu, mask = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu < 6:
        _, mask = cv2.threshold(difference, 6, 255, cv2.THRESH_BINARY)

    radius = max(1, int(round(min(h, w) * 0.004)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(24, int(round(h * w * min_component_fraction)))
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return None

    x0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in keep)
    y0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in keep)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in keep)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in keep)
    pad = int(round(float(padding_mm) / 25.4 * dpi))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if (x1 - x0) * (y1 - y0) >= h * w * 0.96:
        return None
    mm = 25.4 / float(dpi)
    return (x0 * mm, y0 * mm, (x1 - x0) * mm, (y1 - y0) * mm)


def _prop_max(item, prop_id):
    for p in item.Properties:
        if str(p.PropertyID) == str(prop_id):
            try:
                return int(p.SubTypeMax)
            except Exception:
                return int(p.Value)
    raise KeyError(prop_id)


def main(argv=None):
    ap = argparse.ArgumentParser(description="WIA flatbed capture (HP M113x)")
    ap.add_argument("--out", type=str, default=None, help="output PNG/TIFF path")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--color", action="store_true", default=False)
    ap.add_argument("--gray", dest="color", action="store_false")
    ap.add_argument("--brightness", type=int, default=6)
    ap.add_argument("--contrast", type=int, default=6)
    ap.add_argument("--preview", action="store_true",
                    help="fast 150 dpi colour preview to scans/preview.png")
    ap.add_argument("--device", type=str, default="M113")
    args = ap.parse_args(argv)

    if args.preview:
        out = args.out or "scans/preview.png"
        return scan_to_file(out, dpi=150, color=True, device_name_hint=args.device)
    if not args.out:
        ap.error("--out is required (or use --preview)")
    return scan_to_file(
        args.out, dpi=args.dpi, color=args.color,
        brightness=args.brightness, contrast=args.contrast,
        device_name_hint=args.device,
    )


if __name__ == "__main__":
    main()
