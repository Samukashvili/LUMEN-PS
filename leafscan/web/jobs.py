"""Background jobs — capture and pipeline runs.

Single worker thread (the scanner is a one-at-a-time resource and a full-res
solve is heavy), so jobs serialize naturally. Each job keeps an in-memory log
buffer that the WebSocket endpoint tails by index — no async/thread bridging
needed for a local single-user app. WIA/COM work initialises COM per worker.
"""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import sessions as S
from .device import _com_init

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumen")
_jobs: dict[str, dict] = {}          # sid -> current/last job
_lock = threading.Lock()


def get_job(sid: str) -> dict | None:
    with _lock:
        return _jobs.get(sid)


def _new_job(sid: str, kind: str) -> dict:
    job = {"sid": sid, "kind": kind, "status": "queued", "log": [], "error": None,
           "result": None}
    with _lock:
        _jobs[sid] = job
    return job


def _log(job: dict, line: str):
    job["log"].append(str(line))


def is_busy(sid: str) -> bool:
    j = get_job(sid)
    return bool(j and j["status"] in ("queued", "running"))


# --------------------------------------------------------------------------- #
def start_capture(sid: str, role: str) -> dict:
    job = _new_job(sid, f"capture:{role}")
    _executor.submit(_run_capture, job, sid, role)
    return job


def _run_capture(job: dict, sid: str, role: str):
    job["status"] = "running"
    try:
        _com_init()
        from .. import capture_wia
        cfg = S.session_config(sid)
        cap = cfg["capture"]
        out = S.scans_dir(sid) / f"{role}.png"
        roi_mm = None
        smart = cap.get("smart_roi", {})
        if role in S.LEAF_ROLES and smart.get("enabled", True):
            preview_dpi = int(smart.get("preview_dpi", 75))
            preview = S.scans_dir(sid) / "previews" / f"{role}.png"
            _log(job, f"[scan] {role}: locator pass at {preview_dpi} dpi ...")
            capture_wia.scan_to_file(
                preview, dpi=preview_dpi, color=cap["color"],
                brightness=cap["brightness"], contrast=cap["contrast"],
                device_name_hint=cap.get("device_name_hint", "M113"),
                verbose=False,
            )
            roi_mm = capture_wia.detect_content_roi_mm(
                preview, preview_dpi,
                padding_mm=float(smart.get("padding_mm", 10.0)),
                min_component_fraction=float(smart.get("min_component_fraction", 0.0005)),
            )
            if roi_mm:
                x, y, w, h = roi_mm
                _log(job, f"[scan] {role}: content ROI x={x:.1f} y={y:.1f} "
                          f"w={w:.1f} h={h:.1f} mm")
            else:
                _log(job, f"[scan] {role}: locator was inconclusive; using full bed")

        scope = "detected area" if roi_mm else "full bed"
        _log(job, f"[scan] {role}: detail pass at {cap['dpi']} dpi ({scope}) ...")
        info = capture_wia.scan_to_file(
            out, dpi=cap["dpi"], color=cap["color"],
            brightness=cap["brightness"], contrast=cap["contrast"],
            roi_mm=roi_mm,
            device_name_hint=cap.get("device_name_hint", "M113"),
            verbose=False,
        )
        info["roi_mm"] = roi_mm
        _log(job, f"[scan] {role}: {info['shape']} depth={info['depth']} "
                  f"distinct={info['distinct_levels']}")
        S.record_scan(sid, role, roi_mm=roi_mm)
        job["result"] = {"role": role, "info": info}
        job["status"] = "done"
        _log(job, f"[scan] {role}: saved.")
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "error"
        _log(job, f"[error] capture failed: {e}")
        _log(job, traceback.format_exc())


# --------------------------------------------------------------------------- #
def start_run(sid: str) -> dict:
    job = _new_job(sid, "run")
    _executor.submit(_run_pipeline_job, job, sid)
    return job


def _run_pipeline_job(job: dict, sid: str):
    job["status"] = "running"
    S.set_status(sid, "processing")
    try:
        from ..cli import run_pipeline
        cfg = S.session_config(sid)
        scans = S.leaf_scan_paths(sid)
        flat = S.scans_dir(sid) / "flat.png"
        c0 = S.scans_dir(sid) / "calib0.png"
        c90 = S.scans_dir(sid) / "calib90.png"
        calib = [str(c0), str(c90)] if (c0.exists() and c90.exists()) else None
        auto_crop = bool(cfg.get("runtime", {}).get("auto_crop", True))

        res = run_pipeline(
            cfg, scans, S.out_dir(sid),
            flat_path=str(flat) if flat.exists() else None,
            calib_paths=calib, scale=cfg["runtime"]["scale"],
            verbose=True, log_fn=lambda s: _log(job, s), auto_crop=auto_crop,
        )
        summary = {
            "az0": res["az0"], "el": res["el"], "thetas": res["thetas"],
            "valid_px": res["valid_px"],
            "residual_means": [round(s["mean"], 4) for s in res["residual"]],
        }
        S.set_status(sid, "done", result=summary)
        job["result"] = summary
        job["status"] = "done"
        _log(job, "[done] pipeline complete.")
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "error"
        S.set_status(sid, "error")
        _log(job, f"[error] pipeline failed: {e}")
        _log(job, traceback.format_exc())
