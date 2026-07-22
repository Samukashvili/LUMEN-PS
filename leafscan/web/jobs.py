"""Background jobs — capture and pipeline runs.

Single worker thread (the scanner is a one-at-a-time resource and a full-res
solve is heavy), so jobs serialize naturally. Each job keeps an in-memory log
buffer that the WebSocket endpoint tails by index — no async/thread bridging
needed for a local single-user app. WIA/COM work initialises COM per worker.
"""
from __future__ import annotations

import threading
import traceback
import multiprocessing as mp
import queue
from concurrent.futures import CancelledError
from concurrent.futures import ThreadPoolExecutor

from . import sessions as S
from .device import _com_init

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumen")
_jobs: dict[str, dict] = {}          # sid -> current/last job
_lock = threading.Lock()
_shutdown_requested = threading.Event()


def get_job(sid: str) -> dict | None:
    with _lock:
        return _jobs.get(sid)


def _new_job(sid: str, kind: str) -> dict:
    if _shutdown_requested.is_set():
        raise RuntimeError("LUMEN-PS is shutting down")
    job = {"sid": sid, "kind": kind, "status": "queued", "log": [], "error": None,
           "result": None, "cancel_requested": threading.Event()}
    with _lock:
        _jobs[sid] = job
    return job


def _log(job: dict, line: str):
    job["log"].append(str(line))


def is_busy(sid: str) -> bool:
    j = get_job(sid)
    return bool(j and j["status"] in ("queued", "running"))


def cancel(sid: str) -> bool:
    """Request cancellation of the active job. Worker code observes the event."""
    with _lock:
        job = _jobs.get(sid)
        if not job or job["status"] not in ("queued", "running"):
            return False
        job["cancel_requested"].set()
        _log(job, "[cancel] cancellation requested...")
        return True


def _check_cancelled(job: dict):
    if _shutdown_requested.is_set() or job["cancel_requested"].is_set():
        raise CancelledError("Cancelled by user")


def shutdown() -> None:
    """Cancel queued work and terminate Python child processes before exit."""
    _shutdown_requested.set()
    with _lock:
        active = [job for job in _jobs.values()
                  if job["status"] in ("queued", "running")]
        for job in active:
            job["cancel_requested"].set()
            _log(job, "[shutdown] application shutdown requested...")

    for process in mp.active_children():
        try:
            process.terminate()
            process.join(2)
        except Exception:
            pass
    _executor.shutdown(wait=False, cancel_futures=True)


def _scan_transfer_worker(result_queue, out_path, kwargs):
    """Own WIA's blocking COM transfer in a process that can be stopped."""
    try:
        _com_init()
        from .. import capture_wia
        result_queue.put(("ok", capture_wia.scan_to_file(out_path, **kwargs)))
    except BaseException as exc:
        result_queue.put(("error", f"{exc}\n{traceback.format_exc()}"))


def _cancellable_scan(job: dict, out_path, **kwargs):
    """Run a driver transfer separately so cancelling does not wait for WIA.

    WIA's ``Item.Transfer`` is a blocking COM call and exposes no dependable
    abort method across drivers.  Terminating its isolated helper immediately
    releases the web worker; the partly written image is removed by the caller.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_scan_transfer_worker, args=(result_queue, str(out_path), kwargs))
    process.start()
    try:
        while process.is_alive():
            process.join(0.1)
            if job["cancel_requested"].is_set():
                process.terminate()
                process.join(2)
                raise CancelledError("Cancelled by user")
        try:
            state, result = result_queue.get(timeout=1)
        except queue.Empty:
            raise RuntimeError("Scanner transfer stopped without returning a result")
        if state != "ok":
            raise RuntimeError(result)
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)
        result_queue.close()


# --------------------------------------------------------------------------- #
def start_capture(sid: str, role: str) -> dict:
    job = _new_job(sid, f"capture:{role}")
    _executor.submit(_run_capture, job, sid, role)
    return job


def _run_capture(job: dict, sid: str, role: str):
    job["status"] = "running"
    try:
        _check_cancelled(job)
        cfg = S.session_config(sid)
        cap = cfg["capture"]
        out = S.scans_dir(sid) / f"{role}.png"
        roi_mm = None
        smart = cap.get("smart_roi", {})
        if role in S.LEAF_ROLES and smart.get("enabled", True):
            preview_dpi = int(smart.get("preview_dpi", 75))
            preview = S.scans_dir(sid) / "previews" / f"{role}.png"
            _log(job, f"[scan] {role}: locator pass at {preview_dpi} dpi ...")
            _cancellable_scan(
                job,
                preview, dpi=preview_dpi, color=cap["color"],
                brightness=cap["brightness"], contrast=cap["contrast"],
                device_name_hint=cap.get("device_name_hint", "M113"),
                verbose=False,
            )
            _check_cancelled(job)
            from .. import capture_wia
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
        info = _cancellable_scan(
            job,
            out, dpi=cap["dpi"], color=cap["color"],
            brightness=cap["brightness"], contrast=cap["contrast"],
            roi_mm=roi_mm,
            device_name_hint=cap.get("device_name_hint", "M113"),
            verbose=False,
        )
        _check_cancelled(job)
        # Record the rectangle the driver ACTUALLY scanned (positions/extents
        # get snapped), not the requested one: run-time bed placement of the
        # ROI captures is only exact with the true geometry.
        actual_roi = info.get("roi_mm_actual") if roi_mm else None
        actual_dpi = int(info["dpi"][0]) if info.get("dpi") else cap["dpi"]
        info["roi_mm"] = actual_roi or roi_mm
        _log(job, f"[scan] {role}: {info['shape']} depth={info['depth']} "
                  f"distinct={info['distinct_levels']}")
        S.record_scan(sid, role, roi_mm=actual_roi or roi_mm, dpi=actual_dpi)
        job["result"] = {"role": role, "info": info}
        job["status"] = "done"
        _log(job, f"[scan] {role}: saved.")
    except CancelledError:
        job["status"] = "cancelled"
        _log(job, f"[cancel] {role}: scan cancelled. Any incomplete scan was discarded.")
        (S.scans_dir(sid) / f"{role}.png").unlink(missing_ok=True)
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
        # Smart-ROI captures record their glass rectangle per scan; hand the
        # geometry to the pipeline so it can rebuild a common bed canvas.
        meta = S.load_meta(sid) or {}
        rois = meta.get("capture_rois") or {}
        capture_rois = [rois.get(r) for r in S.LEAF_ROLES]
        dpis = meta.get("capture_dpis") or {}
        # per-scan dpi list: geometry placement must honour the dpi each ROI
        # was actually captured at, not one session-wide value
        capture_dpi = [dpis.get(r) or cfg["capture"]["dpi"] for r in S.LEAF_ROLES]

        res = run_pipeline(
            cfg, scans, S.out_dir(sid),
            flat_path=str(flat) if flat.exists() else None,
            calib_paths=calib, scale=cfg["runtime"]["scale"],
            verbose=True, log_fn=lambda s: _log(job, s), auto_crop=auto_crop,
            cancel_check=lambda: _check_cancelled(job),
            capture_rois=capture_rois, capture_dpi=capture_dpi,
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
    except CancelledError:
        job["status"] = "cancelled"
        S.set_status(sid, "ready" if S.ready_to_run(sid) else "capturing")
        _log(job, "[cancel] reconstruction cancelled. Partial output files may remain in the selected folder.")
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "error"
        S.set_status(sid, "error")
        _log(job, f"[error] pipeline failed: {e}")
        _log(job, traceback.format_exc())
