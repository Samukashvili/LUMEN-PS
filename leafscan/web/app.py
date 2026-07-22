"""LUMEN-PS FastAPI app — serves the UI and bridges to the pipeline.

Local, single-user. Sync endpoints run in Starlette's threadpool (so WIA/COM
work is off the event loop); the pipeline/capture run in a dedicated worker via
:mod:`jobs`. The WebSocket tails the active job's log buffer by index.
"""
from __future__ import annotations

import asyncio
import io as _io
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import device, jobs
from . import sessions as S

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

app = FastAPI(title="LUMEN-PS")
APP_VERSION = "2026.07-scan-controls-v2"


# ---- models ---------------------------------------------------------------- #
class NewSession(BaseModel):
    name: str = "Untitled leaf"


class CaptureReq(BaseModel):
    role: str


class ConfigReq(BaseModel):
    overrides: dict


class OutputDirReq(BaseModel):
    path: str | None = None


class DeleteSessionReq(BaseModel):
    delete_files: bool = False


# ---- pages / static -------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/version")
def api_version():
    return {"version": APP_VERSION}


# ---- device ---------------------------------------------------------------- #
@app.get("/api/device")
def get_device():
    hint = S.default_config()["capture"].get("device_name_hint", "M113")
    return device.device_status(hint)


# ---- sessions -------------------------------------------------------------- #
@app.get("/api/sessions")
def api_list_sessions():
    return S.list_sessions()


@app.post("/api/sessions")
def api_create_session(body: NewSession):
    return S.create_session(body.name)


@app.get("/api/sessions/{sid}")
def api_get_session(sid: str):
    m = S.load_meta(sid)
    if not m:
        raise HTTPException(404, "Session not found")
    m["ready"] = S.ready_to_run(sid)
    m["busy"] = jobs.is_busy(sid)
    return m


@app.get("/api/sessions/{sid}/config")
def api_get_config(sid: str):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    return {"config": S.session_config(sid),
            "overrides": S.load_meta(sid).get("config_overrides", {}),
            "defaults": S.default_config()}


@app.put("/api/sessions/{sid}/config")
def api_set_config(sid: str, body: ConfigReq):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    S.set_overrides(sid, body.overrides)
    return {"config": S.session_config(sid)}


@app.put("/api/sessions/{sid}/output-dir")
def api_set_output_dir(sid: str, body: OutputDirReq):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    try:
        meta = S.set_output_dir(sid, body.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"output_dir": meta.get("output_dir"),
            "effective_output_dir": str(S.out_dir(sid))}


@app.post("/api/choose-output-dir")
def api_choose_output_dir():
    """Open the native folder chooser on this local machine."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(parent=root, title="Choose result save folder")
        root.destroy()
        return {"path": path or None}
    except Exception as exc:
        raise HTTPException(500, f"Could not open the folder chooser: {exc}") from exc


@app.delete("/api/sessions/{sid}")
def api_delete_session(sid: str, body: DeleteSessionReq):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    if jobs.is_busy(sid):
        raise HTTPException(409, "Cancel or wait for the active job before deleting this session")
    S.delete_session(sid, body.delete_files)
    return {"status": "deleted", "delete_files": body.delete_files}


# ---- capture / run --------------------------------------------------------- #
@app.post("/api/sessions/{sid}/capture")
def api_capture(sid: str, body: CaptureReq):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    if body.role not in S.ALL_ROLES:
        raise HTTPException(400, f"Unknown role {body.role}")
    if jobs.is_busy(sid):
        raise HTTPException(409, "A job is already running for this session")
    job = jobs.start_capture(sid, body.role)
    return {"status": job["status"], "kind": job["kind"]}


@app.post("/api/sessions/{sid}/import/{role}")
def api_import_scan(sid: str, role: str, file: UploadFile = File(...)):
    """Import an externally captured image into one primary scan slot."""
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    if role not in S.LEAF_ROLES:
        raise HTTPException(400, f"External import is only available for k0-k3, not {role}")
    if jobs.is_busy(sid):
        raise HTTPException(409, "Wait for the active scanner or reconstruction job to finish")
    try:
        meta = S.import_scan(sid, role, file.file)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        file.file.close()
    meta["ready"] = S.ready_to_run(sid)
    return meta


@app.delete("/api/sessions/{sid}/import/{role}")
def api_remove_imported_scan(sid: str, role: str):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    if jobs.is_busy(sid):
        raise HTTPException(409, "Wait for the active scanner or reconstruction job to finish")
    try:
        meta = S.remove_imported_scan(sid, role)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    meta["ready"] = S.ready_to_run(sid)
    return meta


def _shutdown_runtime() -> None:
    """Stop workers and then terminate this local web-server process."""
    jobs.shutdown()
    time.sleep(0.4)
    os._exit(0)


@app.post("/api/shutdown")
def api_shutdown(background_tasks: BackgroundTasks):
    # Background tasks begin only after Starlette sends the response body.
    background_tasks.add_task(_shutdown_runtime)
    return {"status": "shutting_down"}


@app.post("/api/sessions/{sid}/reset-scans")
def api_reset_scans(sid: str):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    if jobs.is_busy(sid):
        raise HTTPException(409, "Wait for the active scanner or reconstruction job to finish")
    return S.reset_scans(sid)


@app.post("/api/sessions/{sid}/run")
def api_run(sid: str):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    if not S.ready_to_run(sid):
        raise HTTPException(400, "Need all four rotated scans (k0–k3) first")
    if jobs.is_busy(sid):
        raise HTTPException(409, "A job is already running for this session")
    job = jobs.start_run(sid)
    return {"status": job["status"]}


@app.get("/api/sessions/{sid}/job")
def api_job(sid: str):
    job = jobs.get_job(sid)
    if not job:
        return {"status": "idle", "log": []}
    return {"status": job["status"], "kind": job["kind"], "log": job["log"],
            "result": job["result"], "error": job["error"]}


@app.post("/api/sessions/{sid}/job/cancel")
def api_cancel_job(sid: str):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    if not jobs.cancel(sid):
        raise HTTPException(409, "No active job to cancel")
    return {"status": "cancelling"}


# ---- images (scans + results, optional on-the-fly downscale) --------------- #
@lru_cache(maxsize=96)
def _render_preview(path_string: str, mtime_ns: int, maxdim: int | None,
                    keep_alpha: bool) -> tuple[bytes, str]:
    """Render and cache a small browser copy; mtime invalidates stale entries."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    path = Path(path_string)
    with Image.open(path) as im:
        # PIL's direct I;16 -> RGB conversion clips values above 255, making
        # 16-bit height maps look solid white. Scale the full uint16 range for
        # browser previews while leaving the original downloadable file intact.
        if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            import numpy as np
            arr = np.asarray(im)
            if arr.dtype == np.uint16:
                arr8 = (arr.astype(np.uint32) * 255 // 65535).astype(np.uint8)
            else:
                lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
                arr8 = np.zeros(arr.shape, dtype=np.uint8) if hi <= lo else np.clip(
                    (arr.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255
                ).astype(np.uint8)
            im = Image.fromarray(arr8, mode="L").convert("RGB")
        else:
            im = im.convert("RGBA" if keep_alpha else "RGB")
        if maxdim:
            im.thumbnail((maxdim, maxdim), Image.LANCZOS)
        buf = _io.BytesIO()
        if keep_alpha:
            im.save(buf, format="PNG", optimize=True)
            media = "image/png"
        else:
            im.save(buf, format="JPEG", quality=88, progressive=True)
            media = "image/jpeg"
    return buf.getvalue(), media


def _serve_image(path: Path, maxdim: int | None, keep_alpha: bool):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found")
    data, media = _render_preview(str(path.resolve()), path.stat().st_mtime_ns,
                                  maxdim, keep_alpha)
    return Response(data, media_type=media,
                    headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/sessions/{sid}/scan/{role}")
def api_scan_image(sid: str, role: str, max: int = 1000):
    return _serve_image(S.scans_dir(sid) / f"{role}.png", max, keep_alpha=False)


@app.get("/api/sessions/{sid}/result/{name:path}")
def api_result_image(sid: str, name: str, max: int = 0,
                     raw: bool = False, download: bool = False):
    # guard against path traversal
    base = S.out_dir(sid).resolve()
    path = (base / name).resolve()
    if base not in path.parents and path != base:
        raise HTTPException(400, "Invalid path")
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found")
    if raw or download or path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return FileResponse(path, filename=path.name if download else None)
    alpha = name.endswith("rgba.png") or name == "alpha.png" or name.startswith("normal_")
    return _serve_image(path, max or None, keep_alpha=alpha)


@app.get("/api/sessions/{sid}/results")
def api_result_manifest(sid: str):
    if not S.load_meta(sid):
        raise HTTPException(404, "Session not found")
    base = S.out_dir(sid)
    files = []
    if base.exists():
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = path.relative_to(base).as_posix()
            files.append({"name": rel, "bytes": path.stat().st_size,
                          "kind": path.suffix.lower().lstrip(".")})
    return {"output_dir": str(base), "files": files}


# ---- live log stream ------------------------------------------------------- #
@app.websocket("/api/sessions/{sid}/stream")
async def ws_stream(ws: WebSocket, sid: str):
    await ws.accept()
    try:
        idx = max(0, int(ws.query_params.get("from", "0")))
    except ValueError:
        idx = 0
    try:
        while True:
            job = jobs.get_job(sid)
            if job:
                lines = job["log"]
                if idx < len(lines):
                    batch = lines[idx:]
                    idx += len(batch)
                    await ws.send_json({"type": "log", "lines": batch})
                if job["status"] in ("done", "error", "cancelled"):
                    await ws.send_json({"type": "status", "status": job["status"],
                                        "kind": job["kind"], "result": job["result"],
                                        "error": job["error"]})
                    # keep socket open briefly so client can read, then stop
                    await asyncio.sleep(0.2)
                    break
            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        return
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# static assets last (so /api and / win)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
