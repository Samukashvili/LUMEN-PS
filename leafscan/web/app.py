"""LUMEN-PS FastAPI app — serves the UI and bridges to the pipeline.

Local, single-user. Sync endpoints run in Starlette's threadpool (so WIA/COM
work is off the event loop); the pipeline/capture run in a dedicated worker via
:mod:`jobs`. The WebSocket tails the active job's log buffer by index.
"""
from __future__ import annotations

import asyncio
import io as _io
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import device, jobs
from . import sessions as S

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

app = FastAPI(title="LUMEN-PS")


# ---- models ---------------------------------------------------------------- #
class NewSession(BaseModel):
    name: str = "Untitled leaf"


class CaptureReq(BaseModel):
    role: str


class ConfigReq(BaseModel):
    overrides: dict


# ---- pages / static -------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


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


# ---- images (scans + results, optional on-the-fly downscale) --------------- #
def _serve_image(path: Path, maxdim: int | None, keep_alpha: bool):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
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
    return Response(buf.getvalue(), media_type=media,
                    headers={"Cache-Control": "no-store"})


@app.get("/api/sessions/{sid}/scan/{role}")
def api_scan_image(sid: str, role: str, max: int = 1000):
    return _serve_image(S.scans_dir(sid) / f"{role}.png", max, keep_alpha=False)


@app.get("/api/sessions/{sid}/result/{name:path}")
def api_result_image(sid: str, name: str, max: int = 0):
    # guard against path traversal
    base = S.out_dir(sid).resolve()
    path = (base / name).resolve()
    if base not in path.parents and path != base:
        raise HTTPException(400, "Invalid path")
    alpha = name.endswith("rgba.png") or name == "alpha.png" or name.startswith("normal_")
    return _serve_image(path, max or None, keep_alpha=alpha)


# ---- live log stream ------------------------------------------------------- #
@app.websocket("/api/sessions/{sid}/stream")
async def ws_stream(ws: WebSocket, sid: str):
    await ws.accept()
    idx = 0
    try:
        while True:
            job = jobs.get_job(sid)
            if job:
                lines = job["log"]
                if idx < len(lines):
                    await ws.send_json({"type": "log", "lines": lines[idx:]})
                    idx = len(lines)
                if job["status"] in ("done", "error"):
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
