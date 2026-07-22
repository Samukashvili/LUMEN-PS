import asyncio
import io
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import UploadFile

import leafscan.web.app as web_app
from leafscan.web import jobs, sessions
from leafscan.web.app import _serve_image


def test_16bit_height_preview_preserves_tonal_range(tmp_path: Path):
    source = np.linspace(0, 65535, 128 * 64, dtype=np.uint16).reshape(64, 128)
    path = tmp_path / "height.png"
    Image.fromarray(source).save(path)

    response = _serve_image(path, 128, keep_alpha=False)
    preview = np.asarray(Image.open(io.BytesIO(response.body)).convert("L"))

    assert preview.min() < 5
    assert preview.max() > 245
    assert np.unique(preview).size > 100


def test_job_stream_does_not_skip_log_appended_during_send(monkeypatch):
    job = {
        "status": "running",
        "kind": "run",
        "log": ["first"],
        "result": None,
        "error": None,
    }

    class FakeWebSocket:
        query_params = {"from": "0"}

        def __init__(self):
            self.messages = []

        async def accept(self):
            pass

        async def send_json(self, message):
            self.messages.append(message)
            if message.get("type") == "log" and message["lines"] == ["first"]:
                job["log"].append("second")

        async def close(self):
            pass

    async def advance_job(delay):
        if delay == 0.15:
            job["status"] = "done"

    ws = FakeWebSocket()
    monkeypatch.setattr(web_app.jobs, "get_job", lambda sid: job)
    monkeypatch.setattr(web_app.asyncio, "sleep", advance_job)

    asyncio.run(web_app.ws_stream(ws, "session"))

    log_batches = [message["lines"] for message in ws.messages if message["type"] == "log"]
    assert log_batches == [["first"], ["second"]]


def test_session_can_use_external_output_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("Output path test")
    destination = tmp_path / "exports" / "leaf-a"

    sessions.set_output_dir(meta["id"], str(destination))
    assert sessions.out_dir(meta["id"]) == destination.resolve()

    sessions.set_output_dir(meta["id"], None)
    assert sessions.out_dir(meta["id"]) == sessions.session_dir(meta["id"]) / "out"


def test_rescan_invalidates_previous_result(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("Rescan test")
    for role in sessions.LEAF_ROLES:
        sessions.record_scan(meta["id"], role)
    sessions.set_status(meta["id"], "done", result={"valid_px": 42})

    updated = sessions.record_scan(meta["id"], "k2", roi_mm=(1, 2, 3, 4))

    assert updated["status"] == "ready"
    assert updated["result"] is None
    assert updated["capture_rois"]["k2"] == [1, 2, 3, 4]


def test_import_scan_normalizes_image_and_updates_session(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("External scan")
    source = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 80, 140)).save(source, format="TIFF")
    source.seek(0)

    updated = sessions.import_scan(meta["id"], "k1", source)

    imported = sessions.scans_dir(meta["id"]) / "k1.png"
    assert imported.exists()
    with Image.open(imported) as image:
        assert image.format == "PNG"
        assert image.size == (32, 24)
    assert updated["scans"]["k1"] is True
    assert updated["capture_sources"]["k1"] == "imported"
    assert updated["capture_rois"]["k1"] is None


def test_import_scan_rejects_non_primary_role(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("External scan")

    try:
        sessions.import_scan(meta["id"], "flat", io.BytesIO(b"not used"))
    except ValueError as exc:
        assert "only available" in str(exc)
    else:
        raise AssertionError("optional roles must not accept external imports")


def test_import_scan_endpoint_accepts_uploaded_image(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("Browser import")
    source = io.BytesIO()
    Image.new("RGB", (20, 12), (90, 40, 10)).save(source, format="PNG")
    source.seek(0)

    result = web_app.api_import_scan(
        meta["id"], "k0", UploadFile(filename="external-scan.png", file=source)
    )

    assert result["capture_sources"]["k0"] == "imported"
    assert result["ready"] is False
    assert (sessions.scans_dir(meta["id"]) / "k0.png").exists()


def test_remove_one_imported_scan_keeps_other_captures(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("Fix one import")
    for role in sessions.LEAF_ROLES:
        image = io.BytesIO()
        Image.new("RGB", (12, 8), (10, 20, 30)).save(image, format="PNG")
        image.seek(0)
        sessions.import_scan(meta["id"], role, image)

    updated = sessions.remove_imported_scan(meta["id"], "k2")

    assert updated["status"] == "capturing"
    assert updated["result"] is None
    assert updated["scans"]["k2"] is False
    assert updated["capture_sources"]["k2"] is None
    assert not (sessions.scans_dir(meta["id"]) / "k2.png").exists()
    assert all((sessions.scans_dir(meta["id"]) / f"{role}.png").exists()
               for role in ("k0", "k1", "k3"))


def test_remove_imported_scan_rejects_scanner_capture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("Scanner capture")
    sessions.record_scan(meta["id"], "k0", source="scanner")

    try:
        sessions.remove_imported_scan(meta["id"], "k0")
    except ValueError as exc:
        assert "not an imported scan" in str(exc)
    else:
        raise AssertionError("scanner captures must not use imported-scan removal")


def test_shutdown_cancels_jobs_and_terminates_children(monkeypatch):
    class FakeExecutor:
        def __init__(self):
            self.shutdown_args = None

        def shutdown(self, **kwargs):
            self.shutdown_args = kwargs

    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.join_timeout = None

        def terminate(self):
            self.terminated = True

        def join(self, timeout):
            self.join_timeout = timeout

    fake_executor, child = FakeExecutor(), FakeProcess()
    event = jobs.threading.Event()
    monkeypatch.setattr(jobs, "_executor", fake_executor)
    monkeypatch.setattr(jobs.mp, "active_children", lambda: [child])
    monkeypatch.setattr(jobs, "_shutdown_requested", jobs.threading.Event())
    monkeypatch.setattr(jobs, "_jobs", {"s": {
        "status": "running", "cancel_requested": event, "log": []
    }})

    jobs.shutdown()

    assert event.is_set()
    assert child.terminated and child.join_timeout == 2
    assert fake_executor.shutdown_args == {"wait": False, "cancel_futures": True}


def test_shutdown_endpoint_schedules_exit_after_response():
    from fastapi import BackgroundTasks

    background = BackgroundTasks()
    response = web_app.api_shutdown(background)

    assert response == {"status": "shutting_down"}
    assert len(background.tasks) == 1
    assert background.tasks[0].func is web_app._shutdown_runtime


def test_reset_scans_returns_session_to_k0(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("Reset test")
    for role in sessions.LEAF_ROLES:
        path = sessions.scans_dir(meta["id"]) / f"{role}.png"
        path.write_bytes(b"scan")
        sessions.record_scan(meta["id"], role, roi_mm=(1, 2, 3, 4))
    sessions.set_status(meta["id"], "done", result={"valid_px": 42})

    reset = sessions.reset_scans(meta["id"])

    assert reset["status"] == "capturing"
    assert reset["result"] is None
    assert not any(reset["scans"].values())
    assert not any(reset["capture_rois"].values())
    assert not any(reset["capture_sources"].values())
    assert not any(sessions.scans_dir(meta["id"]).glob("k*.png"))


def test_remove_session_from_ui_keeps_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("Keep files")
    scan = sessions.scans_dir(meta["id"]) / "k0.png"
    scan.write_bytes(b"scan")

    sessions.delete_session(meta["id"], delete_files=False)

    assert not sessions.load_meta(meta["id"])
    assert scan.exists()


def test_remove_session_with_files_removes_external_output(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path / "sessions")
    meta = sessions.create_session("Delete files")
    destination = tmp_path / "exports" / "leaf-a"
    destination.mkdir(parents=True)
    (destination / "normal_gl.png").write_bytes(b"result")
    sessions.set_output_dir(meta["id"], str(destination))

    sessions.delete_session(meta["id"], delete_files=True)

    assert not sessions.session_dir(meta["id"]).exists()
    assert not destination.exists()
