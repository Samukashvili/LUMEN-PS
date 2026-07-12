import io
from pathlib import Path

import numpy as np
from PIL import Image

from leafscan.web import sessions
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
    assert not any(sessions.scans_dir(meta["id"]).glob("k*.png"))
