"""Session library — one folder per captured leaf.

sessions/<id>/
  session.json   name, timestamps, status, scan roles present, config overrides
  scans/         k0..k3.png (+ optional flat.png, calib0.png, calib90.png)
  out/           pipeline outputs + qa/
"""
from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSIONS_DIR = REPO_ROOT / "sessions"
CONFIG_PATH = REPO_ROOT / "leafscan" / "config.yaml"

LEAF_ROLES = ["k0", "k1", "k2", "k3"]
OPTIONAL_ROLES = ["flat", "calib0", "calib90"]
ALL_ROLES = LEAF_ROLES + OPTIONAL_ROLES


# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "leaf"


def default_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
def session_dir(sid: str) -> Path:
    return SESSIONS_DIR / sid


def scans_dir(sid: str) -> Path:
    return session_dir(sid) / "scans"


def out_dir(sid: str) -> Path:
    return session_dir(sid) / "out"


def _meta_path(sid: str) -> Path:
    return session_dir(sid) / "session.json"


def load_meta(sid: str) -> dict | None:
    p = _meta_path(sid)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(meta: dict) -> dict:
    meta["updated"] = _now()
    d = session_dir(meta["id"])
    d.mkdir(parents=True, exist_ok=True)
    with open(_meta_path(meta["id"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def create_session(name: str) -> dict:
    ts = time.strftime("%Y%m%d-%H%M%S")
    sid = f"{_slugify(name)}-{ts}"
    meta = {
        "id": sid,
        "name": name.strip() or "Untitled leaf",
        "created": _now(),
        "updated": _now(),
        "status": "capturing",          # capturing | ready | processing | done | error
        "scans": {r: False for r in ALL_ROLES},
        "config_overrides": {},
        "result": None,                  # summary dict after a run
    }
    scans_dir(sid).mkdir(parents=True, exist_ok=True)
    return save_meta(meta)


def list_sessions() -> list[dict]:
    if not SESSIONS_DIR.exists():
        return []
    out = []
    for d in SESSIONS_DIR.iterdir():
        if d.is_dir():
            m = load_meta(d.name)
            if m:
                out.append(m)
    out.sort(key=lambda m: m.get("created", ""), reverse=True)
    return out


def record_scan(sid: str, role: str) -> dict:
    meta = load_meta(sid)
    meta["scans"][role] = True
    if all(meta["scans"][r] for r in LEAF_ROLES) and meta["status"] == "capturing":
        meta["status"] = "ready"
    return save_meta(meta)


def set_status(sid: str, status: str, result: dict | None = None) -> dict:
    meta = load_meta(sid)
    meta["status"] = status
    if result is not None:
        meta["result"] = result
    return save_meta(meta)


def set_overrides(sid: str, overrides: dict) -> dict:
    meta = load_meta(sid)
    meta["config_overrides"] = overrides or {}
    return save_meta(meta)


def session_config(sid: str) -> dict:
    meta = load_meta(sid)
    return deep_merge(default_config(), meta.get("config_overrides", {}))


def leaf_scan_paths(sid: str) -> list[Path]:
    return [scans_dir(sid) / f"{r}.png" for r in LEAF_ROLES]


def ready_to_run(sid: str) -> bool:
    return all((scans_dir(sid) / f"{r}.png").exists() for r in LEAF_ROLES)
