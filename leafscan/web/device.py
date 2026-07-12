"""Scanner status for the boot screen — thin read-only wrapper over WIA.

Reuses the already-verified enumeration in :mod:`leafscan.capture_wia`. Never
raises to the caller: returns a status dict so the UI can show a clear,
actionable state whether or not a scanner is present.
"""
from __future__ import annotations

from .. import capture_wia as cw


def _com_init():
    """Initialise COM for the current thread (WIA needs it). Safe to call often."""
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass


def device_status(name_hint: str = "M113") -> dict:
    """Return {connected, name, dpi_options, depth_options, color, note}."""
    _com_init()
    try:
        dev = cw.connect(name_hint)
        item = dev.Items.Item(1)

        def prop(pid):
            for p in item.Properties:
                if str(p.PropertyID) == str(pid):
                    return p
            return None

        def options(pid):
            p = prop(pid)
            if p is None:
                return []
            try:
                return [int(x) for x in p.SubTypeValues]
            except Exception:
                try:
                    return [int(p.Value)]
                except Exception:
                    return []

        name = ""
        for p in dev.Properties:
            if p.Name in ("Name", "Description"):
                name = str(p.Value)
                break

        depth = options(cw.WIA_IPA_DEPTH)
        return {
            "connected": True,
            "name": name or "Flatbed scanner",
            "dpi_options": options(cw.WIA_IPS_XRES),
            "depth_options": depth,
            "color": cw.DEPTH_COLOR in depth,
            "max_bit_depth": max(depth) if depth else None,
            "note": None,
        }
    except Exception as e:  # no device, driver busy, COM not initialised, etc.
        return {
            "connected": False,
            "name": None,
            "dpi_options": [],
            "depth_options": [],
            "color": False,
            "max_bit_depth": None,
            "note": f"No scanner detected ({e}). Connect the flatbed, close other "
                    f"scan software, and reload.",
        }
