"""WIA capability probing without attached scanner hardware."""
from types import SimpleNamespace

from leafscan import capture_wia as cw
from leafscan.web import device


class RangeProperty:
    def __init__(self, pid, minimum, maximum, step, value):
        self.PropertyID = pid
        self.SubType = 1
        self.SubTypeMin = minimum
        self.SubTypeMax = maximum
        self.SubTypeStep = step
        self.Value = value

    @property
    def SubTypeValues(self):
        raise RuntimeError("SubTypeValues is unavailable for a WIA range")


class ListProperty:
    def __init__(self, pid, values):
        self.PropertyID = pid
        self.SubType = 2
        self.SubTypeValues = values
        self.Value = values[0]


def _status(monkeypatch, properties):
    item = SimpleNamespace(Properties=properties)
    scanner = SimpleNamespace(
        Items=SimpleNamespace(Item=lambda _: item),
        Properties=[SimpleNamespace(Name="Name", Value="Simulated flatbed")],
    )
    monkeypatch.setattr(device.cw, "connect", lambda _: scanner)
    monkeypatch.setattr(device, "_com_init", lambda: None)
    return device.device_status()


def test_range_resolution_exposes_detail_choices_instead_of_current_value(monkeypatch):
    status = _status(monkeypatch, [
        RangeProperty(cw.WIA_IPS_XRES, 100, 1200, 1, 100),
        RangeProperty(cw.WIA_IPS_YRES, 100, 1200, 1, 100),
        ListProperty(cw.WIA_IPA_DEPTH, [8, 24]),
    ])
    assert status["connected"]
    assert status["dpi_options"] == [100, 150, 200, 300, 600, 1200]
    assert status["max_bit_depth"] == 24


def test_range_resolution_respects_driver_step(monkeypatch):
    status = _status(monkeypatch, [
        RangeProperty(cw.WIA_IPS_XRES, 100, 1200, 100, 100),
        ListProperty(cw.WIA_IPS_YRES, [100, 300, 600]),
    ])
    assert status["dpi_options"] == [100, 200, 300, 600, 1200]


def test_list_resolution_keeps_driver_choices(monkeypatch):
    status = _status(monkeypatch, [
        ListProperty(cw.WIA_IPS_XRES, [1200, 300, 600]),
        ListProperty(cw.WIA_IPS_YRES, [300, 600, 1200]),
    ])
    assert status["dpi_options"] == [1200, 300, 600]
