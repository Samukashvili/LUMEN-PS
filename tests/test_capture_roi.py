from pathlib import Path

import numpy as np
from PIL import Image

from leafscan.capture_wia import detect_content_roi_mm


def test_detect_content_roi_mm_finds_subject_with_padding(tmp_path: Path):
    image = np.full((300, 400, 3), 246, dtype=np.uint8)
    image[80:220, 110:310] = (52, 76, 43)
    path = tmp_path / "locator.png"
    Image.fromarray(image).save(path)

    roi = detect_content_roi_mm(path, dpi=100, padding_mm=5)

    assert roi is not None
    x, y, width, height = roi
    assert 22 < x < 28
    assert 15 < y < 22
    assert 58 < width < 64
    assert 45 < height < 50


def test_detect_content_roi_mm_ignores_bed_edge_artifacts(tmp_path: Path):
    """A dark sliver hugging the preview border (calibration strip / lid seam)
    must not drag the ROI out to the full bed."""
    image = np.full((300, 400, 3), 246, dtype=np.uint8)
    image[80:220, 110:310] = (52, 76, 43)     # the subject
    image[-3:, :] = (30, 30, 30)              # bed bottom-edge artifact
    image[:, :2] = (40, 40, 40)               # lid seam on the left
    path = tmp_path / "locator.png"
    Image.fromarray(image).save(path)

    roi = detect_content_roi_mm(path, dpi=100, padding_mm=5)

    assert roi is not None
    x, y, width, height = roi
    assert 22 < x < 28
    assert 15 < y < 22
    assert 58 < width < 64
    assert 45 < height < 50


def test_detect_content_roi_mm_falls_back_for_blank_bed(tmp_path: Path):
    path = tmp_path / "blank.png"
    Image.fromarray(np.full((200, 300, 3), 240, dtype=np.uint8)).save(path)
    assert detect_content_roi_mm(path, dpi=75) is None
