import numpy as np

from leafscan.align import segment_leaf


def _subject_with_white_detail():
    luma = np.ones((120, 140), dtype=np.float32)
    luma[20:100, 25:115] = 0.25
    luma[45:75, 55:85] = 1.0
    return luma


def test_subject_mask_fills_enclosed_white_detail_by_default():
    mask = segment_leaf(_subject_with_white_detail(), close_radius=0, open_radius=0)

    assert mask[30, 35]
    assert mask[60, 70]
    assert not mask[5, 5]


def test_optional_hole_detection_preserves_legacy_white_removal():
    mask = segment_leaf(
        _subject_with_white_detail(), close_radius=0, open_radius=0,
        detect_interior_holes=True,
    )

    assert mask[30, 35]
    assert not mask[60, 70]
    assert not mask[5, 5]


def test_silhouette_fill_handles_subject_touching_image_edge():
    luma = _subject_with_white_detail()
    luma[20:100, :26] = 0.25

    mask = segment_leaf(luma, close_radius=0, open_radius=0)

    assert mask[60, 70]
    assert not mask[5, 130]
