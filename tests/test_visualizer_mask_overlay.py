"""Tests for track-mask visualization."""

import numpy as np

from core.visualizer import Visualizer


def test_mask_overlay_only_changes_active_roi_pixels() -> None:
    visualizer = Visualizer(
        {
            "show_window": False,
            "save_video": False,
            "save_screenshot": False,
            "mask_alpha": 0.5,
            "mask_color": [0, 200, 100],
        }
    )
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    result = visualizer._overlay_roi_mask(image, mask, (2, 2, 8, 6))
    assert result.shape == (8, 10, 3)
    assert np.array_equal(result[0, 0], [0, 0, 0])
    assert np.array_equal(result[3, 4], [0, 100, 50])
    assert np.array_equal(result[2, 2], [0, 0, 0])
