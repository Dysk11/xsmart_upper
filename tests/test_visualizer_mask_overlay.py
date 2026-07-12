"""Tests for track-mask visualization."""

import numpy as np
from types import SimpleNamespace

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


def test_canvas_renders_fork_corners_without_stopping_ui() -> None:
    visualizer = Visualizer({"show_window": False, "save_video": False, "save_screenshot": False})
    fork = SimpleNamespace(
        left_points=[], right_points=[], left_corner=(20, 20), right_corner=(70, 25),
        left_detected=True, right_detected=True, confirm_frames=2,
    )
    detection = SimpleNamespace(
        centerline_points=[(45, 70), (48, 30)],
        left_boundary_points=[(20, 70), (22, 20)],
        right_boundary_points=[(70, 70), (72, 20)],
        filtered_mask=np.ones((80, 100), dtype=np.uint8) * 255,
        fork_result=fork,
        segmentation_status="ok",
        segmentation_confidence=0.9,
    )
    tracked = SimpleNamespace(centerline_points=detection.centerline_points)
    avoidance = SimpleNamespace(
        shifted_centerline_points=detection.centerline_points,
        target_point_roi=(50.0, 40.0), mode="normal", avoid_bias_px=0.0,
        final_lateral_error_px=0.0,
    )
    control = SimpleNamespace(steer_deg=0.0)
    canvas = visualizer._build_canvas(
        frame=np.zeros((120, 160, 3), dtype=np.uint8),
        roi_rect=(30, 30, 130, 110),
        detection_result=detection,
        tracked_state=tracked,
        control_command=control,
        fps_value=60.0,
        avoidance_result=avoidance,
    )
    assert canvas.shape == (120, 160, 3)
    assert np.any(canvas[44:57, 44:57] != 0)
