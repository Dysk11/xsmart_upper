"""Regression tests for the bottom debug panel layout."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.visualization.visualizer import Visualizer
from utils.image_utils import measure_text_width, wrap_text_lines


def make_detection(reason: str = "geometry left=False right=False") -> SimpleNamespace:
    fork_result = SimpleNamespace(
        left_detected=False,
        right_detected=False,
        confirm_frames=0,
        selected_direction=None,
        reason=reason,
        left_centerline_points=[],
        right_centerline_points=[],
        left_corner=None,
        right_corner=None,
    )
    return SimpleNamespace(
        filtered_mask=np.zeros((480, 640), dtype=np.uint8),
        centerline_points=[],
        left_boundary_points=[],
        right_boundary_points=[],
        fork_result=fork_result,
        segmentation_status="ok",
        segmentation_confidence=0.81,
    )


def test_mixed_text_and_long_tokens_wrap_to_measured_width() -> None:
    max_width = 150
    wrapped = wrap_text_lines(
        [
            "岔路 reason=geometry_with_a_very_long_unbroken_diagnostic_identifier",
            "OCR识别结果 左转",
        ],
        max_width=max_width,
        font_size=18,
    )

    assert len(wrapped) > 2
    assert all(measure_text_width(line, font_size=18) <= max_width for line in wrapped)
    assert "".join(part.replace(" ", "") for part in wrapped).startswith("岔路reason=")


def test_debug_panel_auto_grows_for_full_long_status() -> None:
    visualizer = Visualizer({"show_window": False, "debug_panel_font_size": 18})
    common = {
        "width": 640,
        "avoidance_result": SimpleNamespace(
            mode="lane_follow",
            avoid_bias_px=0.0,
            final_lateral_error_px=-107.0,
        ),
        "control_command": SimpleNamespace(steer_deg=-35.0),
        "fps_value": 59.9,
        "gold_result": SimpleNamespace(active=True, reason="coin target active"),
        "path_marker_result": SimpleNamespace(active=True, reason="Go/Stop path marker active"),
        "ocr_result": SimpleNamespace(
            frame_id=10,
            event_id=1,
            error="",
            text="turn left after the extremely_long_unbroken_marker_identifier",
            confidence=0.95,
            inference_ms=12.4,
        ),
    }
    short_panel = visualizer._build_debug_panel(
        **common,
        detection_result=make_detection("ok"),
        blocking_result=SimpleNamespace(reason="no blocking"),
    )
    long_panel = visualizer._build_debug_panel(
        **common,
        detection_result=make_detection(
            "geometry selected by frame center because candidate distances remained stable "
            "through_multiple_consecutive_bottom_track_regions"
        ),
        blocking_result=SimpleNamespace(
            reason="blocking object overlaps the projected lane corridor for multiple rows"
        ),
    )

    assert short_panel.shape[1] == 640
    assert long_panel.shape[1] == 640
    assert long_panel.shape[0] > short_panel.shape[0]


def test_canvas_keeps_camera_frame_on_top_and_panel_below() -> None:
    visualizer = Visualizer({"show_window": False, "debug_panel_font_size": 18})
    frame = np.full((480, 640, 3), 110, dtype=np.uint8)
    canvas = visualizer._build_canvas(
        frame=frame,
        roi_rect=(10, 200, 630, 470),
        detection_result=make_detection(),
        tracked_state=SimpleNamespace(centerline_points=[]),
        control_command=SimpleNamespace(steer_deg=0.0),
        fps_value=60.0,
    )

    assert canvas.shape[1] == frame.shape[1]
    assert canvas.shape[0] > frame.shape[0]
    assert np.array_equal(canvas[:100, :100], frame[:100, :100])
    assert not np.array_equal(canvas[480:], frame[: canvas.shape[0] - 480])


def test_debug_panel_handles_missing_optional_results() -> None:
    visualizer = Visualizer({"show_window": False})
    panel = visualizer._build_debug_panel(
        width=640,
        avoidance_result=None,
        blocking_result=None,
        control_command=None,  # type: ignore[arg-type]
        fps_value=0.0,
        gold_result=None,
        path_marker_result=None,
        detection_result=None,
        ocr_result=None,
    )

    assert panel.shape[1] == 640
    assert panel.shape[0] > 0
    assert panel.dtype == np.uint8
