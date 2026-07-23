"""Regression tests for the bottom debug panel layout."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.object.pedestrian_safety import PedestrianSafetyResult
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
        "target_result": SimpleNamespace(
            target_point_roi=(213.0, 80.0),
            target_lateral_error_px=-107.0,
            reason="fixed-height lane target",
        ),
        "control_command": SimpleNamespace(mode="NORMAL", steer_deg=-35.0),
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
        pedestrian_safety_result=SimpleNamespace(
            latched=False,
            armed=True,
            human_count=0,
            frozen_target_x_frame=None,
            target_region="none",
            cooldown_remaining_sec=0.0,
            reason="armed; no qualifying pedestrian center in ROI",
        ),
    )
    long_panel = visualizer._build_debug_panel(
        **common,
        detection_result=make_detection(
            "geometry selected by frame center because candidate distances remained stable "
            "through_multiple_consecutive_bottom_track_regions"
        ),
        pedestrian_safety_result=SimpleNamespace(
            latched=True,
            armed=False,
            human_count=2,
            frozen_target_x_frame=318.0,
            target_region="center",
            cooldown_remaining_sec=0.0,
            reason=(
                "latched triggering pedestrian still waits to cross the frozen target "
                "through_multiple_consecutive_detection_results"
            ),
        ),
    )

    assert short_panel.shape[1] == 640
    assert long_panel.shape[1] == 640
    assert long_panel.shape[0] > short_panel.shape[0]


def test_debug_panel_omits_camera_legend_header(monkeypatch) -> None:
    drawn_lines: list[str] = []

    def capture_lines(image, lines, **kwargs):
        drawn_lines.extend(lines)
        return image

    monkeypatch.setattr(
        "core.visualization.visualizer.draw_text_lines",
        capture_lines,
    )
    visualizer = Visualizer({"show_window": False, "debug_panel_font_size": 18})
    visualizer._build_debug_panel(
        width=640,
        target_result=None,
        pedestrian_safety_result=None,
        control_command=None,  # type: ignore[arg-type]
        fps_value=0.0,
    )

    drawn_text = "\n".join(drawn_lines)
    assert "窗口1：原始画面" not in drawn_text
    assert "绿线=原中心线" not in drawn_text
    assert "洋红线=左岔中线" not in drawn_text


def test_canvas_keeps_camera_frame_size_without_embedded_debug_panel() -> None:
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
    assert canvas.shape[0] == frame.shape[0]
    assert np.array_equal(canvas[:100, :100], frame[:100, :100])


def test_pedestrian_regions_and_frozen_target_are_drawn() -> None:
    visualizer = Visualizer({"show_window": False})
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = PedestrianSafetyResult(
        stop_required=True,
        latched=True,
        armed=False,
        center_region_frame=(30.0, 10.0, 70.0, 90.0),
        frozen_target_x_frame=50.0,
        target_region="center",
        tracked_center_frame=(45.0, 60.0),
        human_count=1,
        cooldown_remaining_sec=0.0,
        reason="test",
    )

    visualizer._draw_pedestrian_regions(frame, result)

    assert tuple(frame[10, 30]) == (0, 255, 255)
    assert tuple(frame[10, 70]) == (0, 255, 255)
    assert tuple(frame[50, 50]) == (0, 0, 255)
    assert tuple(frame[60, 45]) == (0, 0, 255)


def test_car_warning_zone_and_avoidance_route_are_drawn() -> None:
    visualizer = Visualizer({"show_window": False})
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = SimpleNamespace(
        active=True,
        shifted_centerline_points=[(69.0, 150.0), (69.0, 50.0)],
        stop_required=False,
        warning_zones=[
            SimpleNamespace(
                bbox_frame=(70.0, 50.0, 130.0, 110.0),
                avoid_side="left",
            )
        ],
    )

    output = visualizer._draw_car_avoidance(
        frame,
        result,  # type: ignore[arg-type]
        roi_offset=(0, 0),
    )

    assert tuple(output[50, 70]) == (0, 128, 255)
    assert tuple(output[150, 69]) == (0, 255, 255)


def test_debug_panel_handles_missing_optional_results() -> None:
    visualizer = Visualizer({"show_window": False})
    panel = visualizer._build_debug_panel(
        width=640,
        target_result=None,
        pedestrian_safety_result=None,
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
