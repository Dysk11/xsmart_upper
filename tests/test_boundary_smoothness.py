"""Regression tests for normal-lane smoother-boundary centerline fallback."""

from __future__ import annotations

import numpy as np
import pytest

from core.lane.detector import LaneDetector


HEIGHT = 80
WIDTH = 160
CENTER_X = WIDTH // 2


def make_detector(**centerline_overrides) -> LaneDetector:
    centerline = {
        "scan_step": 1,
        "min_valid_points": 6,
        "default_lane_width_px": 60,
        "perspective_width_top_px": 30,
        "perspective_width_bottom_px": 60,
        "enable_boundary_smoothness_fallback": True,
        "boundary_smoothness_residual_threshold_px": 3.0,
        "boundary_smoothness_tie_margin_px": 0.1,
        "boundary_smoothness_confirm_frames": 2,
        "lookahead_ratio": 0.45,
    }
    centerline.update(centerline_overrides)
    return LaneDetector(
        {
            "boundary": {
                "gradient_jump_ratio": 1.0,
                "gradient_step_ratio": 1.0,
                "max_single_side_gap_rows": 0,
                "min_run_width_px": 6,
            },
            "centerline": centerline,
            "confidence": {
                "lost_threshold": 0.0,
                "expected_area_ratio": 0.01,
                "residual_tolerance_px": 40.0,
            },
            "fork": {
                "corner_span_rows": 2,
                "outward_jump_ratio": 0.08,
                "min_lost_rows": 3,
                "corner_min_y_ratio": 0.05,
                "corner_max_y_ratio": 0.72,
                "corner_side_margin_ratio": 0.02,
                "confirm_frames": 1,
                "release_frames": 1,
                "split_enter_ratio": 0.60,
                "split_exit_ratio": 0.35,
                "split_min_rows": 5,
                "smoothness_residual_threshold_px": 3.0,
                "smoothness_tie_margin_px": 0.1,
            },
        }
    )


def perspective_width(y: int) -> float:
    return 30.0 + 30.0 * float(y) / float(HEIGHT - 1)


def make_lane_mask(left_amplitude: float = 0.0, right_amplitude: float = 0.0) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(HEIGHT):
        half_width = 0.5 * perspective_width(y)
        sign = 1.0 if (y // 2) % 2 else -1.0
        left = int(round(CENTER_X - half_width + sign * left_amplitude))
        right = int(round(CENTER_X + half_width + sign * right_amplitude))
        mask[y, max(0, left) : min(WIDTH - 1, right) + 1] = 255
    return mask


def sample_x(points, y: int) -> float:
    return float(min(points, key=lambda point: abs(point[1] - y))[0])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"boundary_smoothness_residual_threshold_px": 0.0}, "threshold_px"),
        ({"boundary_smoothness_residual_threshold_px": float("inf")}, "threshold_px"),
        ({"boundary_smoothness_tie_margin_px": -0.1}, "tie_margin_px"),
        ({"boundary_smoothness_tie_margin_px": float("nan")}, "tie_margin_px"),
        ({"boundary_smoothness_confirm_frames": 0}, "confirm_frames"),
    ],
)
def test_rejects_invalid_boundary_smoothness_config(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_detector(**overrides)


def test_two_smooth_boundaries_keep_normal_midpoint() -> None:
    detector = make_detector()
    result = detector.detect_from_mask(make_lane_mask())

    assert not result.fork_result.fork_detected
    assert detector._normal_centerline_mode == "normal"
    assert sample_x(result.centerline_points, 20) == pytest.approx(CENTER_X, abs=1)


@pytest.mark.parametrize(
    ("left_amplitude", "right_amplitude", "expected_mode"),
    [(6.0, 0.0, "right"), (0.0, 6.0, "left")],
)
def test_single_rough_boundary_switches_after_two_frames(
    left_amplitude: float,
    right_amplitude: float,
    expected_mode: str,
) -> None:
    detector = make_detector()
    mask = make_lane_mask(left_amplitude, right_amplitude)

    first = detector.detect_from_mask(mask)
    assert detector._normal_centerline_mode == "normal"
    second = detector.detect_from_mask(mask)

    assert detector._normal_centerline_mode == expected_mode
    assert first.centerline_points
    assert sample_x(second.centerline_points, 20) == pytest.approx(CENTER_X, abs=1)


def test_two_rough_boundaries_choose_lower_residual() -> None:
    detector = make_detector()
    assert detector._choose_normal_centerline_mode(7.0, 4.0) == "right"
    assert detector._choose_normal_centerline_mode(4.0, 7.0) == "left"


def test_similarly_rough_boundaries_keep_normal_midpoint() -> None:
    detector = make_detector()
    assert detector._choose_normal_centerline_mode(4.0, 4.1) == "normal"


def test_insufficient_measured_points_do_not_trigger_fallback() -> None:
    detector = make_detector()
    points = [(50, y) for y in range(10)]
    lost = [True] * 5 + [False] * 5

    assert detector._boundary_smoothness_residual(points, lost) is None
    assert detector._choose_normal_centerline_mode(None, 5.0) == "normal"


def test_lost_row_uses_original_center_during_boundary_rebuild() -> None:
    detector = make_detector(boundary_smoothness_confirm_frames=1)
    left = [
        (20 + (8 if index % 2 else -8), 79 - index)
        for index in range(7)
    ]
    right = [(140, y) for _x, y in left]
    centers = [(80, y) for _x, y in left]
    rebuilt = detector._apply_boundary_smoothness_fallback(
        left_points=left,
        right_points=right,
        raw_centers=centers,
        left_lost=[False] * 7,
        right_lost=[False, False, True, False, False, False, False],
        shape=(HEIGHT, WIDTH),
    )
    assert detector._normal_centerline_mode == "right"
    assert rebuilt[2] == centers[2]


def test_mode_recovery_and_side_change_are_debounced() -> None:
    detector = make_detector()

    assert detector._update_normal_centerline_mode("left", 1.0, 6.0) == "normal"
    assert detector._update_normal_centerline_mode("left", 1.0, 6.0) == "left"

    # Both boundaries recover: keep the still-smooth active side for one frame.
    assert detector._update_normal_centerline_mode("normal", 1.0, 1.0) == "left"
    assert detector._update_normal_centerline_mode("normal", 1.0, 1.0) == "normal"

    assert detector._update_normal_centerline_mode("left", 1.0, 6.0) == "normal"
    assert detector._update_normal_centerline_mode("left", 1.0, 6.0) == "left"

    # When the active side becomes rough, use the normal midpoint during confirmation.
    assert detector._update_normal_centerline_mode("right", 6.0, 1.0) == "normal"
    assert detector._update_normal_centerline_mode("right", 6.0, 1.0) == "right"


def test_fork_detection_resets_normal_boundary_mode() -> None:
    detector = make_detector(boundary_smoothness_confirm_frames=1)
    detector._normal_centerline_mode = "left"
    fork_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(HEIGHT):
        width = perspective_width(y)
        if y >= 40:
            left = int(round(CENTER_X - 0.5 * width))
            right = int(round(CENTER_X + 0.5 * width))
            fork_mask[y, left : right + 1] = 255
        else:
            for center in (CENTER_X - 0.7 * width, CENTER_X + 0.7 * width):
                left = int(round(center - 0.5 * width))
                right = int(round(center + 0.5 * width))
                fork_mask[y, max(0, left) : min(WIDTH - 1, right) + 1] = 255

    result = detector.detect_from_mask(fork_mask, route_direction="left")

    assert result.fork_result.fork_detected
    assert result.fork_result.selected_direction == "left"
    assert detector._normal_centerline_mode == "normal"
