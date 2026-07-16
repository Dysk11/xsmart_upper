"""Regression tests for perspective-inferred fork centerlines."""

from __future__ import annotations

import numpy as np
import pytest

from core.lane.detector import LaneDetector


HEIGHT = 80
WIDTH = 160
CENTER_X = WIDTH // 2


def make_detector(**overrides) -> LaneDetector:
    config = {
        "boundary": {
            "gradient_jump_ratio": 1.0,
            "gradient_step_ratio": 1.0,
            "max_single_side_gap_rows": 0,
            "min_run_width_px": 6,
        },
        "centerline": {
            "scan_step": 1,
            "min_valid_points": 4,
            "default_lane_width_px": 60,
            "perspective_width_top_px": 30,
            "perspective_width_bottom_px": 60,
            "lookahead_ratio": 0.45,
        },
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
        },
    }
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    return LaneDetector(config)


def perspective_width(y: int) -> float:
    return 30.0 + 30.0 * float(y) / float(HEIGHT - 1)


def fill_run(mask: np.ndarray, y: int, center_x: float, run_width: float) -> None:
    left = max(0, int(round(center_x - 0.5 * run_width)))
    right = min(mask.shape[1] - 1, int(round(center_x + 0.5 * run_width)))
    mask[y, left : right + 1] = 255


def make_normal_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(HEIGHT):
        fill_run(mask, y, CENTER_X, perspective_width(y))
    return mask


def make_fork_mask(split_y: int = 40) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(HEIGHT):
        width = perspective_width(y)
        if y >= split_y:
            fill_run(mask, y, CENTER_X, width)
            continue
        half_separation = 0.70 * width
        fill_run(mask, y, CENTER_X - half_separation, width)
        fill_run(mask, y, CENTER_X + half_separation, width)
    return mask


def make_close_fork_mask(split_y: int = 40) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(HEIGHT):
        width = perspective_width(y)
        if y >= split_y:
            fill_run(mask, y, CENTER_X, width)
            continue
        fill_run(mask, y, CENTER_X - 18, 10)
        fill_run(mask, y, CENTER_X + 18, 10)
    return mask


def sample_x(points, y: int) -> float:
    point = min(points, key=lambda value: abs(value[1] - y))
    return float(point[0])


def test_fixed_perspective_width_interpolates_top_middle_and_bottom() -> None:
    detector = make_detector()

    assert detector._perspective_lane_width(0, 101) == pytest.approx(30.0)
    assert detector._perspective_lane_width(50, 101) == pytest.approx(45.0)
    assert detector._perspective_lane_width(100, 101) == pytest.approx(60.0)


@pytest.mark.parametrize(
    ("centerline", "message"),
    [
        ({"perspective_width_top_px": 0}, "top_px"),
        ({"perspective_width_bottom_px": 0}, "bottom_px"),
        (
            {"perspective_width_top_px": 61, "perspective_width_bottom_px": 60},
            "greater than or equal",
        ),
    ],
)
def test_rejects_invalid_perspective_width_config(centerline, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_detector(centerline=centerline)


def test_explicit_direction_selects_inferred_left_or_right_centerline() -> None:
    mask = make_fork_mask()
    left_result = make_detector().detect_from_mask(mask, route_direction="left")
    right_result = make_detector().detect_from_mask(mask, route_direction="right")

    assert left_result.fork_result.fork_detected
    assert right_result.fork_result.fork_detected
    assert left_result.fork_result.selected_direction == "left"
    assert right_result.fork_result.selected_direction == "right"
    assert left_result.fork_result.left_centerline_points
    assert left_result.fork_result.right_centerline_points
    assert sample_x(left_result.centerline_points, 20) < CENTER_X
    assert sample_x(right_result.centerline_points, 20) > CENTER_X
    assert sample_x(left_result.centerline_points, 70) == pytest.approx(CENTER_X, abs=2)
    assert sample_x(right_result.centerline_points, 70) == pytest.approx(CENTER_X, abs=2)


def test_close_fork_region_keeps_only_the_normal_centerline() -> None:
    result = make_detector().detect_from_mask(make_close_fork_mask())

    assert result.fork_result.fork_detected
    assert result.fork_result.left_centerline_points == []
    assert result.fork_result.right_centerline_points == []
    assert result.fork_result.selected_direction is None
    assert sample_x(result.centerline_points, 20) == pytest.approx(CENTER_X, abs=2)


def test_current_branch_is_held_until_fork_release_and_explicit_route_can_override() -> None:
    detector = make_detector()
    fork_mask = make_fork_mask()

    first = detector.detect_from_mask(fork_mask)
    held = detector.detect_from_mask(fork_mask)
    overridden = detector.detect_from_mask(fork_mask, route_direction="right")
    released = detector.detect_from_mask(make_normal_mask())

    assert first.fork_result.selected_direction == "left"
    assert held.fork_result.selected_direction == "left"
    assert overridden.fork_result.selected_direction == "right"
    assert sample_x(overridden.centerline_points, 20) > CENTER_X
    assert not released.fork_result.fork_detected
    assert released.fork_result.selected_direction is None
    assert detector._held_fork_direction is None


def test_isolated_outer_run_does_not_create_a_fork_or_double_centerline() -> None:
    mask = make_normal_mask()
    mask[10, 5:16] = 255

    result = make_detector().detect_from_mask(mask)

    assert not result.fork_result.fork_detected
    assert result.fork_result.left_centerline_points == []
    assert result.fork_result.right_centerline_points == []
    assert sample_x(result.centerline_points, 20) == pytest.approx(CENTER_X, abs=2)
