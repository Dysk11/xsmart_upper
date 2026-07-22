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
            "roughness_threshold_px": 3.0,
            "roughness_tie_margin_px": 0.1,
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


def make_rough_fork_mask(
    left_amplitude: float = 0.0,
    right_amplitude: float = 0.0,
    split_y: int = 40,
) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(HEIGHT):
        width = perspective_width(y)
        if y >= split_y:
            fill_run(mask, y, CENTER_X, width)
            continue
        half_separation = 0.70 * width
        sign = 1.0 if (y // 2) % 2 else -1.0
        fill_run(mask, y, CENTER_X - half_separation + sign * left_amplitude, width)
        fill_run(mask, y, CENTER_X + half_separation + sign * right_amplitude, width)
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


@pytest.mark.parametrize(
    ("fork", "message"),
    [
        ({"roughness_threshold_px": 0.0}, "threshold_px"),
        ({"roughness_threshold_px": float("inf")}, "threshold_px"),
        ({"roughness_tie_margin_px": -0.1}, "tie_margin_px"),
        ({"roughness_tie_margin_px": float("nan")}, "tie_margin_px"),
    ],
)
def test_rejects_invalid_fork_roughness_config(fork, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_detector(fork=fork)


def test_second_difference_distinguishes_smooth_curves_from_jagged_lines() -> None:
    detector = make_detector()
    straight = [(50, y) for y in range(40)]
    gentle_curve = [(int(round(50.0 + 0.02 * (y - 20) ** 2)), y) for y in range(40)]
    jagged = [(50 + (4 if (y // 2) % 2 else -4), y) for y in range(40)]

    assert detector._fork_roughness(straight) == pytest.approx(0.0)
    assert detector._fork_roughness(gentle_curve) < 1.0
    assert detector._fork_roughness(jagged) > 3.0
    assert detector._fork_roughness([(50, 1), (50, 0)]) is None


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


def test_symmetric_fork_does_not_select_a_current_branch() -> None:
    detector = make_detector()

    result = detector.detect_from_mask(
        make_fork_mask(),
        vehicle_center_x=CENTER_X,
    )

    assert result.fork_result.fork_detected
    assert result.fork_result.selected_direction is None
    assert detector._held_fork_direction is None
    assert "distances too close" in result.fork_result.reason


@pytest.mark.parametrize(
    ("vehicle_center_x", "expected_direction"),
    [
        (50.0, "left"),
        (110.0, "right"),
    ],
)
def test_branch_closest_to_frame_center_is_selected(
    vehicle_center_x: float,
    expected_direction: str,
) -> None:
    result = make_detector().detect_from_mask(
        make_fork_mask(),
        vehicle_center_x=vehicle_center_x,
    )

    assert result.fork_result.selected_direction == expected_direction
    assert f"selected {expected_direction} branch by frame center" in result.fork_result.reason


@pytest.mark.parametrize(
    ("left_amplitude", "right_amplitude", "vehicle_center_x", "expected_direction"),
    [
        (4.0, 0.0, 40.0, "right"),
        (0.0, 4.0, 120.0, "left"),
    ],
)
def test_single_rough_branch_is_hidden_and_smoother_branch_is_selected(
    left_amplitude: float,
    right_amplitude: float,
    vehicle_center_x: float,
    expected_direction: str,
) -> None:
    result = make_detector().detect_from_mask(
        make_rough_fork_mask(left_amplitude, right_amplitude),
        vehicle_center_x=vehicle_center_x,
    )
    fork = result.fork_result
    rejected_direction = "right" if expected_direction == "left" else "left"

    assert fork.selected_direction == expected_direction
    assert fork.rejected_direction == rejected_direction
    assert getattr(fork, f"{rejected_direction}_centerline_points") == []
    assert getattr(fork, f"{expected_direction}_centerline_points")
    assert "branch by roughness" in fork.reason
    assert f"rejected={rejected_direction}" in fork.reason


def test_two_rough_branches_discard_the_one_with_larger_roughness() -> None:
    result = make_detector().detect_from_mask(
        make_rough_fork_mask(left_amplitude=8.0, right_amplitude=4.0),
        vehicle_center_x=40.0,
    )
    fork = result.fork_result

    assert fork.left_roughness_px > fork.right_roughness_px > 3.0
    assert fork.selected_direction == "right"
    assert fork.rejected_direction == "left"
    assert fork.left_centerline_points == []
    assert fork.right_centerline_points


def test_similarly_rough_branches_fall_back_to_frame_center_selection() -> None:
    result = make_detector().detect_from_mask(
        make_rough_fork_mask(left_amplitude=4.0, right_amplitude=4.0),
        vehicle_center_x=40.0,
    )
    fork = result.fork_result

    assert fork.left_roughness_px > 3.0
    assert fork.right_roughness_px > 3.0
    assert abs(
        fork.left_roughness_px - fork.right_roughness_px
    ) <= 0.1
    assert fork.selected_direction == "left"
    assert fork.rejected_direction is None
    assert fork.left_centerline_points
    assert fork.right_centerline_points
    assert "branch by frame center" in fork.reason


def test_threshold_is_strictly_greater_and_invalid_scores_fall_back() -> None:
    detector = make_detector()

    assert detector._choose_fork_direction_by_roughness(3.0, 0.0) == (None, None)
    assert detector._choose_fork_direction_by_roughness(None, 4.0) == (None, None)


def test_requested_rough_branch_is_not_filtered() -> None:
    result = make_detector().detect_from_mask(
        make_rough_fork_mask(left_amplitude=4.0),
        route_direction="left",
        vehicle_center_x=120.0,
    )
    fork = result.fork_result

    assert fork.left_roughness_px > 3.0
    assert fork.selected_direction == "left"
    assert fork.rejected_direction is None
    assert fork.left_centerline_points
    assert "branch (requested)" in fork.reason


def test_held_branch_is_not_reconsidered_by_roughness() -> None:
    detector = make_detector()

    first = detector.detect_from_mask(make_fork_mask(), vehicle_center_x=40.0)
    held = detector.detect_from_mask(
        make_rough_fork_mask(left_amplitude=4.0),
        vehicle_center_x=120.0,
    )

    assert first.fork_result.selected_direction == "left"
    assert held.fork_result.left_roughness_px > 3.0
    assert held.fork_result.selected_direction == "left"
    assert held.fork_result.rejected_direction is None
    assert held.fork_result.left_centerline_points
    assert "branch (held)" in held.fork_result.reason


def test_frame_center_selection_is_held_until_release_and_explicit_route_can_override() -> None:
    detector = make_detector()
    fork_mask = make_fork_mask()

    first = detector.detect_from_mask(fork_mask, vehicle_center_x=110.0)
    held = detector.detect_from_mask(fork_mask, vehicle_center_x=50.0)
    overridden = detector.detect_from_mask(
        fork_mask,
        route_direction="left",
        vehicle_center_x=110.0,
    )
    released = detector.detect_from_mask(make_normal_mask())

    assert first.fork_result.selected_direction == "right"
    assert held.fork_result.selected_direction == "right"
    assert overridden.fork_result.selected_direction == "left"
    assert sample_x(overridden.centerline_points, 20) < CENTER_X
    assert not released.fork_result.fork_detected
    assert released.fork_result.selected_direction is None
    assert detector._held_fork_direction is None


def test_frame_center_distance_margin_requires_more_than_one_pixel_difference() -> None:
    detector = make_detector()
    left_candidates = [(60, 20), (60, 10)]
    right_candidates = [(100, 20), (100, 10)]

    tied, left_score, right_score = detector._choose_current_fork_direction(
        80.5,
        left_candidates,
        right_candidates,
    )
    right, _, _ = detector._choose_current_fork_direction(
        80.6,
        left_candidates,
        right_candidates,
    )
    left, _, _ = detector._choose_current_fork_direction(
        79.4,
        left_candidates,
        right_candidates,
    )

    assert tied is None
    assert abs(left_score - right_score) == pytest.approx(1.0)
    assert right == "right"
    assert left == "left"


def test_isolated_outer_run_does_not_create_a_fork_or_double_centerline() -> None:
    mask = make_normal_mask()
    mask[10, 5:16] = 255

    result = make_detector().detect_from_mask(mask)

    assert not result.fork_result.fork_detected
    assert result.fork_result.left_centerline_points == []
    assert result.fork_result.right_centerline_points == []
    assert sample_x(result.centerline_points, 20) == pytest.approx(CENTER_X, abs=2)
