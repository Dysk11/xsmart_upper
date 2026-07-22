"""Unit tests for the eight-neighbor lane-boundary tracer."""

from __future__ import annotations

import numpy as np

from core.lane.detector import LaneDetector


HEIGHT = 12
WIDTH = 32


def make_detector(
    *,
    min_run_width_px: int = 1,
    max_single_side_gap_rows: int = 1,
) -> LaneDetector:
    return LaneDetector(
        {
            "boundary": {
                "gradient_jump_ratio": 0.2,
                "max_single_side_gap_rows": max_single_side_gap_rows,
                "min_run_width_px": min_run_width_px,
            },
            "centerline": {"min_valid_points": 3},
            "confidence": {"lost_threshold": 0.0},
        }
    )


def extract(detector: LaneDetector, mask: np.ndarray):
    return detector._extract_row_boundaries(mask)


def test_traces_straight_lane_from_bottom_to_top() -> None:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[:, 8:24] = 255

    left, right, centers, left_lost, right_lost, *_ = extract(
        make_detector(), mask
    )

    assert left == [(8, y) for y in range(HEIGHT - 1, -1, -1)]
    assert right == [(23, y) for y in range(HEIGHT - 1, -1, -1)]
    assert centers == [(16, y) for y in range(HEIGHT - 1, -1, -1)]
    assert not any(left_lost)
    assert not any(right_lost)


def test_traces_diagonal_lane_with_eight_neighbor_steps() -> None:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    expected_left: dict[int, int] = {}
    expected_right: dict[int, int] = {}
    for y in range(HEIGHT):
        left = 4 + y // 2
        right = left + 11
        mask[y, left : right + 1] = 255
        expected_left[y] = left
        expected_right[y] = right

    left, right, _centers, left_lost, right_lost, *_ = extract(
        make_detector(), mask
    )

    assert left == [(expected_left[y], y) for y in range(HEIGHT - 1, -1, -1)]
    assert right == [(expected_right[y], y) for y in range(HEIGHT - 1, -1, -1)]
    assert not any(left_lost)
    assert not any(right_lost)


def test_continues_above_horizontal_boundary_step() -> None:
    mask = np.zeros((6, WIDTH), dtype=np.uint8)
    mask[3:6, 9:23] = 255
    mask[0:3, 5:23] = 255

    left, right, _centers, left_lost, right_lost, *_ = extract(
        make_detector(max_single_side_gap_rows=0), mask
    )

    assert left == [(9, 5), (9, 4), (9, 3), (5, 2), (5, 1), (5, 0)]
    assert right == [(22, y) for y in range(5, -1, -1)]
    assert not any(left_lost)
    assert not any(right_lost)


def test_resumes_after_short_gap_and_marks_only_gap_row_lost() -> None:
    mask = np.zeros((5, WIDTH), dtype=np.uint8)
    mask[3:5, 6:20] = 255
    mask[0:2, 7:21] = 255

    left, right, _centers, left_lost, right_lost, *_ = extract(
        make_detector(max_single_side_gap_rows=1), mask
    )

    assert left == [(6, 4), (6, 3), (6, 2), (7, 1), (7, 0)]
    assert right == [(19, 4), (19, 3), (19, 2), (20, 1), (20, 0)]
    assert left_lost == [False, False, True, False, False]
    assert right_lost == [False, False, True, False, False]


def test_empty_mask_has_no_seed_or_boundaries() -> None:
    result = extract(make_detector(), np.zeros((HEIGHT, WIDTH), dtype=np.uint8))

    assert result[0] == []
    assert result[1] == []
    assert result[2] == []
    assert np.count_nonzero(result[5]) == 0


def test_seed_search_skips_empty_bottom_rows() -> None:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[0:8, 7:25] = 255

    left, right, _centers, *_ = extract(make_detector(), mask)

    assert left[0] == (7, 7)
    assert right[0] == (24, 7)
    assert left[-1] == (7, 0)
    assert right[-1] == (24, 0)


def test_too_narrow_foreground_does_not_create_seed() -> None:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[:, 15:17] = 255

    result = extract(make_detector(min_run_width_px=3), mask)

    assert result[0] == []
    assert result[1] == []


def test_bottom_seed_uses_vehicle_center_not_previous_centerline() -> None:
    detector = make_detector()
    detector.last_centerline_points = [(27, HEIGHT - 1), (27, 0)]
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[:, 2:10] = 255
    mask[:, 22:30] = 255

    left, right, centers, *_ = detector._extract_row_boundaries(
        mask,
        bottom_center_x=6.0,
    )

    assert left[0] == (2, HEIGHT - 1)
    assert right[0] == (9, HEIGHT - 1)
    assert centers[0] == (6, HEIGHT - 1)


def test_trace_is_bounded_by_three_times_roi_height() -> None:
    detector = make_detector()
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[:, 5:27] = 255
    padded = np.pad(mask > 0, 1).astype(np.uint8)

    points = detector._trace_eight_neighbor_boundary(
        mask,
        padded,
        (5, HEIGHT - 1),
        "left",
    )

    assert len(points) <= HEIGHT * 3
