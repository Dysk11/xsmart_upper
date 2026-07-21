"""Regression tests for track runs that reach an ROI edge."""

from __future__ import annotations

import numpy as np

from core.lane.detector import LaneDetector


WIDTH = 20


def make_detector(*, max_single_side_gap_rows: int = 0) -> LaneDetector:
    return LaneDetector(
        {
            "boundary": {
                "max_single_side_gap_rows": max_single_side_gap_rows,
                "min_run_width_px": 1,
            }
        }
    )


def extract(
    detector: LaneDetector,
    mask: np.ndarray,
    *,
    bottom_center_x: float | None = None,
):
    return detector._extract_row_boundaries(
        mask,
        bottom_center_x=bottom_center_x,
    )[:5]


def test_left_roi_edge_is_a_valid_boundary() -> None:
    mask = np.zeros((1, WIDTH), dtype=np.uint8)
    mask[0, 0:13] = 255

    left, right, centers, left_lost, right_lost = extract(make_detector(), mask)

    assert left == [(0, 0)]
    assert right == [(12, 0)]
    assert centers == [(6, 0)]
    assert left_lost == [False]
    assert right_lost == [False]


def test_right_roi_edge_is_a_valid_boundary() -> None:
    mask = np.zeros((1, WIDTH), dtype=np.uint8)
    mask[0, 7:WIDTH] = 255

    left, right, centers, left_lost, right_lost = extract(make_detector(), mask)

    assert left == [(7, 0)]
    assert right == [(WIDTH - 1, 0)]
    assert centers == [(13, 0)]
    assert left_lost == [False]
    assert right_lost == [False]


def test_both_roi_edges_are_valid_boundaries() -> None:
    mask = np.full((1, WIDTH), 255, dtype=np.uint8)

    left, right, centers, left_lost, right_lost = extract(make_detector(), mask)

    assert left == [(0, 0)]
    assert right == [(WIDTH - 1, 0)]
    assert centers == [(WIDTH // 2, 0)]
    assert left_lost == [False]
    assert right_lost == [False]


def test_empty_row_reuses_boundaries_and_marks_both_sides_lost() -> None:
    mask = np.zeros((3, WIDTH), dtype=np.uint8)
    mask[2, 0:13] = 255
    mask[0, 1:14] = 255

    left, right, centers, left_lost, right_lost = extract(
        make_detector(max_single_side_gap_rows=1),
        mask,
    )

    assert left == [(0, 2), (0, 1), (1, 0)]
    assert right == [(12, 2), (12, 1), (13, 0)]
    assert centers == [(6, 2), (6, 1), (7, 0)]
    assert left_lost == [False, True, False]
    assert right_lost == [False, True, False]


def test_bottom_row_chooses_run_midpoint_closest_to_frame_center() -> None:
    detector = make_detector()
    detector.last_centerline_points = [(17, 0)]
    mask = np.zeros((3, WIDTH), dtype=np.uint8)
    mask[:, 1:6] = 255
    mask[:, 14:20] = 255

    left, right, centers, _left_lost, _right_lost = extract(
        detector,
        mask,
        bottom_center_x=5.0,
    )

    assert left[0] == (1, 2)
    assert right[0] == (5, 2)
    assert centers[0] == (3, 2)


def test_non_fork_detection_applies_bottom_frame_center_selection() -> None:
    detector = make_detector()
    detector.last_centerline_points = [(17, 7)]
    mask = np.zeros((8, WIDTH), dtype=np.uint8)
    mask[:, 1:6] = 255
    mask[:, 14:20] = 255

    result = detector.detect_from_mask(mask, vehicle_center_x=5.0)

    assert not result.fork_result.fork_detected
    assert result.centerline_points[0][0] == 3


def test_bottom_row_uses_mapped_frame_center_instead_of_roi_midpoint() -> None:
    mask = np.zeros((1, WIDTH), dtype=np.uint8)
    mask[0, 1:6] = 255
    mask[0, 11:16] = 255

    _left, _right, centers, _left_lost, _right_lost = extract(
        make_detector(),
        mask,
        bottom_center_x=15.0,
    )

    assert centers == [(13, 0)]


def test_bottom_row_equal_distance_uses_previous_centerline_as_tiebreaker() -> None:
    detector = make_detector()
    detector.last_centerline_points = [(16, 0)]
    mask = np.zeros((1, WIDTH), dtype=np.uint8)
    mask[0, 1:6] = 255
    mask[0, 15:20] = 255

    _left, _right, centers, _left_lost, _right_lost = extract(
        detector,
        mask,
        bottom_center_x=10.0,
    )

    assert centers == [(17, 0)]


def test_only_first_valid_bottom_row_uses_frame_center_selection() -> None:
    detector = make_detector()
    mask = np.zeros((3, WIDTH), dtype=np.uint8)
    mask[2, 1:6] = 255
    mask[2, 14:19] = 255
    mask[1, 2:7] = 255
    mask[1, 8:13] = 255

    _left, _right, centers, _left_lost, _right_lost = extract(
        detector,
        mask,
        bottom_center_x=4.0,
    )

    assert centers[:2] == [(3, 2), (4, 1)]
