"""Regression tests for track runs that reach an ROI edge."""

from __future__ import annotations

import numpy as np

from core.lane.detector import LaneDetector


WIDTH = 20


def make_detector(
    *,
    max_single_side_gap_rows: int = 0,
    switch_threshold_px: float = 80.0,
    switch_confirm_frames: int = 3,
) -> LaneDetector:
    return LaneDetector(
        {
            "boundary": {
                "max_single_side_gap_rows": max_single_side_gap_rows,
                "min_run_width_px": 1,
            },
            "track_selection": {
                "switch_threshold_px": switch_threshold_px,
                "switch_confirm_frames": switch_confirm_frames,
                "pending_tolerance_px": 5,
            },
        }
    )


def extract(detector: LaneDetector, mask: np.ndarray):
    return detector._extract_row_boundaries(mask)[:5]


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


def test_left_roi_edge_remains_measured_while_tracing_upward() -> None:
    mask = np.zeros((6, WIDTH), dtype=np.uint8)
    mask[:, 0:13] = 255

    left, right, _centers, left_lost, right_lost = extract(
        make_detector(), mask
    )

    assert left == [(0, y) for y in range(5, -1, -1)]
    assert right == [(12, y) for y in range(5, -1, -1)]
    assert not any(left_lost)
    assert not any(right_lost)


def test_right_roi_edge_remains_measured_while_tracing_upward() -> None:
    mask = np.zeros((6, WIDTH), dtype=np.uint8)
    mask[:, 7:WIDTH] = 255

    left, right, _centers, left_lost, right_lost = extract(
        make_detector(), mask
    )

    assert left == [(7, y) for y in range(5, -1, -1)]
    assert right == [(WIDTH - 1, y) for y in range(5, -1, -1)]
    assert not any(left_lost)
    assert not any(right_lost)


def test_both_roi_edges_remain_measured_while_tracing_upward() -> None:
    mask = np.full((6, WIDTH), 255, dtype=np.uint8)

    left, right, _centers, left_lost, right_lost = extract(
        make_detector(), mask
    )

    assert left == [(0, y) for y in range(5, -1, -1)]
    assert right == [(WIDTH - 1, y) for y in range(5, -1, -1)]
    assert not any(left_lost)
    assert not any(right_lost)


def test_boundary_can_reach_roi_edge_and_continue_upward() -> None:
    mask = np.zeros((6, WIDTH), dtype=np.uint8)
    expected_left = {5: 3, 4: 2, 3: 1, 2: 0, 1: 0, 0: 0}
    for y, left_x in expected_left.items():
        mask[y, left_x : left_x + 11] = 255

    left, _right, _centers, left_lost, _right_lost = extract(
        make_detector(), mask
    )

    assert left == [(expected_left[y], y) for y in range(5, -1, -1)]
    assert not any(left_lost)


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


def test_bottom_row_chooses_run_nearest_mapped_vehicle_center() -> None:
    detector = make_detector()
    detector.last_centerline_points = [(17, 2), (17, 1), (17, 0)]
    mask = np.zeros((3, WIDTH), dtype=np.uint8)
    mask[:, 1:6] = 255
    mask[:, 14:20] = 255

    _left, _right, centers, _left_lost, _right_lost = detector._extract_row_boundaries(
        mask,
        bottom_center_x=4.0,
    )[:5]

    assert centers[0] == (3, 2)


def test_large_normal_track_switch_requires_consecutive_confirmation() -> None:
    detector = make_detector(switch_threshold_px=20.0, switch_confirm_frames=3)
    previous = [(10, 10), (10, 5), (10, 0)]
    candidate = [(100, 10), (100, 5), (100, 0)]
    detector.last_centerline_points = list(previous)

    first, first_reason = detector._stabilize_normal_track_switch(candidate)
    second, second_reason = detector._stabilize_normal_track_switch(candidate)
    third, third_reason = detector._stabilize_normal_track_switch(candidate)

    assert first == previous
    assert second == previous
    assert "confirm=1/3" in str(first_reason)
    assert "confirm=2/3" in str(second_reason)
    assert third == candidate
    assert third_reason is None


def test_alternating_large_track_candidates_never_confirm() -> None:
    detector = make_detector(switch_threshold_px=20.0, switch_confirm_frames=3)
    previous = [(50, 10), (50, 0)]
    detector.last_centerline_points = list(previous)

    for candidate_x in (100, 0, 100, 0):
        held, reason = detector._stabilize_normal_track_switch(
            [(candidate_x, 10), (candidate_x, 0)]
        )
        assert held == previous
        assert "confirm=1/3" in str(reason)
