"""Regression tests for ROI-edge-triggered single-side lane following."""

from __future__ import annotations

import pytest

from core.lane.detector import LaneDetector


HEIGHT = 7
WIDTH = 160
CENTER_X = WIDTH // 2


def make_detector() -> LaneDetector:
    return LaneDetector(
        {
            "centerline": {
                "scan_step": 1,
                "min_valid_points": 3,
                "default_lane_width_px": 60,
                "perspective_width_top_px": 30,
                "perspective_width_bottom_px": 60,
            }
        }
    )


def perspective_width(y: int) -> float:
    return 30.0 + 30.0 * float(y) / float(HEIGHT - 1)


def make_boundaries():
    left = []
    right = []
    centers = []
    for y in range(HEIGHT - 1, -1, -1):
        half_width = 0.5 * perspective_width(y)
        left_x = int(round(CENTER_X - half_width))
        right_x = int(round(CENTER_X + half_width))
        left.append((left_x, y))
        right.append((right_x, y))
        centers.append((int(round(0.5 * (left_x + right_x))), y))
    return left, right, centers


def midpoints(left, right):
    return [
        (int(round(0.5 * (left_x + right_x))), y)
        for (left_x, y), (right_x, _right_y) in zip(left, right)
    ]


def rebuild(
    detector: LaneDetector,
    left,
    right,
    centers,
    *,
    left_lost=None,
    right_lost=None,
):
    row_count = len(centers)
    return detector._apply_roi_edge_single_side_fallback(
        left_points=left,
        right_points=right,
        raw_centers=centers,
        left_lost=left_lost or [False] * row_count,
        right_lost=right_lost or [False] * row_count,
        shape=(HEIGHT, WIDTH),
    )


def test_any_measured_left_edge_row_uses_right_boundary_for_whole_frame() -> None:
    detector = make_detector()
    left, right, centers = make_boundaries()
    left = [(x + 8, y) for x, y in left]
    left[3] = (0, left[3][1])
    centers = midpoints(left, right)

    rebuilt = rebuild(detector, left, right, centers)

    for (rebuilt_x, y), (right_x, _right_y) in zip(rebuilt, right):
        expected_x = right_x - 0.5 * perspective_width(y)
        assert rebuilt_x == pytest.approx(expected_x, abs=0.5)


def test_any_measured_right_edge_row_uses_left_boundary_for_whole_frame() -> None:
    detector = make_detector()
    left, right, centers = make_boundaries()
    right = [(x - 8, y) for x, y in right]
    right[2] = (WIDTH - 1, right[2][1])
    centers = midpoints(left, right)

    rebuilt = rebuild(detector, left, right, centers)

    for (rebuilt_x, y), (left_x, _left_y) in zip(rebuilt, left):
        expected_x = left_x + 0.5 * perspective_width(y)
        assert rebuilt_x == pytest.approx(expected_x, abs=0.5)


def test_both_sides_touching_edges_keep_original_midpoints() -> None:
    detector = make_detector()
    left, right, centers = make_boundaries()
    left[1] = (0, left[1][1])
    right[5] = (WIDTH - 1, right[5][1])

    assert rebuild(detector, left, right, centers) == centers


def test_neither_side_touching_edges_keeps_original_midpoints() -> None:
    detector = make_detector()
    left, right, centers = make_boundaries()

    assert rebuild(detector, left, right, centers) == centers


def test_edge_coordinate_on_inferred_gap_row_does_not_trigger() -> None:
    detector = make_detector()
    left, right, centers = make_boundaries()
    left[3] = (0, left[3][1])
    left_lost = [False] * HEIGHT
    left_lost[3] = True

    assert rebuild(
        detector,
        left,
        right,
        centers,
        left_lost=left_lost,
    ) == centers


def test_inferred_support_row_keeps_midpoint_during_single_side_rebuild() -> None:
    detector = make_detector()
    left, right, centers = make_boundaries()
    left[0] = (0, left[0][1])
    centers[3] = (centers[3][0] + 9, centers[3][1])
    right_lost = [False] * HEIGHT
    right_lost[4] = True

    rebuilt = rebuild(
        detector,
        left,
        right,
        centers,
        right_lost=right_lost,
    )

    assert rebuilt[4] == centers[4]
    assert rebuilt[3] != centers[3]


def test_edge_mode_applies_and_releases_without_cross_frame_state() -> None:
    detector = make_detector()
    left, right, centers = make_boundaries()
    edge_left = list(left)
    edge_centers = list(centers)
    edge_left[0] = (0, edge_left[0][1])
    edge_centers[0] = (right[0][0] // 2, edge_centers[0][1])

    edge_result = rebuild(detector, edge_left, right, edge_centers)
    normal_result = rebuild(detector, left, right, centers)

    assert edge_result != edge_centers
    assert normal_result == centers
