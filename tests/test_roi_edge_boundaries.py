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
