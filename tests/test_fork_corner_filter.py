"""Regression tests for false LF/RF markers near the ROI edges."""

from core.lane.detector import LaneDetector


def make_detector() -> LaneDetector:
    return LaneDetector(
        {
            "fork": {
                "corner_span_rows": 2,
                "outward_jump_ratio": 0.08,
                "min_lost_rows": 3,
                "corner_min_y_ratio": 0.05,
                "corner_max_y_ratio": 0.72,
                "corner_side_margin_ratio": 0.02,
                "confirm_frames": 1,
                "release_frames": 1,
            }
        }
    )


def make_boundaries(height: int = 100):
    left = [(30, height - 1 - index) for index in range(height)]
    right = [(70, height - 1 - index) for index in range(height)]
    return left, right


def detect(detector, left, right, left_lost=None, right_lost=None):
    row_count = len(left)
    return detector._geometric_fork_result(
        left,
        right,
        left_lost or [False] * row_count,
        right_lost or [False] * row_count,
        [],
        [],
        (row_count, 100),
    )


def test_rejects_outward_corner_in_lower_edge_band() -> None:
    detector = make_detector()
    left, right = make_boundaries()
    left[19] = (15, 80)
    right[19] = (85, 80)

    result = detect(detector, left, right)

    assert result.left_corner is None
    assert result.right_corner is None
    assert not result.left_detected
    assert not result.right_detected


def test_rejects_reconstructed_boundary_as_corner() -> None:
    detector = make_detector()
    left, right = make_boundaries()
    candidate_index = 35
    left[candidate_index] = (15, 64)
    left_lost = [False] * len(left)
    left_lost[candidate_index] = True
    left_lost[50:53] = [True, True, True]

    result = detect(detector, left, right, left_lost=left_lost)

    assert result.left_corner is None
    assert not result.left_detected


def test_keeps_interior_measured_corner_detection() -> None:
    detector = make_detector()
    left, right = make_boundaries()
    candidate_index = 35
    left[candidate_index] = (15, 64)
    left_lost = [False] * len(left)
    left_lost[50:53] = [True, True, True]

    result = detect(detector, left, right, left_lost=left_lost)

    assert result.left_corner == (15, 64)
    assert result.left_detected
