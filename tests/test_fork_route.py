"""Tests for fork sign holding and branch selection."""

from __future__ import annotations

import pytest

from core.blocking_analyzer import BlockingAnalyzer, DetectedObject, attach_roi_bboxes
from core.fork_route_planner import ForkRoutePlanner


def test_fork_sign_class_names_are_case_insensitive() -> None:
    planner = ForkRoutePlanner({"min_confidence": 0.4})
    result = planner.update([DetectedObject("left", 0.8, (10, 10, 40, 50))])
    assert result.active
    assert result.requested_direction == "left"


def test_fork_sign_prefers_confidence_and_lower_position() -> None:
    planner = ForkRoutePlanner({"min_confidence": 0.4})
    result = planner.update(
        [
            DetectedObject("Left", 0.7, (10, 10, 40, 40)),
            DetectedObject("Right", 0.9, (10, 5, 40, 30)),
        ]
    )
    assert result.requested_direction == "right"


def test_fork_sign_tie_prefers_lower_position() -> None:
    planner = ForkRoutePlanner({"min_confidence": 0.4})
    result = planner.update(
        [
            DetectedObject("Left", 0.8, (10, 10, 40, 40)),
            DetectedObject("Right", 0.8, (10, 80, 40, 130)),
        ]
    )
    assert result.requested_direction == "right"


def test_fork_direction_holds_until_fork_clears() -> None:
    planner = ForkRoutePlanner({"sign_hold_frames": 2, "clear_after_no_fork_frames": 2})
    assert planner.update([DetectedObject("Right", 0.9, (10, 10, 40, 50))]).requested_direction == "right"
    assert planner.update([], fork_detected=True).requested_direction == "right"
    assert planner.update([], fork_detected=False).active
    cleared = planner.update([], fork_detected=False)
    assert not cleared.active


def test_left_right_signs_do_not_trigger_blocking() -> None:
    objects = attach_roi_bboxes(
        [DetectedObject("Left", 0.9, (180, 120, 260, 220))],
        (100, 60, 500, 300),
        400,
        240,
    )
    filtered = [obj for obj in objects if obj.class_name.casefold() in {"car", "human"}]
    result = BlockingAnalyzer({}).analyze(filtered, [(170, 230), (190, 190), (220, 150)], 400, 240)
    assert not result.need_avoid


def test_blocking_analyzer_ignores_left_right_by_config() -> None:
    objects = attach_roi_bboxes(
        [DetectedObject("Right", 0.9, (250, 150, 360, 290))],
        (100, 60, 500, 300),
        400,
        240,
    )
    result = BlockingAnalyzer({}).analyze(objects, [(170, 230), (190, 190), (220, 150)], 400, 240)
    assert not result.need_avoid


def test_lane_detector_selects_requested_fork_branch() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from core.lane_detector import LaneDetector

    detector = LaneDetector(
        {
            "hsv": {"lower": [90, 60, 40], "upper": [130, 255, 255]},
            "connected_components": {"min_area": 20, "min_height": 3, "min_component_chain_score": 1.0},
            "centerline": {"scan_step": 4, "min_valid_points": 2},
            "confidence": {"lost_threshold": 0.0},
        }
    )
    roi = np.zeros((220, 300, 3), dtype=np.uint8)
    blue = (255, 0, 0)
    cv2.rectangle(roi, (135, 190), (165, 210), blue, -1)
    cv2.rectangle(roi, (130, 145), (160, 165), blue, -1)
    cv2.rectangle(roi, (70, 90), (100, 110), blue, -1)
    cv2.rectangle(roi, (200, 90), (230, 110), blue, -1)

    left = detector.detect(roi, route_direction="left")
    right = detector.detect(roi, route_direction="right")

    assert left.fork_result.fork_detected
    assert left.fork_result.selected_direction == "left"
    assert right.fork_result.selected_direction == "right"
    assert left.centerline_points[-1][0] < right.centerline_points[-1][0]


def test_lane_detector_keeps_default_route_without_sign() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from core.lane_detector import LaneDetector

    detector = LaneDetector(
        {
            "hsv": {"lower": [90, 60, 40], "upper": [130, 255, 255]},
            "connected_components": {"min_area": 20, "min_height": 3, "min_component_chain_score": 1.0},
            "centerline": {"scan_step": 4, "min_valid_points": 2},
            "confidence": {"lost_threshold": 0.0},
        }
    )
    roi = np.zeros((220, 300, 3), dtype=np.uint8)
    blue = (255, 0, 0)
    cv2.rectangle(roi, (135, 190), (165, 210), blue, -1)
    cv2.rectangle(roi, (130, 145), (160, 165), blue, -1)
    cv2.rectangle(roi, (70, 90), (100, 110), blue, -1)
    cv2.rectangle(roi, (200, 90), (230, 110), blue, -1)

    result = detector.detect(roi)

    assert result.fork_result.fork_detected
    assert result.fork_result.requested_direction is None
    assert result.fork_result.selected_direction is None
