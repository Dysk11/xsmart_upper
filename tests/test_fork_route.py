"""Geometric fork recognition and obstacle-regression tests."""

from __future__ import annotations

import cv2
import numpy as np

from core.blocking_analyzer import BlockingAnalyzer, DetectedObject, attach_roi_bboxes
from core.lane_detector import LaneDetector


def _detector() -> LaneDetector:
    return LaneDetector(
        {
            "centerline": {"scan_step": 4, "min_valid_points": 4, "default_lane_width_px": 50},
            "confidence": {"lost_threshold": 0.0, "expected_area_ratio": 0.01},
            "boundary": {"min_run_width_px": 4},
            "fork": {"confirm_frames": 1, "release_frames": 2, "min_lost_rows": 0},
        }
    )


def _fork_mask(direction: str) -> np.ndarray:
    mask = np.zeros((220, 300), dtype=np.uint8)
    cv2.rectangle(mask, (130, 105), (170, 219), 255, -1)
    if direction in {"left", "both"}:
        cv2.rectangle(mask, (55, 25), (100, 100), 255, -1)
    if direction in {"right", "both"}:
        cv2.rectangle(mask, (200, 25), (245, 100), 255, -1)
    cv2.rectangle(mask, (130, 25), (170, 104), 255, -1)
    return mask


def test_left_fork_is_reported_without_selecting_route() -> None:
    result = _detector().detect_from_mask(_fork_mask("left"))
    assert result.fork_result.left_detected
    assert not result.fork_result.right_detected
    assert result.fork_result.selected_direction is None


def test_right_fork_is_reported_without_selecting_route() -> None:
    result = _detector().detect_from_mask(_fork_mask("right"))
    assert result.fork_result.right_detected
    assert not result.fork_result.left_detected
    assert result.fork_result.requested_direction is None


def test_both_forks_are_reported_independently() -> None:
    result = _detector().detect_from_mask(_fork_mask("both"))
    assert result.fork_result.left_detected
    assert result.fork_result.right_detected


def test_plain_curve_does_not_trigger_fork() -> None:
    mask = np.zeros((220, 300), dtype=np.uint8)
    points = np.asarray([(130 + int(25 * (1 - y / 219)), y) for y in range(220)], dtype=np.int32)
    cv2.polylines(mask, [points], False, 255, 42)
    result = _detector().detect_from_mask(mask)
    assert not result.fork_result.fork_detected


def test_non_blocking_classes_remain_ignored() -> None:
    objects = attach_roi_bboxes(
        [DetectedObject("road_sign", 0.9, (180, 120, 260, 220))],
        (100, 60, 500, 300), 400, 240,
    )
    result = BlockingAnalyzer({}).analyze(objects, [(170, 230), (190, 190)], 400, 240)
    assert not result.need_avoid
