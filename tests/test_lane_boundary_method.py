"""Synthetic coverage for the row-boundary lane method."""

from __future__ import annotations

import cv2
import numpy as np

from core.lane_detector import ARTICLE_ROW_WEIGHTS, LaneDetector


def _detector() -> LaneDetector:
    return LaneDetector(
        {
            "centerline": {"scan_step": 2, "min_valid_points": 5, "default_lane_width_px": 60},
            "confidence": {"lost_threshold": 0.0, "expected_area_ratio": 0.01},
            "boundary": {
                "gradient_jump_ratio": 0.05,
                "gradient_step_ratio": 0.025,
                "max_single_side_gap_rows": 8,
                "min_run_width_px": 4,
            },
            "fork": {"confirm_frames": 1, "release_frames": 1, "min_lost_rows": 2},
        }
    )


def test_article_weight_table_has_one_weight_per_source_row() -> None:
    assert ARTICLE_ROW_WEIGHTS.shape == (120,)
    assert ARTICLE_ROW_WEIGHTS[0] == 0
    assert ARTICLE_ROW_WEIGHTS[73] == 10


def test_vectorized_row_runs_preserve_edges_gaps_and_minimum_width() -> None:
    mask = np.zeros((4, 12), dtype=np.uint8)
    mask[0, 0:4] = 255
    mask[0, 6:12] = 255
    mask[1, 2:5] = 255
    mask[2, 3:8] = 255
    mask[2, 10:12] = 255

    assert _detector()._build_row_runs(mask) == [
        [(0, 3), (6, 11)],
        [],
        [(3, 7)],
        [],
    ]


def test_straight_lane_center_is_near_zero_error() -> None:
    mask = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(mask, (125, 0), (195, 199), 255, -1)
    result = _detector().detect_from_mask(mask)
    assert abs(result.lateral_error_px) <= 1.0
    assert abs(result.heading_error_deg) <= 1.0
    assert not result.is_lane_lost


def test_gradient_limit_suppresses_single_row_center_spike() -> None:
    mask = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(mask, (125, 0), (195, 199), 255, -1)
    mask[110, :] = 0
    mask[110, 230:300] = 255
    result = _detector().detect_from_mask(mask)
    xs = [x for x, _y in result.centerline_points]
    assert max(xs) - min(xs) < 25


def test_short_missing_rows_are_bridged_but_long_loss_is_not() -> None:
    short = np.zeros((200, 320), dtype=np.uint8)
    cv2.rectangle(short, (125, 0), (195, 199), 255, -1)
    short[90:95] = 0
    short_result = _detector().detect_from_mask(short)
    assert len(short_result.centerline_points) > 80

    long = short.copy()
    long[70:110] = 0
    long_result = _detector().detect_from_mask(long)
    assert len(long_result.centerline_points) < len(short_result.centerline_points)


def test_temporal_filter_reduces_abrupt_lateral_change() -> None:
    detector = _detector()
    centered = np.zeros((200, 320), dtype=np.uint8)
    shifted = np.zeros_like(centered)
    cv2.rectangle(centered, (125, 0), (195, 199), 255, -1)
    cv2.rectangle(shifted, (185, 0), (255, 199), 255, -1)
    detector.detect_from_mask(centered)
    detector.detect_from_mask(centered)
    result = detector.detect_from_mask(shifted)
    assert 0 < result.lateral_error_px < 60
