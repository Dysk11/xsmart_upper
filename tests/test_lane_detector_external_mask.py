"""Regression tests for externally supplied segmentation masks."""

import cv2
import numpy as np

from core.lane_detector import LaneDetector


def test_external_mask_preserves_segmentation_metadata_and_builds_lane() -> None:
    detector = LaneDetector(
        {
            "connected_components": {"min_area": 20, "min_height": 5, "max_components": 3},
            "centerline": {"scan_step": 4, "min_valid_points": 4, "default_lane_width_px": 60},
            "confidence": {"lost_threshold": 0.01, "expected_area_ratio": 0.01},
        }
    )
    mask = np.zeros((240, 640), dtype=np.uint8)
    cv2.rectangle(mask, (270, 20), (370, 239), 255, -1)
    result = detector.detect_from_mask(mask, segmentation_confidence=0.87, segmentation_status="ok")
    assert result.mask.shape == mask.shape
    assert result.segmentation_confidence == 0.87
    assert result.segmentation_status == "ok"
    assert result.centerline_points
    assert not result.is_lane_lost


def test_empty_external_mask_returns_lane_lost() -> None:
    detector = LaneDetector({})
    result = detector.detect_from_mask(
        np.zeros((200, 300), dtype=np.uint8),
        segmentation_status="no_detection",
    )
    assert result.is_lane_lost
    assert result.segmentation_status == "no_detection"
