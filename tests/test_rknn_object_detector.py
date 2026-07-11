"""Tests for PP-YOLOE RKNN output decoding."""

from __future__ import annotations

import numpy as np

from core.rknn_object_detector import LetterboxInfo, RknnObjectDetector


CLASS_NAMES = ["car", "coin", "Go", "human", "road_sign", "speed_limit", "Stop"]


def make_detector(score_threshold: float = 0.2) -> RknnObjectDetector:
    return RknnObjectDetector(
        {
            "class_names": CLASS_NAMES,
            "score_threshold": score_threshold,
            "nms_threshold": 0.45,
        }
    )


def identity_letterbox() -> LetterboxInfo:
    return LetterboxInfo(1.0, 0.0, 0.0, 640, 480)


def test_ppyoloe_two_output_contract_decodes_xyxy_boxes() -> None:
    boxes = np.zeros((1, 6300, 4), dtype=np.float32)
    scores = np.zeros((1, 7, 6300), dtype=np.float32)
    boxes[0, 10] = [100.0, 120.0, 220.0, 260.0]
    scores[0, 1, 10] = 0.91

    detections = make_detector()._postprocess(
        [boxes, scores], identity_letterbox(), (480, 640)
    )

    assert len(detections) == 1
    assert detections[0].class_name == "coin"
    assert detections[0].confidence == np.float32(0.91)
    assert detections[0].bbox_frame == (100, 120, 220, 260)


def test_ppyoloe_two_output_contract_applies_class_aware_nms() -> None:
    boxes = np.asarray([[[10, 20, 110, 120], [12, 22, 108, 118], [12, 22, 108, 118]]], dtype=np.float32)
    scores = np.zeros((1, 7, 3), dtype=np.float32)
    scores[0, 0, 0] = 0.9
    scores[0, 0, 1] = 0.8
    scores[0, 3, 2] = 0.7

    detections = make_detector()._postprocess(
        [boxes, scores], identity_letterbox(), (480, 640)
    )

    assert [(item.class_name, round(item.confidence, 1)) for item in detections] == [
        ("car", 0.9),
        ("human", 0.7),
    ]


def test_ppyoloe_two_output_contract_rejects_low_scores() -> None:
    boxes = np.asarray([[[10, 20, 110, 120]]], dtype=np.float32)
    scores = np.zeros((1, 7, 1), dtype=np.float32)
    scores[0, 0, 0] = 0.19

    assert make_detector()._postprocess(
        [boxes, scores], identity_letterbox(), (480, 640)
    ) == []
