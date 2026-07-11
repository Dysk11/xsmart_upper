"""Tests for RKNN YOLOv5-seg lane postprocessing."""

from __future__ import annotations

import numpy as np

from core.rknn_lane_segmenter import (
    EXPECTED_OUTPUT_SHAPES,
    LetterboxInfo,
    RknnLaneSegmenter,
)


def make_segmenter(**overrides) -> RknnLaneSegmenter:
    config = {
        "enable": True,
        "score_threshold": 0.25,
        "nms_threshold": 0.45,
        "mask_threshold": 0.5,
    }
    config.update(overrides)
    return RknnLaneSegmenter(config)


def test_decode_requires_exact_seven_output_contract() -> None:
    segmenter = make_segmenter()
    outputs = [np.zeros(shape, dtype=np.float32) for shape in EXPECTED_OUTPUT_SHAPES]
    predictions, prototypes = segmenter.decode(outputs)
    assert predictions.shape == (1, 18900, 38)
    assert prototypes.shape == (1, 32, 120, 160)


def test_decode_rejects_wrong_output_shape() -> None:
    segmenter = make_segmenter()
    outputs = [np.zeros(shape, dtype=np.float32) for shape in EXPECTED_OUTPUT_SHAPES]
    outputs[-1] = np.zeros((1, 32, 60, 80), dtype=np.float32)
    try:
        segmenter.decode(outputs)
    except ValueError as error:
        assert "unexpected RKNN lane output shapes" in str(error)
    else:
        raise AssertionError("invalid output shape was accepted")


def test_nms_suppresses_overlapping_lower_score_box() -> None:
    segmenter = make_segmenter()
    boxes = np.asarray([[10, 10, 110, 110], [12, 12, 108, 108], [200, 200, 250, 250]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    assert segmenter._nms(boxes, scores) == [0, 2]


def test_postprocess_reconstructs_and_unletterboxes_mask() -> None:
    segmenter = make_segmenter()
    predictions = np.zeros((1, 2, 38), dtype=np.float32)
    predictions[0, 0, :6] = [320, 240, 320, 240, 0.9, 0.9]
    predictions[0, 0, 6] = 10.0
    predictions[0, 1, :6] = [320, 240, 318, 238, 0.8, 0.8]
    predictions[0, 1, 6] = 10.0
    prototypes = np.zeros((1, 32, 120, 160), dtype=np.float32)
    prototypes[0, 0] = 1.0
    segmenter.decode = lambda outputs: (predictions, prototypes)  # type: ignore[method-assign]
    info = LetterboxInfo(0.8, 0, 0, 640, 480, 800, 600)
    result = segmenter.postprocess([], info)
    assert result.status == "ok"
    assert result.mask.shape == (600, 800)
    assert result.confidence > 0.8
    assert len(result.instances) == 1
    assert np.count_nonzero(result.mask) > 0
    assert result.mask[300, 400] == 255


def test_preprocess_and_restore_non_four_by_three_frame() -> None:
    segmenter = make_segmenter()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tensor, info = segmenter._preprocess(frame)
    assert tensor.shape == (1, 480, 640, 3)
    assert (info.pad_x, info.pad_y, info.resized_width, info.resized_height) == (0, 60, 640, 360)
    model_mask = np.zeros((480, 640), dtype=np.uint8)
    model_mask[60:420] = 255
    restored = segmenter._restore_mask(model_mask, info)
    assert restored.shape == (720, 1280)
    assert np.all(restored == 255)


def test_missing_runtime_returns_explicit_empty_result() -> None:
    segmenter = make_segmenter(model_path="definitely-missing.rknn")
    result = segmenter.segment(np.zeros((480, 640, 3), dtype=np.uint8))
    assert result.status == "runtime_unavailable"
    assert result.mask.shape == (480, 640)
    assert np.count_nonzero(result.mask) == 0
