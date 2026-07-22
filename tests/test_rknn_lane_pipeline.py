from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from core.lane.rknn_segmenter import (
    LetterboxInfo,
    RknnLaneSegmenter,
    SegmentationResult,
)
from core.runtime.app import UpperMachineApp


def test_lane_inference_can_be_completed_after_next_stage_boundary(monkeypatch) -> None:
    segmenter = RknnLaneSegmenter({"enable": True, "model_path": "unused.rknn"})
    segmenter._runtime_ready = True

    class FakeRuntime:
        @staticmethod
        def inference(inputs):
            assert inputs[0].shape == (1, 480, 640, 3)
            return [np.asarray([1.0], dtype=np.float32)]

    segmenter._rknn = FakeRuntime()
    info = LetterboxInfo(1.0, 0, 0, 640, 480, 640, 480)
    monkeypatch.setattr(
        segmenter,
        "_preprocess",
        lambda frame: (np.zeros((1, 480, 640, 3), dtype=np.uint8), info),
    )
    expected = SegmentationResult(np.full((480, 640), 255, dtype=np.uint8), [], 0.8, "ok")
    monkeypatch.setattr(segmenter, "postprocess", lambda outputs, letterbox: expected)

    inference = segmenter.infer(np.zeros((480, 640, 3), dtype=np.uint8))
    assert inference.outputs is not None
    result, timing = segmenter.complete(inference)

    assert result is expected
    assert set(timing) == {
        "preprocess_ms",
        "inference_ms",
        "postprocess_queue_ms",
        "postprocess_ms",
        "total_ms",
    }
    assert timing["total_ms"] >= timing["inference_ms"]


def test_vectorized_lane_masks_match_per_instance_reference(monkeypatch) -> None:
    segmenter = RknnLaneSegmenter(
        {
            "enable": True,
            "input_size": [640, 480],
            "score_threshold": 0.3,
            "nms_threshold": 0.45,
            "mask_threshold": 0.5,
            "max_instances": 3,
        }
    )
    predictions = np.zeros((1, 3, 38), dtype=np.float32)
    predictions[0, 0, :6] = [180, 180, 200, 160, 0.95, 0.95]
    predictions[0, 0, 6] = 2.0
    predictions[0, 1, :6] = [470, 300, 180, 180, 0.90, 0.90]
    predictions[0, 1, 7] = 2.0
    predictions[0, 2, :6] = [320, 240, 100, 100, 0.1, 0.1]

    prototypes = np.zeros((1, 32, 120, 160), dtype=np.float32)
    prototypes[0, 0, :, :80] = 1.0
    prototypes[0, 0, :, 80:] = -1.0
    prototypes[0, 1, :60, :] = -1.0
    prototypes[0, 1, 60:, :] = 1.0
    monkeypatch.setattr(segmenter, "decode", lambda outputs: (predictions, prototypes))

    info = LetterboxInfo(1.0, 0, 0, 640, 480, 640, 480)
    result = segmenter.postprocess([], info)

    candidates = predictions[0, :2]
    boxes = segmenter._xywh_to_xyxy(candidates[:, :4])
    proto = prototypes[0].reshape(32, -1)
    expected = np.zeros((480, 640), dtype=np.uint8)
    for index in (0, 1):
        logits = (candidates[index, 6:] @ proto).reshape(120, 160)
        instance_mask = cv2.resize(logits, (640, 480), interpolation=cv2.INTER_LINEAR) >= 0.0
        x1, y1, x2, y2 = segmenter._clip_box(boxes[index])
        crop = expected[y1:y2, x1:x2]
        crop[instance_mask[y1:y2, x1:x2]] = 255

    np.testing.assert_array_equal(result.mask, expected)
    assert len(result.instances) == 2


def test_lane_worker_selection_is_round_robin_with_pipeline_capacity() -> None:
    app = UpperMachineApp.__new__(UpperMachineApp)
    app.lane_worker_inflight = [0, 0]
    app.lane_pipeline_depth = 3
    app.next_lane_worker_index = 0

    selected = []
    for _ in range(6):
        worker = app._next_available_lane_worker()
        assert worker is not None
        selected.append(worker)
        app.lane_worker_inflight[worker] += 1

    assert selected == [0, 1, 0, 1, 0, 1]
    assert app._next_available_lane_worker() is None

    app.lane_worker_inflight[1] -= 1
    assert app._next_available_lane_worker() == 1


def test_worker_timing_accumulator_ignores_non_numeric_values() -> None:
    totals = {"inference_ms": 3.0}

    UpperMachineApp._accumulate_timing(
        totals,
        {"inference_ms": 4.5, "postprocess_ms": 2, "status": "ok"},
    )

    assert totals == {"inference_ms": 7.5, "postprocess_ms": 2.0}


def test_lane_timing_summary_includes_wait_geometry_and_bridge(capsys) -> None:
    app = UpperMachineApp.__new__(UpperMachineApp)
    app.lane_timing_enabled = True
    app.lane_timing_interval = 1
    app.lane_timing_count = 0
    app.lane_timing_total_ms = 0.0
    app.lane_timing_roi_ms = 0.0
    app.lane_timing_detect_ms = 0.0
    app.lane_timing_track_ms = 0.0
    app.lane_timing_camera_wait_ms = 0.0
    app.lane_timing_segmentation_wait_ms = 0.0
    app.lane_timing_geometry_ms = 0.0
    app.lane_timing_bridge_ms = 0.0
    app.lane_timing_max_ms = 0.0
    app.camera = SimpleNamespace(
        color_conversion_count=1,
        color_conversion_total_ms=0.5,
        last_color_conversion_ms=0.5,
    )

    app._record_lane_timing(
        total_ms=20.0,
        roi_ms=1.0,
        detect_ms=17.0,
        track_ms=2.0,
        camera_wait_ms=3.0,
        segmentation_wait_ms=8.0,
        geometry_ms=6.0,
        bridge_ms=4.0,
    )

    output = capsys.readouterr().out
    assert "camera_wait_ms=3.00" in output
    assert "segmentation_wait_ms=8.00" in output
    assert "geometry_ms=6.00" in output
    assert "bridge_ms=4.00" in output
    assert app.lane_timing_count == 0
    assert app.lane_timing_camera_wait_ms == 0.0
    assert app.lane_timing_segmentation_wait_ms == 0.0
    assert app.lane_timing_geometry_ms == 0.0
    assert app.lane_timing_bridge_ms == 0.0
