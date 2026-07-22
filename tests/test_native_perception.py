from __future__ import annotations

from pathlib import Path

import numpy as np

from core.native.ocr import NativeRoadSignOcrSession
from core.native.runtime import NativePerceptionBackend
from core.object.blocking import DetectedObject
from core.ocr.recognizer import OcrResult


class FakeEngine:
    def __init__(self, config):
        self.config = config
        self.opened = False
        self.submissions = []
        self.ocr_result = None

    def open(self):
        self.opened = True

    def next_frame(self, want_bgr=True):
        return {
            "ok": True,
            "frame_id": 7,
            "captured_at": 12.5,
            "width": 640,
            "height": 480,
            "frame_bgr": np.zeros((480, 640, 3), dtype=np.uint8) if want_bgr else None,
            "lane": {
                "mask": np.zeros((480, 640), dtype=np.uint8),
                "instances": [{"bbox": (1, 2, 3, 4), "confidence": 0.9}],
                "confidence": 0.9,
                "status": "ok",
                "timing": {"total_ms": 4.0},
            },
            "detections": [
                {"class_name": "road_sign", "confidence": 0.8, "bbox": (100, 100, 240, 180)}
            ],
            "object_timing": {"total_ms": 3.0},
            "total_ms": 6.0,
        }

    def submit_ocr(self, trigger_id, frame_id, bbox):
        self.submissions.append((trigger_id, frame_id, tuple(bbox)))
        return True

    def poll_ocr(self):
        result, self.ocr_result = self.ocr_result, None
        return result

    def close(self):
        self.opened = False


class FakeModule:
    def __init__(self):
        self.engine = None

    def NativePerception(self, config):
        self.engine = FakeEngine(config)
        return self.engine


class MemoryLogger:
    def __init__(self):
        self.results = []

    def append(self, result):
        self.results.append(result)


def runtime_config():
    return {
        "camera": {"mode": "shared_memory", "shared_memory_name": "shm_ar_video"},
        "native_perception": {"enable": True, "ring_buffer_size": 3},
        "rknn_lane_segmenter": {
            "model_path": "models/lane/lane.rknn",
            "input_size": [640, 480],
        },
        "rknn_object_detector": {
            "model_path": "models/object/object.rknn",
            "input_size": [640, 480],
            "class_names": ["road_sign"],
        },
        "ocr": {
            "enable": True,
            "det_model_path": "models/ocr/det.rknn",
            "rec_model_path": "models/ocr/rec.rknn",
            "character_dict_path": "models/ocr/keys.txt",
            "class_names": ["road_sign"],
            "bbox_min_confidence": 0.5,
            "bbox_min_width_px": 100,
            "bbox_min_height_px": 50,
            "accept_score": 0.6,
        },
    }


def test_native_backend_maps_packet_without_color_roundtrip(tmp_path: Path):
    module = FakeModule()
    backend = NativePerceptionBackend(runtime_config(), tmp_path, want_bgr=False, module=module)
    backend.open()

    success, frame = backend.read()

    assert success
    assert frame.shape == (480, 640, 3)
    assert backend.frame_id == 7
    assert backend.last_segmentation.status == "ok"
    assert backend.last_detections[0].class_name == "road_sign"
    assert module.engine.config["lane_model_path"] == str(tmp_path / "models/lane/lane.rknn")


def test_native_ocr_requires_road_sign_and_confirmed_fork(tmp_path: Path):
    module = FakeModule()
    backend = NativePerceptionBackend(runtime_config(), tmp_path, want_bgr=False, module=module)
    backend.open()
    backend.read()
    logger = MemoryLogger()
    triggers = []
    session = NativeRoadSignOcrSession(
        runtime_config()["ocr"],
        backend,
        tmp_path,
        trigger_callback=triggers.append,
        event_logger=logger,
    )
    detections = [DetectedObject("road_sign", 0.8, (100, 100, 240, 180))]

    session.update((480, 640), 7, detections, allow_inference=False)
    assert module.engine.submissions == []

    session.update((480, 640), 7, detections, allow_inference=True)
    assert len(module.engine.submissions) == 1
    assert triggers[0].frame_id == 7

    module.engine.ocr_result = {
        "trigger_id": 1,
        "frame_id": 7,
        "source_bbox": module.engine.submissions[0][2],
        "text": "左转",
        "confidence": 0.9,
        "inference_ms": 12.0,
        "error": None,
        "items": [],
    }
    result = session.update((480, 640), 8, detections, allow_inference=True)
    assert result is not None
    assert result.text == "左转"
    assert result.event_id == 1
    assert len(logger.results) == 1


def test_native_ocr_does_not_retry_after_confirmed_fork_disappears(tmp_path: Path):
    module = FakeModule()
    backend = NativePerceptionBackend(runtime_config(), tmp_path, want_bgr=False, module=module)
    backend.open()
    backend.read()
    session = NativeRoadSignOcrSession(
        runtime_config()["ocr"],
        backend,
        tmp_path,
        event_logger=MemoryLogger(),
    )
    detections = [DetectedObject("road_sign", 0.8, (100, 100, 240, 180))]

    session.update((480, 640), 7, detections, allow_inference=True)
    module.engine.ocr_result = {
        "trigger_id": 1,
        "frame_id": 7,
        "source_bbox": module.engine.submissions[0][2],
        "text": "",
        "confidence": 0.0,
        "inference_ms": 12.0,
        "error": None,
        "items": [],
    }
    session.update((480, 640), 8, detections, allow_inference=False)
    session.update((480, 640), 9, detections, allow_inference=False)

    assert len(module.engine.submissions) == 1
