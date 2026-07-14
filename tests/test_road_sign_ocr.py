from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.blocking_analyzer import DetectedObject
from core.ocr import OcrResult
from core.road_sign_ocr import RoadSignOcrSession, select_road_sign_crop


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
SIGN = [DetectedObject("road_sign", 0.90, (100, 100, 200, 150))]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeRecognizer:
    def __init__(self, results: list[OcrResult]) -> None:
        self.results = list(results)
        self.calls = 0
        self.images: list[np.ndarray] = []
        self.closed = False

    def recognize(self, image: np.ndarray, frame_id: int) -> OcrResult:
        self.calls += 1
        self.images.append(image.copy())
        return replace(self.results.pop(0), frame_id=frame_id)

    def close(self) -> None:
        self.closed = True


class FakeLogger:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.results: list[OcrResult] = []

    def append(self, result: OcrResult) -> None:
        if self.failures:
            self.failures -= 1
            raise OSError("disk unavailable")
        self.results.append(result)


def config() -> dict[str, object]:
    return {
        "enable": True,
        "class_names": ["road_sign"],
        "bbox_min_confidence": 0.5,
        "bbox_min_width_px": 96,
        "bbox_min_height_px": 48,
        "bbox_padding_ratio": 0.1,
        "accept_score": 0.8,
        "retry_interval_sec": 0.5,
        "cooldown_seconds": 20.0,
    }


def test_crop_requires_class_confidence_and_96_by_48_size() -> None:
    detections = [
        DetectedObject("car", 0.99, (0, 0, 300, 300)),
        DetectedObject("road_sign", 0.90, (0, 0, 95, 100)),
        DetectedObject("road_sign", 0.89, (100, 100, 196, 148)),
    ]
    crop = select_road_sign_crop(FRAME, detections, {"road_sign"}, 0.5, 96, 48, 0.1)
    assert crop is not None
    assert crop.bbox == (90, 95, 206, 153)
    assert crop.image.shape == (58, 116, 3)


def test_crop_prefers_confidence_then_area_and_clips_boundaries() -> None:
    detections = [
        DetectedObject("road_sign", 0.80, (100, 100, 300, 200)),
        DetectedObject("road_sign", 0.90, (-10, -5, 100, 50)),
    ]
    crop = select_road_sign_crop(FRAME, detections, {"road_sign"}, 0.5, 96, 48, 0.1)
    assert crop is not None
    assert crop.bbox == (0, 0, 111, 56)


def test_high_confidence_logs_once_then_cools_down_for_20_seconds() -> None:
    clock = FakeClock()
    recognizer = FakeRecognizer([
        OcrResult(text="右道", confidence=0.90),
        OcrResult(text="左道", confidence=0.92),
    ])
    logger = FakeLogger()
    session = RoadSignOcrSession(config(), recognizer=recognizer, event_logger=logger, clock=clock)

    first = session.update(FRAME, 1, SIGN)
    assert first is not None and first.text == "右道"
    assert first.event_id == 1
    assert len(logger.results) == 1

    clock.now = 19.999
    assert session.update(FRAME, 2, SIGN) == first
    assert recognizer.calls == 1
    clock.now = 20.0
    second = session.update(FRAME, 3, SIGN)
    assert second is not None and second.text == "左道"
    assert second.event_id == 2
    assert recognizer.calls == 2


def test_custom_ocr_cooldown_is_loaded_from_config() -> None:
    clock = FakeClock()
    recognizer = FakeRecognizer([
        OcrResult(text="右道", confidence=0.90),
        OcrResult(text="左道", confidence=0.92),
    ])
    custom = config()
    custom["cooldown_seconds"] = 3.0
    session = RoadSignOcrSession(custom, recognizer=recognizer, event_logger=FakeLogger(), clock=clock)

    assert session.update(FRAME, 1, SIGN) is not None
    clock.now = 2.999
    session.update(FRAME, 2, SIGN)
    assert recognizer.calls == 1
    clock.now = 3.0
    session.update(FRAME, 3, SIGN)
    assert recognizer.calls == 2


def test_negative_ocr_cooldown_is_rejected() -> None:
    invalid = config()
    invalid["cooldown_seconds"] = -1
    with pytest.raises(ValueError, match="cooldown_seconds"):
        RoadSignOcrSession(invalid, recognizer=FakeRecognizer([]), event_logger=FakeLogger())


def test_low_confidence_retries_without_logging_or_cooldown() -> None:
    clock = FakeClock()
    recognizer = FakeRecognizer([
        OcrResult(text="模糊", confidence=0.79),
        OcrResult(text="清晰", confidence=0.88),
    ])
    logger = FakeLogger()
    session = RoadSignOcrSession(config(), recognizer=recognizer, event_logger=logger, clock=clock)
    assert session.update(FRAME, 1, SIGN) is None
    assert session.last_attempt is not None
    assert session.last_attempt.text == "模糊"
    assert session.last_attempt.confidence == 0.79
    clock.now = 0.49
    assert session.update(FRAME, 2, SIGN) is None
    assert recognizer.calls == 1
    clock.now = 0.5
    result = session.update(FRAME, 3, SIGN)
    assert result is not None and result.text == "清晰"
    assert len(logger.results) == 1


def test_log_failure_retries_write_without_repeating_ocr() -> None:
    clock = FakeClock()
    recognizer = FakeRecognizer([OcrResult(text="右道", confidence=0.90)])
    logger = FakeLogger(failures=1)
    session = RoadSignOcrSession(config(), recognizer=recognizer, event_logger=logger, clock=clock)
    assert session.update(FRAME, 1, SIGN) is None
    assert recognizer.calls == 1
    clock.now = 0.5
    result = session.update(FRAME, 2, [])
    assert result is not None and result.text == "右道"
    assert recognizer.calls == 1
    assert len(logger.results) == 1


def test_session_close_releases_recognizer() -> None:
    recognizer = FakeRecognizer([])
    session = RoadSignOcrSession(config(), recognizer=recognizer, event_logger=FakeLogger())
    session.close()
    assert recognizer.closed
