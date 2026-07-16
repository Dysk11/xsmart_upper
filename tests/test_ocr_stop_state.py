from __future__ import annotations

import numpy as np

from core.object.blocking import DetectedObject
from core.ocr.recognizer import OcrResult
from core.ocr.road_sign import OcrStopLatch, OcrTrigger, RoadSignOcrSession


class FakeRecognizer:
    def __init__(self, results: list[OcrResult]) -> None:
        self.results = list(results)
        self.call_count = 0

    def recognize(self, _image: np.ndarray, _frame_id: int) -> OcrResult:
        self.call_count += 1
        return self.results.pop(0)

    def close(self) -> None:
        return None


class FakeLogger:
    def append(self, _result: OcrResult) -> None:
        return None


def make_detection() -> DetectedObject:
    return DetectedObject(
        class_name="road_sign",
        confidence=0.9,
        bbox_frame=(20, 20, 120, 100),
    )


def test_ocr_retries_share_trigger_and_rearm_after_candidate_disappears() -> None:
    now = [0.0]
    triggers: list[OcrTrigger] = []
    session = RoadSignOcrSession(
        {
            "enable": True,
            "class_names": ["road_sign"],
            "bbox_min_width_px": 1,
            "bbox_min_height_px": 1,
            "retry_interval_sec": 0.5,
            "cooldown_seconds": 0.0,
            "accept_score": 0.6,
        },
        recognizer=FakeRecognizer(
            [
                OcrResult(frame_id=1, error="retry"),
                OcrResult(frame_id=2, text="左转", confidence=0.9),
                OcrResult(frame_id=4, text="右转", confidence=0.9),
            ]
        ),
        event_logger=FakeLogger(),
        clock=lambda: now[0],
        trigger_callback=triggers.append,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    session.update(frame, 1, [make_detection()])
    now[0] = 0.5
    accepted = session.update(frame, 2, [make_detection()])

    assert [item.trigger_id for item in triggers] == [1]
    assert triggers[0].started_at == 0.0
    assert accepted is not None and accepted.trigger_id == 1

    session.update(frame, 3, [])
    now[0] = 0.6
    second = session.update(frame, 4, [make_detection()])
    assert [item.trigger_id for item in triggers] == [1, 2]
    assert triggers[1].started_at == 0.6
    assert second is not None and second.trigger_id == 2


def test_stop_latch_duplicate_trigger_does_not_reset_timeout() -> None:
    latch = OcrStopLatch(timeout_sec=20.0)

    assert latch.start(7, 100.0)
    assert not latch.start(7, 115.0)
    assert latch.expire_if_needed(119.99) is None
    assert latch.expire_if_needed(120.0) == 7
    assert not latch.active
    assert not latch.start(7, 121.0)
    assert latch.start(8, 121.0)


def test_road_sign_does_not_start_ocr_until_fork_allows_inference() -> None:
    triggers: list[OcrTrigger] = []
    recognizer = FakeRecognizer([OcrResult(frame_id=2, text="left", confidence=0.9)])
    session = RoadSignOcrSession(
        {
            "enable": True,
            "class_names": ["road_sign"],
            "bbox_min_width_px": 1,
            "bbox_min_height_px": 1,
            "cooldown_seconds": 20.0,
            "accept_score": 0.6,
        },
        recognizer=recognizer,
        event_logger=FakeLogger(),
        trigger_callback=triggers.append,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    assert session.update(frame, 1, [make_detection()], allow_inference=False) is None
    assert recognizer.call_count == 0
    assert triggers == []

    accepted = session.update(frame, 2, [make_detection()], allow_inference=True)
    assert recognizer.call_count == 1
    assert len(triggers) == 1
    assert accepted is not None and accepted.trigger_id == triggers[0].trigger_id


def test_confirmed_fork_without_road_sign_does_not_start_ocr() -> None:
    triggers: list[OcrTrigger] = []
    recognizer = FakeRecognizer([OcrResult(frame_id=1, text="left", confidence=0.9)])
    session = RoadSignOcrSession(
        {"enable": True, "class_names": ["road_sign"]},
        recognizer=recognizer,
        event_logger=FakeLogger(),
        trigger_callback=triggers.append,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    assert session.update(frame, 1, [], allow_inference=True) is None
    assert recognizer.call_count == 0
    assert triggers == []


def test_started_ocr_cycle_keeps_retrying_if_fork_detection_flickers() -> None:
    now = [0.0]
    triggers: list[OcrTrigger] = []
    recognizer = FakeRecognizer(
        [
            OcrResult(frame_id=1, error="retry"),
            OcrResult(frame_id=2, text="right", confidence=0.9),
        ]
    )
    session = RoadSignOcrSession(
        {
            "enable": True,
            "class_names": ["road_sign"],
            "bbox_min_width_px": 1,
            "bbox_min_height_px": 1,
            "retry_interval_sec": 0.5,
            "cooldown_seconds": 0.0,
            "accept_score": 0.6,
        },
        recognizer=recognizer,
        event_logger=FakeLogger(),
        clock=lambda: now[0],
        trigger_callback=triggers.append,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    session.update(frame, 1, [make_detection()], allow_inference=True)
    now[0] = 0.5
    accepted = session.update(frame, 2, [make_detection()], allow_inference=False)

    assert recognizer.call_count == 2
    assert len(triggers) == 1
    assert accepted is not None and accepted.trigger_id == triggers[0].trigger_id
