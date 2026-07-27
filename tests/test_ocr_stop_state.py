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


def make_detection(
    bbox: tuple[int, int, int, int] = (20, 20, 120, 100),
    *,
    confidence: float = 0.9,
) -> DetectedObject:
    return DetectedObject(
        class_name="road_sign",
        confidence=confidence,
        bbox_frame=bbox,
    )


def make_session(
    results: list[OcrResult],
    *,
    now: list[float] | None = None,
    triggers: list[OcrTrigger] | None = None,
    **config: object,
) -> tuple[RoadSignOcrSession, FakeRecognizer]:
    recognizer = FakeRecognizer(results)
    session_config: dict[str, object] = {
        "enable": True,
        "class_names": ["road_sign"],
        "bbox_min_confidence": 0.5,
        "bbox_min_width_px": 1,
        "bbox_min_height_px": 1,
        "retry_interval_sec": 0.5,
        "cooldown_seconds": 0.0,
        "stop_timeout_sec": 20.0,
        "accept_score": 0.6,
    }
    session_config.update(config)
    session_kwargs: dict[str, object] = {
        "recognizer": recognizer,
        "event_logger": FakeLogger(),
        "trigger_callback": triggers.append if triggers is not None else None,
    }
    if now is not None:
        session_kwargs["clock"] = lambda: now[0]
    session = RoadSignOcrSession(session_config, **session_kwargs)
    return session, recognizer


def test_ocr_retries_share_trigger_and_rearm_after_candidate_disappears() -> None:
    now = [0.0]
    triggers: list[OcrTrigger] = []
    session, recognizer = make_session(
        [
            OcrResult(frame_id=1, error="retry"),
            OcrResult(frame_id=2, text="left", confidence=0.9),
            OcrResult(frame_id=4, text="right", confidence=0.9),
        ],
        now=now,
        triggers=triggers,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    session.update(frame, 1, [make_detection()])
    now[0] = 0.5
    accepted = session.update(frame, 2, [make_detection()])

    assert recognizer.call_count == 2
    assert [item.trigger_id for item in triggers] == [1]
    assert triggers[0].started_at == 0.0
    assert accepted is not None and accepted.trigger_id == 1

    session.update(frame, 3, [])
    now[0] = 0.6
    second = session.update(frame, 4, [make_detection()])

    assert recognizer.call_count == 3
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


def test_missing_or_ineligible_road_sign_does_not_start_ocr() -> None:
    triggers: list[OcrTrigger] = []
    session, recognizer = make_session(
        [OcrResult(frame_id=3, text="left", confidence=0.9)],
        triggers=triggers,
        bbox_min_confidence=0.8,
        bbox_min_width_px=100,
        bbox_min_height_px=50,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    session.update(frame, 1, [])
    session.update(
        frame,
        2,
        [make_detection((20, 20, 80, 60), confidence=0.9)],
    )
    session.update(
        frame,
        3,
        [make_detection((20, 20, 130, 90), confidence=0.7)],
    )

    assert recognizer.call_count == 0
    assert triggers == []


def test_horizontal_edge_waits_then_starts_ocr_immediately_on_full_entry() -> None:
    triggers: list[OcrTrigger] = []
    session, recognizer = make_session(
        [OcrResult(frame_id=3, text="left", confidence=0.9)],
        triggers=triggers,
        bbox_min_width_px=80,
        bbox_min_height_px=50,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    session.update(frame, 1, [make_detection((0, 20, 100, 100))])
    session.update(frame, 2, [make_detection((60, 20, 159, 100))])

    assert recognizer.call_count == 0
    assert triggers == []

    accepted = session.update(
        frame,
        3,
        [make_detection((1, 20, 158, 100))],
    )

    assert recognizer.call_count == 1
    assert len(triggers) == 1
    assert triggers[0].frame_id == 3
    assert accepted is not None
    assert accepted.trigger_id == triggers[0].trigger_id


def test_top_and_bottom_edges_do_not_block_ocr() -> None:
    triggers: list[OcrTrigger] = []
    session, recognizer = make_session(
        [OcrResult(frame_id=1, text="right", confidence=0.9)],
        triggers=triggers,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    accepted = session.update(
        frame,
        1,
        [make_detection((1, 0, 158, 139))],
    )

    assert recognizer.call_count == 1
    assert len(triggers) == 1
    assert accepted is not None


def test_edge_frame_pauses_retry_without_creating_another_trigger() -> None:
    now = [0.0]
    triggers: list[OcrTrigger] = []
    session, recognizer = make_session(
        [
            OcrResult(frame_id=1, error="retry"),
            OcrResult(frame_id=3, text="right", confidence=0.9),
        ],
        now=now,
        triggers=triggers,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    session.update(frame, 1, [make_detection((1, 20, 158, 100))])
    now[0] = 0.5
    session.update(frame, 2, [make_detection((0, 20, 100, 100))])

    assert recognizer.call_count == 1
    assert len(triggers) == 1

    accepted = session.update(
        frame,
        3,
        [make_detection((1, 20, 158, 100))],
    )

    assert recognizer.call_count == 2
    assert len(triggers) == 1
    assert accepted is not None
    assert accepted.trigger_id == triggers[0].trigger_id


def test_edge_highest_priority_candidate_discards_entire_frame() -> None:
    triggers: list[OcrTrigger] = []
    session, recognizer = make_session(
        [OcrResult(frame_id=2, text="left", confidence=0.9)],
        triggers=triggers,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)
    edge_candidate = make_detection(
        (0, 20, 100, 100),
        confidence=0.9,
    )
    inside_candidate = make_detection(
        (20, 20, 120, 100),
        confidence=0.8,
    )

    session.update(frame, 1, [edge_candidate, inside_candidate])

    assert recognizer.call_count == 0
    assert triggers == []

    accepted = session.update(frame, 2, [inside_candidate])

    assert recognizer.call_count == 1
    assert len(triggers) == 1
    assert accepted is not None


def test_timeout_suppresses_same_visible_sign_until_it_disappears() -> None:
    now = [0.0]
    triggers: list[OcrTrigger] = []
    session, recognizer = make_session(
        [
            OcrResult(frame_id=1, error="retry"),
            OcrResult(frame_id=4, text="left", confidence=0.9),
        ],
        now=now,
        triggers=triggers,
        retry_interval_sec=100.0,
    )
    frame = np.zeros((140, 160, 3), dtype=np.uint8)

    session.update(frame, 1, [make_detection()])
    now[0] = 20.0
    session.update(frame, 2, [make_detection()])
    now[0] = 20.1
    session.update(frame, 3, [])
    accepted = session.update(frame, 4, [make_detection()])

    assert recognizer.call_count == 2
    assert [item.trigger_id for item in triggers] == [1, 2]
    assert accepted is not None
    assert accepted.trigger_id == 2
