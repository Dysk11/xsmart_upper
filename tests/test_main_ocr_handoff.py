from core.ocr import OcrResult
from main import OcrBBoxDisplayTimer, consume_ocr_event


def test_consume_ocr_event_marks_each_event_new_once() -> None:
    worker = OcrResult(text="右道", event_id=1, locked=True)
    first, seen = consume_ocr_event(OcrResult(), 0, worker)
    repeated, seen = consume_ocr_event(first, seen, worker)
    assert first.is_new
    assert not repeated.is_new
    assert seen == 1


def test_consume_ocr_event_ignores_stale_worker_event() -> None:
    previous = OcrResult(text="左道", event_id=2, locked=True)
    result, seen = consume_ocr_event(previous, 2, OcrResult(text="旧", event_id=1))
    assert result.text == "左道"
    assert not result.is_new
    assert seen == 2


def test_ocr_bbox_timer_expires_at_one_second_and_new_attempt_resets_it() -> None:
    timer = OcrBBoxDisplayTimer(hold_seconds=1.0)
    first = OcrResult(frame_id=10, source_bbox=(1, 2, 30, 40))
    second = OcrResult(frame_id=11, source_bbox=(5, 6, 35, 45))

    assert timer.observe(first, now=20.0)
    assert timer.is_visible(first, now=20.999)
    assert not timer.is_visible(first, now=21.0)
    assert not timer.observe(first, now=21.2)

    assert timer.observe(second, now=21.2)
    assert timer.is_visible(second, now=22.199)
    assert not timer.is_visible(second, now=22.2)


def test_ocr_bbox_timer_requires_a_valid_bbox() -> None:
    timer = OcrBBoxDisplayTimer(hold_seconds=1.0)
    result = OcrResult(frame_id=3)
    assert timer.observe(result, now=5.0)
    assert not timer.is_visible(result, now=5.1)
