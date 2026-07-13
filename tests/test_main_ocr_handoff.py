from core.ocr import OcrResult
from main import consume_ocr_event


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
