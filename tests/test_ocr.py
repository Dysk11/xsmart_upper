from __future__ import annotations

import json

import numpy as np
import pytest

from core.ocr import OcrEventLogger, OcrRecognizer, OcrResult, OcrTextItem, merge_ocr_text, ocr_result_confidence


def test_merge_and_confidence_filter_low_score_items() -> None:
    items = (
        OcrTextItem("右道", 0.90, ((0, 0), (1, 0), (1, 1), (0, 1))),
        OcrTextItem("忽略", 0.20, ((0, 10), (1, 10), (1, 11), (0, 11))),
        OcrTextItem("直行", 0.80, ((20, 0), (21, 0), (21, 1), (20, 1))),
    )
    assert merge_ocr_text(items, 0.5) == "右道直行"
    assert ocr_result_confidence(items, 0.5) == pytest.approx(0.85)


def test_prepare_frame_preserves_aspect_ratio_on_square_canvas() -> None:
    recognizer = OcrRecognizer({"enable": False, "input_width": 100, "input_height": 100})
    image = np.full((20, 40, 3), 255, dtype=np.uint8)
    prepared = recognizer._prepare_frame(image)
    assert prepared.shape == (100, 100, 3)
    assert np.all(prepared[0, 50] == 114)
    assert np.all(prepared[50, 50] == 255)
    assert np.all(prepared[-1, 50] == 114)


def test_event_logger_writes_utf8_jsonl(tmp_path) -> None:
    logger = OcrEventLogger(tmp_path)
    result = OcrResult(
        text="右道直行",
        confidence=0.91,
        detection_confidence=0.88,
        frame_id=12,
        event_id=1,
        source_bbox=(1, 2, 101, 52),
        inference_ms=8.5,
        locked=True,
    )
    path = logger.append(result)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["text"] == "右道直行"
    assert record["source_bbox"] == [1, 2, 101, 52]
    assert record["confidence"] == 0.91
