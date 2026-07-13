"""PPOCR integration for the upper-machine runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence
import sys
import time


Box = tuple[tuple[float, float], ...]
FrameBBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class OcrTextItem:
    text: str
    score: float
    box: Box


@dataclass(frozen=True)
class OcrResult:
    text: str = ""
    items: tuple[OcrTextItem, ...] = ()
    frame_id: int = 0
    error: str | None = None
    confidence: float = 0.0
    event_id: int = 0
    is_new: bool = False
    source_bbox: FrameBBox | None = None
    detection_confidence: float = 0.0
    inference_ms: float = 0.0
    locked: bool = False


class OcrEventLogger:
    """Append accepted OCR events to one UTF-8 JSONL file per run."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.file_path: Path | None = None

    def append(self, result: OcrResult) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.file_path is None:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            self.file_path = self.output_dir / f"ocr_events_{timestamp}.jsonl"
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event_id": result.event_id,
            "frame_id": result.frame_id,
            "text": result.text,
            "confidence": result.confidence,
            "detection_confidence": result.detection_confidence,
            "source_bbox": list(result.source_bbox) if result.source_bbox is not None else None,
            "inference_ms": result.inference_ms,
        }
        with self.file_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self.file_path


def merge_ocr_text(items: Iterable[OcrTextItem], min_score: float = 0.7) -> str:
    """Return high-confidence OCR text in top-to-bottom, left-to-right order."""

    accepted = [item for item in items if item.text.strip() and item.score >= min_score]
    accepted.sort(key=lambda item: (item.box[0][1], item.box[0][0]))
    return "".join(item.text.strip() for item in accepted)


def ocr_result_confidence(items: Iterable[OcrTextItem], min_score: float = 0.7) -> float:
    """Return character-count-weighted confidence for accepted OCR text."""

    accepted = [item for item in items if item.text.strip() and item.score >= min_score]
    total_chars = sum(len(item.text.strip()) for item in accepted)
    if total_chars == 0:
        return 0.0
    weighted_score = sum(item.score * len(item.text.strip()) for item in accepted)
    return float(weighted_score / total_chars)


class OcrRecognizer:
    """One-shot PPOCR runner controlled by the road-sign event session."""

    def __init__(self, config: dict[str, Any] | None = None, project_root: Path | None = None) -> None:
        self.config = config or {}
        self.project_root = project_root or Path.cwd()
        self.enabled = bool(self.config.get("enable", False))
        self.min_score = float(self.config.get("min_score", 0.7))
        self.input_size = (
            int(self.config.get("input_width", 480)),
            int(self.config.get("input_height", 480)),
        )
        self._system: Any | None = None
        self.last_result = OcrResult()

    def recognize(self, frame: Any, frame_id: int) -> OcrResult:
        """Run one PPOCR inference when the event session explicitly requests it."""

        if not self.enabled:
            return self.last_result

        started = time.perf_counter()
        try:
            system = self._load_system()
            boxes, recognition = system.run(self._prepare_frame(frame))
            items = tuple(
                OcrTextItem(
                    text=str(result[0][0]),
                    score=float(result[0][1]),
                    box=_to_box(box),
                )
                for box, result in zip(boxes or [], recognition or [])
            )
            result = OcrResult(
                text=merge_ocr_text(items, self.min_score),
                items=items,
                frame_id=frame_id,
                confidence=ocr_result_confidence(items, self.min_score),
                inference_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # OCR must not stop lane following.
            result = OcrResult(
                frame_id=frame_id,
                error=f"{type(exc).__name__}: {exc}",
                inference_ms=(time.perf_counter() - started) * 1000.0,
            )
        self.last_result = result
        return result

    def close(self) -> None:
        if self._system is None:
            return
        for name in ("text_detector", "text_recognizer"):
            component = getattr(self._system, name, None)
            model = getattr(component, "model", None)
            release = getattr(model, "release", None)
            if callable(release):
                release()
        self._system = None

    def _load_system(self) -> Any:
        if self._system is not None:
            return self._system

        system_python = self._resolve_path(
            self.config.get("system_python_dir", "third_party/ppocr/PPOCR-System/python")
        )
        if str(system_python) not in sys.path:
            sys.path.insert(0, str(system_python))
        from ppocr_system import TextSystem  # type: ignore[import-not-found]

        args = SimpleNamespace(
            det_model_path=str(self._resolve_path(self.config["det_model_path"])),
            rec_model_path=str(self._resolve_path(self.config["rec_model_path"])),
            target=str(self.config.get("target", "rk3588")),
            device_id=self.config.get("device_id"),
            core_mask=str(self.config.get("core_mask", "NPU_CORE_1")),
        )
        self._system = TextSystem(args)
        return self._system

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _prepare_frame(self, frame: Any) -> Any:
        shape = getattr(frame, "shape", ())
        if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
            return frame
        import cv2
        import numpy as np

        height, width = shape[:2]
        side = max(height, width)
        canvas_shape = (side, side) + tuple(shape[2:])
        canvas = np.full(canvas_shape, 114, dtype=frame.dtype)
        top = (side - height) // 2
        left = (side - width) // 2
        canvas[top : top + height, left : left + width] = frame
        if tuple(canvas.shape[:2]) == (self.input_size[1], self.input_size[0]):
            return canvas
        return cv2.resize(canvas, self.input_size, interpolation=cv2.INTER_LINEAR)


def _to_box(box: Sequence[Sequence[float]]) -> Box:
    return tuple((float(point[0]), float(point[1])) for point in box)
