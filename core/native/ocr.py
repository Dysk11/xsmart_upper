"""Event-gated asynchronous OCR state for the native RKNNRT backend."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, floor
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from core.object.blocking import DetectedObject
from core.ocr.recognizer import OcrEventLogger, OcrResult
from core.ocr.road_sign import OcrTrigger

from .runtime import NativePerceptionBackend


def select_road_sign_bbox(
    frame_shape: tuple[int, int],
    detections: Sequence[DetectedObject],
    class_names: set[str],
    min_confidence: float,
    min_width_px: int,
    min_height_px: int,
    padding_ratio: float,
) -> tuple[tuple[int, int, int, int], float] | None:
    height, width = frame_shape
    accepted = {name.casefold() for name in class_names}
    candidates = []
    for obj in detections:
        x1, y1, x2, y2 = obj.bbox_frame
        if (
            obj.class_name.casefold() not in accepted
            or obj.confidence < min_confidence
            or x2 - x1 < min_width_px
            or y2 - y1 < min_height_px
        ):
            continue
        candidates.append((float(obj.confidence), (x2 - x1) * (y2 - y1), obj))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for confidence, _area, obj in candidates:
        x1, y1, x2, y2 = obj.bbox_frame
        pad_x = (x2 - x1) * max(0.0, padding_ratio)
        pad_y = (y2 - y1) * max(0.0, padding_ratio)
        bbox = (
            max(0, min(width, int(floor(x1 - pad_x)))),
            max(0, min(height, int(floor(y1 - pad_y)))),
            max(0, min(width, int(ceil(x2 + pad_x)))),
            max(0, min(height, int(ceil(y2 + pad_y)))),
        )
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            return bbox, confidence
    return None


class NativeRoadSignOcrSession:
    """Submit OCR only for a road sign on a Python-confirmed fork frame."""

    def __init__(
        self,
        config: dict[str, Any],
        backend: NativePerceptionBackend,
        project_root: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        trigger_callback: Callable[[OcrTrigger], None] | None = None,
        event_logger: Any | None = None,
    ) -> None:
        self.enabled = bool(config.get("enable", False))
        self.backend = backend
        self.clock = clock
        self.trigger_callback = trigger_callback
        self.class_names = {str(name).casefold() for name in config.get("class_names", ["road_sign"])}
        self.min_confidence = float(config.get("bbox_min_confidence", 0.50))
        self.min_width = max(1, int(config.get("bbox_min_width_px", 96)))
        self.min_height = max(1, int(config.get("bbox_min_height_px", 48)))
        self.padding_ratio = max(0.0, float(config.get("bbox_padding_ratio", 0.10)))
        self.accept_score = float(config.get("accept_score", 0.60))
        self.retry_interval = max(0.0, float(config.get("retry_interval_sec", 0.50)))
        self.cooldown = max(0.0, float(config.get("cooldown_seconds", 20.0)))
        output_dir = Path(str(config.get("output_dir", "outputs/logs/ocr")))
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        self.event_logger = event_logger or OcrEventLogger(output_dir)
        self.last_attempt: OcrResult | None = None
        self.last_result: OcrResult | None = None
        self.last_error: str | None = None
        self._trigger_counter = 0
        self._active_trigger_id = 0
        self._event_counter = 0
        self._pending = False
        self._cycle_completed = False
        self._next_retry_at = 0.0
        self._cooldown_until = 0.0

    def update(
        self,
        frame_shape: tuple[int, int],
        frame_id: int,
        detections: Sequence[DetectedObject],
        *,
        allow_inference: bool,
    ) -> OcrResult | None:
        if not self.enabled:
            return None
        now = self.clock()
        completed = self.backend.poll_ocr()
        if completed is not None:
            self._pending = False
            previous_attempt = self.last_attempt
            candidate = replace(
                completed,
                detection_confidence=(
                    previous_attempt.detection_confidence
                    if previous_attempt is not None
                    and previous_attempt.trigger_id == completed.trigger_id
                    else completed.detection_confidence
                ),
            )
            self.last_attempt = candidate
            if candidate.error or not candidate.text or candidate.confidence < self.accept_score:
                self.last_error = candidate.error
                self._next_retry_at = now + self.retry_interval
            else:
                self._event_counter += 1
                accepted = replace(candidate, event_id=self._event_counter, locked=True)
                try:
                    self.event_logger.append(accepted)
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._next_retry_at = now + self.retry_interval
                    return self.last_result
                self.last_result = accepted
                self.last_attempt = accepted
                self.last_error = None
                self._cycle_completed = True
                self._cooldown_until = now + self.cooldown
                self._next_retry_at = self._cooldown_until

        # A completed in-flight job may still be collected after the fork disappears,
        # but never submit a new/retry job unless this frame still confirms the gate.
        if not allow_inference:
            if not self._pending:
                self._active_trigger_id = 0
                self._cycle_completed = False
            return self.last_result
        selection = select_road_sign_bbox(
            frame_shape,
            detections,
            self.class_names,
            self.min_confidence,
            self.min_width,
            self.min_height,
            self.padding_ratio,
        )
        if selection is None:
            if not self._pending:
                self._active_trigger_id = 0
                self._cycle_completed = False
            return self.last_result
        if self._pending or self._cycle_completed or now < self._next_retry_at or now < self._cooldown_until:
            return self.last_result

        bbox, detection_confidence = selection
        if self._active_trigger_id == 0:
            self._trigger_counter += 1
            self._active_trigger_id = self._trigger_counter
        if not self.backend.submit_ocr(self._active_trigger_id, frame_id, bbox):
            self._next_retry_at = now + self.retry_interval
            return self.last_result
        self._pending = True
        self.last_attempt = OcrResult(
            frame_id=frame_id,
            source_bbox=bbox,
            detection_confidence=detection_confidence,
            trigger_id=self._active_trigger_id,
        )
        if self.trigger_callback is not None:
            self.trigger_callback(
                OcrTrigger(
                    trigger_id=self._active_trigger_id,
                    frame_id=frame_id,
                    started_at=now,
                )
            )
        return self.last_result

    def close(self) -> None:
        return None
