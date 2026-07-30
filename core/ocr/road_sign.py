"""YOLO-driven road-sign cropping and RKNN OCR event handling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, floor
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from core.object.blocking import DetectedObject
from core.ocr.recognizer import OcrEventLogger, OcrRecognizer, OcrResult, FrameBBox


@dataclass(frozen=True)
class RoadSignCrop:
    """A road-sign image cropped from the exact object-detection frame."""

    image: np.ndarray
    bbox: FrameBBox
    detection_confidence: float
    detection_area_px: float


@dataclass(frozen=True)
class RoadSignCandidate:
    """The strongest road-sign detection that meets OCR size/score gates."""

    bbox: FrameBBox
    detection_confidence: float
    detection_area_px: float


@dataclass(frozen=True)
class OcrTrigger:
    """Notification emitted when a qualifying road sign starts a stop cycle."""

    trigger_id: int
    frame_id: int
    started_at: float


@dataclass(frozen=True)
class OcrCycleCompletion:
    """Reliable notification that an OCR stop cycle has no usable result."""

    trigger_id: int
    frame_id: int
    completed_at: float
    reason: str


class OcrStopLatch:
    """Latch vehicle stop from OCR start until API completion or total timeout."""

    def __init__(self, timeout_sec: float) -> None:
        self.timeout_sec = float(timeout_sec)
        if self.timeout_sec <= 0:
            raise ValueError("ocr.stop_timeout_sec must be greater than zero")
        self.active_trigger_id = 0
        self.started_at = 0.0
        self.closed_trigger_ids: set[int] = set()

    @property
    def active(self) -> bool:
        return self.active_trigger_id > 0

    def start(self, trigger_id: int, now: float) -> bool:
        trigger_id = int(trigger_id)
        if trigger_id <= 0 or trigger_id in self.closed_trigger_ids:
            return False
        if trigger_id == self.active_trigger_id:
            return False
        if self.active_trigger_id > 0:
            self.closed_trigger_ids.add(self.active_trigger_id)
        self.active_trigger_id = trigger_id
        self.started_at = float(now)
        return True

    def owns(self, trigger_id: int) -> bool:
        return self.active and int(trigger_id) == self.active_trigger_id

    def complete(self, trigger_id: int) -> bool:
        if not self.owns(trigger_id):
            return False
        self.closed_trigger_ids.add(self.active_trigger_id)
        self.active_trigger_id = 0
        self.started_at = 0.0
        return True

    def expire_if_needed(self, now: float) -> int | None:
        if not self.active or float(now) - self.started_at < self.timeout_sec:
            return None
        expired_trigger_id = self.active_trigger_id
        self.complete(expired_trigger_id)
        return expired_trigger_id


def select_road_sign_candidate(
    detections: Sequence[DetectedObject],
    class_names: set[str],
    min_confidence: float,
    min_width_px: int,
    min_height_px: int,
) -> RoadSignCandidate | None:
    """Choose the strongest road-sign box that meets OCR score and size gates."""

    accepted_names = {name.casefold() for name in class_names}
    candidates: list[tuple[float, float, DetectedObject]] = []
    for obj in detections:
        if obj.class_name.casefold() not in accepted_names:
            continue
        confidence = float(obj.confidence)
        if confidence < min_confidence:
            continue
        x1, y1, x2, y2 = obj.bbox_frame
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < min_width_px or box_height < min_height_px:
            continue
        candidates.append((confidence, float(box_width * box_height), obj))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not candidates:
        return None
    confidence, area, obj = candidates[0]
    return RoadSignCandidate(
        bbox=obj.bbox_frame,
        detection_confidence=confidence,
        detection_area_px=area,
    )


def crop_road_sign_candidate(
    frame: np.ndarray,
    candidate: RoadSignCandidate,
    padding_ratio: float,
) -> RoadSignCrop | None:
    """Crop one selected road sign while preserving its original detection data."""

    shape = getattr(frame, "shape", ())
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        return None

    frame_height, frame_width = int(shape[0]), int(shape[1])
    padding_ratio = max(0.0, float(padding_ratio))
    x1, y1, x2, y2 = candidate.bbox
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    left = max(0, min(frame_width, int(floor(x1 - pad_x))))
    top = max(0, min(frame_height, int(floor(y1 - pad_y))))
    right = max(0, min(frame_width, int(ceil(x2 + pad_x))))
    bottom = max(0, min(frame_height, int(ceil(y2 + pad_y))))
    if right <= left or bottom <= top:
        return None
    image = frame[top:bottom, left:right]
    if not image.size:
        return None
    return RoadSignCrop(
        image.copy(),
        (left, top, right, bottom),
        candidate.detection_confidence,
        candidate.detection_area_px,
    )


def select_road_sign_crop(
    frame: np.ndarray,
    detections: Sequence[DetectedObject],
    class_names: set[str],
    min_confidence: float,
    min_width_px: int,
    min_height_px: int,
    padding_ratio: float,
) -> RoadSignCrop | None:
    """Choose and safely crop the strongest sufficiently large road sign."""

    candidate = select_road_sign_candidate(
        detections,
        class_names,
        min_confidence,
        min_width_px,
        min_height_px,
    )
    if candidate is None:
        return None
    return crop_road_sign_candidate(frame, candidate, padding_ratio)


class RoadSignOcrSession:
    """Run OCR on eligible road signs and enforce a global success cooldown."""

    def __init__(
        self,
        config: dict[str, Any],
        project_root: Path | None = None,
        recognizer: Any | None = None,
        event_logger: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        trigger_callback: Callable[[OcrTrigger], None] | None = None,
        completion_callback: Callable[[OcrCycleCompletion], None] | None = None,
    ) -> None:
        self.config = config
        self.enabled = bool(config.get("enable", False))
        self.class_names = {str(name).casefold() for name in config.get("class_names", ["road_sign"])}
        self.bbox_min_confidence = float(config.get("bbox_min_confidence", 0.50))
        self.bbox_min_width_px = max(1, int(config.get("bbox_min_width_px", 96)))
        self.bbox_min_height_px = max(1, int(config.get("bbox_min_height_px", 48)))
        self.bbox_padding_ratio = max(0.0, float(config.get("bbox_padding_ratio", 0.10)))
        self.accept_score = float(config.get("accept_score", 0.60))
        self.retry_interval_sec = float(config.get("retry_interval_sec", 0.50))
        self.cooldown_seconds = float(config.get("cooldown_seconds", 20.0))
        self.stop_timeout_sec = float(config.get("stop_timeout_sec", 20.0))
        if self.retry_interval_sec < 0:
            raise ValueError("ocr.retry_interval_sec must not be negative")
        if self.cooldown_seconds < 0:
            raise ValueError("ocr.cooldown_seconds must not be negative")
        if self.stop_timeout_sec <= 0:
            raise ValueError("ocr.stop_timeout_sec must be greater than zero")
        self.clock = clock
        self.trigger_callback = trigger_callback
        self.completion_callback = completion_callback

        root = project_root or Path.cwd()
        output_dir = Path(str(config.get("output_dir", "outputs/logs")))
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        self.recognizer = recognizer or OcrRecognizer(config, project_root=root)
        self.event_logger = event_logger or OcrEventLogger(output_dir)

        self._event_counter = 0
        self._next_retry_at = 0.0
        self._cooldown_until = 0.0
        self._pending_log_result: OcrResult | None = None
        self._last_result: OcrResult | None = None
        self.last_attempt: OcrResult | None = None
        self.last_error: str | None = None
        self._trigger_counter = 0
        self._active_trigger_id = 0
        self._trigger_started_at = 0.0
        self._cycle_completed = False

    def update(
        self,
        frame: np.ndarray,
        frame_id: int,
        detections: Sequence[DetectedObject],
    ) -> OcrResult | None:
        """Advance OCR state using detections and pixels from the same frame."""

        if not self.enabled:
            return None
        now = self.clock()

        if (
            self._active_trigger_id > 0
            and not self._cycle_completed
            and now - self._trigger_started_at >= self.stop_timeout_sec
        ):
            self._cycle_completed = True

        if self._pending_log_result is not None:
            if now >= self._next_retry_at:
                self._publish_pending(now)
            return self._last_result

        candidate = select_road_sign_candidate(
            detections,
            self.class_names,
            self.bbox_min_confidence,
            self.bbox_min_width_px,
            self.bbox_min_height_px,
        )
        if candidate is None:
            if self._cycle_completed:
                self._reset_cycle()
            return self._last_result
        if now < self._cooldown_until or self._cycle_completed:
            return self._last_result

        shape = getattr(frame, "shape", ())
        frame_width = int(shape[1]) if len(shape) >= 2 else 0
        x1, _, x2, _ = candidate.bbox
        if frame_width <= 0 or x1 <= 0 or x2 >= frame_width - 1:
            return self._last_result

        crop = crop_road_sign_candidate(
            frame,
            candidate,
            self.bbox_padding_ratio,
        )
        if crop is None:
            return self._last_result

        if self._active_trigger_id == 0:
            self._trigger_counter += 1
            self._active_trigger_id = self._trigger_counter
            self._trigger_started_at = now
            if self.trigger_callback is not None:
                self.trigger_callback(
                    OcrTrigger(
                        trigger_id=self._active_trigger_id,
                        frame_id=frame_id,
                        started_at=now,
                    )
                )

        if now < self._next_retry_at:
            return self._last_result

        raw_result = self.recognizer.recognize(crop.image, frame_id)
        candidate = replace(
            raw_result,
            source_bbox=crop.bbox,
            detection_confidence=crop.detection_confidence,
            event_id=0,
            is_new=False,
            locked=False,
            trigger_id=self._active_trigger_id,
        )
        self.last_attempt = candidate
        if candidate.error or not candidate.text or candidate.confidence < self.accept_score:
            self.last_error = candidate.error
            if candidate.error:
                reason = "error"
            elif not candidate.text:
                reason = "empty"
            else:
                reason = "low_confidence"
            self._cycle_completed = True
            self._cooldown_until = now + self.cooldown_seconds
            self._next_retry_at = self._cooldown_until
            if self.completion_callback is not None:
                self.completion_callback(
                    OcrCycleCompletion(
                        trigger_id=self._active_trigger_id,
                        frame_id=frame_id,
                        completed_at=now,
                        reason=reason,
                    )
                )
            return self._last_result

        self._event_counter += 1
        self._pending_log_result = replace(
            candidate,
            event_id=self._event_counter,
            locked=True,
        )
        self._cycle_completed = True
        self._publish_pending(now)
        return self._last_result

    def would_run_recognizer(
        self,
        frame: np.ndarray,
        detections: Sequence[DetectedObject],
    ) -> bool:
        """Return whether update() would enter the NPU OCR recognizer now."""

        if (
            not self.enabled
            or self._pending_log_result is not None
            or self._cycle_completed
        ):
            return False
        now = self.clock()
        if now < self._cooldown_until or now < self._next_retry_at:
            return False
        candidate = select_road_sign_candidate(
            detections,
            self.class_names,
            self.bbox_min_confidence,
            self.bbox_min_width_px,
            self.bbox_min_height_px,
        )
        if candidate is None:
            return False
        shape = getattr(frame, "shape", ())
        frame_width = int(shape[1]) if len(shape) >= 2 else 0
        x1, _, x2, _ = candidate.bbox
        if frame_width <= 0 or x1 <= 0 or x2 >= frame_width - 1:
            return False
        return (
            crop_road_sign_candidate(
                frame,
                candidate,
                self.bbox_padding_ratio,
            )
            is not None
        )

    def _reset_cycle(self) -> None:
        self._active_trigger_id = 0
        self._trigger_started_at = 0.0
        self._cycle_completed = False
        self._next_retry_at = 0.0

    def _publish_pending(self, now: float) -> None:
        assert self._pending_log_result is not None
        try:
            self.event_logger.append(self._pending_log_result)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._next_retry_at = now + self.retry_interval_sec
            return
        self._last_result = self._pending_log_result
        self.last_attempt = self._last_result
        self._pending_log_result = None
        self._cooldown_until = now + self.cooldown_seconds
        self._next_retry_at = self._cooldown_until
        self.last_error = None

    def close(self) -> None:
        self.recognizer.close()
