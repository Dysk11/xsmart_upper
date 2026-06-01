"""Fork sign state and route direction selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from core.blocking_analyzer import DetectedObject


@dataclass(frozen=True)
class ForkRouteResult:
    """Held fork direction derived from Left/Right sign detections."""

    active: bool
    requested_direction: str | None
    sign_object: DetectedObject | None
    confidence: float
    hold_frames_left: int
    seen_fork: bool
    no_fork_frames: int
    reason: str


class ForkRoutePlanner:
    """Keeps the latest fork sign direction until the selected fork has cleared."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.left_class_name = str(config.get("left_class_name", "Left")).casefold()
        self.right_class_name = str(config.get("right_class_name", "Right")).casefold()
        self.min_confidence = float(config.get("min_confidence", 0.45))
        self.sign_hold_frames = int(config.get("sign_hold_frames", 90))
        self.clear_after_no_fork_frames = int(config.get("clear_after_no_fork_frames", 8))
        self.invert_direction = bool(config.get("invert_direction", False))

        self._direction: str | None = None
        self._sign_object: DetectedObject | None = None
        self._confidence = 0.0
        self._hold_frames_left = 0
        self._seen_fork = False
        self._no_fork_frames = 0

    def update(
        self,
        objects: Sequence[DetectedObject],
        fork_detected: bool | None = None,
    ) -> ForkRouteResult:
        if not self.enabled:
            return self._result("disabled")

        sign = None if fork_detected is not None else self._select_sign(objects)
        if sign is not None:
            direction = self._direction_for_class(sign.class_name)
            if self.invert_direction and direction is not None:
                direction = "right" if direction == "left" else "left"
            self._direction = direction
            self._sign_object = sign
            self._confidence = float(sign.confidence)
            self._hold_frames_left = self.sign_hold_frames
            self._seen_fork = False
            self._no_fork_frames = 0
            return self._result(f"sign {sign.class_name} conf={sign.confidence:.2f}")

        if self._direction is None:
            return self._result("no sign")

        if fork_detected is True:
            self._seen_fork = True
            self._no_fork_frames = 0
            self._hold_frames_left = self.sign_hold_frames
            return self._result("holding through fork")

        if fork_detected is False and self._seen_fork:
            self._no_fork_frames += 1
            if self._no_fork_frames >= self.clear_after_no_fork_frames:
                self.clear()
                return self._result("fork cleared")
            return self._result(f"waiting clear {self._no_fork_frames}/{self.clear_after_no_fork_frames}")

        if fork_detected is False:
            return self._result("waiting for fork")

        if self._hold_frames_left > 0:
            self._hold_frames_left -= 1
            return self._result(f"holding sign {self._hold_frames_left} frames")

        self.clear()
        return self._result("hold expired")

    def clear(self) -> None:
        self._direction = None
        self._sign_object = None
        self._confidence = 0.0
        self._hold_frames_left = 0
        self._seen_fork = False
        self._no_fork_frames = 0

    def _select_sign(self, objects: Sequence[DetectedObject]) -> DetectedObject | None:
        best: tuple[float, DetectedObject] | None = None
        for obj in objects:
            if self._direction_for_class(obj.class_name) is None:
                continue
            if obj.confidence < self.min_confidence:
                continue
            _, _, x2, y2 = obj.bbox_frame
            x1, y1, _, _ = obj.bbox_frame
            area = max(0, x2 - x1) * max(0, y2 - y1)
            score = float(obj.confidence) * 10.0 + float(y2) * 0.01 + float(area) * 0.0001
            if best is None or score > best[0]:
                best = (score, obj)
        return best[1] if best is not None else None

    def _direction_for_class(self, class_name: str) -> str | None:
        name = class_name.casefold()
        if name == self.left_class_name:
            return "left"
        if name == self.right_class_name:
            return "right"
        return None

    def _result(self, reason: str) -> ForkRouteResult:
        return ForkRouteResult(
            active=self._direction is not None,
            requested_direction=self._direction,
            sign_object=self._sign_object,
            confidence=self._confidence,
            hold_frames_left=self._hold_frames_left,
            seen_fork=self._seen_fork,
            no_fork_frames=self._no_fork_frames,
            reason=reason,
        )
