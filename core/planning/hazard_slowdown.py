"""One-gear slowdown policy for detected road hazards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from core.io.protocol import validate_moving_speed_state
from core.object.blocking import DetectedObject


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class HazardPresence:
    """Hazard classes found in one accepted object-detection result."""

    human: bool = False
    car: bool = False
    road_sign: bool = False

    @property
    def any(self) -> bool:
        return self.human or self.car or self.road_sign

    @property
    def generic_slowdown(self) -> bool:
        """Return hazards that still use the one-gear hold controller."""

        return self.human


def detect_hazard_presence(
    objects: Sequence[DetectedObject],
    avoidance_roi_rect: tuple[int, int, int, int],
    pedestrian_min_box_area_px: float,
) -> HazardPresence:
    """Classify hazards using the same ROI contracts as the safety planners."""

    avoid_x1, avoid_y1, avoid_x2, avoid_y2 = (
        float(value) for value in avoidance_roi_rect
    )
    min_human_area = max(0.0, float(pedestrian_min_box_area_px))
    human = False
    car = False
    road_sign = False

    for obj in objects:
        class_name = obj.class_name.casefold()
        x1, y1, x2, y2 = (float(value) for value in obj.bbox_frame)
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        if class_name == "road_sign":
            road_sign = True
        elif class_name == "car":
            # Edge contact counts, matching CarAvoidancePlanner.
            car = not (
                x2 < avoid_x1
                or x1 > avoid_x2
                or y2 < avoid_y1
                or y1 > avoid_y2
            ) or car
        elif class_name == "human":
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            area = (x2 - x1) * (y2 - y1)
            human = (
                area >= min_human_area
                and avoid_x1 <= center_x <= avoid_x2
                and avoid_y1 <= center_y <= avoid_y2
            ) or human

    return HazardPresence(human=human, car=car, road_sign=road_sign)


class HazardSlowdownController:
    """Hold a one-gear slowdown after fresh detections and stateful hazards."""

    def __init__(self, hold_sec: float = 1.0) -> None:
        self.hold_sec = float(hold_sec)
        if not math.isfinite(self.hold_sec) or self.hold_sec < 0.0:
            raise ValueError(
                "hazard_slowdown.hold_sec must be finite and non-negative"
            )
        self._hold_until = 0.0
        self._last_detection_result_id: int | None = None
        self._stateful_hazard_active = False
        self.last_presence = HazardPresence()

    def observe_ai_result(
        self,
        objects: Sequence[DetectedObject],
        avoidance_roi_rect: tuple[int, int, int, int],
        pedestrian_min_box_area_px: float,
        detection_result_id: int,
        now_monotonic: float,
    ) -> HazardPresence:
        """Refresh detection-driven slowdown exactly once per AI result."""

        result_id = int(detection_result_id)
        if result_id == self._last_detection_result_id:
            return self.last_presence
        self._last_detection_result_id = result_id
        previous_presence = self.last_presence
        self.last_presence = detect_hazard_presence(
            objects=objects,
            avoidance_roi_rect=avoidance_roi_rect,
            pedestrian_min_box_area_px=pedestrian_min_box_area_px,
        )
        if (
            self.last_presence.generic_slowdown
            or previous_presence.generic_slowdown
        ):
            self._extend(float(now_monotonic))
        return self.last_presence

    def observe_stateful_hazards(
        self,
        now_monotonic: float,
        *,
        pedestrian_active: bool = False,
    ) -> None:
        """Keep slowdown held throughout latched safety/planning states."""

        active = pedestrian_active
        if active or self._stateful_hazard_active:
            self._extend(float(now_monotonic))
        self._stateful_hazard_active = active

    def active(self, now_monotonic: float) -> bool:
        return float(now_monotonic) < self._hold_until

    def _extend(self, now_monotonic: float) -> None:
        self._hold_until = max(self._hold_until, now_monotonic + self.hold_sec)


class RoadSignApproachController:
    """Apply an independent moving gear until one sign starts its OCR cycle."""

    def __init__(self, speed_state: int, hold_sec: float = 1.0) -> None:
        self.speed_state = validate_moving_speed_state(
            speed_state,
            "ocr.approach_speed_state",
        )
        self.hold_sec = float(hold_sec)
        if not math.isfinite(self.hold_sec) or self.hold_sec < 0.0:
            raise ValueError(
                "hazard_slowdown.hold_sec must be finite and non-negative"
            )
        self._hold_until = 0.0
        self._last_detection_result_id: int | None = None
        self._last_road_sign_present = False
        self._blocked_until_absent = False

    def observe(
        self,
        road_sign_present: bool,
        detection_result_id: int,
        now_monotonic: float,
    ) -> None:
        """Advance approach/rearm state exactly once per object result."""

        result_id = int(detection_result_id)
        if result_id == self._last_detection_result_id:
            return
        self._last_detection_result_id = result_id
        present = bool(road_sign_present)
        now = float(now_monotonic)

        if self._blocked_until_absent:
            self._hold_until = 0.0
            if not present:
                self._blocked_until_absent = False
            self._last_road_sign_present = present
            return

        if present or self._last_road_sign_present:
            self._hold_until = max(self._hold_until, now + self.hold_sec)
        self._last_road_sign_present = present

    def mark_ocr_started(self) -> None:
        """Stop approach speed until a fresh result confirms sign absence."""

        self._blocked_until_absent = True
        self._hold_until = 0.0

    def active(self, now_monotonic: float) -> bool:
        return (
            not self._blocked_until_absent
            and float(now_monotonic) < self._hold_until
        )

    def merge_speed_state(
        self,
        current_override: int | None,
        now_monotonic: float,
    ) -> int | None:
        """Merge approach gear with another moving override using the safer gear."""

        if not self.active(now_monotonic):
            return current_override
        if current_override is None:
            return self.speed_state
        return min(
            validate_moving_speed_state(
                current_override,
                "speed_state_override",
            ),
            self.speed_state,
        )
