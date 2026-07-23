"""Stateful pedestrian crossing-based stop analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from core.object.blocking import DetectedObject


BBox = tuple[float, float, float, float]
BBoxInt = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass(frozen=True)
class PedestrianSafetyResult:
    """Current pedestrian crossing state for planning and visualization."""

    stop_required: bool
    latched: bool
    armed: bool
    center_region_frame: BBox
    frozen_target_x_frame: float | None
    target_region: str
    tracked_center_frame: Point | None
    human_count: int
    cooldown_remaining_sec: float
    reason: str


class PedestrianSafetyAnalyzer:
    """Wait for one triggering pedestrian to cross a frozen lane target."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.min_box_area_px = float(config.get("min_box_area_px", 600.0))
        self.rearm_cooldown_sec = float(config.get("rearm_cooldown_sec", 3.0))
        if self.min_box_area_px < 0.0:
            raise ValueError("pedestrian_safety.min_box_area_px must be non-negative")
        if self.rearm_cooldown_sec < 0.0:
            raise ValueError(
                "pedestrian_safety.rearm_cooldown_sec must be non-negative"
            )

        region_config = config.get("center_region", {})
        self.left_ratio = float(region_config.get("left_ratio", 0.30))
        self.right_ratio = float(region_config.get("right_ratio", 0.70))
        self._validate_region_ratios()

        self.latched = False
        self.cooldown_until = 0.0
        self.last_processed_result_id: int | None = None
        self.frozen_target_x_frame: float | None = None
        self.target_region = "none"
        self.tracked_center_frame: Point | None = None

    def analyze(
        self,
        objects: Sequence[DetectedObject],
        roi_rect: BBoxInt,
        target_point_roi: Point,
        detection_result_id: int,
        now_monotonic: float,
    ) -> PedestrianSafetyResult:
        """Update state only once per AI result while exposing live cooldown time."""

        now = float(now_monotonic)
        center_region = self._center_region_frame(roi_rect)
        humans = [
            obj for obj in objects
            if obj.class_name.casefold() == "human"
        ]

        if not self.enabled:
            self._reset(clear_processed_result=True)
            return self._result(
                center_region,
                humans,
                now,
                "disabled",
            )

        result_id = int(detection_result_id)
        is_new_result = result_id != self.last_processed_result_id
        if is_new_result:
            self.last_processed_result_id = result_id

        cooldown_remaining = self._cooldown_remaining(now)
        if not self.latched and cooldown_remaining > 0.0:
            return self._result(
                center_region,
                humans,
                now,
                f"cooldown; remaining={cooldown_remaining:.2f}s",
            )

        if not is_new_result:
            reason = (
                "latched; waiting for a new AI result"
                if self.latched
                else "armed; waiting for a new AI result"
            )
            return self._result(center_region, humans, now, reason)

        if self.latched:
            return self._update_latched_track(
                humans=humans,
                center_region=center_region,
                now=now,
            )

        self._clear_crossing_state()
        triggering = [
            obj for obj in humans
            if self._bbox_area(obj.bbox_frame) >= self.min_box_area_px
            and self._center_is_in_roi(self._bbox_center(obj.bbox_frame), roi_rect)
        ]
        if not triggering:
            return self._result(
                center_region,
                humans,
                now,
                "armed; no qualifying pedestrian center in ROI",
            )

        trigger = min(triggering, key=self._trigger_sort_key)
        roi_x1, _roi_y1, roi_x2, _roi_y2 = [float(value) for value in roi_rect]
        target_x = max(
            roi_x1,
            min(roi_x2, roi_x1 + float(target_point_roi[0])),
        )
        self.frozen_target_x_frame = target_x
        self.target_region = self._classify_target_region(
            target_x_frame=target_x,
            center_region=center_region,
        )
        self.tracked_center_frame = self._bbox_center(trigger.bbox_frame)
        self.latched = True
        trigger_area = self._bbox_area(trigger.bbox_frame)
        return self._result(
            center_region,
            humans,
            now,
            (
                f"triggered; area={trigger_area:.0f}px^2 "
                f"region={self.target_region} target_x={target_x:.1f}"
            ),
        )

    def _update_latched_track(
        self,
        humans: Sequence[DetectedObject],
        center_region: BBox,
        now: float,
    ) -> PedestrianSafetyResult:
        if not humans or self.tracked_center_frame is None:
            return self._result(
                center_region,
                humans,
                now,
                "latched; triggering pedestrian missing",
            )

        previous_center = self.tracked_center_frame
        tracked = min(
            humans,
            key=lambda obj: self._association_sort_key(
                obj,
                previous_center=previous_center,
            ),
        )
        current_center = self._bbox_center(tracked.bbox_frame)
        crossed = self._has_crossed_target(
            previous_x=previous_center[0],
            current_x=current_center[0],
        )
        self.tracked_center_frame = current_center
        if not crossed:
            return self._result(
                center_region,
                humans,
                now,
                (
                    f"latched; tracking region={self.target_region} "
                    f"x={current_center[0]:.1f}"
                ),
            )

        released_region = self.target_region
        self.latched = False
        self.cooldown_until = now + self.rearm_cooldown_sec
        return self._result(
            center_region,
            humans,
            now,
            (
                f"released; pedestrian crossed {released_region} target; "
                f"cooldown={self.rearm_cooldown_sec:.1f}s"
            ),
        )

    def _has_crossed_target(self, previous_x: float, current_x: float) -> bool:
        target_x = self.frozen_target_x_frame
        if target_x is None:
            return False
        if self.target_region == "left":
            return previous_x >= target_x and current_x < target_x
        if self.target_region == "right":
            return previous_x <= target_x and current_x > target_x
        if self.target_region == "center":
            return (
                previous_x < target_x < current_x
                or previous_x > target_x > current_x
            )
        return False

    def _validate_region_ratios(self) -> None:
        if not 0.0 <= self.left_ratio < self.right_ratio <= 1.0:
            raise ValueError(
                "pedestrian_safety.center_region requires "
                "0 <= left_ratio < right_ratio <= 1"
            )

    def _center_region_frame(self, roi_rect: BBoxInt) -> BBox:
        roi_x1, roi_y1, roi_x2, roi_y2 = [float(value) for value in roi_rect]
        roi_width = max(0.0, roi_x2 - roi_x1)
        return (
            roi_x1 + roi_width * self.left_ratio,
            roi_y1,
            roi_x1 + roi_width * self.right_ratio,
            roi_y2,
        )

    @staticmethod
    def _classify_target_region(
        target_x_frame: float,
        center_region: BBox,
    ) -> str:
        center_left, _top, center_right, _bottom = center_region
        if target_x_frame < center_left:
            return "left"
        if target_x_frame > center_right:
            return "right"
        return "center"

    @staticmethod
    def _normalized_bbox(bbox: BBoxInt) -> BBox:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def _bbox_area(self, bbox: BBoxInt) -> float:
        x1, y1, x2, y2 = self._normalized_bbox(bbox)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _bbox_center(self, bbox: BBoxInt) -> Point:
        x1, y1, x2, y2 = self._normalized_bbox(bbox)
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @staticmethod
    def _center_is_in_roi(center: Point, roi_rect: BBoxInt) -> bool:
        center_x, center_y = center
        roi_x1, roi_y1, roi_x2, roi_y2 = [float(value) for value in roi_rect]
        return roi_x1 <= center_x < roi_x2 and roi_y1 <= center_y < roi_y2

    def _trigger_sort_key(
        self,
        obj: DetectedObject,
    ) -> tuple[float, float, tuple[int, int, int, int]]:
        return (
            -self._bbox_area(obj.bbox_frame),
            -float(obj.confidence),
            tuple(int(value) for value in obj.bbox_frame),
        )

    def _association_sort_key(
        self,
        obj: DetectedObject,
        previous_center: Point,
    ) -> tuple[float, tuple[int, int, int, int]]:
        center_x, center_y = self._bbox_center(obj.bbox_frame)
        distance = math.hypot(
            center_x - previous_center[0],
            center_y - previous_center[1],
        )
        return distance, tuple(int(value) for value in obj.bbox_frame)

    def _cooldown_remaining(self, now: float) -> float:
        return max(0.0, self.cooldown_until - now)

    def _clear_crossing_state(self) -> None:
        self.frozen_target_x_frame = None
        self.target_region = "none"
        self.tracked_center_frame = None
        self.cooldown_until = 0.0

    def _reset(self, clear_processed_result: bool) -> None:
        self.latched = False
        self._clear_crossing_state()
        if clear_processed_result:
            self.last_processed_result_id = None

    def _result(
        self,
        center_region: BBox,
        humans: Sequence[DetectedObject],
        now: float,
        reason: str,
    ) -> PedestrianSafetyResult:
        cooldown_remaining = self._cooldown_remaining(now)
        return PedestrianSafetyResult(
            stop_required=self.latched,
            latched=self.latched,
            armed=bool(
                self.enabled
                and not self.latched
                and cooldown_remaining <= 0.0
            ),
            center_region_frame=center_region,
            frozen_target_x_frame=self.frozen_target_x_frame,
            target_region=self.target_region,
            tracked_center_frame=self.tracked_center_frame,
            human_count=len(humans),
            cooldown_remaining_sec=cooldown_remaining,
            reason=reason,
        )
