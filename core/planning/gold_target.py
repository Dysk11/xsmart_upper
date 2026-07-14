"""Coin target selection and approach planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from core.object.blocking import DetectedObject
from utils.math_utils import clamp


Point = Tuple[float, float]


@dataclass
class GoldTargetResult:
    """Control target derived from a coin/Gold detection."""

    active: bool
    target_object: DetectedObject | None
    target_point_roi: Point
    final_lateral_error_px: float
    final_heading_error_deg: float
    confidence: float
    speed_limit: float | None
    reason: str
    using_hold: bool = False


class GoldTargetPlanner:
    """Turns coin detections into a temporary target point before lane following."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.class_names = {
            str(name).casefold()
            for name in config.get("class_names", ["coin", "Gold"])
        }
        self.min_confidence = float(config.get("min_confidence", 0.35))
        self.hold_frames = int(config.get("hold_frames", 4))
        self.approach_speed_limit = self._optional_float(config.get("approach_speed_limit", 0.85))
        self.close_speed_limit = self._optional_float(config.get("close_speed_limit", 0.45))
        self.close_y_ratio = float(config.get("close_y_ratio", 0.82))
        self.max_above_roi_ratio = float(config.get("max_above_roi_ratio", 0.8))
        self.aim_at = str(config.get("aim_at", "bottom_center")).lower()

        self._last_active: GoldTargetResult | None = None
        self._miss_frames = 0

    def plan(
        self,
        objects: Sequence[DetectedObject],
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
    ) -> GoldTargetResult:
        if not self.enabled:
            return self._empty(roi_width, roi_height, "disabled")

        gold = self._select_gold(objects)
        if gold is None:
            if self._last_active is not None and self._miss_frames < self.hold_frames:
                self._miss_frames += 1
                held = self._last_active
                return GoldTargetResult(
                    active=True,
                    target_object=held.target_object,
                    target_point_roi=held.target_point_roi,
                    final_lateral_error_px=held.final_lateral_error_px,
                    final_heading_error_deg=held.final_heading_error_deg,
                    confidence=max(0.0, held.confidence * (0.75 ** self._miss_frames)),
                    speed_limit=held.speed_limit,
                    reason=f"hold coin target, miss_frames={self._miss_frames}",
                    using_hold=True,
                )
            self._last_active = None
            self._miss_frames = 0
            return self._empty(roi_width, roi_height, "no coin")

        self._miss_frames = 0
        result = self._build_result(gold, roi_rect, roi_width, roi_height)
        self._last_active = result
        return result

    def _select_gold(self, objects: Sequence[DetectedObject]) -> DetectedObject | None:
        best: tuple[float, DetectedObject] | None = None
        for obj in objects:
            if obj.class_name.casefold() not in self.class_names:
                continue
            if obj.confidence < self.min_confidence:
                continue
            x1, y1, x2, y2 = obj.bbox_frame
            area = max(0, x2 - x1) * max(0, y2 - y1)
            score = float(obj.confidence) + float(y2) * 1e-3 + math.sqrt(float(area)) * 1e-4
            if best is None or score > best[0]:
                best = (score, obj)
        return best[1] if best is not None else None

    def _build_result(
        self,
        gold: DetectedObject,
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
    ) -> GoldTargetResult:
        x1, y1, x2, y2 = [float(value) for value in gold.bbox_frame]
        roi_x1, roi_y1, _, _ = [float(value) for value in roi_rect]
        target_x_frame = (x1 + x2) * 0.5
        if self.aim_at == "center":
            target_y_frame = (y1 + y2) * 0.5
        else:
            target_y_frame = y2

        target_x_roi = clamp(target_x_frame - roi_x1, 0.0, float(max(0, roi_width - 1)))
        min_y = -float(roi_height) * self.max_above_roi_ratio
        target_y_roi = clamp(target_y_frame - roi_y1, min_y, float(max(0, roi_height - 1)))

        ego_x = float(roi_width) * 0.5
        ego_y = float(roi_height - 1)
        dx = target_x_roi - ego_x
        dy = max(1.0, ego_y - target_y_roi)
        heading_error_deg = math.degrees(math.atan2(dx, dy))

        speed_limit = self.approach_speed_limit
        if target_y_roi >= float(roi_height) * self.close_y_ratio:
            speed_limit = self.close_speed_limit

        return GoldTargetResult(
            active=True,
            target_object=gold,
            target_point_roi=(float(target_x_roi), float(target_y_roi)),
            final_lateral_error_px=float(dx),
            final_heading_error_deg=float(heading_error_deg),
            confidence=float(gold.confidence),
            speed_limit=speed_limit,
            reason=(
                f"coin target at roi=({target_x_roi:.1f},{target_y_roi:.1f}), "
                f"conf={gold.confidence:.2f}"
            ),
        )

    def _empty(self, roi_width: int, roi_height: int, reason: str) -> GoldTargetResult:
        return GoldTargetResult(
            active=False,
            target_object=None,
            target_point_roi=(float(roi_width) * 0.5, float(max(0, roi_height - 1))),
            final_lateral_error_px=0.0,
            final_heading_error_deg=0.0,
            confidence=0.0,
            speed_limit=None,
            reason=reason,
        )

    def _optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        return float(value)
