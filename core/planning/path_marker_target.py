"""Plan a continuous path through Go/Stop detections that interrupt the lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from core.object.blocking import DetectedObject
from utils.math_utils import clamp


Point = Tuple[float, float]


@dataclass
class PathMarkerTargetResult:
    """A Go/Stop target and the virtual centerline connected through its box."""

    active: bool
    target_object: DetectedObject | None
    target_class_name: str
    target_point_roi: Point
    connected_centerline_points: list[Point]
    lower_anchor_roi: Point | None
    upper_anchor_roi: Point | None
    final_lateral_error_px: float
    final_heading_error_deg: float
    confidence: float
    reason: str
    using_hold: bool = False
    using_historical_upper: bool = False


class PathMarkerTargetPlanner:
    """Connect lane segments through the center of a detected Go/Stop marker."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.class_names = {
            str(name).casefold()
            for name in config.get("class_names", ["Go", "Stop"])
        }
        self.min_confidence = float(config.get("min_confidence", 0.20))
        self.hold_frames = max(0, int(config.get("hold_frames", 4)))
        self.connection_margin_px = max(0.0, float(config.get("connection_margin_px", 12.0)))
        self.interpolation_step_px = max(1.0, float(config.get("interpolation_step_px", 4.0)))
        self.release_y_ratio = clamp(float(config.get("release_y_ratio", 0.92)), 0.0, 1.0)

        self._last_target: DetectedObject | None = None
        self._miss_frames = 0

    def plan(
        self,
        objects: Sequence[DetectedObject],
        centerline_points: Sequence[Tuple[float, float]],
        historical_centerline_points: Sequence[Tuple[float, float]],
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
    ) -> PathMarkerTargetResult:
        if not self.enabled:
            self._clear_hold()
            return self._empty(roi_width, roi_height, "disabled")

        target, saw_matching = self._select_target(objects, roi_rect, roi_height)
        using_hold = False
        if target is None:
            if saw_matching:
                self._clear_hold()
                return self._empty(roi_width, roi_height, "path marker passed release line")
            if self._last_target is None or self._miss_frames >= self.hold_frames:
                self._clear_hold()
                return self._empty(roi_width, roi_height, "no Go/Stop path marker")
            self._miss_frames += 1
            target = self._last_target
            using_hold = True
        else:
            self._last_target = target
            self._miss_frames = 0

        result = self._build_result(
            target=target,
            centerline_points=centerline_points,
            historical_centerline_points=historical_centerline_points,
            roi_rect=roi_rect,
            roi_width=roi_width,
            roi_height=roi_height,
            using_hold=using_hold,
        )
        if using_hold:
            result.confidence *= 0.75 ** self._miss_frames
            result.reason = f"hold {result.reason}, miss_frames={self._miss_frames}"
        return result

    def _select_target(
        self,
        objects: Sequence[DetectedObject],
        roi_rect: tuple[int, int, int, int],
        roi_height: int,
    ) -> tuple[DetectedObject | None, bool]:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_rect
        release_y = float(roi_height) * self.release_y_ratio
        candidates: list[tuple[float, DetectedObject]] = []
        saw_matching = False
        for obj in objects:
            if obj.class_name.casefold() not in self.class_names or obj.confidence < self.min_confidence:
                continue
            x1, y1, x2, y2 = obj.bbox_frame
            if x2 <= roi_x1 or x1 >= roi_x2:
                continue
            if y1 >= roi_y2:
                saw_matching = True
                continue
            if y2 <= roi_y1:
                continue
            saw_matching = True
            center_y_roi = 0.5 * (float(y1) + float(y2)) - float(roi_y1)
            if center_y_roi >= release_y:
                continue
            area = max(0, x2 - x1) * max(0, y2 - y1)
            score = float(obj.confidence) + float(y2) * 1e-3 + math.sqrt(float(area)) * 1e-4
            candidates.append((score, obj))
        if not candidates:
            return None, saw_matching
        return max(candidates, key=lambda item: item[0])[1], saw_matching

    def _build_result(
        self,
        target: DetectedObject,
        centerline_points: Sequence[Tuple[float, float]],
        historical_centerline_points: Sequence[Tuple[float, float]],
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
        using_hold: bool,
    ) -> PathMarkerTargetResult:
        frame_x1, frame_y1, frame_x2, frame_y2 = [float(value) for value in target.bbox_frame]
        roi_x1, roi_y1, _, _ = [float(value) for value in roi_rect]
        center = (
            clamp(0.5 * (frame_x1 + frame_x2) - roi_x1, 0.0, float(max(0, roi_width - 1))),
            clamp(0.5 * (frame_y1 + frame_y2) - roi_y1, 0.0, float(max(0, roi_height - 1))),
        )
        bbox_top = frame_y1 - roi_y1
        bbox_bottom = frame_y2 - roi_y1
        lower_limit = bbox_bottom + self.connection_margin_px
        upper_limit = bbox_top - self.connection_margin_px

        current = self._normalize_points(centerline_points, roi_width, roi_height)
        historical = self._normalize_points(historical_centerline_points, roi_width, roi_height)
        lower_candidates = [point for point in current if point[1] >= lower_limit]
        lower_anchor = min(lower_candidates, key=lambda point: point[1]) if lower_candidates else (
            float(roi_width) * 0.5,
            float(max(0, roi_height - 1)),
        )
        upper_candidates = [point for point in current if point[1] <= upper_limit]
        using_historical_upper = False
        if upper_candidates:
            upper_anchor: Point | None = max(upper_candidates, key=lambda point: point[1])
        else:
            historical_upper = [point for point in historical if point[1] <= upper_limit]
            upper_anchor = max(historical_upper, key=lambda point: point[1]) if historical_upper else None
            using_historical_upper = upper_anchor is not None

        lower_points = [point for point in current if point[1] >= lower_anchor[1]]
        upper_source = historical if using_historical_upper else current
        upper_points = (
            [point for point in upper_source if upper_anchor is not None and point[1] <= upper_anchor[1]]
            if upper_anchor is not None
            else []
        )
        connected = sorted(lower_points, key=lambda point: point[1], reverse=True)
        if not connected or connected[-1] != lower_anchor:
            connected.append(lower_anchor)
        connected.extend(self._interpolate(lower_anchor, center)[1:])
        if upper_anchor is not None:
            connected.extend(self._interpolate(center, upper_anchor)[1:])
            connected.extend(sorted(upper_points, key=lambda point: point[1], reverse=True))
        connected = self._deduplicate_rows(connected)

        ego_x = float(roi_width) * 0.5
        ego_y = float(max(0, roi_height - 1))
        dx = center[0] - ego_x
        heading = math.degrees(math.atan2(dx, max(1.0, ego_y - center[1])))
        confidence = float(target.confidence)
        if upper_anchor is None:
            confidence *= 0.65
        elif using_historical_upper:
            confidence *= 0.80

        upper_status = "historical upper anchor" if using_historical_upper else (
            "current upper anchor" if upper_anchor is not None else "no upper anchor"
        )
        return PathMarkerTargetResult(
            active=True,
            target_object=target,
            target_class_name=target.class_name,
            target_point_roi=center,
            connected_centerline_points=connected,
            lower_anchor_roi=lower_anchor,
            upper_anchor_roi=upper_anchor,
            final_lateral_error_px=float(dx),
            final_heading_error_deg=float(heading),
            confidence=clamp(confidence, 0.0, 1.0),
            reason=(
                f"{target.class_name} path target at roi=({center[0]:.1f},{center[1]:.1f}); "
                f"{upper_status}"
            ),
            using_hold=using_hold,
            using_historical_upper=using_historical_upper,
        )

    def _normalize_points(
        self,
        points: Sequence[Tuple[float, float]],
        roi_width: int,
        roi_height: int,
    ) -> list[Point]:
        normalized = [
            (
                clamp(float(x), 0.0, float(max(0, roi_width - 1))),
                clamp(float(y), 0.0, float(max(0, roi_height - 1))),
            )
            for x, y in points
        ]
        return self._deduplicate_rows(sorted(normalized, key=lambda point: point[1], reverse=True))

    def _interpolate(self, start: Point, end: Point) -> list[Point]:
        vertical_distance = abs(end[1] - start[1])
        steps = max(1, int(math.ceil(vertical_distance / self.interpolation_step_px)))
        return [
            (
                start[0] + (end[0] - start[0]) * index / steps,
                start[1] + (end[1] - start[1]) * index / steps,
            )
            for index in range(steps + 1)
        ]

    @staticmethod
    def _deduplicate_rows(points: Sequence[Point]) -> list[Point]:
        result: list[Point] = []
        seen_rows: set[int] = set()
        for x, y in sorted(points, key=lambda point: point[1], reverse=True):
            row = int(round(y))
            if row in seen_rows:
                continue
            seen_rows.add(row)
            result.append((float(x), float(y)))
        return result

    def _empty(self, roi_width: int, roi_height: int, reason: str) -> PathMarkerTargetResult:
        return PathMarkerTargetResult(
            active=False,
            target_object=None,
            target_class_name="",
            target_point_roi=(float(roi_width) * 0.5, float(max(0, roi_height - 1))),
            connected_centerline_points=[],
            lower_anchor_roi=None,
            upper_anchor_roi=None,
            final_lateral_error_px=0.0,
            final_heading_error_deg=0.0,
            confidence=0.0,
            reason=reason,
        )

    def _clear_hold(self) -> None:
        self._last_target = None
        self._miss_frames = 0
