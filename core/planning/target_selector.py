"""Polyline based target point selection for lane following."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from utils.math_utils import clamp


Point = Tuple[float, float]


@dataclass
class TargetPointResult:
    """Target point selected along the lane centerline in ROI coordinates."""

    target_point_roi: Point
    target_index: int
    lookahead_px: float
    target_lateral_error_px: float
    target_heading_error_deg: float
    confidence: float
    reason: str


class TargetSelector:
    """Selects a lookahead target by accumulating arc length on a centerline polyline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.base_lookahead_px = float(config.get("base_lookahead_px", 120.0))
        self.min_lookahead_px = float(config.get("min_lookahead_px", 70.0))
        self.max_lookahead_px = float(config.get("max_lookahead_px", 180.0))
        self.curvature_gain = float(config.get("curvature_gain", 80.0))
        self.low_confidence_lookahead_px = float(config.get("low_confidence_lookahead_px", 80.0))
        self.low_confidence_threshold = float(config.get("low_confidence_threshold", 0.45))

    def select(
        self,
        centerline_points: Sequence[Tuple[float, float]],
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        curvature: float,
        shifted_centerline_points: Sequence[Tuple[float, float]] | None = None,
    ) -> TargetPointResult:
        points = self._sort_near_to_far(shifted_centerline_points or centerline_points)
        ego_x = float(roi_width) * 0.5
        ego_y = float(roi_height - 1)
        ego_point = (ego_x, ego_y)

        if not points:
            return TargetPointResult(
                target_point_roi=ego_point,
                target_index=-1,
                lookahead_px=0.0,
                target_lateral_error_px=0.0,
                target_heading_error_deg=0.0,
                confidence=0.0,
                reason="lost: no centerline points",
            )

        lookahead, lookahead_confidence, lookahead_reason = self._compute_lookahead(
            lane_confidence=lane_confidence,
            curvature=curvature,
        )

        if len(points) == 1:
            target = points[0]
            return self._build_result(
                target_point=target,
                target_index=0,
                lookahead_px=lookahead,
                ego_point=ego_point,
                confidence=min(lane_confidence, lookahead_confidence) * 0.45,
                reason=f"fallback: only one centerline point; {lookahead_reason}",
            )

        nearest_index = self._nearest_point_index(points, ego_point)
        target, target_index = self._walk_arc_length(points, nearest_index, lookahead)
        reason = f"arc lookahead from index {nearest_index}; {lookahead_reason}"
        confidence = min(lane_confidence, lookahead_confidence)
        return self._build_result(
            target_point=target,
            target_index=target_index,
            lookahead_px=lookahead,
            ego_point=ego_point,
            confidence=confidence,
            reason=reason,
        )

    def select_on_polyline_with_min_y(
        self,
        centerline_points: Sequence[Tuple[float, float]],
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        curvature: float,
        min_target_y: float,
    ) -> TargetPointResult:
        """Select a target on the polyline, forcing it to be at or above min_target_y."""

        result = self.select(
            centerline_points=centerline_points,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=lane_confidence,
            curvature=curvature,
        )
        points = self._sort_near_to_far(centerline_points)
        if not points or result.target_point_roi[1] <= min_target_y:
            return result

        target, index = self._sample_by_y(points, min_target_y)
        ego_point = (float(roi_width) * 0.5, float(roi_height - 1))
        forced = self._build_result(
            target_point=target,
            target_index=index,
            lookahead_px=result.lookahead_px,
            ego_point=ego_point,
            confidence=result.confidence,
            reason=f"forced before obstacle at y={min_target_y:.1f}; {result.reason}",
        )
        return forced

    def _compute_lookahead(self, lane_confidence: float, curvature: float) -> Tuple[float, float, str]:
        if lane_confidence < self.low_confidence_threshold:
            lookahead = clamp(
                self.low_confidence_lookahead_px,
                self.min_lookahead_px,
                self.max_lookahead_px,
            )
            return lookahead, lane_confidence * 0.6, "low lane confidence lookahead"

        lookahead = self.base_lookahead_px - self.curvature_gain * abs(float(curvature))
        lookahead = clamp(lookahead, self.min_lookahead_px, self.max_lookahead_px)
        return lookahead, lane_confidence, "curvature adjusted lookahead"

    def _build_result(
        self,
        target_point: Point,
        target_index: int,
        lookahead_px: float,
        ego_point: Point,
        confidence: float,
        reason: str,
    ) -> TargetPointResult:
        target_x, target_y = target_point
        ego_x, ego_y = ego_point
        dx = target_x - ego_x
        dy = ego_y - target_y
        heading_error_deg = math.degrees(math.atan2(dx, max(1.0, dy)))
        return TargetPointResult(
            target_point_roi=(float(target_x), float(target_y)),
            target_index=int(target_index),
            lookahead_px=float(lookahead_px),
            target_lateral_error_px=float(dx),
            target_heading_error_deg=float(heading_error_deg),
            confidence=clamp(float(confidence), 0.0, 1.0),
            reason=reason,
        )

    def _sort_near_to_far(
        self,
        points: Sequence[Tuple[float, float]],
    ) -> list[Point]:
        return sorted([(float(x), float(y)) for x, y in points], key=lambda item: item[1], reverse=True)

    def _nearest_point_index(self, points: Sequence[Point], ego_point: Point) -> int:
        ego_x, ego_y = ego_point
        best_index = 0
        best_distance_sq = float("inf")
        for index, (x, y) in enumerate(points):
            distance_sq = (x - ego_x) ** 2 + (y - ego_y) ** 2
            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_index = index
        return best_index

    def _walk_arc_length(
        self,
        points: Sequence[Point],
        start_index: int,
        lookahead_px: float,
    ) -> tuple[Point, int]:
        if start_index >= len(points) - 1:
            return points[-1], len(points) - 1

        accumulated = 0.0
        previous = points[start_index]
        for index in range(start_index + 1, len(points)):
            current = points[index]
            segment_length = math.dist(previous, current)
            if accumulated + segment_length >= lookahead_px:
                ratio = (lookahead_px - accumulated) / max(segment_length, 1e-6)
                x = previous[0] + (current[0] - previous[0]) * ratio
                y = previous[1] + (current[1] - previous[1]) * ratio
                return (float(x), float(y)), index
            accumulated += segment_length
            previous = current
        return points[-1], len(points) - 1

    def _sample_by_y(self, points: Sequence[Point], target_y: float) -> tuple[Point, int]:
        if target_y >= points[0][1]:
            return points[0], 0
        if target_y <= points[-1][1]:
            return points[-1], len(points) - 1

        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            if y1 >= target_y >= y2 and y1 != y2:
                ratio = (target_y - y1) / (y2 - y1)
                x = x1 + (x2 - x1) * ratio
                return (float(x), float(target_y)), index + 1
        return points[-1], len(points) - 1
