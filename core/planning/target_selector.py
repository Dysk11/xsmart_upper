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
    """Select a target at a fixed ROI height on a centerline polyline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.fixed_target_y = float(config.get("fixed_target_y", 80.0))
        self.max_extrapolation_y_px = max(
            0.0, float(config.get("max_extrapolation_y_px", 40.0))
        )
        self.max_extrapolation_x_px = max(
            0.0, float(config.get("max_extrapolation_x_px", 60.0))
        )

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

        # Keep curvature in the public call signature for compatibility. Target
        # height is intentionally independent of curvature and lane confidence.
        _ = curvature
        target_y = clamp(self.fixed_target_y, 0.0, float(max(0, roi_height - 1)))
        target, target_index, sample_reason = self._sample_at_fixed_y(
            points=points,
            target_y=target_y,
            roi_width=roi_width,
        )
        lookahead = max(0.0, ego_y - float(target[1]))

        if len(points) == 1:
            return self._build_result(
                target_point=target,
                target_index=target_index,
                lookahead_px=lookahead,
                ego_point=ego_point,
                confidence=lane_confidence * 0.45,
                reason=f"fixed target y={target_y:.1f}; {sample_reason}",
            )

        return self._build_result(
            target_point=target,
            target_index=target_index,
            lookahead_px=lookahead,
            ego_point=ego_point,
            confidence=lane_confidence,
            reason=f"fixed target y={target_y:.1f}; {sample_reason}",
        )

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

    def _sample_at_fixed_y(
        self,
        points: Sequence[Point],
        target_y: float,
        roi_width: int,
    ) -> tuple[Point, int, str]:
        """Interpolate or extrapolate centerline x while keeping y fixed."""

        max_x = float(max(0, roi_width - 1))
        if len(points) == 1:
            x = clamp(points[0][0], 0.0, max_x)
            return (float(x), float(target_y)), 0, "single-point x fallback"

        # Prefer the segment that brackets the configured height.
        for index in range(len(points) - 1):
            x1, y1 = points[index]
            x2, y2 = points[index + 1]
            if y1 >= target_y >= y2 and y1 != y2:
                x = self._linear_x_at_y(x1, y1, x2, y2, target_y)
                x = clamp(x, 0.0, max_x)
                return (float(x), float(target_y)), index + 1, "centerline interpolation"

        # The centerline is shorter than the requested height. Select the two
        # closest points with distinct y values and extend that local segment.
        candidates = sorted(
            enumerate(points),
            key=lambda item: (abs(item[1][1] - target_y), item[0]),
        )
        first_index, (x1, y1) = candidates[0]
        extrapolation_y = abs(float(target_y) - float(y1))
        if extrapolation_y > self.max_extrapolation_y_px:
            x = clamp(x1, 0.0, max_x)
            return (
                (float(x), float(y1)),
                first_index,
                f"visible endpoint fallback; extrapolation_y={extrapolation_y:.1f}px",
            )
        for _second_index, (x2, y2) in candidates[1:]:
            if y2 == y1:
                continue
            extrapolated_x = self._linear_x_at_y(x1, y1, x2, y2, target_y)
            extrapolation_x = abs(extrapolated_x - x1)
            if (
                extrapolation_x > self.max_extrapolation_x_px
                or extrapolated_x < 0.0
                or extrapolated_x > max_x
            ):
                x = clamp(x1, 0.0, max_x)
                return (
                    (float(x), float(y1)),
                    first_index,
                    f"visible endpoint fallback; extrapolation_x={extrapolation_x:.1f}px",
                )
            x = clamp(extrapolated_x, 0.0, max_x)
            return (
                (float(x), float(target_y)),
                first_index,
                "centerline endpoint extrapolation",
            )

        # Degenerate input where every point lies on the same image row.
        x = clamp(x1, 0.0, max_x)
        return (float(x), float(target_y)), first_index, "same-row x fallback"

    def _linear_x_at_y(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        target_y: float,
    ) -> float:
        ratio = (target_y - y1) / (y2 - y1)
        return float(x1 + (x2 - x1) * ratio)
