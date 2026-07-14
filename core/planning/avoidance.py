"""Generate local avoidance targets by shifting lane centerline segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from core.object.blocking import BlockingAnalysisResult
from core.planning.target_selector import TargetPointResult, TargetSelector
from utils.math_utils import clamp


Point = Tuple[float, float]


@dataclass
class AvoidanceTargetResult:
    """Final target and shifted path after local obstacle handling."""

    mode: str
    shifted_centerline_points: list[Point]
    target_point_roi: Point
    avoid_bias_px: float
    final_lateral_error_px: float
    final_heading_error_deg: float
    confidence: float
    reason: str


class AvoidanceTargetPlanner:
    """Applies a smooth local lateral bias near blocking objects."""

    def __init__(self, config: Dict[str, Any], target_selector: TargetSelector) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.max_avoid_bias_px = float(config.get("max_avoid_bias_px", 110.0))
        self.bias_alpha = float(config.get("bias_alpha", 0.25))
        self.return_decay = float(config.get("return_decay", 0.85))
        self.clear_frames_threshold = int(config.get("clear_frames", 8))
        self.front_margin_px = float(config.get("front_margin_px", 45.0))
        self.rear_margin_px = float(config.get("rear_margin_px", 35.0))
        self.target_front_margin_px = float(config.get("target_front_margin_px", 35.0))
        self.min_lane_confidence_to_start_avoid = float(
            config.get("min_lane_confidence_to_start_avoid", 0.45)
        )
        self.max_target_jump_px = float(config.get("max_target_jump_px", 50.0))

        self.target_selector = target_selector
        self.previous_avoid_bias_px = 0.0
        self.clear_frames = 0
        self.mode = "lane_follow"
        self.last_target_point: Point | None = None

    def plan(
        self,
        centerline_points: Sequence[Tuple[float, float]],
        normal_target: TargetPointResult,
        blocking_result: BlockingAnalysisResult,
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        curvature: float,
    ) -> AvoidanceTargetResult:
        points = self._sort_near_to_far(centerline_points)
        if not self.enabled or not points:
            return self._lane_follow(points, normal_target, "disabled or no centerline")

        if blocking_result.too_close:
            self.previous_avoid_bias_px *= self.return_decay
            return AvoidanceTargetResult(
                mode="too_close",
                shifted_centerline_points=points,
                target_point_roi=normal_target.target_point_roi,
                avoid_bias_px=self.previous_avoid_bias_px,
                final_lateral_error_px=normal_target.target_lateral_error_px,
                final_heading_error_deg=normal_target.target_heading_error_deg,
                confidence=min(normal_target.confidence, 0.35),
                reason=f"too close; {blocking_result.reason}",
            )

        can_start_avoid = lane_confidence >= self.min_lane_confidence_to_start_avoid
        if blocking_result.need_avoid and blocking_result.blocking_object and can_start_avoid:
            self.clear_frames = 0
            desired_bias = self._desired_bias(blocking_result.recommended_avoid_side)
            self.previous_avoid_bias_px = (
                self.bias_alpha * desired_bias
                + (1.0 - self.bias_alpha) * self.previous_avoid_bias_px
            )
            mode = (
                "avoid_left"
                if blocking_result.recommended_avoid_side == "left"
                else "avoid_right"
            )
            return self._avoid(
                mode=mode,
                points=points,
                normal_target=normal_target,
                blocking_result=blocking_result,
                roi_width=roi_width,
                roi_height=roi_height,
                lane_confidence=lane_confidence,
                curvature=curvature,
            )

        if blocking_result.need_avoid and not can_start_avoid:
            self.previous_avoid_bias_px *= self.return_decay
            shifted_points = self._shift_points_without_object(points, self.previous_avoid_bias_px)
            target = self.target_selector.select(
                centerline_points=points,
                shifted_centerline_points=shifted_points,
                roi_width=roi_width,
                roi_height=roi_height,
                lane_confidence=lane_confidence,
                curvature=curvature,
            )
            return self._from_target(
                mode="return_center",
                shifted_points=shifted_points,
                target=target,
                reason="low lane confidence, avoid suppressed",
            )

        self.clear_frames += 1
        if abs(self.previous_avoid_bias_px) > 5.0 and self.clear_frames < self.clear_frames_threshold:
            self.previous_avoid_bias_px *= self.return_decay
            shifted_points = self._shift_points_without_object(points, self.previous_avoid_bias_px)
            target = self.target_selector.select(
                centerline_points=points,
                shifted_centerline_points=shifted_points,
                roi_width=roi_width,
                roi_height=roi_height,
                lane_confidence=lane_confidence,
                curvature=curvature,
            )
            return self._from_target(
                mode="return_center",
                shifted_points=shifted_points,
                target=target,
                reason=f"returning center, clear_frames={self.clear_frames}",
            )

        self.previous_avoid_bias_px = 0.0
        self.clear_frames = min(self.clear_frames, self.clear_frames_threshold)
        return self._lane_follow(points, normal_target, blocking_result.reason)

    def _avoid(
        self,
        mode: str,
        points: list[Point],
        normal_target: TargetPointResult,
        blocking_result: BlockingAnalysisResult,
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        curvature: float,
    ) -> AvoidanceTargetResult:
        assert blocking_result.blocking_object is not None
        _, y1, _, y2 = blocking_result.blocking_object.bbox_roi
        influence_y_min = y1 - self.front_margin_px
        influence_y_max = y2 + self.rear_margin_px
        shifted_points = self._shift_points_near_obstacle(
            points=points,
            influence_y_min=influence_y_min,
            influence_y_max=influence_y_max,
            avoid_bias_px=self.previous_avoid_bias_px,
            roi_width=roi_width,
        )

        min_target_y = y1 - self.target_front_margin_px
        target = self.target_selector.select_on_polyline_with_min_y(
            centerline_points=shifted_points,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=lane_confidence,
            curvature=curvature,
            min_target_y=min_target_y,
        )
        target = self._limit_target_jump(target, roi_width=roi_width, roi_height=roi_height)
        reason = (
            f"{mode}: bias={self.previous_avoid_bias_px:.1f}px, "
            f"influence_y=[{influence_y_min:.0f},{influence_y_max:.0f}], "
            f"{blocking_result.reason}"
        )
        return self._from_target(
            mode=mode,
            shifted_points=shifted_points,
            target=target,
            reason=reason,
        )

    def _lane_follow(
        self,
        points: list[Point],
        normal_target: TargetPointResult,
        reason: str,
    ) -> AvoidanceTargetResult:
        self.mode = "lane_follow"
        self.last_target_point = normal_target.target_point_roi
        return AvoidanceTargetResult(
            mode="lane_follow",
            shifted_centerline_points=points,
            target_point_roi=normal_target.target_point_roi,
            avoid_bias_px=0.0,
            final_lateral_error_px=normal_target.target_lateral_error_px,
            final_heading_error_deg=normal_target.target_heading_error_deg,
            confidence=normal_target.confidence,
            reason=f"lane follow; {reason}",
        )

    def _from_target(
        self,
        mode: str,
        shifted_points: list[Point],
        target: TargetPointResult,
        reason: str,
    ) -> AvoidanceTargetResult:
        self.mode = mode
        self.last_target_point = target.target_point_roi
        return AvoidanceTargetResult(
            mode=mode,
            shifted_centerline_points=shifted_points,
            target_point_roi=target.target_point_roi,
            avoid_bias_px=self.previous_avoid_bias_px,
            final_lateral_error_px=target.target_lateral_error_px,
            final_heading_error_deg=target.target_heading_error_deg,
            confidence=target.confidence,
            reason=reason,
        )

    def _desired_bias(self, recommended_avoid_side: str) -> float:
        if recommended_avoid_side == "left":
            return -self.max_avoid_bias_px
        if recommended_avoid_side == "right":
            return self.max_avoid_bias_px
        return 0.0

    def _shift_points_near_obstacle(
        self,
        points: Sequence[Point],
        influence_y_min: float,
        influence_y_max: float,
        avoid_bias_px: float,
        roi_width: int,
    ) -> list[Point]:
        if influence_y_max <= influence_y_min:
            return [(float(x), float(y)) for x, y in points]

        shifted: list[Point] = []
        ramp = max(1.0, (influence_y_max - influence_y_min) * 0.25)
        for x, y in points:
            weight = self._cosine_influence_weight(y, influence_y_min, influence_y_max, ramp)
            shifted_x = clamp(x + weight * avoid_bias_px, 0.0, float(max(0, roi_width - 1)))
            shifted.append((float(shifted_x), float(y)))
        return shifted

    def _shift_points_without_object(
        self,
        points: Sequence[Point],
        avoid_bias_px: float,
    ) -> list[Point]:
        if abs(avoid_bias_px) < 1e-3:
            return [(float(x), float(y)) for x, y in points]
        return [(float(x + avoid_bias_px), float(y)) for x, y in points]

    def _cosine_influence_weight(
        self,
        y: float,
        influence_y_min: float,
        influence_y_max: float,
        ramp: float,
    ) -> float:
        if y < influence_y_min - ramp or y > influence_y_max + ramp:
            return 0.0
        if influence_y_min <= y <= influence_y_max:
            return 1.0
        if y < influence_y_min:
            ratio = (y - (influence_y_min - ramp)) / ramp
        else:
            ratio = ((influence_y_max + ramp) - y) / ramp
        ratio = clamp(ratio, 0.0, 1.0)
        return 0.5 - 0.5 * math.cos(math.pi * ratio)

    def _limit_target_jump(
        self,
        target: TargetPointResult,
        roi_width: int,
        roi_height: int,
    ) -> TargetPointResult:
        if self.last_target_point is None:
            return target
        prev_x, prev_y = self.last_target_point
        target_x, target_y = target.target_point_roi
        dx = clamp(target_x - prev_x, -self.max_target_jump_px, self.max_target_jump_px)
        dy = clamp(target_y - prev_y, -self.max_target_jump_px, self.max_target_jump_px)
        limited_point = (prev_x + dx, prev_y + dy)
        if limited_point == target.target_point_roi:
            return target

        ego_x = float(roi_width) * 0.5
        ego_y = float(roi_height - 1)
        final_dx = limited_point[0] - ego_x
        final_heading = math.degrees(math.atan2(final_dx, max(1.0, ego_y - limited_point[1])))
        return TargetPointResult(
            target_point_roi=limited_point,
            target_index=target.target_index,
            lookahead_px=target.lookahead_px,
            target_lateral_error_px=final_dx,
            target_heading_error_deg=final_heading,
            confidence=target.confidence * 0.9,
            reason=f"target jump limited; {target.reason}",
        )

    def _sort_near_to_far(self, points: Sequence[Tuple[float, float]]) -> list[Point]:
        return sorted([(float(x), float(y)) for x, y in points], key=lambda item: item[1], reverse=True)
