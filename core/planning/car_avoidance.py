"""Lightweight car avoidance using expanded boxes and smooth lateral constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from core.object.blocking import DetectedObject
from core.planning.target_selector import TargetPointResult, TargetSelector
from utils.math_utils import clamp


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CarWarningZone:
    """One expanded car box expressed in frame and lane-ROI coordinates."""

    bbox_frame: BBox
    bbox_roi: BBox
    car_center_x_frame: float
    avoid_side: str
    confidence: float


@dataclass
class CarAvoidanceResult:
    """Final collision-free route and control state for car avoidance."""

    active: bool
    mode: str
    warning_zones: list[CarWarningZone]
    shifted_centerline_points: list[Point]
    target_result: TargetPointResult
    edge_limited: bool
    stop_required: bool
    reason: str


class CarAvoidancePlanner:
    """Build a smooth constrained polyline without search or curve fitting."""

    def __init__(self, config: Dict[str, Any], target_selector: TargetSelector) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.box_scale = float(config.get("box_scale", 1.5))
        self.clearance_px = float(config.get("clearance_px", 1.0))
        self.transition_margin_px = float(config.get("transition_margin_px", 40.0))
        self.edge_slow_margin_px = float(config.get("edge_slow_margin_px", 20.0))
        if self.box_scale < 1.0:
            raise ValueError("car_avoidance.box_scale must be >= 1.0")
        if self.clearance_px < 1.0:
            raise ValueError("car_avoidance.clearance_px must be >= 1.0")
        if self.transition_margin_px <= 0.0:
            raise ValueError("car_avoidance.transition_margin_px must be > 0")
        if self.edge_slow_margin_px < 0.0:
            raise ValueError("car_avoidance.edge_slow_margin_px must be >= 0")
        self.target_selector = target_selector

    def plan(
        self,
        objects: Sequence[DetectedObject],
        centerline_points: Sequence[Tuple[float, float]],
        candidate_route_points: Sequence[Tuple[float, float]],
        candidate_target_roi: Point,
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        normal_target: TargetPointResult,
    ) -> CarAvoidanceResult:
        base_route = self._normalize_route(
            centerline_points,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        if not self.enabled:
            return self._inactive(base_route, normal_target, "disabled")

        zones = self._build_warning_zones(
            objects=objects,
            candidate_target_roi=candidate_target_roi,
            roi_rect=roi_rect,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        if not zones:
            return self._inactive(base_route, normal_target, "no car warning zone in ROI")

        candidate_route = self._normalize_route(
            candidate_route_points,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        relevant = [
            zone
            for zone in zones
            if self.polyline_intersects_rect(candidate_route, zone.bbox_roi)
            or self.polyline_intersects_rect(base_route, zone.bbox_roi)
        ]
        if not relevant:
            return self._inactive(
                base_route,
                normal_target,
                "all car warning zones clear",
                warning_zones=zones,
            )
        if not base_route:
            return self._stop(
                base_route,
                normal_target,
                relevant,
                "car warning zone blocks candidate route but no lane centerline is available",
            )

        shifted = base_route
        while True:
            shifted, feasible, failure_reason = self._constrain_route(
                base_route,
                relevant,
                roi_width=roi_width,
                roi_height=roi_height,
            )
            if not feasible:
                return self._stop(base_route, normal_target, relevant, failure_reason)

            newly_relevant = [
                zone
                for zone in zones
                if zone not in relevant
                and self.polyline_intersects_rect(shifted, zone.bbox_roi)
            ]
            if not newly_relevant:
                break
            relevant.extend(newly_relevant)

        colliding = [
            zone
            for zone in zones
            if self.polyline_intersects_rect(shifted, zone.bbox_roi)
        ]
        if colliding:
            return self._stop(
                base_route,
                normal_target,
                relevant,
                "final segment-to-warning-zone validation failed",
            )

        target = self.target_selector.select(
            centerline_points=shifted,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=lane_confidence,
        )
        max_x = float(max(0, roi_width - 1))
        edge_limited = any(
            x <= self.edge_slow_margin_px
            or x >= max_x - self.edge_slow_margin_px
            for x, _ in shifted
        )
        mode = "CAR_AVOID_EDGE" if edge_limited else "CAR_AVOID"
        sides = ",".join(zone.avoid_side for zone in relevant)
        return CarAvoidanceResult(
            active=True,
            mode=mode,
            warning_zones=relevant,
            shifted_centerline_points=shifted,
            target_result=target,
            edge_limited=edge_limited,
            stop_required=False,
            reason=f"{mode}: cars={len(relevant)} sides={sides}",
        )

    def _build_warning_zones(
        self,
        objects: Sequence[DetectedObject],
        candidate_target_roi: Point,
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
    ) -> list[CarWarningZone]:
        roi_x1, roi_y1, _, _ = roi_rect
        target_x_frame = float(roi_x1) + float(candidate_target_roi[0])
        max_x = float(max(0, roi_width - 1))
        max_y = float(max(0, roi_height - 1))
        zones: list[CarWarningZone] = []
        for obj in objects:
            if obj.class_name.casefold() != "car":
                continue
            x1, y1, x2, y2 = [float(value) for value in obj.bbox_frame]
            if x2 <= x1 or y2 <= y1:
                continue
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            half_width = 0.5 * (x2 - x1) * self.box_scale
            half_height = 0.5 * (y2 - y1) * self.box_scale
            expanded_frame = (
                center_x - half_width,
                center_y - half_height,
                center_x + half_width,
                center_y + half_height,
            )
            raw_roi = (
                expanded_frame[0] - float(roi_x1),
                expanded_frame[1] - float(roi_y1),
                expanded_frame[2] - float(roi_x1),
                expanded_frame[3] - float(roi_y1),
            )
            if raw_roi[2] < 0.0 or raw_roi[0] > max_x:
                continue
            if raw_roi[3] < 0.0 or raw_roi[1] > max_y:
                continue
            clipped_roi = (
                clamp(raw_roi[0], 0.0, max_x),
                clamp(raw_roi[1], 0.0, max_y),
                clamp(raw_roi[2], 0.0, max_x),
                clamp(raw_roi[3], 0.0, max_y),
            )
            avoid_side = "left" if target_x_frame <= center_x else "right"
            zones.append(
                CarWarningZone(
                    bbox_frame=expanded_frame,
                    bbox_roi=clipped_roi,
                    car_center_x_frame=center_x,
                    avoid_side=avoid_side,
                    confidence=float(obj.confidence),
                )
            )
        return zones

    def _constrain_route(
        self,
        base_route: Sequence[Point],
        zones: Sequence[CarWarningZone],
        roi_width: int,
        roi_height: int,
    ) -> tuple[list[Point], bool, str]:
        anchor_ys = {float(y) for _, y in base_route}
        max_y = float(max(0, roi_height - 1))
        for zone in zones:
            _, top, _, bottom = zone.bbox_roi
            anchor_ys.update(
                {
                    clamp(top - self.transition_margin_px, 0.0, max_y),
                    clamp(top, 0.0, max_y),
                    clamp(bottom, 0.0, max_y),
                    clamp(bottom + self.transition_margin_px, 0.0, max_y),
                }
            )

        route: list[Point] = []
        max_x = float(max(0, roi_width - 1))
        for y in sorted(anchor_ys, reverse=True):
            base_x = self._interpolate_x(base_route, y)
            lower = 0.0
            upper = max_x
            for zone in zones:
                left, top, right, bottom = zone.bbox_roi
                weight = self._influence_weight(y, top, bottom)
                if weight <= 0.0:
                    continue
                if zone.avoid_side == "left":
                    safe_x = left - self.clearance_px
                    desired = base_x + weight * (min(base_x, safe_x) - base_x)
                    upper = min(upper, desired)
                else:
                    safe_x = right + self.clearance_px
                    desired = base_x + weight * (max(base_x, safe_x) - base_x)
                    lower = max(lower, desired)
            if lower > upper + 1e-6:
                return (
                    list(base_route),
                    False,
                    f"no feasible x at roi_y={y:.1f}: lower={lower:.1f} upper={upper:.1f}",
                )
            route.append((float(clamp(base_x, lower, upper)), float(y)))
        return self._deduplicate_rows(route), True, ""

    def _influence_weight(self, y: float, top: float, bottom: float) -> float:
        if top <= y <= bottom:
            return 1.0
        distance = top - y if y < top else y - bottom
        if distance >= self.transition_margin_px:
            return 0.0
        ratio = 1.0 - distance / self.transition_margin_px
        ratio = clamp(ratio, 0.0, 1.0)
        return ratio * ratio * (3.0 - 2.0 * ratio)

    def _normalize_route(
        self,
        points: Sequence[Tuple[float, float]],
        roi_width: int,
        roi_height: int,
    ) -> list[Point]:
        max_x = float(max(0, roi_width - 1))
        max_y = float(max(0, roi_height - 1))
        normalized = [
            (clamp(float(x), 0.0, max_x), clamp(float(y), 0.0, max_y))
            for x, y in points
        ]
        return self._deduplicate_rows(normalized)

    @staticmethod
    def _deduplicate_rows(points: Sequence[Point]) -> list[Point]:
        result: list[Point] = []
        seen: set[float] = set()
        for x, y in sorted(points, key=lambda point: point[1], reverse=True):
            key = round(float(y), 6)
            if key in seen:
                continue
            seen.add(key)
            result.append((float(x), float(y)))
        return result

    @staticmethod
    def _interpolate_x(points: Sequence[Point], target_y: float) -> float:
        ordered = sorted(points, key=lambda point: point[1], reverse=True)
        if target_y >= ordered[0][1]:
            return float(ordered[0][0])
        if target_y <= ordered[-1][1]:
            return float(ordered[-1][0])
        for (x1, y1), (x2, y2) in zip(ordered, ordered[1:]):
            if y1 >= target_y >= y2:
                if y1 == y2:
                    return float(0.5 * (x1 + x2))
                ratio = (target_y - y1) / (y2 - y1)
                return float(x1 + (x2 - x1) * ratio)
        return float(ordered[-1][0])

    @classmethod
    def polyline_intersects_rect(cls, points: Sequence[Point], rect: BBox) -> bool:
        if not points:
            return False
        if any(cls._point_in_rect(point, rect) for point in points):
            return True
        return any(
            cls._segment_intersects_rect(start, end, rect)
            for start, end in zip(points, points[1:])
        )

    @staticmethod
    def _point_in_rect(point: Point, rect: BBox) -> bool:
        x, y = point
        left, top, right, bottom = rect
        return left <= x <= right and top <= y <= bottom

    @classmethod
    def _segment_intersects_rect(cls, start: Point, end: Point, rect: BBox) -> bool:
        left, top, right, bottom = rect
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        t_min = 0.0
        t_max = 1.0
        for p, q in (
            (-dx, x1 - left),
            (dx, right - x1),
            (-dy, y1 - top),
            (dy, bottom - y1),
        ):
            if abs(p) < 1e-12:
                if q < 0.0:
                    return False
                continue
            ratio = q / p
            if p < 0.0:
                t_min = max(t_min, ratio)
            else:
                t_max = min(t_max, ratio)
            if t_min > t_max:
                return False
        return True

    @staticmethod
    def _inactive(
        base_route: list[Point],
        normal_target: TargetPointResult,
        reason: str,
        warning_zones: Sequence[CarWarningZone] = (),
    ) -> CarAvoidanceResult:
        return CarAvoidanceResult(
            active=False,
            mode="LANE_FOLLOW",
            warning_zones=list(warning_zones),
            shifted_centerline_points=base_route,
            target_result=normal_target,
            edge_limited=False,
            stop_required=False,
            reason=reason,
        )

    @staticmethod
    def _stop(
        _base_route: list[Point],
        normal_target: TargetPointResult,
        zones: list[CarWarningZone],
        reason: str,
    ) -> CarAvoidanceResult:
        return CarAvoidanceResult(
            active=True,
            mode="CAR_AVOID_STOP",
            warning_zones=list(zones),
            shifted_centerline_points=[],
            target_result=normal_target,
            edge_limited=False,
            stop_required=True,
            reason=reason,
        )
