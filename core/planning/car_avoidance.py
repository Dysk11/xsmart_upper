"""Lightweight car avoidance using perspective warning polygons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from core.object.blocking import DetectedObject
from core.planning.target_selector import TargetPointResult, TargetSelector
from utils.math_utils import clamp


Point = Tuple[float, float]
Polygon = Tuple[Point, ...]


@dataclass(frozen=True)
class CarWarningZone:
    """One perspective car warning polygon in frame and lane-ROI coordinates."""

    polygon_frame: Polygon
    polygon_roi: Polygon
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

    def __init__(
        self,
        config: Dict[str, Any],
        target_selector: TargetSelector,
        perspective_config: Dict[str, Any],
    ) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.clearance_px = float(config.get("clearance_px", 1.0))
        self.transition_margin_px = float(config.get("transition_margin_px", 40.0))
        self.edge_slow_margin_px = float(config.get("edge_slow_margin_px", 20.0))
        self.perspective_width_top_px = float(
            perspective_config.get("perspective_width_top_px", 30.0)
        )
        self.perspective_width_bottom_px = float(
            perspective_config.get("perspective_width_bottom_px", 60.0)
        )
        if self.perspective_width_top_px <= 0.0:
            raise ValueError(
                "centerline.perspective_width_top_px must be greater than zero"
            )
        if self.perspective_width_bottom_px <= 0.0:
            raise ValueError(
                "centerline.perspective_width_bottom_px must be greater than zero"
            )
        if self.perspective_width_bottom_px < self.perspective_width_top_px:
            raise ValueError(
                "centerline.perspective_width_bottom_px must be greater than or equal to "
                "centerline.perspective_width_top_px"
            )
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
            if self.polyline_intersects_polygon(candidate_route, zone.polygon_roi)
            or self.polyline_intersects_polygon(base_route, zone.polygon_roi)
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
                and self.polyline_intersects_polygon(shifted, zone.polygon_roi)
            ]
            if not newly_relevant:
                break
            relevant.extend(newly_relevant)

        colliding = [
            zone
            for zone in zones
            if self.polyline_intersects_polygon(shifted, zone.polygon_roi)
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
            box_roi = (
                x1 - float(roi_x1),
                y1 - float(roi_y1),
                x2 - float(roi_x1),
                y2 - float(roi_y1),
            )
            top_offset = 0.5 * self._perspective_width(box_roi[1], roi_height)
            bottom_offset = 0.5 * self._perspective_width(box_roi[3], roi_height)
            polygon_roi: Polygon = (
                (box_roi[0] - top_offset, box_roi[1] - top_offset),
                (box_roi[2] + top_offset, box_roi[1] - top_offset),
                (box_roi[2] + bottom_offset, box_roi[3] + bottom_offset),
                (box_roi[0] - bottom_offset, box_roi[3] + bottom_offset),
            )
            polygon_frame: Polygon = tuple(
                (point_x + float(roi_x1), point_y + float(roi_y1))
                for point_x, point_y in polygon_roi
            )
            polygon_xs = [point[0] for point in polygon_roi]
            polygon_ys = [point[1] for point in polygon_roi]
            if max(polygon_xs) < 0.0 or min(polygon_xs) > max_x:
                continue
            if max(polygon_ys) < 0.0 or min(polygon_ys) > max_y:
                continue
            avoid_side = "left" if target_x_frame <= center_x else "right"
            zones.append(
                CarWarningZone(
                    polygon_frame=polygon_frame,
                    polygon_roi=polygon_roi,
                    car_center_x_frame=center_x,
                    avoid_side=avoid_side,
                    confidence=float(obj.confidence),
                )
            )
        return zones

    def _perspective_width(self, y: float, roi_height: int) -> float:
        """Interpolate the shared lane width at one ROI row."""

        ratio = (
            0.0
            if roi_height <= 1
            else clamp(float(y) / float(roi_height - 1), 0.0, 1.0)
        )
        return self.perspective_width_top_px + (
            self.perspective_width_bottom_px - self.perspective_width_top_px
        ) * ratio

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
            top, bottom = self._polygon_vertical_bounds(zone.polygon_roi)
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
                top, bottom = self._polygon_vertical_bounds(zone.polygon_roi)
                weight = self._influence_weight(y, top, bottom)
                if weight <= 0.0:
                    continue
                left, right = self._polygon_horizontal_bounds(
                    zone.polygon_roi,
                    y,
                    clamp_y=True,
                )
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
    def polyline_intersects_polygon(
        cls,
        points: Sequence[Point],
        polygon: Sequence[Point],
    ) -> bool:
        if not points or len(polygon) < 3:
            return False
        if any(cls._point_in_convex_polygon(point, polygon) for point in points):
            return True
        polygon_edges = list(zip(polygon, (*polygon[1:], polygon[0])))
        return any(
            cls._segments_intersect(start, end, edge_start, edge_end)
            for start, end in zip(points, points[1:])
            for edge_start, edge_end in polygon_edges
        )

    @staticmethod
    def _point_in_convex_polygon(
        point: Point,
        polygon: Sequence[Point],
    ) -> bool:
        signs: list[float] = []
        px, py = point
        for start, end in zip(polygon, (*polygon[1:], polygon[0])):
            cross = (end[0] - start[0]) * (py - start[1]) - (
                end[1] - start[1]
            ) * (px - start[0])
            if abs(cross) > 1e-9:
                signs.append(cross)
        return not signs or all(value > 0.0 for value in signs) or all(
            value < 0.0 for value in signs
        )

    @classmethod
    def _segments_intersect(
        cls,
        first_start: Point,
        first_end: Point,
        second_start: Point,
        second_end: Point,
    ) -> bool:
        o1 = cls._orientation(first_start, first_end, second_start)
        o2 = cls._orientation(first_start, first_end, second_end)
        o3 = cls._orientation(second_start, second_end, first_start)
        o4 = cls._orientation(second_start, second_end, first_end)
        if o1 * o2 < -1e-9 and o3 * o4 < -1e-9:
            return True
        return (
            abs(o1) <= 1e-9
            and cls._point_on_segment(second_start, first_start, first_end)
        ) or (
            abs(o2) <= 1e-9
            and cls._point_on_segment(second_end, first_start, first_end)
        ) or (
            abs(o3) <= 1e-9
            and cls._point_on_segment(first_start, second_start, second_end)
        ) or (
            abs(o4) <= 1e-9
            and cls._point_on_segment(first_end, second_start, second_end)
        )

    @staticmethod
    def _orientation(start: Point, end: Point, point: Point) -> float:
        return (end[0] - start[0]) * (point[1] - start[1]) - (
            end[1] - start[1]
        ) * (point[0] - start[0])

    @staticmethod
    def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
        return (
            min(start[0], end[0]) - 1e-9
            <= point[0]
            <= max(start[0], end[0]) + 1e-9
            and min(start[1], end[1]) - 1e-9
            <= point[1]
            <= max(start[1], end[1]) + 1e-9
        )

    @staticmethod
    def _polygon_vertical_bounds(polygon: Sequence[Point]) -> tuple[float, float]:
        ys = [float(point[1]) for point in polygon]
        return min(ys), max(ys)

    @classmethod
    def _polygon_horizontal_bounds(
        cls,
        polygon: Sequence[Point],
        y: float,
        *,
        clamp_y: bool = False,
    ) -> tuple[float, float]:
        top, bottom = cls._polygon_vertical_bounds(polygon)
        sample_y = clamp(float(y), top, bottom) if clamp_y else float(y)
        intersections: list[float] = []
        for start, end in zip(polygon, (*polygon[1:], polygon[0])):
            x1, y1 = start
            x2, y2 = end
            if sample_y < min(y1, y2) - 1e-9 or sample_y > max(y1, y2) + 1e-9:
                continue
            if abs(y2 - y1) <= 1e-9:
                if abs(sample_y - y1) <= 1e-9:
                    intersections.extend((float(x1), float(x2)))
                continue
            ratio = (sample_y - y1) / (y2 - y1)
            intersections.append(float(x1 + (x2 - x1) * ratio))
        if not intersections:
            raise ValueError(f"polygon has no horizontal intersection at y={sample_y}")
        return min(intersections), max(intersections)

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
