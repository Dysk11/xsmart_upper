"""Lightweight car avoidance using selected track boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from core.lane.detector import LaneBoundaryRow
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
    """Final boundary-following route and control state for car avoidance."""

    active: bool
    mode: str
    warning_zones: list[CarWarningZone]
    shifted_centerline_points: list[Point]
    target_result: TargetPointResult
    edge_limited: bool
    stop_required: bool
    reason: str
    boundary_route_points: list[Point]
    recovery_progress: float


class CarAvoidancePlanner:
    """Follow a selected track boundary with spatial and temporal smoothstep."""

    def __init__(
        self,
        config: Dict[str, Any],
        target_selector: TargetSelector,
        max_boundary_gap_rows: int = 12,
    ) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.box_scale = float(config.get("box_scale", 1.5))
        self.transition_margin_px = float(config.get("transition_margin_px", 40.0))
        self.edge_slow_margin_px = float(config.get("edge_slow_margin_px", 20.0))
        self.release_duration_s = float(config.get("release_duration_s", 1.0))
        self.max_boundary_gap_rows = int(max_boundary_gap_rows)
        if self.box_scale < 1.0:
            raise ValueError("car_avoidance.box_scale must be >= 1.0")
        if self.transition_margin_px <= 0.0:
            raise ValueError("car_avoidance.transition_margin_px must be > 0")
        if self.edge_slow_margin_px < 0.0:
            raise ValueError("car_avoidance.edge_slow_margin_px must be >= 0")
        if self.release_duration_s <= 0.0:
            raise ValueError("car_avoidance.release_duration_s must be > 0")
        if self.max_boundary_gap_rows < 0:
            raise ValueError("lane_geometry.boundary.max_single_side_gap_rows must be >= 0")
        self.target_selector = target_selector

        self._last_detection_result_id: int | None = None
        self._last_live_zones: list[CarWarningZone] = []
        self._last_active_route: list[Point] = []
        self._last_active_base_route: list[Point] = []
        self._last_boundary_route: list[Point] = []
        self._release_start_monotonic: float | None = None
        self._recovery_offsets: list[Point] = []

    def plan(
        self,
        objects: Sequence[DetectedObject],
        centerline_points: Sequence[Tuple[float, float]],
        candidate_route_points: Sequence[Tuple[float, float]],
        candidate_target_roi: Point,
        track_boundary_rows: Sequence[LaneBoundaryRow],
        detection_result_id: int,
        now_monotonic: float,
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        normal_target: TargetPointResult,
    ) -> CarAvoidanceResult:
        del candidate_target_roi  # Car side is intentionally based on the lane centerline.
        now = float(now_monotonic)
        base_route = self._normalize_route(
            centerline_points,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        if not self.enabled:
            self._clear_state()
            return self._inactive(base_route, normal_target, "disabled")

        detection_id = int(detection_result_id)
        new_detection = detection_id != self._last_detection_result_id
        zones = self._build_warning_zones(
            objects=objects,
            base_route=base_route,
            roi_rect=roi_rect,
            roi_width=roi_width,
            roi_height=roi_height,
        )
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

        if new_detection:
            self._last_detection_result_id = detection_id
        elif (
            not relevant
            and self._release_start_monotonic is None
            and self._last_active_route
        ):
            # Reusing a cached detector result must not confirm that the car vanished.
            relevant = list(self._last_live_zones)

        if relevant:
            self._release_start_monotonic = None
            self._recovery_offsets = []
            return self._plan_live_avoidance(
                base_route=base_route,
                track_boundary_rows=track_boundary_rows,
                relevant=relevant,
                all_zones=zones or relevant,
                roi_width=roi_width,
                roi_height=roi_height,
                lane_confidence=lane_confidence,
                normal_target=normal_target,
            )

        if new_detection and self._last_active_route and self._release_start_monotonic is None:
            self._start_recovery(now)

        if self._release_start_monotonic is not None:
            return self._plan_recovery(
                base_route=base_route,
                now_monotonic=now,
                roi_width=roi_width,
                roi_height=roi_height,
                lane_confidence=lane_confidence,
                normal_target=normal_target,
            )

        return self._inactive(
            base_route,
            normal_target,
            "no relevant car warning zone",
            warning_zones=zones,
        )

    def _plan_live_avoidance(
        self,
        base_route: list[Point],
        track_boundary_rows: Sequence[LaneBoundaryRow],
        relevant: list[CarWarningZone],
        all_zones: Sequence[CarWarningZone],
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        normal_target: TargetPointResult,
    ) -> CarAvoidanceResult:
        if not base_route:
            self._clear_route_state()
            return self._stop(
                normal_target,
                relevant,
                "car warning zone blocks route but no lane centerline is available",
            )

        planning_zones = list(relevant)
        while True:
            shifted, boundary_route, feasible, failure_reason = self._build_boundary_route(
                base_route=base_route,
                boundary_rows=track_boundary_rows,
                zones=planning_zones,
                roi_width=roi_width,
                roi_height=roi_height,
            )
            if not feasible:
                self._clear_route_state()
                return self._stop(normal_target, planning_zones, failure_reason)
            newly_relevant = [
                zone
                for zone in all_zones
                if zone not in planning_zones
                and self.polyline_intersects_rect(shifted, zone.bbox_roi)
            ]
            if not newly_relevant:
                break
            planning_zones.extend(newly_relevant)

        colliding = [
            zone
            for zone in all_zones
            if self.polyline_intersects_rect(shifted, zone.bbox_roi)
        ]
        if colliding:
            self._clear_route_state()
            return self._stop(
                normal_target,
                planning_zones,
                "final segment-to-warning-zone validation failed",
            )

        target = self.target_selector.select(
            centerline_points=shifted,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=lane_confidence,
        )
        edge_limited = self._is_edge_limited(shifted, roi_width)
        mode = "CAR_AVOID_EDGE" if edge_limited else "CAR_AVOID"
        sides = ",".join(zone.avoid_side for zone in planning_zones)

        self._last_live_zones = list(planning_zones)
        self._last_active_route = list(shifted)
        self._last_active_base_route = list(base_route)
        self._last_boundary_route = list(boundary_route)
        return CarAvoidanceResult(
            active=True,
            mode=mode,
            warning_zones=list(planning_zones),
            shifted_centerline_points=shifted,
            target_result=target,
            edge_limited=edge_limited,
            stop_required=False,
            reason=f"{mode}: cars={len(planning_zones)} sides={sides}",
            boundary_route_points=boundary_route,
            recovery_progress=0.0,
        )

    def _build_warning_zones(
        self,
        objects: Sequence[DetectedObject],
        base_route: Sequence[Point],
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
    ) -> list[CarWarningZone]:
        roi_x1, roi_y1, _, _ = roi_rect
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
            car_center_y_roi = center_y - float(roi_y1)
            lane_x_roi = (
                self._interpolate_x(base_route, car_center_y_roi)
                if base_route
                else center_x - float(roi_x1)
            )
            lane_x_frame = float(roi_x1) + lane_x_roi
            avoid_side = "left" if lane_x_frame <= center_x else "right"
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

    def _build_boundary_route(
        self,
        base_route: Sequence[Point],
        boundary_rows: Sequence[LaneBoundaryRow],
        zones: Sequence[CarWarningZone],
        roi_width: int,
        roi_height: int,
    ) -> tuple[list[Point], list[Point], bool, str]:
        max_x = float(max(0, roi_width - 1))
        max_y = float(max(0, roi_height - 1))
        anchor_ys = {float(y) for _, y in base_route}
        for zone in zones:
            _, top, _, bottom = zone.bbox_roi
            start = clamp(top - self.transition_margin_px, 0.0, max_y)
            end = clamp(bottom + self.transition_margin_px, 0.0, max_y)
            anchor_ys.update({start, top, bottom, end})
            anchor_ys.update(float(y) for y in range(int(start), int(end) + 1))

        boundary_cache: dict[tuple[int, str], float | None] = {}

        def boundary_x(y: float, side: str) -> float | None:
            cache_key = (int(round(y * 1000.0)), side)
            if cache_key not in boundary_cache:
                boundary_cache[cache_key] = self._interpolate_boundary_x(
                    boundary_rows,
                    target_y=y,
                    side=side,
                )
            return boundary_cache[cache_key]

        # Every pixel row inside a warning box must have a trustworthy boundary.
        for zone in zones:
            _, top, _, bottom = zone.bbox_roi
            core_ys = {top, bottom}
            core_ys.update(
                float(y)
                for y in range(
                    max(0, int(top)),
                    min(int(max_y), int(bottom)) + 1,
                )
            )
            for y in core_ys:
                if boundary_x(y, zone.avoid_side) is None:
                    return (
                        list(base_route),
                        [],
                        False,
                        f"missing {zone.avoid_side} track boundary at roi_y={y:.1f}",
                    )

        route: list[Point] = []
        selected_boundary: list[Point] = []
        for y in sorted(anchor_ys, reverse=True):
            base_x = self._interpolate_x(base_route, y)
            core_zones = [
                zone
                for zone in zones
                if zone.bbox_roi[1] <= y <= zone.bbox_roi[3]
            ]
            core_sides = {zone.avoid_side for zone in core_zones}
            if len(core_sides) > 1:
                return (
                    list(base_route),
                    [],
                    False,
                    f"overlapping cars require opposite boundaries at roi_y={y:.1f}",
                )

            if core_zones:
                side = core_zones[0].avoid_side
                selected_x = boundary_x(y, side)
                if selected_x is None:
                    return (
                        list(base_route),
                        [],
                        False,
                        f"missing {side} track boundary at roi_y={y:.1f}",
                    )
                route_x = selected_x
                selected_boundary.append((float(selected_x), float(y)))
            else:
                weighted_offsets: list[tuple[float, float]] = []
                for zone in zones:
                    _, top, _, bottom = zone.bbox_roi
                    weight = self._influence_weight(y, top, bottom)
                    if weight <= 0.0:
                        continue
                    selected_x = boundary_x(y, zone.avoid_side)
                    if selected_x is None:
                        edge_y = top if y < top else bottom
                        selected_x = boundary_x(edge_y, zone.avoid_side)
                    if selected_x is None:
                        continue
                    weighted_offsets.append((weight, selected_x - base_x))
                if weighted_offsets:
                    weight_sum = sum(weight for weight, _ in weighted_offsets)
                    combined_offset = sum(
                        weight * offset for weight, offset in weighted_offsets
                    ) / max(1.0, weight_sum)
                    route_x = base_x + combined_offset
                else:
                    route_x = base_x

            if route_x < 0.0 or route_x > max_x:
                return (
                    list(base_route),
                    [],
                    False,
                    f"selected track boundary leaves ROI at roi_y={y:.1f}",
                )
            route.append((float(route_x), float(y)))

        return (
            self._deduplicate_rows(route),
            self._deduplicate_rows(selected_boundary),
            True,
            "",
        )

    def _interpolate_boundary_x(
        self,
        rows: Sequence[LaneBoundaryRow],
        target_y: float,
        side: str,
    ) -> float | None:
        if side not in {"left", "right"}:
            return None
        samples = sorted(
            (
                (
                    float(row.left_x if side == "left" else row.right_x),
                    float(row.y),
                )
                for row in rows
                if (row.left_valid if side == "left" else row.right_valid)
            ),
            key=lambda point: point[1],
        )
        if not samples:
            return None

        for x, y in samples:
            if abs(y - target_y) <= 1e-6:
                return x

        lower: Point | None = None
        upper: Point | None = None
        for sample in samples:
            if sample[1] < target_y:
                lower = sample
                continue
            if sample[1] > target_y:
                upper = sample
                break
        if lower is None or upper is None:
            return None
        missing_rows = max(0, int(round(upper[1] - lower[1])) - 1)
        if missing_rows > self.max_boundary_gap_rows:
            return None
        ratio = (target_y - lower[1]) / (upper[1] - lower[1])
        return float(lower[0] + (upper[0] - lower[0]) * ratio)

    def _start_recovery(self, now_monotonic: float) -> None:
        anchor_ys = {
            float(y)
            for _x, y in self._last_active_route + self._last_active_base_route
        }
        self._recovery_offsets = [
            (
                self._interpolate_x(self._last_active_route, y)
                - self._interpolate_x(self._last_active_base_route, y),
                y,
            )
            for y in sorted(anchor_ys, reverse=True)
        ]
        self._release_start_monotonic = float(now_monotonic)

    def _plan_recovery(
        self,
        base_route: list[Point],
        now_monotonic: float,
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        normal_target: TargetPointResult,
    ) -> CarAvoidanceResult:
        assert self._release_start_monotonic is not None
        elapsed = max(0.0, now_monotonic - self._release_start_monotonic)
        progress = clamp(elapsed / self.release_duration_s, 0.0, 1.0)
        if progress >= 1.0:
            old_zones = list(self._last_live_zones)
            self._clear_route_state()
            return self._inactive(
                base_route,
                normal_target,
                "car avoidance recovery complete",
                warning_zones=old_zones,
            )
        if not base_route or not self._recovery_offsets:
            self._clear_route_state()
            return self._stop(
                normal_target,
                self._last_live_zones,
                "normal lane centerline unavailable during car avoidance recovery",
            )

        smooth_progress = self._smoothstep(progress)
        remaining_weight = 1.0 - smooth_progress
        anchor_ys = {
            float(y) for _x, y in base_route + self._recovery_offsets
        }
        max_x = float(max(0, roi_width - 1))
        route = [
            (
                clamp(
                    self._interpolate_x(base_route, y)
                    + remaining_weight
                    * self._interpolate_x(self._recovery_offsets, y),
                    0.0,
                    max_x,
                ),
                y,
            )
            for y in sorted(anchor_ys, reverse=True)
        ]
        route = self._deduplicate_rows(route)
        target = self.target_selector.select(
            centerline_points=route,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=lane_confidence,
        )
        edge_limited = self._is_edge_limited(route, roi_width)
        return CarAvoidanceResult(
            active=True,
            mode="CAR_AVOID_RECOVERY",
            warning_zones=list(self._last_live_zones),
            shifted_centerline_points=route,
            target_result=target,
            edge_limited=edge_limited,
            stop_required=False,
            reason=f"CAR_AVOID_RECOVERY: progress={progress:.3f}",
            boundary_route_points=list(self._last_boundary_route),
            recovery_progress=float(progress),
        )

    def _influence_weight(self, y: float, top: float, bottom: float) -> float:
        if top <= y <= bottom:
            return 1.0
        distance = top - y if y < top else y - bottom
        if distance >= self.transition_margin_px:
            return 0.0
        ratio = 1.0 - distance / self.transition_margin_px
        return self._smoothstep(clamp(ratio, 0.0, 1.0))

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    def _is_edge_limited(self, route: Sequence[Point], roi_width: int) -> bool:
        max_x = float(max(0, roi_width - 1))
        return any(
            x <= self.edge_slow_margin_px
            or x >= max_x - self.edge_slow_margin_px
            for x, _ in route
        )

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
        if not points:
            return 0.0
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

    def _clear_state(self) -> None:
        self._last_detection_result_id = None
        self._clear_route_state()

    def _clear_route_state(self) -> None:
        self._last_live_zones = []
        self._last_active_route = []
        self._last_active_base_route = []
        self._last_boundary_route = []
        self._release_start_monotonic = None
        self._recovery_offsets = []

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
            boundary_route_points=[],
            recovery_progress=1.0,
        )

    @staticmethod
    def _stop(
        normal_target: TargetPointResult,
        zones: Sequence[CarWarningZone],
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
            boundary_route_points=[],
            recovery_progress=0.0,
        )
