"""Low-complexity car avoidance using one locked track boundary."""

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
    """One original car detection box in frame and lane-ROI coordinates."""

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
    locked_side: str | None
    transition_phase: str
    transition_progress: float


class CarAvoidancePlanner:
    """Blend between the normal route and one episode-locked track boundary."""

    def __init__(
        self,
        config: Dict[str, Any],
        target_selector: TargetSelector,
        max_boundary_gap_rows: int = 12,
    ) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.entry_duration_s = float(config.get("entry_duration_s", 1.0))
        self.release_duration_s = float(config.get("release_duration_s", 1.0))
        self.edge_slow_margin_px = float(config.get("edge_slow_margin_px", 20.0))
        self.max_boundary_gap_rows = int(max_boundary_gap_rows)
        if self.entry_duration_s <= 0.0:
            raise ValueError("car_avoidance.entry_duration_s must be > 0")
        if self.release_duration_s <= 0.0:
            raise ValueError("car_avoidance.release_duration_s must be > 0")
        if self.edge_slow_margin_px < 0.0:
            raise ValueError("car_avoidance.edge_slow_margin_px must be >= 0")
        if self.max_boundary_gap_rows < 0:
            raise ValueError(
                "lane_geometry.boundary.max_single_side_gap_rows must be >= 0"
            )
        self.target_selector = target_selector

        self._last_detection_result_id: int | None = None
        self._last_detection_zones: list[CarWarningZone] = []
        self._last_live_zones: list[CarWarningZone] = []
        self._last_output_route: list[Point] = []
        self._last_boundary_route: list[Point] = []
        self._locked_side: str | None = None
        self._phase = "inactive"
        self._transition_start_monotonic: float | None = None
        self._transition_start_offsets: list[Point] = []

    def needs_track_boundary_rows(
        self,
        objects: Sequence[DetectedObject],
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
    ) -> bool:
        """Return whether this frame can enter live boundary-following avoidance."""

        if not self.enabled:
            return False
        return bool(
            self._build_original_zones(
                objects=objects,
                roi_rect=roi_rect,
                roi_width=roi_width,
                roi_height=roi_height,
            )
        )

    def plan(
        self,
        objects: Sequence[DetectedObject],
        centerline_points: Sequence[Tuple[float, float]],
        track_boundary_rows: Sequence[LaneBoundaryRow],
        detection_result_id: int,
        now_monotonic: float,
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        normal_target: TargetPointResult,
    ) -> CarAvoidanceResult:
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
        detected_zones = self._build_original_zones(
            objects=objects,
            roi_rect=roi_rect,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        if new_detection:
            self._last_detection_result_id = detection_id
            self._last_detection_zones = list(detected_zones)
        # Reusing an AI result must preserve whether that result contained a car.
        zones = list(self._last_detection_zones)

        if zones:
            if self._phase == "inactive":
                if not base_route:
                    return self._stop(
                        normal_target=normal_target,
                        zones=zones,
                        reason=(
                            "car box intersects ROI but no lane centerline is available"
                        ),
                        route=[],
                        boundary_route=[],
                        transition_phase="entry",
                        transition_progress=0.0,
                    )
                primary = self._select_primary_zone(zones)
                self._locked_side = self._choose_avoid_side(
                    primary,
                    base_route=base_route,
                    roi_x1=float(roi_rect[0]),
                    roi_y1=float(roi_rect[1]),
                )
                self._start_transition(
                    phase="entry",
                    now_monotonic=now,
                    base_route=base_route,
                    start_route=base_route,
                )
            elif self._phase == "recovery":
                # Re-enter from the route currently being commanded, without a jump.
                start_route = self._last_output_route or base_route
                self._start_transition(
                    phase="entry",
                    now_monotonic=now,
                    base_route=base_route,
                    start_route=start_route,
                )

            if self._locked_side is None:
                return self._stop(
                    normal_target=normal_target,
                    zones=zones,
                    reason="car avoidance side is unavailable",
                    route=[],
                    boundary_route=[],
                    transition_phase="entry",
                    transition_progress=0.0,
                )

            zones = [
                CarWarningZone(
                    bbox_frame=zone.bbox_frame,
                    bbox_roi=zone.bbox_roi,
                    car_center_x_frame=zone.car_center_x_frame,
                    avoid_side=self._locked_side,
                    confidence=zone.confidence,
                )
                for zone in zones
            ]
            self._last_live_zones = list(zones)
            return self._plan_live_avoidance(
                base_route=base_route,
                track_boundary_rows=track_boundary_rows,
                zones=zones,
                now_monotonic=now,
                roi_width=roi_width,
                roi_height=roi_height,
                lane_confidence=lane_confidence,
                normal_target=normal_target,
            )

        if (
            new_detection
            and self._phase in {"entry", "hold"}
            and self._transition_start_monotonic is not None
        ):
            self._start_transition(
                phase="recovery",
                now_monotonic=now,
                base_route=base_route,
                start_route=self._last_output_route or base_route,
            )

        if self._phase == "recovery":
            return self._plan_recovery(
                base_route=base_route,
                now_monotonic=now,
                roi_width=roi_width,
                roi_height=roi_height,
                lane_confidence=lane_confidence,
                normal_target=normal_target,
            )

        return self._inactive(base_route, normal_target, "no original car box in ROI")

    def _plan_live_avoidance(
        self,
        base_route: list[Point],
        track_boundary_rows: Sequence[LaneBoundaryRow],
        zones: list[CarWarningZone],
        now_monotonic: float,
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        normal_target: TargetPointResult,
    ) -> CarAvoidanceResult:
        if not base_route:
            return self._stop(
                normal_target=normal_target,
                zones=zones,
                reason="lane centerline unavailable during car avoidance",
                route=[],
                boundary_route=[],
                transition_phase=self._phase,
                transition_progress=0.0,
            )
        assert self._locked_side in {"left", "right"}

        boundary_route, failure_reason = self._build_boundary_route(
            base_route=base_route,
            boundary_rows=track_boundary_rows,
            side=self._locked_side,
            roi_height=roi_height,
        )
        if not boundary_route:
            return self._stop(
                normal_target=normal_target,
                zones=zones,
                reason=failure_reason,
                route=[],
                boundary_route=[],
                transition_phase=self._phase,
                transition_progress=0.0,
            )

        if self._phase == "entry":
            progress = self._transition_progress(
                now_monotonic,
                self.entry_duration_s,
            )
            weight = self._smoothstep(progress)
            route = self._blend_to_route(
                base_route=base_route,
                destination_route=boundary_route,
                start_offsets=self._transition_start_offsets,
                destination_weight=weight,
            )
            if progress >= 1.0:
                self._phase = "hold"
        else:
            progress = 1.0
            route = list(boundary_route)

        self._last_output_route = list(route)
        self._last_boundary_route = list(boundary_route)
        phase = "hold" if self._phase == "hold" else "entry"
        target = self.target_selector.select(
            centerline_points=route,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=lane_confidence,
        )
        colliding = [
            zone
            for zone in zones
            if self.polyline_intersects_rect(route, zone.bbox_roi)
        ]
        if colliding:
            return self._stop(
                normal_target=target,
                zones=zones,
                reason=(
                    "transition route intersects an original car box: "
                    f"cars={len(colliding)}"
                ),
                route=route,
                boundary_route=boundary_route,
                transition_phase=phase,
                transition_progress=progress,
            )

        edge_limited = self._is_edge_limited(route, roi_width)
        mode = "CAR_AVOID_EDGE" if edge_limited else "CAR_AVOID"
        return CarAvoidanceResult(
            active=True,
            mode=mode,
            warning_zones=list(zones),
            shifted_centerline_points=route,
            target_result=target,
            edge_limited=edge_limited,
            stop_required=False,
            reason=(
                f"{mode}: side={self._locked_side} phase={phase} "
                f"progress={progress:.3f} cars={len(zones)}"
            ),
            boundary_route_points=boundary_route,
            locked_side=self._locked_side,
            transition_phase=phase,
            transition_progress=float(progress),
        )

    def _plan_recovery(
        self,
        base_route: list[Point],
        now_monotonic: float,
        roi_width: int,
        roi_height: int,
        lane_confidence: float,
        normal_target: TargetPointResult,
    ) -> CarAvoidanceResult:
        if not base_route or not self._transition_start_offsets:
            return self._stop(
                normal_target=normal_target,
                zones=self._last_live_zones,
                reason="normal lane centerline unavailable during car avoidance recovery",
                route=[],
                boundary_route=self._last_boundary_route,
                transition_phase="recovery",
                transition_progress=0.0,
            )

        progress = self._transition_progress(
            now_monotonic,
            self.release_duration_s,
        )
        if progress >= 1.0:
            old_zones = list(self._last_live_zones)
            self._clear_route_state()
            return self._inactive(
                base_route,
                normal_target,
                "car avoidance recovery complete",
                warning_zones=old_zones,
            )

        remaining_weight = 1.0 - self._smoothstep(progress)
        max_x = float(max(0, roi_width - 1))
        offset_xs = self._aligned_x_values(
            self._transition_start_offsets,
            base_route,
        )
        route = [
            (
                clamp(
                    base_x + remaining_weight * offset_x,
                    0.0,
                    max_x,
                ),
                y,
            )
            for (base_x, y), offset_x in zip(base_route, offset_xs)
        ]
        self._last_output_route = list(route)
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
            locked_side=self._locked_side,
            transition_phase="recovery",
            transition_progress=float(progress),
        )

    def _build_original_zones(
        self,
        objects: Sequence[DetectedObject],
        roi_rect: tuple[int, int, int, int],
        roi_width: int,
        roi_height: int,
    ) -> list[CarWarningZone]:
        roi_x1, roi_y1, _, _ = roi_rect
        max_x = float(max(0, roi_width - 1))
        max_y = float(max(0, roi_height - 1))
        roi_extent_x = float(max(0, roi_width))
        roi_extent_y = float(max(0, roi_height))
        zones: list[CarWarningZone] = []
        for obj in objects:
            if obj.class_name.casefold() != "car":
                continue
            x1, y1, x2, y2 = [float(value) for value in obj.bbox_frame]
            if x2 <= x1 or y2 <= y1:
                continue
            raw_roi = (
                x1 - float(roi_x1),
                y1 - float(roi_y1),
                x2 - float(roi_x1),
                y2 - float(roi_y1),
            )
            # Inclusive comparisons make edge contact count as entering the ROI.
            if raw_roi[2] < 0.0 or raw_roi[0] > roi_extent_x:
                continue
            if raw_roi[3] < 0.0 or raw_roi[1] > roi_extent_y:
                continue
            clipped_roi = (
                clamp(raw_roi[0], 0.0, max_x),
                clamp(raw_roi[1], 0.0, max_y),
                clamp(raw_roi[2], 0.0, max_x),
                clamp(raw_roi[3], 0.0, max_y),
            )
            zones.append(
                CarWarningZone(
                    bbox_frame=(x1, y1, x2, y2),
                    bbox_roi=clipped_roi,
                    car_center_x_frame=0.5 * (x1 + x2),
                    avoid_side=self._locked_side or "unlocked",
                    confidence=float(obj.confidence),
                )
            )
        return zones

    @staticmethod
    def _select_primary_zone(zones: Sequence[CarWarningZone]) -> CarWarningZone:
        return max(
            zones,
            key=lambda zone: (
                zone.bbox_frame[3],
                zone.confidence,
            ),
        )

    def _choose_avoid_side(
        self,
        primary: CarWarningZone,
        base_route: Sequence[Point],
        roi_x1: float,
        roi_y1: float,
    ) -> str:
        car_center_y_roi = (
            0.5 * (primary.bbox_frame[1] + primary.bbox_frame[3])
            - roi_y1
        )
        lane_x_frame = roi_x1 + self._interpolate_x(base_route, car_center_y_roi)
        return "left" if lane_x_frame <= primary.car_center_x_frame else "right"

    def _build_boundary_route(
        self,
        base_route: Sequence[Point],
        boundary_rows: Sequence[LaneBoundaryRow],
        side: str,
        roi_height: int,
    ) -> tuple[list[Point], str]:
        dense_boundary = self._build_dense_boundary(
            boundary_rows,
            side=side,
            roi_height=roi_height,
        )
        route: list[Point] = []
        for _, y in base_route:
            boundary_x = self._sample_dense_boundary(dense_boundary, y)
            if boundary_x is None:
                return [], f"missing {side} track boundary at roi_y={y:.1f}"
            route.append((float(boundary_x), float(y)))
        return route, ""

    def _build_dense_boundary(
        self,
        rows: Sequence[LaneBoundaryRow],
        side: str,
        roi_height: int,
    ) -> list[float | None]:
        if side not in {"left", "right"} or roi_height <= 0:
            return []
        dense: list[float | None] = [None] * roi_height
        for row in rows:
            y = int(row.y)
            valid = row.left_valid if side == "left" else row.right_valid
            if not valid or y < 0 or y >= roi_height:
                continue
            dense[y] = float(row.left_x if side == "left" else row.right_x)

        valid_ys = [y for y, value in enumerate(dense) if value is not None]
        for lower_y, upper_y in zip(valid_ys, valid_ys[1:]):
            missing_rows = upper_y - lower_y - 1
            if missing_rows <= 0 or missing_rows > self.max_boundary_gap_rows:
                continue
            lower_x = dense[lower_y]
            upper_x = dense[upper_y]
            assert lower_x is not None and upper_x is not None
            span = float(upper_y - lower_y)
            for y in range(lower_y + 1, upper_y):
                ratio = float(y - lower_y) / span
                dense[y] = lower_x + (upper_x - lower_x) * ratio
        return dense

    @staticmethod
    def _sample_dense_boundary(
        dense: Sequence[float | None],
        target_y: float,
    ) -> float | None:
        if not dense or target_y < 0.0 or target_y > float(len(dense) - 1):
            return None
        lower_y = int(target_y)
        upper_y = min(lower_y + 1, len(dense) - 1)
        lower_x = dense[lower_y]
        if abs(target_y - float(lower_y)) <= 1e-6:
            return float(lower_x) if lower_x is not None else None
        upper_x = dense[upper_y]
        if lower_x is None or upper_x is None:
            return None
        if lower_y == upper_y:
            return float(lower_x)
        ratio = target_y - float(lower_y)
        return float(lower_x + (upper_x - lower_x) * ratio)

    def _start_transition(
        self,
        phase: str,
        now_monotonic: float,
        base_route: Sequence[Point],
        start_route: Sequence[Point],
    ) -> None:
        self._phase = phase
        self._transition_start_monotonic = float(now_monotonic)
        start_xs = self._aligned_x_values(start_route, base_route)
        self._transition_start_offsets = [
            (start_x - base_x, y)
            for (base_x, y), start_x in zip(base_route, start_xs)
        ]

    def _transition_progress(
        self,
        now_monotonic: float,
        duration_s: float,
    ) -> float:
        if self._transition_start_monotonic is None:
            return 0.0
        elapsed = max(0.0, now_monotonic - self._transition_start_monotonic)
        return clamp(elapsed / duration_s, 0.0, 1.0)

    def _blend_to_route(
        self,
        base_route: Sequence[Point],
        destination_route: Sequence[Point],
        start_offsets: Sequence[Point],
        destination_weight: float,
    ) -> list[Point]:
        offset_xs = self._aligned_x_values(start_offsets, base_route)
        destination_xs = self._aligned_x_values(destination_route, base_route)
        return [
            (
                (base_x + offset_x)
                + destination_weight
                * (destination_x - (base_x + offset_x)),
                y,
            )
            for (base_x, y), offset_x, destination_x in zip(
                base_route,
                offset_xs,
                destination_xs,
            )
        ]

    @classmethod
    def _aligned_x_values(
        cls,
        points: Sequence[Point],
        reference_route: Sequence[Point],
    ) -> list[float]:
        if len(points) == len(reference_route) and all(
            abs(point[1] - reference[1]) <= 1e-6
            for point, reference in zip(points, reference_route)
        ):
            return [float(x) for x, _ in points]
        ordered = sorted(points, key=lambda point: point[1], reverse=True)
        return [
            cls._interpolate_ordered_x(ordered, y)
            for _, y in reference_route
        ]

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
        ordered = sorted(points, key=lambda point: point[1], reverse=True)
        return CarAvoidancePlanner._interpolate_ordered_x(ordered, target_y)

    @staticmethod
    def _interpolate_ordered_x(
        ordered: Sequence[Point],
        target_y: float,
    ) -> float:
        if not ordered:
            return 0.0
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
        self._last_detection_zones = []
        self._clear_route_state()

    def _clear_route_state(self) -> None:
        self._last_live_zones = []
        self._last_output_route = []
        self._last_boundary_route = []
        self._locked_side = None
        self._phase = "inactive"
        self._transition_start_monotonic = None
        self._transition_start_offsets = []

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
            locked_side=None,
            transition_phase="inactive",
            transition_progress=1.0,
        )

    def _stop(
        self,
        normal_target: TargetPointResult,
        zones: Sequence[CarWarningZone],
        reason: str,
        route: Sequence[Point],
        boundary_route: Sequence[Point],
        transition_phase: str,
        transition_progress: float,
    ) -> CarAvoidanceResult:
        return CarAvoidanceResult(
            active=True,
            mode="CAR_AVOID_STOP",
            warning_zones=list(zones),
            shifted_centerline_points=list(route),
            target_result=normal_target,
            edge_limited=False,
            stop_required=True,
            reason=reason,
            boundary_route_points=list(boundary_route),
            locked_side=self._locked_side,
            transition_phase=transition_phase,
            transition_progress=float(transition_progress),
        )
