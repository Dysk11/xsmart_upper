"""Stateful pedestrian danger-zone stop analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from core.object.blocking import DetectedObject


BBox = tuple[float, float, float, float]
BBoxInt = tuple[int, int, int, int]


@dataclass(frozen=True)
class PedestrianSafetyResult:
    """Current pedestrian safety state for planning and visualization."""

    stop_required: bool
    latched: bool
    danger_zone_frame: BBox
    human_count: int
    overlapping_count: int
    reason: str


class PedestrianSafetyAnalyzer:
    """Latch a stop while pedestrians overlap a fixed ROI-relative danger zone."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.min_box_area_px = float(config.get("min_box_area_px", 600.0))
        if self.min_box_area_px < 0.0:
            raise ValueError("pedestrian_safety.min_box_area_px must be non-negative")

        zone_config = config.get("danger_zone", {})
        self.left_ratio = float(zone_config.get("left_ratio", 0.30))
        self.right_ratio = float(zone_config.get("right_ratio", 0.70))
        self.top_ratio = float(zone_config.get("top_ratio", 0.00))
        self.bottom_ratio = float(zone_config.get("bottom_ratio", 1.00))
        self._validate_zone_ratios()
        self.latched = False

    def analyze(
        self,
        objects: Sequence[DetectedObject],
        roi_rect: BBoxInt,
    ) -> PedestrianSafetyResult:
        danger_zone = self._danger_zone_frame(roi_rect)
        if not self.enabled:
            self.latched = False
            return self._result(danger_zone, [], 0, "disabled")

        humans = [
            obj for obj in objects
            if obj.class_name.casefold() == "human"
        ]
        overlapping = [
            obj for obj in humans
            if self._has_positive_overlap(obj.bbox_frame, danger_zone)
        ]

        if not self.latched:
            triggering = [
                obj for obj in overlapping
                if self._bbox_area(obj.bbox_frame) >= self.min_box_area_px
            ]
            if triggering:
                self.latched = True
                largest_area = max(self._bbox_area(obj.bbox_frame) for obj in triggering)
                return self._result(
                    danger_zone,
                    humans,
                    len(overlapping),
                    (
                        "pedestrian entered danger zone; "
                        f"trigger_area={largest_area:.0f}px^2"
                    ),
                )
            return self._result(
                danger_zone,
                humans,
                len(overlapping),
                "no qualifying pedestrian overlaps danger zone",
            )

        if not humans:
            return self._result(
                danger_zone,
                humans,
                0,
                "latched; no human in latest detection result",
            )
        if overlapping:
            return self._result(
                danger_zone,
                humans,
                len(overlapping),
                "latched; pedestrian still overlaps danger zone",
            )

        self.latched = False
        return self._result(
            danger_zone,
            humans,
            0,
            "released; all detected pedestrians are clear of danger zone",
        )

    def _validate_zone_ratios(self) -> None:
        if not 0.0 <= self.left_ratio < self.right_ratio <= 1.0:
            raise ValueError(
                "pedestrian_safety.danger_zone requires "
                "0 <= left_ratio < right_ratio <= 1"
            )
        if not 0.0 <= self.top_ratio < self.bottom_ratio <= 1.0:
            raise ValueError(
                "pedestrian_safety.danger_zone requires "
                "0 <= top_ratio < bottom_ratio <= 1"
            )

    def _danger_zone_frame(self, roi_rect: BBoxInt) -> BBox:
        roi_x1, roi_y1, roi_x2, roi_y2 = [float(value) for value in roi_rect]
        roi_width = max(0.0, roi_x2 - roi_x1)
        roi_height = max(0.0, roi_y2 - roi_y1)
        return (
            roi_x1 + roi_width * self.left_ratio,
            roi_y1 + roi_height * self.top_ratio,
            roi_x1 + roi_width * self.right_ratio,
            roi_y1 + roi_height * self.bottom_ratio,
        )

    @staticmethod
    def _normalized_bbox(bbox: BBoxInt) -> BBox:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def _bbox_area(self, bbox: BBoxInt) -> float:
        x1, y1, x2, y2 = self._normalized_bbox(bbox)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _has_positive_overlap(self, bbox: BBoxInt, zone: BBox) -> bool:
        x1, y1, x2, y2 = self._normalized_bbox(bbox)
        zone_x1, zone_y1, zone_x2, zone_y2 = zone
        overlap_width = min(x2, zone_x2) - max(x1, zone_x1)
        overlap_height = min(y2, zone_y2) - max(y1, zone_y1)
        return overlap_width > 0.0 and overlap_height > 0.0

    def _result(
        self,
        danger_zone: BBox,
        humans: Sequence[DetectedObject],
        overlapping_count: int,
        reason: str,
    ) -> PedestrianSafetyResult:
        return PedestrianSafetyResult(
            stop_required=self.latched,
            latched=self.latched,
            danger_zone_frame=danger_zone,
            human_count=len(humans),
            overlapping_count=int(overlapping_count),
            reason=reason,
        )
