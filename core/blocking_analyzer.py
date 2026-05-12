"""Analyze whether detected objects block the current lane corridor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
BBoxInt = Tuple[int, int, int, int]


@dataclass
class DetectedObject:
    """A vehicle/person detection box with frame and optional ROI coordinates."""

    class_name: str
    confidence: float
    bbox_frame: BBoxInt
    bbox_roi: Optional[BBoxInt] = None


@dataclass
class BlockingAnalysisResult:
    """Blocking analysis for the most relevant obstacle."""

    need_avoid: bool
    blocking_object: DetectedObject | None
    blocking_score: float
    obstacle_center_x: float
    lane_center_x_at_obstacle: float
    danger_left: float
    danger_right: float
    recommended_avoid_side: str
    too_close: bool
    reason: str


class BlockingAnalyzer:
    """Checks bbox overlap with a lane-centered danger corridor."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.corridor_half_width_px = float(config.get("corridor_half_width_px", 90.0))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.45))
        self.min_box_area = float(config.get("min_box_area", 600.0))
        self.near_y_ratio = float(config.get("near_y_ratio", 0.42))
        self.too_close_y_ratio = float(config.get("too_close_y_ratio", 0.82))
        self.blocking_score_threshold = float(config.get("blocking_score_threshold", 0.25))
        self.side_deadband_px = float(config.get("side_deadband_px", 25.0))

    def analyze(
        self,
        objects: Sequence[DetectedObject],
        centerline_points: Sequence[Tuple[float, float]],
        roi_width: int,
        roi_height: int,
    ) -> BlockingAnalysisResult:
        if not self.enabled:
            return self._empty("disabled")
        if not objects:
            return self._empty("no objects")
        if not centerline_points:
            return self._empty("no centerline")

        best: tuple[float, float, BlockingAnalysisResult] | None = None
        for detected_object in objects:
            if detected_object.bbox_roi is None:
                continue
            result = self._analyze_object(
                detected_object=detected_object,
                centerline_points=centerline_points,
                roi_width=roi_width,
                roi_height=roi_height,
            )
            if result is None:
                continue
            y2 = result.blocking_object.bbox_roi[3] if result.blocking_object else 0.0
            key = (result.blocking_score, y2)
            if best is None or key > (best[0], best[1]):
                best = (key[0], key[1], result)

        if best is None:
            return self._empty("no object overlaps lane corridor")
        return best[2]

    def _analyze_object(
        self,
        detected_object: DetectedObject,
        centerline_points: Sequence[Tuple[float, float]],
        roi_width: int,
        roi_height: int,
    ) -> BlockingAnalysisResult | None:
        if detected_object.bbox_roi is None:
            return None
        x1, y1, x2, y2 = self._normalize_bbox(detected_object.bbox_roi, roi_width, roi_height)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        if detected_object.confidence < self.confidence_threshold:
            return None
        if area < self.min_box_area:
            return None
        if y2 < float(roi_height) * self.near_y_ratio:
            return None

        obstacle_center_x = (x1 + x2) * 0.5
        lane_center_x = interpolate_centerline_x(centerline_points, y2)
        danger_left = lane_center_x - self.corridor_half_width_px
        danger_right = lane_center_x + self.corridor_half_width_px
        overlap = max(0.0, min(x2, danger_right) - max(x1, danger_left))
        blocking_score = overlap / max(1.0, width)
        if blocking_score < self.blocking_score_threshold:
            return None

        avoid_side = self._choose_avoid_side(obstacle_center_x, lane_center_x, roi_width)
        too_close = y2 >= float(roi_height) * self.too_close_y_ratio and blocking_score >= self.blocking_score_threshold
        normalized_object = DetectedObject(
            class_name=detected_object.class_name,
            confidence=float(detected_object.confidence),
            bbox_frame=detected_object.bbox_frame,
            bbox_roi=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
        )
        reason = (
            f"{detected_object.class_name} overlaps lane corridor at y={y2:.0f}, "
            f"score={blocking_score:.2f}, avoid {avoid_side}"
        )
        return BlockingAnalysisResult(
            need_avoid=True,
            blocking_object=normalized_object,
            blocking_score=float(blocking_score),
            obstacle_center_x=float(obstacle_center_x),
            lane_center_x_at_obstacle=float(lane_center_x),
            danger_left=float(danger_left),
            danger_right=float(danger_right),
            recommended_avoid_side=avoid_side,
            too_close=bool(too_close),
            reason=reason,
        )

    def _choose_avoid_side(self, obstacle_center_x: float, lane_center_x: float, roi_width: int) -> str:
        if obstacle_center_x < lane_center_x - self.side_deadband_px:
            return "right"
        if obstacle_center_x > lane_center_x + self.side_deadband_px:
            return "left"

        left_space = obstacle_center_x
        right_space = float(roi_width) - obstacle_center_x
        return "left" if left_space > right_space else "right"

    def _normalize_bbox(self, bbox: BBox, roi_width: int, roi_height: int) -> BBox:
        x1, y1, x2, y2 = [float(value) for value in bbox]
        x1 = max(0.0, min(float(roi_width - 1), x1))
        x2 = max(0.0, min(float(roi_width - 1), x2))
        y1 = max(0.0, min(float(roi_height - 1), y1))
        y2 = max(0.0, min(float(roi_height - 1), y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def _empty(self, reason: str) -> BlockingAnalysisResult:
        return BlockingAnalysisResult(
            need_avoid=False,
            blocking_object=None,
            blocking_score=0.0,
            obstacle_center_x=0.0,
            lane_center_x_at_obstacle=0.0,
            danger_left=0.0,
            danger_right=0.0,
            recommended_avoid_side="none",
            too_close=False,
            reason=reason,
        )


def interpolate_centerline_x(
    centerline_points: Sequence[Tuple[float, float]],
    target_y: float,
) -> float:
    """Interpolate lane center x at a given ROI y coordinate."""

    points = sorted([(float(x), float(y)) for x, y in centerline_points], key=lambda item: item[1], reverse=True)
    if not points:
        return 0.0
    if len(points) == 1:
        return points[0][0]
    if target_y >= points[0][1]:
        return points[0][0]
    if target_y <= points[-1][1]:
        return points[-1][0]

    for index in range(len(points) - 1):
        x1, y1 = points[index]
        x2, y2 = points[index + 1]
        if y1 >= target_y >= y2 and y1 != y2:
            ratio = (target_y - y1) / (y2 - y1)
            return float(x1 + (x2 - x1) * ratio)
    return points[-1][0]


def frame_bbox_to_roi_bbox(
    bbox_frame: BBoxInt,
    roi_rect: BBoxInt,
    roi_width: int,
    roi_height: int,
) -> Optional[BBoxInt]:
    """Clip a resized-frame bbox into ROI coordinates.

    Args:
        bbox_frame: (x1, y1, x2, y2) in resized_frame coordinates.
        roi_rect: (roi_x1, roi_y1, roi_x2, roi_y2) in resized_frame coordinates.
        roi_width: ROI width in pixels.
        roi_height: ROI height in pixels.

    Returns:
        (x1, y1, x2, y2) in ROI coordinates, or None if there is no intersection.
    """

    x1, y1, x2, y2 = [int(round(value)) for value in bbox_frame]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    roi_x1, roi_y1, roi_x2, roi_y2 = [int(round(value)) for value in roi_rect]
    inter_x1 = max(x1, roi_x1)
    inter_y1 = max(y1, roi_y1)
    inter_x2 = min(x2, roi_x2)
    inter_y2 = min(y2, roi_y2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return None

    out_x1 = max(0, min(roi_width - 1, inter_x1 - roi_x1))
    out_y1 = max(0, min(roi_height - 1, inter_y1 - roi_y1))
    out_x2 = max(0, min(roi_width, inter_x2 - roi_x1))
    out_y2 = max(0, min(roi_height, inter_y2 - roi_y1))

    if out_x1 >= out_x2 or out_y1 >= out_y2:
        return None
    return int(out_x1), int(out_y1), int(out_x2), int(out_y2)


def attach_roi_bboxes(
    objects: Sequence[DetectedObject],
    roi_rect: BBoxInt,
    roi_width: int,
    roi_height: int,
) -> list[DetectedObject]:
    """Attach clipped ROI bboxes to frame-coordinate detections."""

    converted: list[DetectedObject] = []
    for obj in objects:
        bbox_roi = frame_bbox_to_roi_bbox(
            bbox_frame=obj.bbox_frame,
            roi_rect=roi_rect,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        if bbox_roi is None:
            continue
        converted.append(
            DetectedObject(
                class_name=obj.class_name,
                confidence=obj.confidence,
                bbox_frame=obj.bbox_frame,
                bbox_roi=bbox_roi,
            )
        )
    return converted
