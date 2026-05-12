"""Offline target selection and local avoidance test with mock bboxes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.avoidance_target_planner import AvoidanceTargetPlanner, AvoidanceTargetResult
from core.blocking_analyzer import BlockingAnalyzer, BlockingAnalysisResult, DetectedObject, attach_roi_bboxes
from core.lane_detector import LaneDetector
from core.lane_tracker import LaneTracker
from core.preprocess import ImagePreprocessor, PreprocessResult
from core.target_selector import TargetPointResult, TargetSelector
from utils.image_utils import draw_centerline, ensure_bgr, stack_images


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_mock_bboxes(values: Sequence[str]) -> list[DetectedObject]:
    objects: list[DetectedObject] = []
    for value in values:
        parts = [item.strip() for item in value.split(",")]
        if len(parts) != 6:
            raise ValueError(f"Invalid --mock-bbox: {value}")
        class_name = parts[0]
        x1, y1, x2, y2 = [float(item) for item in parts[1:5]]
        confidence = float(parts[5])
        objects.append(
            DetectedObject(
                class_name=class_name,  # type: ignore[arg-type]
                confidence=confidence,
                bbox_frame=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            )
        )
    return objects


class OfflineAvoidanceTester:
    def __init__(self, config: dict, mock_objects: list[DetectedObject]) -> None:
        self.preprocessor = ImagePreprocessor(config.get("preprocess", {}))
        self.detector = LaneDetector(config.get("detector", {}))
        self.tracker = LaneTracker(config.get("tracker", {}))
        self.target_selector = TargetSelector(config.get("target_selector", {}))
        self.blocking_analyzer = BlockingAnalyzer(config.get("blocking_analyzer", {}))
        self.avoidance_planner = AvoidanceTargetPlanner(
            config.get("avoidance_target_planner", {}),
            target_selector=self.target_selector,
        )
        self.mock_objects = mock_objects

    def process_frame(self, frame):
        preprocess_result = self.preprocessor.process(frame)
        detection_result = self.detector.detect(preprocess_result.roi_frame)
        tracked_state = self.tracker.update(detection_result)
        roi_height, roi_width = preprocess_result.roi_frame.shape[:2]
        centerline_points = tracked_state.centerline_points or detection_result.centerline_points
        objects_with_roi = attach_roi_bboxes(
            objects=self.mock_objects,
            roi_rect=preprocess_result.roi_rect,
            roi_width=roi_width,
            roi_height=roi_height,
        )

        normal_target = self.target_selector.select(
            centerline_points=centerline_points,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=tracked_state.confidence,
            curvature=tracked_state.curvature,
        )
        blocking_result = self.blocking_analyzer.analyze(
            objects=objects_with_roi,
            centerline_points=centerline_points,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        avoidance_result = self.avoidance_planner.plan(
            centerline_points=centerline_points,
            normal_target=normal_target,
            blocking_result=blocking_result,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=tracked_state.confidence,
            curvature=tracked_state.curvature,
        )
        print_debug(normal_target, blocking_result, avoidance_result)
        return build_debug_canvas(
            preprocess_result=preprocess_result,
            normal_target=normal_target,
            blocking_result=blocking_result,
            avoidance_result=avoidance_result,
            mask=detection_result.mask,
            centerline_points=centerline_points,
            mock_objects=objects_with_roi,
        )


def print_debug(
    target: TargetPointResult,
    blocking: BlockingAnalysisResult,
    avoidance: AvoidanceTargetResult,
) -> None:
    print(
        "mode={mode} target=({tx:.1f},{ty:.1f}) lane_center_x={lc:.1f} "
        "score={score:.2f} avoid={side} bias={bias:.1f} final_error={err:.1f}".format(
            mode=avoidance.mode,
            tx=avoidance.target_point_roi[0],
            ty=avoidance.target_point_roi[1],
            lc=blocking.lane_center_x_at_obstacle,
            score=blocking.blocking_score,
            side=blocking.recommended_avoid_side,
            bias=avoidance.avoid_bias_px,
            err=avoidance.final_lateral_error_px,
        )
    )
    _ = target


def build_debug_canvas(
    preprocess_result: PreprocessResult,
    normal_target: TargetPointResult,
    blocking_result: BlockingAnalysisResult,
    avoidance_result: AvoidanceTargetResult,
    mask,
    centerline_points,
    mock_objects: Sequence[DetectedObject],
):
    x1, y1, _, _ = preprocess_result.roi_rect
    original = preprocess_result.resized_frame.copy()
    cv2.rectangle(
        original,
        (x1, y1),
        (preprocess_result.roi_rect[2], preprocess_result.roi_rect[3]),
        (0, 255, 255),
        2,
    )
    original = draw_centerline(original, centerline_points, color=(0, 255, 0), offset=(x1, y1))
    original = draw_centerline(
        original,
        avoidance_result.shifted_centerline_points,
        color=(255, 255, 0),
        offset=(x1, y1),
    )
    draw_point(original, normal_target.target_point_roi, (x1, y1), (0, 180, 255), "N")
    draw_point(original, avoidance_result.target_point_roi, (x1, y1), (0, 255, 255), "A")
    for obj in mock_objects:
        draw_object(original, obj, (x1, y1), (0, 165, 255))
    if blocking_result.blocking_object is not None:
        danger_left = int(round(blocking_result.danger_left + x1))
        danger_right = int(round(blocking_result.danger_right + x1))
        cv2.line(original, (danger_left, y1), (danger_left, preprocess_result.roi_rect[3]), (0, 255, 255), 1)
        cv2.line(original, (danger_right, y1), (danger_right, preprocess_result.roi_rect[3]), (0, 255, 255), 1)
        lane_x = int(round(blocking_result.lane_center_x_at_obstacle + x1))
        y_bottom = int(round(blocking_result.blocking_object.bbox_roi[3] + y1))
        cv2.circle(original, (lane_x, y_bottom), 6, (255, 0, 255), -1)
        cv2.putText(
            original,
            blocking_result.reason,
            (12, max(28, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    mask_panel = ensure_bgr(mask)
    mask_panel = draw_centerline(mask_panel, centerline_points, color=(0, 255, 255))
    return stack_images(
        [original, mask_panel],
        cols=2,
        cell_size=(max(320, original.shape[1] // 2), max(180, original.shape[0] // 2)),
    )


def draw_point(image, point, offset, color, label: str) -> None:
    x = int(round(point[0] + offset[0]))
    y = int(round(point[1] + offset[1]))
    cv2.circle(image, (x, y), 8, color, -1)
    cv2.putText(image, label, (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_object(image, obj: DetectedObject, offset, color) -> None:
    x1, y1, x2, y2 = obj.bbox_frame
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(
        image,
        f"{obj.class_name}:{obj.confidence:.2f}",
        (int(x1), int(y1) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    _ = offset


def run_image(tester: OfflineAvoidanceTester, image_path: Path) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    canvas = tester.process_frame(frame)
    cv2.imshow("Target And Avoidance Test", canvas)
    cv2.waitKey(0)


def run_video(tester: OfflineAvoidanceTester, video_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        canvas = tester.process_frame(frame)
        cv2.imshow("Target And Avoidance Test", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
    capture.release()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline target and avoidance tester")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument(
        "--mock-bbox",
        action="append",
        default=[],
        help='Mock resized_frame bbox, e.g. "vehicle,x1,y1,x2,y2,0.9"',
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.image and not args.video:
        raise SystemExit("Use --image or --video")
    config = load_config(Path(args.config))
    mock_objects = parse_mock_bboxes(args.mock_bbox)
    tester = OfflineAvoidanceTester(config=config, mock_objects=mock_objects)
    if args.image:
        run_image(tester, Path(args.image))
    else:
        run_video(tester, Path(args.video))
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
