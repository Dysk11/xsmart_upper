"""调试画面可视化模块。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.object.blocking import DetectedObject
from core.object.pedestrian_safety import PedestrianSafetyResult
from core.planning.car_avoidance import CarAvoidanceResult
from core.planning.gold_target import GoldTargetResult
from core.planning.path_marker_target import PathMarkerTargetResult
from core.lane.detector import LaneDetectionResult
from core.lane.tracker import TrackedLaneState
from core.ocr.recognizer import OcrResult
from core.planning.high_level import ControlCommand
from core.planning.target_selector import TargetPointResult
from utils.image_utils import draw_centerline, draw_text_lines, wrap_text_lines


class Visualizer:
    """负责生成调试画面、显示窗口与保存视频截图。"""

    def __init__(self, config: dict[str, Any]) -> None:
        """读取可视化配置并初始化状态。

        输入:
            config: visualizer 对应配置字典。

        输出:
            无返回值。
        """

        self.show_window = bool(config.get("show_window", True))
        self.window_name = str(config.get("window_name", "X-SmartCar Upper"))
        self.debug_window_name = str(config.get("debug_window_name", "X-SmartCar Debug"))
        self.save_video = bool(config.get("save_video", False))
        self.record_without_ui = bool(config.get("record_without_ui", False))
        self.save_screenshot = bool(config.get("save_screenshot", True))
        self.save_dir = Path(str(config.get("save_dir", "outputs/visual")))
        self.video_name = str(config.get("video_name", "debug.mp4"))
        self.video_fourcc = str(config.get("video_fourcc", "mp4v"))
        self.video_fps = float(config.get("video_fps", 25.0))
        self.font_path = str(config.get("font_path", ""))
        self.font_size = int(config.get("font_size", 22))
        self.debug_panel_font_size = int(config.get("debug_panel_font_size", 18))
        self.mask_alpha = float(config.get("mask_alpha", 0.35))
        self.mask_color = tuple(int(value) for value in config.get("mask_color", [0, 180, 255]))

        self.video_writer: cv2.VideoWriter | None = None
        if self.save_video or self.save_screenshot:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        frame: np.ndarray,
        roi_rect: tuple[int, int, int, int],
        detection_result: LaneDetectionResult,
        tracked_state: TrackedLaneState,
        control_command: ControlCommand,
        fps_value: float,
        target_result: TargetPointResult | None = None,
        pedestrian_safety_result: PedestrianSafetyResult | None = None,
        detected_objects: list[DetectedObject] | None = None,
        gold_result: GoldTargetResult | None = None,
        path_marker_result: PathMarkerTargetResult | None = None,
        car_avoidance_result: CarAvoidanceResult | None = None,
        ocr_result: OcrResult | None = None,
        show_ocr_bbox: bool = True,
    ) -> bool:
        """生成一帧调试画面，并根据配置显示或保存。

        输入:
            frame: 原始相机帧。
            detection_result: 当前帧检测结果。
            tracked_state: 当前帧时序平滑状态。
            control_command: 当前帧高层控制量。
            fps_value: 当前主循环 FPS。

        输出:
            返回布尔值，True 表示继续运行，False 表示用户请求退出。
        """

        canvas = self._build_canvas(
            frame=frame,
            roi_rect=roi_rect,
            detection_result=detection_result,
            tracked_state=tracked_state,
            control_command=control_command,
            fps_value=fps_value,
            target_result=target_result,
            pedestrian_safety_result=pedestrian_safety_result,
            detected_objects=detected_objects,
            gold_result=gold_result,
            path_marker_result=path_marker_result,
            car_avoidance_result=car_avoidance_result,
            ocr_result=ocr_result,
            show_ocr_bbox=show_ocr_bbox,
        )

        if self.save_video:
            video_frame = frame if self.record_without_ui else canvas
            self._write_video(video_frame)

        if self.show_window:
            cv2.imshow(self.window_name, canvas)
            debug_panel = self._build_debug_panel(
                width=frame.shape[1],
                target_result=target_result,
                pedestrian_safety_result=pedestrian_safety_result,
                control_command=control_command,
                fps_value=fps_value,
                gold_result=gold_result,
                path_marker_result=path_marker_result,
                car_avoidance_result=car_avoidance_result,
                detection_result=detection_result,
                ocr_result=ocr_result,
            )
            cv2.imshow(self.debug_window_name, debug_panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return False
            if key in (ord("s"), ord("S")) and self.save_screenshot:
                self._save_screenshot(canvas)

        return True

    def close(self) -> None:
        """关闭窗口与视频写出器。

        输入:
            无。

        输出:
            无返回值。
        """

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.show_window:
            cv2.destroyAllWindows()

    def _build_canvas(
        self,
        frame: np.ndarray,
        roi_rect: tuple[int, int, int, int],
        detection_result: LaneDetectionResult,
        tracked_state: TrackedLaneState,
        control_command: ControlCommand,
        fps_value: float,
        target_result: TargetPointResult | None = None,
        pedestrian_safety_result: PedestrianSafetyResult | None = None,
        detected_objects: list[DetectedObject] | None = None,
        gold_result: GoldTargetResult | None = None,
        path_marker_result: PathMarkerTargetResult | None = None,
        car_avoidance_result: CarAvoidanceResult | None = None,
        ocr_result: OcrResult | None = None,
        show_ocr_bbox: bool = True,
    ) -> np.ndarray:
        """将原图、ROI、掩膜和状态文字合成为一张调试大图。

        输入:
            frame: 原始相机帧。
            detection_result: 当前帧检测结果。
            tracked_state: 当前帧时序平滑状态。
            control_command: 当前帧高层控制量。
            fps_value: 当前主循环 FPS。

        输出:
            返回用于显示或保存的拼接调试图像。
        """

        x1, y1, x2, y2 = roi_rect
        if car_avoidance_result is not None and car_avoidance_result.active:
            centerline_points = tracked_state.centerline_points
        else:
            centerline_points = (
                tracked_state.centerline_points or detection_result.centerline_points
            )

        original_panel = frame.copy()
        self._overlay_roi_mask(original_panel, detection_result.filtered_mask, (x1, y1, x2, y2))
        cv2.rectangle(original_panel, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.line(
            original_panel,
            (frame.shape[1] // 2, 0),
            (frame.shape[1] // 2, frame.shape[0] - 1),
            (60, 60, 255),
            1,
        )
        original_panel = draw_centerline(original_panel, centerline_points, color=(0, 255, 0), offset=(x1, y1))
        fork_lane = detection_result.fork_result
        original_panel = draw_centerline(
            original_panel, detection_result.left_boundary_points,
            color=(255, 80, 80), radius=1, thickness=1, offset=(x1, y1),
        )
        original_panel = draw_centerline(
            original_panel, detection_result.right_boundary_points,
            color=(80, 80, 255), radius=1, thickness=1, offset=(x1, y1),
        )
        if fork_lane.left_centerline_points:
            original_panel = draw_centerline(
                original_panel,
                fork_lane.left_centerline_points,
                color=(255, 0, 255),
                radius=2,
                thickness=1,
                offset=(x1, y1),
            )
        if fork_lane.right_centerline_points:
            original_panel = draw_centerline(
                original_panel,
                fork_lane.right_centerline_points,
                color=(0, 128, 255),
                radius=2,
                thickness=1,
                offset=(x1, y1),
            )
        for corner, color, label in (
            (fork_lane.left_corner, (255, 0, 255), "LF"),
            (fork_lane.right_corner, (0, 128, 255), "RF"),
        ):
            if corner is not None:
                point = (int(corner[0] + x1), int(corner[1] + y1))
                cv2.rectangle(
                    original_panel,
                    (point[0] - 6, point[1] - 6),
                    (point[0] + 6, point[1] + 6),
                    color,
                    2,
                )
                cv2.putText(
                    original_panel,
                    label,
                    (point[0] + 8, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                )
        avoidance_active = bool(
            car_avoidance_result is not None and car_avoidance_result.active
        )
        if path_marker_result is not None and path_marker_result.active and not avoidance_active:
            original_panel = draw_centerline(
                original_panel,
                path_marker_result.connected_centerline_points,
                color=(200, 0, 255),
                radius=3,
                thickness=2,
                offset=(x1, y1),
            )
            if path_marker_result.lower_anchor_roi is not None:
                self._draw_target_point(
                    original_panel,
                    path_marker_result.lower_anchor_roi,
                    offset=(x1, y1),
                    color=(255, 128, 0),
                    label="L",
                )
            if path_marker_result.upper_anchor_roi is not None:
                self._draw_target_point(
                    original_panel,
                    path_marker_result.upper_anchor_roi,
                    offset=(x1, y1),
                    color=(255, 128, 0),
                    label="U",
                )
        if target_result is not None:
            self._draw_target_point(
                original_panel,
                target_result.target_point_roi,
                offset=(x1, y1),
                color=(0, 180, 255),
                label="N",
            )
        if pedestrian_safety_result is not None:
            self._draw_pedestrian_regions(
                original_panel,
                pedestrian_safety_result,
            )
        if detected_objects:
            self._draw_detected_objects(original_panel, detected_objects)
        if (
            car_avoidance_result is not None
            and car_avoidance_result.warning_zones
        ):
            original_panel = self._draw_car_avoidance(
                original_panel,
                car_avoidance_result,
                roi_offset=(x1, y1),
            )
        if show_ocr_bbox and ocr_result is not None and ocr_result.source_bbox is not None:
            bx1, by1, bx2, by2 = ocr_result.source_bbox
            cv2.rectangle(original_panel, (bx1, by1), (bx2, by2), (255, 0, 255), 3)
        if gold_result is not None and gold_result.active and not avoidance_active:
            self._draw_target_point(
                original_panel,
                gold_result.target_point_roi,
                offset=(x1, y1),
                color=(0, 215, 255),
                label="G",
            )
        if path_marker_result is not None and path_marker_result.active and not avoidance_active:
            self._draw_target_point(
                original_panel,
                path_marker_result.target_point_roi,
                offset=(x1, y1),
                color=(200, 0, 255),
                label="P",
            )
        return original_panel

    def _overlay_roi_mask(
        self,
        image: np.ndarray,
        roi_mask: np.ndarray,
        roi_rect: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Blend a binary mask into the ROI without modifying pixels outside it."""

        x1, y1, x2, y2 = roi_rect
        if roi_mask.shape[:2] != (y2 - y1, x2 - x1):
            roi_mask = cv2.resize(roi_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        active = roi_mask > 0
        if np.any(active):
            roi_panel = image[y1:y2, x1:x2]
            color = np.asarray(self.mask_color, dtype=np.float32)
            roi_panel[active] = np.clip(
                roi_panel[active].astype(np.float32) * (1.0 - self.mask_alpha)
                + color * self.mask_alpha,
                0,
                255,
            ).astype(np.uint8)
        return image

    def _draw_target_point(
        self,
        image: np.ndarray,
        point: tuple[float, float],
        offset: tuple[int, int],
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        x = int(round(point[0] + offset[0]))
        y = int(round(point[1] + offset[1]))
        cv2.circle(image, (x, y), 8, color, -1)
        cv2.circle(image, (x, y), 12, (0, 0, 0), 2)
        cv2.putText(image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _draw_detected_objects(
        self,
        image: np.ndarray,
        objects: list[DetectedObject],
    ) -> None:
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox_frame
            is_gold = obj.class_name.casefold() in {"gold", "coin"}
            color = (0, 215, 255) if is_gold else (0, 165, 255)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{obj.class_name} {obj.confidence:.2f}"
            label_y = max(18, int(y1) - 6)
            cv2.putText(
                image,
                label,
                (int(x1), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_car_avoidance(
        self,
        image: np.ndarray,
        result: CarAvoidanceResult,
        roi_offset: tuple[int, int],
    ) -> np.ndarray:
        """Draw expanded warning boxes and the final constrained route."""

        output = image
        boundary_route = getattr(result, "boundary_route_points", [])
        if boundary_route:
            output = draw_centerline(
                output,
                boundary_route,
                color=(255, 0, 255),
                radius=1,
                thickness=2,
                offset=roi_offset,
            )
        if result.active:
            output = draw_centerline(
                output,
                result.shifted_centerline_points,
                color=(0, 255, 255),
                radius=2,
                thickness=3,
                offset=roi_offset,
            )
        color = (0, 0, 255) if result.stop_required else (0, 128, 255)
        for zone in result.warning_zones:
            x1, y1, x2, y2 = zone.bbox_frame
            top_left = (int(round(x1)), int(round(y1)))
            bottom_right = (int(round(x2)), int(round(y2)))
            cv2.rectangle(output, top_left, bottom_right, color, 2)
            cv2.putText(
                output,
                f"CAR {zone.avoid_side.upper()}",
                (top_left[0], max(18, top_left[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return output

    def _draw_pedestrian_regions(
        self,
        image: np.ndarray,
        result: PedestrianSafetyResult,
    ) -> None:
        center_left, y1, center_right, y2 = result.center_region_frame
        boundary_color = (0, 255, 255)
        top = int(round(y1))
        bottom = int(round(y2))
        left = int(round(center_left))
        right = int(round(center_right))
        cv2.line(image, (left, top), (left, bottom), boundary_color, 2)
        cv2.line(image, (right, top), (right, bottom), boundary_color, 2)

        label_y = max(18, top + 20)
        for label, label_x in (
            ("LEFT", max(2, left - 55)),
            ("CENTER", left + 5),
            ("RIGHT", right + 5),
        ):
            cv2.putText(
                image,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                boundary_color,
                2,
                cv2.LINE_AA,
            )

        if result.frozen_target_x_frame is not None:
            target_x = int(round(result.frozen_target_x_frame))
            target_color = (0, 0, 255) if result.stop_required else (255, 255, 0)
            cv2.line(
                image,
                (target_x, top),
                (target_x, bottom),
                target_color,
                3,
            )
            cv2.putText(
                image,
                f"TARGET {result.target_region.upper()}",
                (target_x + 5, min(bottom - 5, label_y + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                target_color,
                2,
                cv2.LINE_AA,
            )

        if result.tracked_center_frame is not None:
            tracked_x, tracked_y = result.tracked_center_frame
            cv2.circle(
                image,
                (int(round(tracked_x)), int(round(tracked_y))),
                7,
                (0, 0, 255),
                -1,
            )

    def _build_debug_panel(
        self,
        width: int,
        target_result: TargetPointResult | None,
        pedestrian_safety_result: PedestrianSafetyResult | None,
        control_command: ControlCommand,
        fps_value: float,
        gold_result: GoldTargetResult | None = None,
        path_marker_result: PathMarkerTargetResult | None = None,
        detection_result: LaneDetectionResult | None = None,
        ocr_result: OcrResult | None = None,
        car_avoidance_result: CarAvoidanceResult | None = None,
    ) -> np.ndarray:
        """Build an opaque, auto-sized two-column status panel."""

        padding = 12
        column_gap = 16
        line_height = self.debug_panel_font_size + 6
        content_width = max(1, int(width) - padding * 2)
        column_width = max(1, (content_width - column_gap) // 2)

        header_lines = wrap_text_lines(
            [
                "窗口1：原始画面",
                "绿线=原中心线 青线=最终路径 紫线=Go/Stop连接 N=普通 A=最终 G=coin P=路径目标",
                "洋红线=左岔中线 橙线=右岔中线 LF/RF=左右岔路拐点",
            ],
            content_width,
            self.font_path,
            self.debug_panel_font_size,
        )

        mode = getattr(control_command, "mode", "n/a")
        final_error = (
            f"{target_result.target_lateral_error_px:.1f}"
            if target_result is not None
            else "n/a"
        )
        steer_deg = getattr(control_command, "steer_deg", None)
        steer_text = f"{float(steer_deg):.2f}" if steer_deg is not None else "n/a"
        segmentation_status = getattr(detection_result, "segmentation_status", "n/a")
        segmentation_confidence = getattr(detection_result, "segmentation_confidence", None)
        segmentation_instance_count = int(
            getattr(detection_result, "segmentation_instance_count", 0)
        )
        track_text = (
            f"track: {segmentation_status} conf={float(segmentation_confidence):.2f} "
            f"instances={segmentation_instance_count}"
            if segmentation_confidence is not None
            else f"track: {segmentation_status} instances={segmentation_instance_count}"
        )
        target_point = getattr(target_result, "target_point_roi", None)
        target_text = (
            f"target: x={float(target_point[0]):.1f} y={float(target_point[1]):.1f}"
            if target_point is not None
            else "target: n/a"
        )
        lane_reason = getattr(target_result, "reason", "n/a")
        left_lines = wrap_text_lines(
            [
                "运行 / 控制",
                f"mode: {mode}  FPS: {fps_value:.1f}",
                track_text,
                f"final_error: {final_error}",
                f"steer_deg: {steer_text}",
                target_text,
                f"lane reason: {lane_reason}",
            ],
            column_width,
            self.font_path,
            self.debug_panel_font_size,
        )

        if pedestrian_safety_result is None:
            pedestrian_status = "pedestrian: unavailable"
            pedestrian_reason = "pedestrian reason: unavailable"
        else:
            frozen_target = (
                f"{pedestrian_safety_result.frozen_target_x_frame:.1f}"
                if pedestrian_safety_result.frozen_target_x_frame is not None
                else "n/a"
            )
            pedestrian_status = (
                f"pedestrian: latched={pedestrian_safety_result.latched} "
                f"armed={pedestrian_safety_result.armed} "
                f"humans={pedestrian_safety_result.human_count}"
            )
            pedestrian_tracking = (
                f"ped target: region={pedestrian_safety_result.target_region} "
                f"x={frozen_target} "
                f"cooldown={pedestrian_safety_result.cooldown_remaining_sec:.2f}s"
            )
            pedestrian_reason = f"pedestrian reason: {pedestrian_safety_result.reason}"
        gold_reason = gold_result.reason if gold_result is not None and gold_result.active else "no coin"
        path_marker_reason = (
            path_marker_result.reason
            if path_marker_result is not None and path_marker_result.active
            else "no Go/Stop path marker"
        )
        fork_summary = "fork: none"
        fork_reason = "reason: no fork"
        if detection_result is not None:
            fork_lane = detection_result.fork_result
            fork_summary = (
                f"fork: left={fork_lane.left_detected} right={fork_lane.right_detected} "
                f"confirm={fork_lane.confirm_frames} selected={fork_lane.selected_direction}"
            )
            fork_reason = f"reason: {fork_lane.reason}"
        right_source_lines = [
            "规划 / 识别",
            fork_summary,
            fork_reason,
            f"{path_marker_reason}",
            f"{gold_reason}",
            pedestrian_status,
            (
                pedestrian_tracking
                if pedestrian_safety_result is not None
                else "ped target: unavailable"
            ),
            pedestrian_reason,
        ]
        if car_avoidance_result is None:
            right_source_lines.append("car avoidance: unavailable")
        else:
            right_source_lines.extend(
                [
                    (
                        f"car avoidance: mode={car_avoidance_result.mode} "
                        f"cars={len(car_avoidance_result.warning_zones)} "
                        f"edge={car_avoidance_result.edge_limited} "
                        f"stop={car_avoidance_result.stop_required} "
                        f"recovery={getattr(car_avoidance_result, 'recovery_progress', 0.0):.2f}"
                    ),
                    f"car reason: {car_avoidance_result.reason}",
                ]
            )
        right_source_lines.extend(self._ocr_status_lines(ocr_result))
        right_lines = wrap_text_lines(
            right_source_lines,
            column_width,
            self.font_path,
            self.debug_panel_font_size,
        )

        header_height = len(header_lines) * line_height
        columns_top = padding + header_height + 10
        panel_height = columns_top + max(len(left_lines), len(right_lines)) * line_height + padding
        panel = np.full((panel_height, int(width), 3), (24, 24, 24), dtype=np.uint8)
        panel = draw_text_lines(
            panel,
            header_lines,
            origin=(padding, padding + self.debug_panel_font_size),
            line_height=line_height,
            background_alpha=0.0,
            font_path=self.font_path,
            font_size=self.debug_panel_font_size,
        )
        separator_y = padding + header_height + 3
        cv2.line(panel, (padding, separator_y), (int(width) - padding - 1, separator_y), (80, 80, 80), 1)
        panel = draw_text_lines(
            panel,
            left_lines,
            origin=(padding, columns_top + self.debug_panel_font_size),
            line_height=line_height,
            background_alpha=0.0,
            font_path=self.font_path,
            font_size=self.debug_panel_font_size,
        )
        panel = draw_text_lines(
            panel,
            right_lines,
            origin=(padding + column_width + column_gap, columns_top + self.debug_panel_font_size),
            line_height=line_height,
            background_alpha=0.0,
            font_path=self.font_path,
            font_size=self.debug_panel_font_size,
        )
        return panel

    @staticmethod
    def _ocr_status_lines(ocr_result: OcrResult | None) -> list[str]:
        if ocr_result is None or ocr_result.frame_id <= 0:
            return []
        status = "accepted" if ocr_result.event_id > 0 else ("error" if ocr_result.error else "candidate")
        text = ocr_result.text or "<empty>"
        return [
            f"OCR[{status}] conf={ocr_result.confidence:.3f} "
            f"{ocr_result.inference_ms:.1f}ms: {text}"
        ]

    def _write_video(self, video_frame: np.ndarray) -> None:
        """将选定的原始画面或调试画面写入视频文件。

        输入:
            video_frame: 当前待录制画面。

        输出:
            无返回值。
        """

        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self.video_fourcc)
            video_path = self.save_dir / self.video_name
            self.video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.video_fps,
                (video_frame.shape[1], video_frame.shape[0]),
            )
        self.video_writer.write(video_frame)

    def _save_screenshot(self, canvas: np.ndarray) -> None:
        """将当前调试画面保存为截图文件。

        输入:
            canvas: 当前帧调试大图。

        输出:
            无返回值，截图将写入 save_dir 目录。
        """

        file_name = f"screenshot_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(self.save_dir / file_name), canvas)
