"""调试画面可视化模块。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.avoidance_target_planner import AvoidanceTargetResult
from core.blocking_analyzer import BlockingAnalysisResult, DetectedObject
from core.gold_target_planner import GoldTargetResult
from core.lane_detector import LaneDetectionResult
from core.lane_tracker import TrackedLaneState
from core.planner import ControlCommand
from core.preprocess import PreprocessResult
from core.target_selector import TargetPointResult
from utils.image_utils import draw_centerline, draw_text_lines, ensure_bgr, stack_images


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
        self.save_video = bool(config.get("save_video", False))
        self.save_screenshot = bool(config.get("save_screenshot", True))
        self.save_dir = Path(str(config.get("save_dir", "outputs/visual")))
        self.video_name = str(config.get("video_name", "debug.mp4"))
        self.video_fourcc = str(config.get("video_fourcc", "mp4v"))
        self.video_fps = float(config.get("video_fps", 25.0))
        self.font_path = str(config.get("font_path", ""))
        self.font_size = int(config.get("font_size", 22))

        self.video_writer: cv2.VideoWriter | None = None
        if self.save_video or self.save_screenshot:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        preprocess_result: PreprocessResult,
        detection_result: LaneDetectionResult,
        tracked_state: TrackedLaneState,
        control_command: ControlCommand,
        fps_value: float,
        target_result: TargetPointResult | None = None,
        blocking_result: BlockingAnalysisResult | None = None,
        avoidance_result: AvoidanceTargetResult | None = None,
        detected_objects: list[DetectedObject] | None = None,
        gold_result: GoldTargetResult | None = None,
    ) -> bool:
        """生成一帧调试画面，并根据配置显示或保存。

        输入:
            preprocess_result: 预处理阶段输出结果。
            detection_result: 当前帧检测结果。
            tracked_state: 当前帧时序平滑状态。
            control_command: 当前帧高层控制量。
            fps_value: 当前主循环 FPS。

        输出:
            返回布尔值，True 表示继续运行，False 表示用户请求退出。
        """

        canvas = self._build_canvas(
            preprocess_result=preprocess_result,
            detection_result=detection_result,
            tracked_state=tracked_state,
            control_command=control_command,
            fps_value=fps_value,
            target_result=target_result,
            blocking_result=blocking_result,
            avoidance_result=avoidance_result,
            detected_objects=detected_objects,
            gold_result=gold_result,
        )

        if self.save_video:
            self._write_video(canvas)

        if self.show_window:
            cv2.imshow(self.window_name, canvas)
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
        preprocess_result: PreprocessResult,
        detection_result: LaneDetectionResult,
        tracked_state: TrackedLaneState,
        control_command: ControlCommand,
        fps_value: float,
        target_result: TargetPointResult | None = None,
        blocking_result: BlockingAnalysisResult | None = None,
        avoidance_result: AvoidanceTargetResult | None = None,
        detected_objects: list[DetectedObject] | None = None,
        gold_result: GoldTargetResult | None = None,
    ) -> np.ndarray:
        """将原图、ROI、掩膜和状态文字合成为一张调试大图。

        输入:
            preprocess_result: 预处理阶段输出结果。
            detection_result: 当前帧检测结果。
            tracked_state: 当前帧时序平滑状态。
            control_command: 当前帧高层控制量。
            fps_value: 当前主循环 FPS。

        输出:
            返回用于显示或保存的拼接调试图像。
        """

        x1, y1, x2, y2 = preprocess_result.roi_rect
        centerline_points = tracked_state.centerline_points or detection_result.centerline_points

        original_panel = preprocess_result.resized_frame.copy()
        cv2.rectangle(original_panel, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.line(
            original_panel,
            (preprocess_result.resized_frame.shape[1] // 2, 0),
            (preprocess_result.resized_frame.shape[1] // 2, preprocess_result.resized_frame.shape[0] - 1),
            (60, 60, 255),
            1,
        )
        original_panel = draw_centerline(original_panel, centerline_points, color=(0, 255, 0), offset=(x1, y1))
        if avoidance_result is not None:
            original_panel = draw_centerline(
                original_panel,
                avoidance_result.shifted_centerline_points,
                color=(255, 255, 0),
                offset=(x1, y1),
            )
        if target_result is not None:
            self._draw_target_point(
                original_panel,
                target_result.target_point_roi,
                offset=(x1, y1),
                color=(0, 180, 255),
                label="N",
            )
        if avoidance_result is not None:
            self._draw_target_point(
                original_panel,
                avoidance_result.target_point_roi,
                offset=(x1, y1),
                color=(0, 255, 255),
                label="A",
            )
        if detected_objects:
            self._draw_detected_objects(original_panel, detected_objects)
        if gold_result is not None and gold_result.active:
            self._draw_target_point(
                original_panel,
                gold_result.target_point_roi,
                offset=(x1, y1),
                color=(0, 215, 255),
                label="G",
            )
        if blocking_result is not None and blocking_result.blocking_object is not None:
            self._draw_blocking_debug(
                original_panel,
                blocking_result,
                roi_offset=(x1, y1),
                roi_height=preprocess_result.roi_frame.shape[0],
            )
        original_panel = draw_text_lines(
            original_panel,
            self._build_original_panel_lines(
                avoidance_result=avoidance_result,
                blocking_result=blocking_result,
                control_command=control_command,
                fps_value=fps_value,
                gold_result=gold_result,
            ) or [
                "窗口1：原始画面",
                "黄色框 = ROI范围  红线 = 车身中心参考线  绿线 = 航道中心线",
            ],
            font_path=self.font_path,
            font_size=self.font_size,
        )

        return original_panel

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
            is_gold = obj.class_name.casefold() == "gold"
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

    def _draw_blocking_debug(
        self,
        image: np.ndarray,
        blocking_result: BlockingAnalysisResult,
        roi_offset: tuple[int, int],
        roi_height: int,
    ) -> None:
        obj = blocking_result.blocking_object
        if obj is None:
            return
        frame_x1, frame_y1, frame_x2, frame_y2 = obj.bbox_frame
        roi_x1, roi_y1, roi_x2, roi_y2 = obj.bbox_roi or (0, 0, 0, 0)
        ox, oy = roi_offset
        cv2.rectangle(
            image,
            (int(frame_x1), int(frame_y1)),
            (int(frame_x2), int(frame_y2)),
            (0, 165, 255),
            2,
        )
        lane_x = int(round(blocking_result.lane_center_x_at_obstacle + ox))
        y_bottom = int(round(roi_y2 + oy))
        cv2.circle(image, (lane_x, y_bottom), 6, (255, 0, 255), -1)
        cv2.line(image, (lane_x, oy), (lane_x, oy + roi_height - 1), (255, 0, 255), 1)
        danger_left = int(round(blocking_result.danger_left + ox))
        danger_right = int(round(blocking_result.danger_right + ox))
        cv2.line(image, (danger_left, oy), (danger_left, oy + roi_height - 1), (0, 255, 255), 1)
        cv2.line(image, (danger_right, oy), (danger_right, oy + roi_height - 1), (0, 255, 255), 1)
        _ = roi_x1
        _ = roi_y1

    def _build_original_panel_lines(
        self,
        avoidance_result: AvoidanceTargetResult | None,
        blocking_result: BlockingAnalysisResult | None,
        control_command: ControlCommand,
        fps_value: float,
        gold_result: GoldTargetResult | None = None,
    ) -> list[str]:
        if avoidance_result is None:
            return []
        blocking_reason = blocking_result.reason if blocking_result is not None else "no blocking"
        gold_reason = gold_result.reason if gold_result is not None and gold_result.active else "no Gold"
        return [
            "窗口1：原始画面",
            "绿线=原中心线 青线=避障/Gold中心线 N=普通目标 A=最终目标 G=Gold",
            f"mode: {avoidance_result.mode}  FPS: {fps_value:.1f}",
            f"bias_px: {avoidance_result.avoid_bias_px:.1f}  final_error: {avoidance_result.final_lateral_error_px:.1f}",
            f"steer_deg: {control_command.steer_deg:.2f}",
            f"{gold_reason}",
            f"{blocking_reason}",
        ]

    def _write_video(self, canvas: np.ndarray) -> None:
        """将调试画面写入视频文件。

        输入:
            canvas: 当前帧调试大图。

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
                (canvas.shape[1], canvas.shape[0]),
            )
        self.video_writer.write(canvas)

    def _save_screenshot(self, canvas: np.ndarray) -> None:
        """将当前调试画面保存为截图文件。

        输入:
            canvas: 当前帧调试大图。

        输出:
            无返回值，截图将写入 save_dir 目录。
        """

        file_name = f"screenshot_{int(time.time() * 1000)}.png"
        cv2.imwrite(str(self.save_dir / file_name), canvas)
