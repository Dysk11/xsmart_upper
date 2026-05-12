"""调试画面可视化模块。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.lane_detector import LaneDetectionResult
from core.lane_tracker import TrackedLaneState
from core.planner import ControlCommand
from core.preprocess import PreprocessResult
from utils.image_utils import draw_centerline, draw_text_lines, ensure_bgr, overlay_mask, stack_images


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
        original_panel = draw_text_lines(
            original_panel,
            [
                "窗口1：原始画面",
                "黄色框 = ROI范围  红线 = 车身中心参考线  绿线 = 航道中心线",
            ],
            font_path=self.font_path,
            font_size=self.font_size,
        )

        # 这里改为显示“稳定蓝色候选掩膜”，也就是调参工具里看到的那种整体蓝色区域。
        # 主航道筛选掩膜会随着组件链长度变化而闪烁，不适合作为调试主窗口的蓝色显示层。
        roi_panel = overlay_mask(preprocess_result.roi_raw_frame, detection_result.mask, color=(255, 0, 0), alpha=0.35)
        roi_panel = draw_centerline(roi_panel, centerline_points, color=(0, 255, 0))
        cv2.line(
            roi_panel,
            (preprocess_result.roi_raw_frame.shape[1] // 2, 0),
            (preprocess_result.roi_raw_frame.shape[1] // 2, preprocess_result.roi_raw_frame.shape[0] - 1),
            (60, 60, 255),
            1,
        )
        roi_panel = draw_text_lines(
            roi_panel,
            [
                "窗口2：ROI区域与稳定蓝色候选区",
                "这里显示调参同款蓝色掩膜，不再因为主链条筛选而忽隐忽现",
            ],
            font_path=self.font_path,
            font_size=self.font_size,
        )

        mask_panel = ensure_bgr(detection_result.mask)
        mask_panel = draw_centerline(mask_panel, detection_result.centerline_points, color=(0, 255, 255))
        mask_panel = draw_text_lines(
            mask_panel,
            [
                "窗口3：稳定蓝色掩膜与中心线",
                "白色区域 = 当前阈值识别到的蓝色航道，黄线 = 实际控制中心线",
            ],
            font_path=self.font_path,
            font_size=self.font_size,
        )

        info_panel = overlay_mask(preprocess_result.roi_frame, detection_result.mask, color=(255, 0, 0), alpha=0.20)
        info_panel = overlay_mask(info_panel, detection_result.filtered_mask, color=(0, 255, 255), alpha=0.16)
        info_panel = draw_centerline(info_panel, centerline_points, color=(0, 255, 0))
        info_panel = draw_text_lines(
            info_panel,
            [
                "窗口4：调试信息总览",
                f"FPS: {fps_value:.1f}",
                f"检测置信度: {tracked_state.confidence:.2f}  丢线计数: {tracked_state.lane_lost_count}",
                f"原始行点数: {detection_result.valid_row_count}  拟合点数: {detection_result.fit_point_count}",
                f"横向误差(px): {tracked_state.lateral_error_px:.2f}",
                f"航向误差(deg): {tracked_state.heading_error_deg:.2f}",
                f"曲率: {tracked_state.curvature:.6f}",
                f"目标速度: {control_command.target_speed:.3f}",
                f"目标转向: {control_command.steer_deg:.3f}",
                f"蓝色像素: {cv2.countNonZero(detection_result.mask)}  主链像素: {cv2.countNonZero(detection_result.filtered_mask)}",
                f"当前模式: {control_command.mode}  是否预测补偿: {'是' if tracked_state.used_prediction else '否'}",
            ],
            font_path=self.font_path,
            font_size=self.font_size,
        )

        cell_width = max(320, preprocess_result.resized_frame.shape[1] // 2)
        cell_height = max(180, preprocess_result.resized_frame.shape[0] // 2)
        return stack_images(
            [original_panel, roi_panel, mask_panel, info_panel],
            cols=2,
            cell_size=(cell_width, cell_height),
        )

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
