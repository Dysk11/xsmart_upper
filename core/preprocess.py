"""图像预处理模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    """保存预处理阶段的中间结果。"""

    original_frame: np.ndarray
    resized_frame: np.ndarray
    roi_raw_frame: np.ndarray
    roi_frame: np.ndarray
    roi_rect: Tuple[int, int, int, int]


class ImagePreprocessor:
    """负责图像缩放、裁剪与基础增强。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """读取预处理配置并初始化处理器。

        输入:
            config: 预处理相关配置字典。

        输出:
            无返回值，内部缓存各类开关与参数。
        """

        self.config = config
        self.resize_config = config.get("resize", {})
        self.roi_config = config.get("roi", {})
        self.gaussian_config = config.get("gaussian_blur", {})
        self.clahe_config = config.get("clahe", {})
        self.brightness_config = config.get("brightness_normalization", {})

    def process(self, frame: np.ndarray) -> PreprocessResult:
        """执行完整预处理流程并返回中间结果。

        输入:
            frame: 原始 BGR 图像。

        输出:
            返回 PreprocessResult，包含缩放结果、ROI 原图、增强后 ROI 和 ROI 位置。
        """

        resized_frame = self._resize_frame(frame)
        roi_raw_frame, roi_rect = self._crop_roi(resized_frame)

        processed_roi = roi_raw_frame.copy()
        if self.brightness_config.get("enable", False):
            processed_roi = self._normalize_brightness(processed_roi)
        if self.clahe_config.get("enable", False):
            processed_roi = self._apply_clahe(processed_roi)
        if self.gaussian_config.get("enable", False):
            processed_roi = self._apply_gaussian_blur(processed_roi)

        return PreprocessResult(
            original_frame=frame,
            resized_frame=resized_frame,
            roi_raw_frame=roi_raw_frame,
            roi_frame=processed_roi,
            roi_rect=roi_rect,
        )

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """根据配置对输入图像进行缩放。

        输入:
            frame: 原始 BGR 图像。

        输出:
            返回缩放后的 BGR 图像。
        """

        if not self.resize_config.get("enable", True):
            return frame.copy()

        width = int(self.resize_config.get("width", frame.shape[1]))
        height = int(self.resize_config.get("height", frame.shape[0]))
        return cv2.resize(frame, (width, height))

    def _crop_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """按照比例配置裁剪出感兴趣区域 ROI。

        输入:
            frame: 缩放后的 BGR 图像。

        输出:
            返回二元组 (roi_frame, roi_rect)，其中 roi_rect 格式为 (x1, y1, x2, y2)。
        """

        height, width = frame.shape[:2]
        top_ratio = float(self.roi_config.get("top_ratio", 0.35))
        bottom_ratio = float(self.roi_config.get("bottom_ratio", 1.0))
        left_ratio = float(self.roi_config.get("left_ratio", 0.0))
        right_ratio = float(self.roi_config.get("right_ratio", 1.0))

        x1 = max(0, min(width - 1, int(width * left_ratio)))
        x2 = max(x1 + 1, min(width, int(width * right_ratio)))
        y1 = max(0, min(height - 1, int(height * top_ratio)))
        y2 = max(y1 + 1, min(height, int(height * bottom_ratio)))

        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _normalize_brightness(self, frame: np.ndarray) -> np.ndarray:
        """通过调节 HSV 的 V 通道对 ROI 做亮度归一化。

        输入:
            frame: ROI BGR 图像。

        输出:
            返回亮度归一化后的 BGR 图像。
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2].astype(np.float32)

        target_mean = float(self.brightness_config.get("target_mean", 150.0))
        min_gain = float(self.brightness_config.get("min_gain", 0.7))
        max_gain = float(self.brightness_config.get("max_gain", 1.35))
        current_mean = float(np.mean(value_channel))
        gain = target_mean / max(current_mean, 1.0)
        gain = float(np.clip(gain, min_gain, max_gain))

        value_channel = np.clip(value_channel * gain, 0.0, 255.0)
        hsv[:, :, 2] = value_channel.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """使用 CLAHE 增强局部对比度，提升蓝色区域稳定性。

        输入:
            frame: ROI BGR 图像。

        输出:
            返回应用 CLAHE 后的 BGR 图像。
        """

        clip_limit = float(self.clahe_config.get("clip_limit", 2.0))
        tile_grid_size = int(self.clahe_config.get("tile_grid_size", 8))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _apply_gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        """通过高斯模糊抑制高频噪声与轻微模糊抖动。

        输入:
            frame: ROI BGR 图像。

        输出:
            返回模糊后的 BGR 图像。
        """

        kernel_size = int(self.gaussian_config.get("kernel_size", 5))
        sigma_x = float(self.gaussian_config.get("sigma_x", 0.0))

        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(1, kernel_size)

        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma_x)
