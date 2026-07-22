"""巡线结果时序平滑模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from core.lane.detector import LaneDetectionResult
from utils.math_utils import ema


@dataclass
class TrackedLaneState:
    """保存时序平滑后的巡线状态。"""

    centerline_points: List[Tuple[int, int]]
    lateral_error_px: float
    heading_error_deg: float
    confidence: float
    is_lane_lost: bool
    lane_lost_count: int
    used_prediction: bool


class LaneTracker:
    """对检测结果做简单时序稳定与短时预测。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """读取平滑与丢线恢复参数。

        输入:
            config: tracker 对应配置字典。

        输出:
            无返回值。
        """

        self.ema_alpha = float(config.get("ema_alpha", 0.35))
        self.recovery_alpha = float(config.get("recovery_alpha", 0.6))
        self.confidence_gate = float(config.get("confidence_gate", 0.45))
        self.max_prediction_frames = int(config.get("max_prediction_frames", 6))
        self.prediction_decay = float(config.get("prediction_decay", 0.88))
        self.min_prediction_confidence = float(config.get("min_prediction_confidence", 0.1))
        self.centerline_alpha = float(config.get("centerline_alpha", 0.25))
        self.max_centerline_point_shift_px = float(
            config.get("max_centerline_point_shift_px", 80.0)
        )

        self.state = TrackedLaneState(
            centerline_points=[],
            lateral_error_px=0.0,
            heading_error_deg=0.0,
            confidence=0.0,
            is_lane_lost=True,
            lane_lost_count=0,
            used_prediction=False,
        )
        self.has_measurement = False
        self.last_valid_centerline: List[Tuple[int, int]] = []

    def update(self, detection_result: LaneDetectionResult, prefer_current: bool = False) -> TrackedLaneState:
        """输入当前帧检测结果并输出时序平滑状态。

        输入:
            detection_result: 单帧检测器输出结果。

        输出:
            返回 TrackedLaneState，包含平滑后的误差、置信度与丢线计数。
        """

        if not detection_result.is_lane_lost and detection_result.confidence >= self.confidence_gate:
            # 当前帧足够可靠，正常用 EMA 做平滑。
            alpha = 1.0 if prefer_current else self.recovery_alpha if self.state.lane_lost_count > 0 else self.ema_alpha
            self.state = self._update_with_measurement(
                detection_result,
                alpha,
                prefer_current=prefer_current,
            )
            self.state.lane_lost_count = 0
            self.state.is_lane_lost = False
            self.state.used_prediction = False
            self.last_valid_centerline = list(detection_result.centerline_points)
            self.has_measurement = True
            return self.state

        if not detection_result.is_lane_lost and detection_result.confidence > 0.0:
            # 当前帧能用但不太稳，降低它的权重，避免坏帧把状态拉飞。
            alpha = 1.0 if prefer_current else max(0.1, self.ema_alpha * detection_result.confidence / max(self.confidence_gate, 1e-6))
            self.state = self._update_with_measurement(
                detection_result,
                alpha,
                prefer_current=prefer_current,
            )
            self.state.lane_lost_count = 0
            self.state.is_lane_lost = False
            self.state.used_prediction = False
            self.last_valid_centerline = list(detection_result.centerline_points)
            self.has_measurement = True
            return self.state

        self.state.lane_lost_count += 1
        if self.has_measurement and self.state.lane_lost_count <= self.max_prediction_frames:
            # 短时间丢线时，先用上一帧结果顶一下，给视觉一点恢复时间。
            decay = self.prediction_decay ** self.state.lane_lost_count
            self.state = TrackedLaneState(
                centerline_points=list(self.last_valid_centerline),
                lateral_error_px=self.state.lateral_error_px * decay,
                heading_error_deg=self.state.heading_error_deg * decay,
                confidence=max(self.min_prediction_confidence, self.state.confidence * decay),
                is_lane_lost=True,
                lane_lost_count=self.state.lane_lost_count,
                used_prediction=True,
            )
            return self.state

        if self.has_measurement:
            # 即使连续丢线时间较长，也保留最后一条已知中心线用于显示与保守过渡。
            # 真正的控制安全由 is_lane_lost 和低速模式保证，这样调试画面不会反复“消失-出现”。
            decay = self.prediction_decay ** min(self.state.lane_lost_count, self.max_prediction_frames)
            self.state = TrackedLaneState(
                centerline_points=list(self.last_valid_centerline),
                lateral_error_px=self.state.lateral_error_px * decay,
                heading_error_deg=self.state.heading_error_deg * decay,
                confidence=0.0,
                is_lane_lost=True,
                lane_lost_count=self.state.lane_lost_count,
                used_prediction=True,
            )
            return self.state

        # 如果一开始就没有拿到过有效测量，再进入完全空状态。
        self.state = TrackedLaneState(
            centerline_points=[],
            lateral_error_px=0.0,
            heading_error_deg=0.0,
            confidence=0.0,
            is_lane_lost=True,
            lane_lost_count=self.state.lane_lost_count,
            used_prediction=False,
        )
        return self.state

    def _update_with_measurement(
        self,
        detection_result: LaneDetectionResult,
        alpha: float,
        prefer_current: bool = False,
    ) -> TrackedLaneState:
        """使用当前帧检测量更新平滑状态。

        输入:
            detection_result: 当前帧检测结果。
            alpha: 当前测量值在 EMA 中的权重。

        输出:
            返回更新后的 TrackedLaneState。
        """

        if not self.has_measurement:
            return TrackedLaneState(
                centerline_points=list(detection_result.centerline_points),
                lateral_error_px=detection_result.lateral_error_px,
                heading_error_deg=detection_result.heading_error_deg,
                confidence=detection_result.confidence,
                is_lane_lost=False,
                lane_lost_count=0,
                used_prediction=False,
            )

        if prefer_current:
            smoothed_centerline = list(detection_result.centerline_points)
        else:
            smoothed_centerline = self._smooth_centerline_points(
                previous_points=self.state.centerline_points,
                current_points=detection_result.centerline_points,
            )

        return TrackedLaneState(
            centerline_points=smoothed_centerline,
            lateral_error_px=ema(self.state.lateral_error_px, detection_result.lateral_error_px, alpha),
            heading_error_deg=ema(self.state.heading_error_deg, detection_result.heading_error_deg, alpha),
            confidence=ema(self.state.confidence, detection_result.confidence, alpha),
            is_lane_lost=False,
            lane_lost_count=0,
            used_prediction=False,
        )

    def _smooth_centerline_points(
        self,
        previous_points: List[Tuple[int, int]],
        current_points: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """对中心线点集做逐点平滑，减小线前端和后端的突然翻折。

        输入:
            previous_points: 上一帧已经平滑过的中心线点集。
            current_points: 当前帧检测器输出的中心线点集。

        输出:
            返回平滑后的中心线点集。
        """

        if not previous_points or not current_points:
            return list(current_points)

        smoothed_points: List[Tuple[int, int]] = []
        for curr_x, curr_y in current_points:
            previous_x = self._sample_previous_centerline_x(previous_points, curr_y)
            delta_x = max(
                -self.max_centerline_point_shift_px,
                min(self.max_centerline_point_shift_px, float(curr_x - previous_x)),
            )
            clamped_x = float(previous_x) + delta_x
            smoothed_x = ema(float(previous_x), clamped_x, self.centerline_alpha)
            smoothed_points.append((int(round(smoothed_x)), int(curr_y)))

        return smoothed_points

    def _sample_previous_centerline_x(
        self,
        previous_points: List[Tuple[int, int]],
        target_y: int,
    ) -> float:
        """按当前帧的纵坐标在上一帧中心线上插值横坐标。

        输入:
            previous_points: 上一帧已经平滑过的中心线点集。
            target_y: 当前需要对齐的纵坐标。

        输出:
            返回上一帧中心线在该纵坐标附近的横坐标估计值。
        """

        if not previous_points:
            return 0.0
        if len(previous_points) == 1:
            return float(previous_points[0][0])

        ordered_points = sorted(previous_points, key=lambda item: item[1], reverse=True)
        if target_y >= ordered_points[0][1]:
            return float(ordered_points[0][0])
        if target_y <= ordered_points[-1][1]:
            return float(ordered_points[-1][0])

        for index in range(len(ordered_points) - 1):
            x1, y1 = ordered_points[index]
            x2, y2 = ordered_points[index + 1]
            if y1 >= target_y >= y2 and y1 != y2:
                ratio = float(target_y - y1) / float(y2 - y1)
                return float(x1 + (x2 - x1) * ratio)

        return float(ordered_points[-1][0])
