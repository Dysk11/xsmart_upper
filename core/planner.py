"""高层巡线控制策略模块。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

from core.lane_tracker import TrackedLaneState
from utils.math_utils import clamp


@dataclass
class ControlCommand:
    """保存上位机输出的高层控制指令。"""

    ts_ms: int
    mode: str
    target_speed: float
    steer_deg: float


@dataclass
class ModuleHints:
    """为后续目标检测、OCR、红绿灯和金币规划模块预留的高层提示接口。"""

    speed_limit: float | None = None
    steer_offset_deg: float = 0.0
    force_mode: str | None = None
    stop: bool = False
    note: str = ""


class HighLevelPlanner:
    """根据巡线状态生成目标速度与目标转向。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """读取控制策略参数。

        输入:
            config: planner 对应配置字典。

        输出:
            无返回值。
        """

        self.lateral_gain = float(config.get("lateral_gain", 0.065))
        self.heading_gain = float(config.get("heading_gain", 0.85))
        self.curvature_gain = float(config.get("curvature_gain", 140.0))
        self.max_steer_deg = float(config.get("max_steer_deg", 28.0))

        self.base_speed = float(config.get("base_speed", 1.6))
        self.max_speed = float(config.get("max_speed", 2.2))
        self.min_speed = float(config.get("min_speed", 0.45))
        self.straight_curvature_threshold = float(config.get("straight_curvature_threshold", 0.0025))
        self.straight_boost_speed = float(config.get("straight_boost_speed", 0.2))
        self.curvature_speed_gain = float(config.get("curvature_speed_gain", 120.0))
        self.heading_speed_gain = float(config.get("heading_speed_gain", 0.03))
        self.confidence_speed_gain = float(config.get("confidence_speed_gain", 0.7))
        self.caution_confidence_threshold = float(config.get("caution_confidence_threshold", 0.55))

        self.lost_speed = float(config.get("lost_speed", 0.25))
        self.lost_steer_decay = float(config.get("lost_steer_decay", 0.6))
        self.last_steer_deg = 0.0

    def plan(
        self,
        tracked_state: TrackedLaneState,
        module_hints: ModuleHints | None = None,
    ) -> ControlCommand:
        """根据平滑后的巡线状态生成一帧高层控制量。

        输入:
            tracked_state: 时序平滑后的巡线状态。
            module_hints: 其他高层模块给出的附加提示，例如限速、强制模式或转向补偿。

        输出:
            返回 ControlCommand，其中只包含目标速度和目标转向等高层量。
        """

        module_hints = module_hints or ModuleHints()
        ts_ms = int(time.time() * 1000)

        if module_hints.stop:
            # 预留给红灯、停车标志等场景：上层模块可以直接要求停车。
            steer_deg = 0.0
            target_speed = 0.0
            mode = module_hints.force_mode or "MODULE_STOP"
        elif tracked_state.is_lane_lost:
            # 丢线时不要激进，速度降下来，方向逐渐回正。
            steer_deg = self.last_steer_deg * self.lost_steer_decay
            target_speed = self.lost_speed
            mode = "LANE_LOST"
        else:
            # 这里只做高层合成，不做底层 PID。
            steer_deg = (
                tracked_state.lateral_error_px * self.lateral_gain
                + tracked_state.heading_error_deg * self.heading_gain
                + tracked_state.curvature * self.curvature_gain
            )
            steer_deg = clamp(steer_deg, -self.max_steer_deg, self.max_steer_deg)

            target_speed = self.base_speed
            target_speed -= abs(tracked_state.curvature) * self.curvature_speed_gain
            target_speed -= abs(tracked_state.heading_error_deg) * self.heading_speed_gain
            target_speed -= (1.0 - tracked_state.confidence) * self.confidence_speed_gain

            if (
                abs(tracked_state.curvature) < self.straight_curvature_threshold
                and tracked_state.confidence >= self.caution_confidence_threshold
            ):
                target_speed += self.straight_boost_speed
                mode = "CRUISE"
            elif tracked_state.confidence < self.caution_confidence_threshold:
                mode = "CAUTION"
            else:
                mode = "NORMAL"

            target_speed = clamp(target_speed, self.min_speed, self.max_speed)

        # 给后续扩展模块保留二次修正能力，例如限速或附加转向偏置。
        steer_deg += module_hints.steer_offset_deg
        steer_deg = clamp(steer_deg, -self.max_steer_deg, self.max_steer_deg)

        if module_hints.speed_limit is not None:
            target_speed = min(target_speed, float(module_hints.speed_limit))

        if module_hints.force_mode:
            mode = module_hints.force_mode

        self.last_steer_deg = steer_deg
        return ControlCommand(
            ts_ms=ts_ms,
            mode=mode,
            target_speed=float(target_speed),
            steer_deg=float(steer_deg),
        )
