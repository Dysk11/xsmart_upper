"""高层巡线控制策略模块。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict

from core.lane.tracker import TrackedLaneState
from utils.math_utils import clamp


DEFAULT_LATERAL_ERROR_THRESHOLDS_PX = (6.0, 13.0, 24.0, 32.0, 55.0, 80.0)
DEFAULT_LATERAL_ERROR_MULTIPLIERS = (0.199, 0.529, 0.99, 1.39, 1.58, 1.88, 2.5)


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


def build_off_track_stop_hint(track_mask_visible: bool) -> ModuleHints | None:
    """Return the highest-priority stop hint when segmentation sees no track."""

    if track_mask_visible:
        return None
    return ModuleHints(
        stop=True,
        force_mode="OFF_TRACK_STOP",
        note="lane segmentation produced an empty ROI mask",
    )


def build_safety_stop_hint(
    track_mask_visible: bool,
    pedestrian_safety_result: Any | None = None,
    road_sign_waiting: bool = False,
) -> ModuleHints | None:
    """Select the active stop request in safety-priority order."""

    off_track_hint = build_off_track_stop_hint(track_mask_visible)
    if off_track_hint is not None:
        return off_track_hint
    if (
        pedestrian_safety_result is not None
        and bool(getattr(pedestrian_safety_result, "stop_required", False))
    ):
        return ModuleHints(
            stop=True,
            force_mode="PEDESTRIAN_WAIT",
            note=str(getattr(pedestrian_safety_result, "reason", "")),
        )
    if road_sign_waiting:
        return ModuleHints(
            stop=True,
            force_mode="ROAD_SIGN_WAIT",
            note="waiting for OCR and road-sign API decision",
        )
    return None


def build_car_avoidance_hint(
    car_avoidance_result: Any | None,
    min_speed: float,
) -> ModuleHints | None:
    """Convert an active car-avoidance result into a control hint."""

    if (
        car_avoidance_result is None
        or not bool(getattr(car_avoidance_result, "active", False))
    ):
        return None
    reason = str(getattr(car_avoidance_result, "reason", ""))
    if bool(getattr(car_avoidance_result, "stop_required", False)):
        return ModuleHints(
            stop=True,
            force_mode="CAR_AVOID_STOP",
            note=reason,
        )
    edge_limited = bool(getattr(car_avoidance_result, "edge_limited", False))
    return ModuleHints(
        speed_limit=float(min_speed) if edge_limited else None,
        force_mode=str(getattr(car_avoidance_result, "mode", "CAR_AVOID")),
        note=reason,
    )


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
        self.max_steer_deg = float(config.get("max_steer_deg", 28.0))
        (
            self.lateral_error_thresholds_px,
            self.lateral_error_multipliers,
        ) = self._parse_lateral_error_amplification(
            config.get("lateral_error_amplification", {})
        )

        self.base_speed = float(config.get("base_speed", 1.6))
        self.max_speed = float(config.get("max_speed", 2.2))
        self.min_speed = float(config.get("min_speed", 0.45))
        self.heading_speed_gain = float(config.get("heading_speed_gain", 0.03))
        self.confidence_speed_gain = float(config.get("confidence_speed_gain", 0.7))
        self.caution_confidence_threshold = float(config.get("caution_confidence_threshold", 0.55))

        self.lost_speed = float(config.get("lost_speed", 0.25))
        self.lost_steer_decay = float(config.get("lost_steer_decay", 0.6))
        self.last_steer_deg = 0.0

    @staticmethod
    def _parse_lateral_error_amplification(
        config: Any,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if not isinstance(config, dict):
            raise ValueError("planner.lateral_error_amplification must be a mapping")

        raw_thresholds = config.get(
            "thresholds_px", DEFAULT_LATERAL_ERROR_THRESHOLDS_PX
        )
        raw_multipliers = config.get(
            "multipliers", DEFAULT_LATERAL_ERROR_MULTIPLIERS
        )
        if not isinstance(raw_thresholds, (list, tuple)):
            raise ValueError(
                "planner.lateral_error_amplification.thresholds_px must be a sequence"
            )
        if not isinstance(raw_multipliers, (list, tuple)):
            raise ValueError(
                "planner.lateral_error_amplification.multipliers must be a sequence"
            )

        try:
            thresholds = tuple(float(value) for value in raw_thresholds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "planner.lateral_error_amplification.thresholds_px must contain numbers"
            ) from exc
        try:
            multipliers = tuple(float(value) for value in raw_multipliers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "planner.lateral_error_amplification.multipliers must contain numbers"
            ) from exc

        if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError(
                "planner.lateral_error_amplification.thresholds_px must be finite and non-negative"
            )
        if any(
            current <= previous
            for previous, current in zip(thresholds, thresholds[1:])
        ):
            raise ValueError(
                "planner.lateral_error_amplification.thresholds_px must be strictly increasing"
            )
        if len(multipliers) != len(thresholds) + 1:
            raise ValueError(
                "planner.lateral_error_amplification.multipliers must contain exactly one more value than thresholds_px"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in multipliers):
            raise ValueError(
                "planner.lateral_error_amplification.multipliers must be finite and non-negative"
            )
        return thresholds, multipliers

    def _amplify_lateral_error(self, lateral_error_px: float) -> float:
        error = float(lateral_error_px)
        absolute_error = abs(error)
        multiplier = self.lateral_error_multipliers[-1]
        for threshold, candidate in zip(
            self.lateral_error_thresholds_px,
            self.lateral_error_multipliers,
        ):
            if absolute_error <= threshold:
                multiplier = candidate
                break
        return error * multiplier

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
            amplified_lateral_error_px = self._amplify_lateral_error(
                tracked_state.lateral_error_px
            )
            steer_deg = (
                amplified_lateral_error_px * self.lateral_gain
                + tracked_state.heading_error_deg * self.heading_gain
            )
            steer_deg = clamp(steer_deg, -self.max_steer_deg, self.max_steer_deg)

            target_speed = self.base_speed
            target_speed -= abs(tracked_state.heading_error_deg) * self.heading_speed_gain
            target_speed -= (1.0 - tracked_state.confidence) * self.confidence_speed_gain

            if tracked_state.confidence < self.caution_confidence_threshold:
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
