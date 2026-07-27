"""高层巡线控制策略模块。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict

from core.io.protocol import validate_moving_speed_state
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
    reduce_one_gear: bool
    speed_state_override: int | None


@dataclass
class ModuleHints:
    """为后续目标检测、OCR、红绿灯和金币规划模块预留的高层提示接口。"""

    speed_limit: float | None = None
    steer_offset_deg: float = 0.0
    force_mode: str | None = None
    stop: bool = False
    reduce_one_gear: bool = False
    speed_state_override: int | None = None
    note: str = ""


def build_safety_stop_hint(
    pedestrian_safety_result: Any | None = None,
    road_sign_waiting: bool = False,
) -> ModuleHints | None:
    """Select the active stop request in safety-priority order."""

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
    *,
    car_present: bool = False,
    speed_state: int | None = None,
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
    speed_state_override = (
        validate_moving_speed_state(
            speed_state,
            "car_avoidance.speed_state",
        )
        if car_present and speed_state is not None
        else None
    )
    return ModuleHints(
        speed_limit=float(min_speed) if edge_limited else None,
        force_mode=str(getattr(car_avoidance_result, "mode", "CAR_AVOID")),
        speed_state_override=speed_state_override,
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
        self.lateral_error_slowdown_threshold_px = float(
            config.get("lateral_error_slowdown_threshold_px", 83.0)
        )
        if (
            not math.isfinite(self.lateral_error_slowdown_threshold_px)
            or self.lateral_error_slowdown_threshold_px < 0.0
        ):
            raise ValueError(
                "planner.lateral_error_slowdown_threshold_px "
                "must be finite and non-negative"
            )
        self.curve_speed_state = validate_moving_speed_state(
            config.get("curve_speed_state", 0x02),
            "planner.curve_speed_state",
        )

        self.line_loss_hold_sec = float(config.get("line_loss_hold_sec", 0.5))
        if (
            not math.isfinite(self.line_loss_hold_sec)
            or self.line_loss_hold_sec < 0.0
        ):
            raise ValueError(
                "planner.line_loss_hold_sec must be finite and non-negative"
            )
        self.last_valid_command: ControlCommand | None = None
        self.line_loss_started_at: float | None = None

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
        *,
        line_lost: bool = False,
        now_monotonic: float | None = None,
    ) -> ControlCommand:
        """根据平滑后的巡线状态生成一帧高层控制量。

        输入:
            tracked_state: 时序平滑后的巡线状态。
            module_hints: 其他高层模块给出的附加提示，例如限速、强制模式或转向补偿。
            line_lost: 原始感知链路补充的丢线信号，例如 ROI 掩膜为空。
            now_monotonic: 可选单调时钟值，主要用于确定性测试。

        输出:
            返回 ControlCommand，其中只包含目标速度和目标转向等高层量。
        """

        module_hints = module_hints or ModuleHints()
        ts_ms = int(time.time() * 1000)
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        effective_line_lost = bool(line_lost or tracked_state.is_lane_lost)

        if effective_line_lost:
            if self.line_loss_started_at is None:
                self.line_loss_started_at = now
        else:
            self.line_loss_started_at = None

        if module_hints.stop:
            # 预留给红灯、停车标志等场景：上层模块可以直接要求停车。
            steer_deg = 0.0
            target_speed = 0.0
            mode = module_hints.force_mode or "MODULE_STOP"
        elif effective_line_lost:
            # 短时丢线保持最后一次有效控制；无历史值或超时后安全停车。
            elapsed = max(0.0, now - float(self.line_loss_started_at))
            if (
                self.last_valid_command is not None
                and elapsed < self.line_loss_hold_sec
            ):
                return ControlCommand(
                    ts_ms=ts_ms,
                    mode="LANE_LOST_HOLD",
                    target_speed=self.last_valid_command.target_speed,
                    steer_deg=self.last_valid_command.steer_deg,
                    reduce_one_gear=self.last_valid_command.reduce_one_gear,
                    speed_state_override=(
                        self.last_valid_command.speed_state_override
                    ),
                )
            return ControlCommand(
                ts_ms=ts_ms,
                mode="OFF_TRACK_STOP",
                target_speed=0.0,
                steer_deg=0.0,
                reduce_one_gear=False,
                speed_state_override=None,
            )
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

        reduce_one_gear = bool(module_hints.reduce_one_gear)
        speed_state_override = None
        if not module_hints.stop:
            if (
                abs(float(tracked_state.lateral_error_px))
                >= self.lateral_error_slowdown_threshold_px
            ):
                speed_state_override = self.curve_speed_state
            if module_hints.speed_state_override is not None:
                hint_speed_state = validate_moving_speed_state(
                    module_hints.speed_state_override,
                    "module_hints.speed_state_override",
                )
                speed_state_override = (
                    hint_speed_state
                    if speed_state_override is None
                    else min(speed_state_override, hint_speed_state)
                )
        command = ControlCommand(
            ts_ms=ts_ms,
            mode=mode,
            target_speed=float(target_speed),
            steer_deg=float(steer_deg),
            reduce_one_gear=reduce_one_gear,
            speed_state_override=speed_state_override,
        )
        if not module_hints.stop and not effective_line_lost:
            self.last_valid_command = command
        return command
