from __future__ import annotations

import pytest

from core.io.protocol import resolve_configured_speed_state
from core.lane.tracker import TrackedLaneState
from core.planning.high_level import HighLevelPlanner, ModuleHints


def make_tracked_state(
    *,
    lane_lost: bool = False,
    lateral_error_px: float = 4.0,
    heading_error_deg: float = 2.0,
) -> TrackedLaneState:
    return TrackedLaneState(
        centerline_points=[(100, 100)],
        lateral_error_px=lateral_error_px,
        heading_error_deg=heading_error_deg,
        confidence=0.9,
        is_lane_lost=lane_lost,
        lane_lost_count=1 if lane_lost else 0,
        used_prediction=lane_lost,
    )


def test_initial_lane_loss_stops_when_no_valid_command_exists() -> None:
    planner = HighLevelPlanner({"line_loss_hold_sec": 0.5})

    command = planner.plan(
        make_tracked_state(lane_lost=True),
        now_monotonic=10.0,
    )
    assert command.mode == "OFF_TRACK_STOP"
    assert command.target_speed == 0.0
    assert command.steer_deg == 0.0


@pytest.mark.parametrize(
    ("tracked_lane_lost", "raw_line_lost"),
    [(True, False), (False, True)],
)
def test_short_line_loss_holds_last_speed_steer_and_gear(
    tracked_lane_lost: bool,
    raw_line_lost: bool,
) -> None:
    planner = HighLevelPlanner(
        {
            "line_loss_hold_sec": 0.5,
            "lateral_error_slowdown_threshold_px": 83.0,
        }
    )
    valid = planner.plan(
        make_tracked_state(lateral_error_px=100.0),
        now_monotonic=10.0,
    )
    planner.plan(
        make_tracked_state(lane_lost=tracked_lane_lost),
        line_lost=raw_line_lost,
        now_monotonic=10.1,
    )

    held = planner.plan(
        make_tracked_state(lane_lost=tracked_lane_lost),
        line_lost=raw_line_lost,
        now_monotonic=10.59,
    )

    assert held.mode == "LANE_LOST_HOLD"
    assert held.target_speed == valid.target_speed
    assert held.steer_deg == valid.steer_deg
    assert held.reduce_one_gear is valid.reduce_one_gear
    assert held.speed_state_override == valid.speed_state_override
    assert resolve_configured_speed_state(
        held.target_speed,
        3,
        reduce_one_gear=held.reduce_one_gear,
        speed_state_override=held.speed_state_override,
    ) == resolve_configured_speed_state(
        valid.target_speed,
        3,
        reduce_one_gear=valid.reduce_one_gear,
        speed_state_override=valid.speed_state_override,
    )


def test_line_loss_stops_exactly_at_timeout() -> None:
    planner = HighLevelPlanner({"line_loss_hold_sec": 0.5})
    planner.plan(make_tracked_state(), now_monotonic=10.0)
    planner.plan(
        make_tracked_state(lane_lost=True),
        now_monotonic=11.0,
    )

    command = planner.plan(
        make_tracked_state(lane_lost=True),
        now_monotonic=11.5,
    )

    assert command.mode == "OFF_TRACK_STOP"
    assert command.target_speed == 0.0
    assert command.steer_deg == 0.0
    assert not command.reduce_one_gear


def test_zero_hold_duration_stops_on_first_lost_frame() -> None:
    planner = HighLevelPlanner({"line_loss_hold_sec": 0.0})
    planner.plan(make_tracked_state(), now_monotonic=10.0)

    command = planner.plan(
        make_tracked_state(lane_lost=True),
        now_monotonic=10.1,
    )

    assert command.mode == "OFF_TRACK_STOP"
    assert command.target_speed == 0.0


def test_line_recovery_resets_loss_timer_and_refreshes_cached_control() -> None:
    planner = HighLevelPlanner({"line_loss_hold_sec": 0.5})
    first = planner.plan(
        make_tracked_state(lateral_error_px=4.0),
        now_monotonic=10.0,
    )
    held_first = planner.plan(
        make_tracked_state(lane_lost=True),
        now_monotonic=10.1,
    )
    recovered = planner.plan(
        make_tracked_state(lateral_error_px=20.0),
        now_monotonic=10.7,
    )
    held_recovered = planner.plan(
        make_tracked_state(lane_lost=True),
        now_monotonic=11.1,
    )

    assert held_first.steer_deg == first.steer_deg
    assert recovered.steer_deg != first.steer_deg
    assert held_recovered.mode == "LANE_LOST_HOLD"
    assert held_recovered.steer_deg == recovered.steer_deg


def test_safety_stop_overrides_hold_without_resetting_loss_timer() -> None:
    planner = HighLevelPlanner({"line_loss_hold_sec": 0.5})
    planner.plan(make_tracked_state(), now_monotonic=10.0)
    safety_hint = ModuleHints(stop=True, force_mode="PEDESTRIAN_WAIT")

    stopped = planner.plan(
        make_tracked_state(lane_lost=True),
        safety_hint,
        now_monotonic=11.0,
    )
    still_lost = planner.plan(
        make_tracked_state(lane_lost=True),
        now_monotonic=11.5,
    )

    assert stopped.mode == "PEDESTRIAN_WAIT"
    assert stopped.target_speed == 0.0
    assert still_lost.mode == "OFF_TRACK_STOP"


def test_non_stop_hints_cannot_modify_held_control() -> None:
    planner = HighLevelPlanner({"line_loss_hold_sec": 0.5})
    valid = planner.plan(
        make_tracked_state(),
        ModuleHints(
            speed_limit=0.45,
            steer_offset_deg=3.0,
            reduce_one_gear=True,
        ),
        now_monotonic=10.0,
    )

    held = planner.plan(
        make_tracked_state(lane_lost=True),
        ModuleHints(
            speed_limit=0.1,
            steer_offset_deg=-10.0,
            reduce_one_gear=False,
        ),
        now_monotonic=10.1,
    )

    assert held.mode == "LANE_LOST_HOLD"
    assert held.target_speed == valid.target_speed
    assert held.steer_deg == valid.steer_deg
    assert held.reduce_one_gear is valid.reduce_one_gear
    assert held.speed_state_override == valid.speed_state_override


def test_curve_avoidance_and_generic_hazard_use_lowest_speed_state() -> None:
    planner = HighLevelPlanner(
        {
            "curve_speed_state": 2,
            "lateral_error_slowdown_threshold_px": 83.0,
        }
    )

    command = planner.plan(
        make_tracked_state(lateral_error_px=100.0),
        ModuleHints(
            reduce_one_gear=True,
            speed_state_override=1,
        ),
    )

    assert command.speed_state_override == 1
    assert command.reduce_one_gear
    assert (
        resolve_configured_speed_state(
            command.target_speed,
            3,
            reduce_one_gear=command.reduce_one_gear,
            speed_state_override=command.speed_state_override,
        )
        == 1
    )


def test_normal_control_uses_only_lateral_and_heading_errors() -> None:
    planner = HighLevelPlanner(
        {
            "lateral_gain": 0.1,
            "heading_gain": 0.5,
            "base_speed": 1.6,
            "heading_speed_gain": 0.03,
            "confidence_speed_gain": 0.7,
            "caution_confidence_threshold": 0.55,
        }
    )

    command = planner.plan(make_tracked_state())

    assert command.mode == "NORMAL"
    assert command.steer_deg == pytest.approx(1.0796)
    assert command.target_speed == pytest.approx(1.47)


@pytest.mark.parametrize(
    ("lateral_error_px", "expected_speed_state"),
    [
        (-100.0, 2),
        (-83.0, 2),
        (-82.999, None),
        (82.999, None),
        (83.0, 2),
        (100.0, 2),
    ],
)
def test_raw_lateral_error_selects_curve_speed_state_at_threshold(
    lateral_error_px: float,
    expected_speed_state: int | None,
) -> None:
    planner = HighLevelPlanner(
        {
            "lateral_error_slowdown_threshold_px": 83.0,
            "curve_speed_state": 2,
            "heading_speed_gain": 0.0,
            "confidence_speed_gain": 0.0,
        }
    )

    command = planner.plan(
        make_tracked_state(
            lateral_error_px=lateral_error_px,
            heading_error_deg=0.0,
        )
    )

    assert not command.reduce_one_gear
    assert command.speed_state_override == expected_speed_state
    assert command.target_speed == pytest.approx(planner.base_speed)


@pytest.mark.parametrize(
    "invalid_threshold",
    [-1.0, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_lateral_error_slowdown_threshold_is_rejected(
    invalid_threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="lateral_error_slowdown_threshold_px",
    ):
        HighLevelPlanner(
            {"lateral_error_slowdown_threshold_px": invalid_threshold}
        )


def test_stop_overrides_curve_speed_state() -> None:
    planner = HighLevelPlanner(
        {"lateral_error_slowdown_threshold_px": 83.0}
    )
    hint = ModuleHints(stop=True, force_mode="OFF_TRACK_STOP")

    command = planner.plan(
        make_tracked_state(lateral_error_px=100.0),
        hint,
    )

    assert not command.reduce_one_gear
    assert command.speed_state_override is None
    assert command.target_speed == 0.0
    assert (
        resolve_configured_speed_state(
            command.target_speed,
            3,
            reduce_one_gear=command.reduce_one_gear,
            speed_state_override=command.speed_state_override,
        )
        == 0
    )


@pytest.mark.parametrize(
    "invalid_curve_state",
    [0, -1, 4, 2.5, True, float("inf"), "bad"],
)
def test_invalid_curve_speed_state_is_rejected(
    invalid_curve_state: object,
) -> None:
    with pytest.raises(ValueError, match="planner.curve_speed_state"):
        HighLevelPlanner({"curve_speed_state": invalid_curve_state})


@pytest.mark.parametrize(
    "invalid_hold_sec",
    [-1.0, float("inf"), float("-inf"), float("nan")],
)
def test_invalid_line_loss_hold_sec_is_rejected(invalid_hold_sec: float) -> None:
    with pytest.raises(ValueError, match="line_loss_hold_sec"):
        HighLevelPlanner({"line_loss_hold_sec": invalid_hold_sec})


@pytest.mark.parametrize(
    ("absolute_error", "multiplier"),
    [
        (0.0, 0.199),
        (6.0, 0.199),
        (6.0001, 0.529),
        (13.0, 0.529),
        (13.0001, 0.99),
        (24.0, 0.99),
        (24.0001, 1.39),
        (32.0, 1.39),
        (32.0001, 1.58),
        (55.0, 1.58),
        (55.0001, 1.88),
        (80.0, 1.88),
        (80.0001, 2.5),
    ],
)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_lateral_error_amplification_uses_inclusive_thresholds_and_preserves_sign(
    absolute_error: float,
    multiplier: float,
    sign: float,
) -> None:
    planner = HighLevelPlanner(
        {
            "lateral_gain": 1.0,
            "heading_gain": 0.0,
            "max_steer_deg": 1000.0,
        }
    )
    lateral_error_px = sign * absolute_error

    command = planner.plan(
        make_tracked_state(
            lateral_error_px=lateral_error_px,
            heading_error_deg=0.0,
        )
    )

    assert command.steer_deg == pytest.approx(lateral_error_px * multiplier)


def test_custom_lateral_error_amplification_is_used_before_heading_term() -> None:
    planner = HighLevelPlanner(
        {
            "lateral_gain": 0.5,
            "heading_gain": 2.0,
            "max_steer_deg": 1000.0,
            "lateral_error_amplification": {
                "thresholds_px": [10.0],
                "multipliers": [0.25, 3.0],
            },
        }
    )

    command = planner.plan(
        make_tracked_state(lateral_error_px=12.0, heading_error_deg=4.0)
    )

    assert command.steer_deg == pytest.approx(12.0 * 3.0 * 0.5 + 4.0 * 2.0)


@pytest.mark.parametrize(
    ("amplification_config", "message"),
    [
        ([], "must be a mapping"),
        ({"thresholds_px": "6,13"}, "thresholds_px must be a sequence"),
        ({"multipliers": "0.2,0.5"}, "multipliers must be a sequence"),
        ({"thresholds_px": [6, "bad"]}, "thresholds_px must contain numbers"),
        (
            {"thresholds_px": [6, 6], "multipliers": [1, 2, 3]},
            "thresholds_px must be strictly increasing",
        ),
        (
            {"thresholds_px": [-1], "multipliers": [1, 2]},
            "thresholds_px must be finite and non-negative",
        ),
        (
            {"thresholds_px": [6], "multipliers": [1]},
            "multipliers must contain exactly one more value",
        ),
        (
            {"thresholds_px": [6], "multipliers": [1, float("inf")]},
            "multipliers must be finite and non-negative",
        ),
    ],
)
def test_invalid_lateral_error_amplification_config_is_rejected(
    amplification_config: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HighLevelPlanner({"lateral_error_amplification": amplification_config})
