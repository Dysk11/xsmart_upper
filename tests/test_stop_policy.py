from __future__ import annotations

import pytest

from core.io.protocol import resolve_configured_speed_state
from core.lane.tracker import TrackedLaneState
from core.planning.high_level import HighLevelPlanner, build_off_track_stop_hint


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


def test_empty_track_mask_stops_immediately_with_highest_priority_mode() -> None:
    planner = HighLevelPlanner({"lost_speed": 0.25})
    hint = build_off_track_stop_hint(track_mask_visible=False)

    assert hint is not None and hint.stop
    command = planner.plan(make_tracked_state(lane_lost=True), hint)
    assert command.mode == "OFF_TRACK_STOP"
    assert command.target_speed == 0.0
    assert command.steer_deg == 0.0


def test_nonempty_track_mask_releases_off_track_stop() -> None:
    planner = HighLevelPlanner({"base_speed": 1.6, "min_speed": 0.45})

    assert build_off_track_stop_hint(track_mask_visible=True) is None
    command = planner.plan(make_tracked_state())
    assert command.mode != "OFF_TRACK_STOP"
    assert command.target_speed > 0.0


def test_geometric_lane_loss_with_visible_mask_keeps_existing_lost_behavior() -> None:
    planner = HighLevelPlanner({"lost_speed": 0.25})

    hint = build_off_track_stop_hint(track_mask_visible=True)
    command = planner.plan(make_tracked_state(lane_lost=True), hint)
    assert command.mode == "LANE_LOST"
    assert command.target_speed == 0.25


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
    ("lateral_error_px", "expected_reduction"),
    [
        (-100.0, True),
        (-83.0, True),
        (-82.999, False),
        (82.999, False),
        (83.0, True),
        (100.0, True),
    ],
)
def test_raw_lateral_error_requests_one_gear_reduction_at_threshold(
    lateral_error_px: float,
    expected_reduction: bool,
) -> None:
    planner = HighLevelPlanner(
        {
            "lateral_error_slowdown_threshold_px": 83.0,
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

    assert command.reduce_one_gear is expected_reduction
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


def test_stop_overrides_lateral_error_gear_reduction() -> None:
    planner = HighLevelPlanner(
        {"lateral_error_slowdown_threshold_px": 83.0}
    )
    hint = build_off_track_stop_hint(track_mask_visible=False)

    command = planner.plan(
        make_tracked_state(lateral_error_px=100.0),
        hint,
    )

    assert command.reduce_one_gear
    assert command.target_speed == 0.0
    assert (
        resolve_configured_speed_state(
            command.target_speed,
            3,
            reduce_one_gear=command.reduce_one_gear,
        )
        == 0
    )


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
