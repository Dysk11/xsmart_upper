from __future__ import annotations

import pytest

from core.lane.tracker import TrackedLaneState
from core.planning.high_level import HighLevelPlanner, build_off_track_stop_hint


def make_tracked_state(*, lane_lost: bool = False) -> TrackedLaneState:
    return TrackedLaneState(
        centerline_points=[(100, 100)],
        lateral_error_px=4.0,
        heading_error_deg=2.0,
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
    assert command.steer_deg == pytest.approx(1.4)
    assert command.target_speed == pytest.approx(1.47)
