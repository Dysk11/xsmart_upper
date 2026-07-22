"""Tests for fixed-height lane and avoidance target selection."""

from __future__ import annotations

import pytest

from core.object.blocking import BlockingAnalysisResult, DetectedObject
from core.planning.avoidance import AvoidanceTargetPlanner
from core.planning.target_selector import TargetSelector


def make_selector() -> TargetSelector:
    return TargetSelector({"fixed_target_y": 80})


@pytest.mark.parametrize("lane_confidence", [0.2, 0.9])
def test_target_height_is_fixed_across_confidence(lane_confidence: float) -> None:
    result = make_selector().select(
        centerline_points=[(120, 20), (100, 140)],
        roi_width=200,
        roi_height=200,
        lane_confidence=lane_confidence,
    )

    assert result.target_point_roi == pytest.approx((110.0, 80.0))
    assert result.lookahead_px == pytest.approx(119.0)


def test_unsorted_centerline_is_interpolated_at_fixed_height() -> None:
    result = make_selector().select(
        centerline_points=[(120, 20), (100, 140), (110, 80)],
        roi_width=200,
        roi_height=200,
        lane_confidence=0.8,
    )

    assert result.target_point_roi == pytest.approx((110.0, 80.0))


def test_short_centerline_is_extrapolated_to_fixed_height() -> None:
    result = make_selector().select(
        centerline_points=[(100, 150), (110, 120)],
        roi_width=200,
        roi_height=200,
        lane_confidence=0.8,
    )

    assert result.target_point_roi == pytest.approx((123.333333, 80.0))
    assert "extrapolation" in result.reason


def test_extrapolation_outside_roi_uses_visible_endpoint() -> None:
    result = make_selector().select(
        centerline_points=[(190, 150), (210, 120)],
        roi_width=200,
        roi_height=200,
        lane_confidence=0.8,
    )

    assert result.target_point_roi == pytest.approx((199.0, 120.0))
    assert "visible endpoint fallback" in result.reason


def test_excessive_vertical_extrapolation_uses_visible_endpoint() -> None:
    result = make_selector().select(
        centerline_points=[(100, 170), (110, 130)],
        roi_width=200,
        roi_height=200,
        lane_confidence=0.8,
    )

    assert result.target_point_roi == pytest.approx((110.0, 130.0))
    assert result.lookahead_px == pytest.approx(69.0)
    assert "extrapolation_y=50.0px" in result.reason


def test_single_point_keeps_x_and_uses_fixed_height() -> None:
    result = make_selector().select(
        centerline_points=[(77, 130)],
        roi_width=200,
        roi_height=200,
        lane_confidence=0.8,
    )

    assert result.target_point_roi == pytest.approx((77.0, 80.0))
    assert result.confidence == pytest.approx(0.8 * 0.45)


def test_no_centerline_preserves_lost_fallback() -> None:
    result = make_selector().select(
        centerline_points=[],
        roi_width=200,
        roi_height=200,
        lane_confidence=0.0,
    )

    assert result.target_point_roi == pytest.approx((100.0, 199.0))
    assert result.lookahead_px == 0.0
    assert result.confidence == 0.0


def test_avoidance_keeps_fixed_height_when_obstacle_is_ahead() -> None:
    selector = make_selector()
    planner = AvoidanceTargetPlanner(
        {
            "enabled": True,
            "max_avoid_bias_px": 20,
            "bias_alpha": 1.0,
            "front_margin_px": 45,
            "rear_margin_px": 35,
            "min_lane_confidence_to_start_avoid": 0.45,
        },
        target_selector=selector,
    )
    centerline = [(100, 180), (100, 140), (100, 100), (100, 60), (100, 20)]
    normal_target = selector.select(centerline, 200, 200, 0.9)
    obstacle = DetectedObject(
        class_name="car",
        confidence=0.9,
        bbox_frame=(80, 100, 120, 140),
        bbox_roi=(80, 100, 120, 140),
    )
    blocking = BlockingAnalysisResult(
        need_avoid=True,
        blocking_object=obstacle,
        blocking_score=0.9,
        obstacle_center_x=100.0,
        lane_center_x_at_obstacle=100.0,
        danger_left=10.0,
        danger_right=190.0,
        recommended_avoid_side="right",
        too_close=False,
        reason="test obstacle",
    )

    result = planner.plan(
        centerline_points=centerline,
        normal_target=normal_target,
        blocking_result=blocking,
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
    )

    assert result.mode == "avoid_right"
    assert result.target_point_roi == pytest.approx((120.0, 80.0))
    assert result.avoid_bias_px == pytest.approx(20.0)
