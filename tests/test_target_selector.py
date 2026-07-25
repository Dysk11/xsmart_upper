"""Tests for fixed-height lane target selection."""

from __future__ import annotations

import pytest

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


def test_steer_angle_uses_target_to_roi_top_centerline_geometry() -> None:
    result = make_selector().select(
        centerline_points=[(140, 0), (120, 80), (100, 199)],
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
    )

    assert result.target_point_roi == pytest.approx((120.0, 80.0))
    assert result.steer_angle_deg == pytest.approx(14.036243)


def test_steer_angle_is_invalid_when_centerline_misses_roi_top() -> None:
    result = make_selector().select(
        centerline_points=[(140, 1), (120, 80), (100, 199)],
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
    )

    assert result.steer_angle_deg is None


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
