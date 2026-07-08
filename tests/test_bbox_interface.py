"""Smoke tests for bbox frame-to-ROI interface and avoidance handoff."""

from __future__ import annotations

from core.avoidance_target_planner import AvoidanceTargetPlanner
from core.blocking_analyzer import (
    BlockingAnalyzer,
    DetectedObject,
    attach_roi_bboxes,
    frame_bbox_to_roi_bbox,
)
from core.gold_target_planner import GoldTargetPlanner
from core.protocol import build_packet, parse_packet
from core.target_selector import TargetSelector


ROI_RECT = (100, 60, 500, 300)
ROI_WIDTH = 400
ROI_HEIGHT = 240


def test_bbox_fully_inside_roi() -> None:
    assert frame_bbox_to_roi_bbox((180, 120, 260, 220), ROI_RECT, ROI_WIDTH, ROI_HEIGHT) == (
        80,
        60,
        160,
        160,
    )


def test_bbox_partially_inside_roi() -> None:
    assert frame_bbox_to_roi_bbox((80, 40, 180, 120), ROI_RECT, ROI_WIDTH, ROI_HEIGHT) == (
        0,
        0,
        80,
        60,
    )


def test_bbox_outside_roi() -> None:
    assert frame_bbox_to_roi_bbox((10, 10, 90, 50), ROI_RECT, ROI_WIDTH, ROI_HEIGHT) is None


def test_bbox_blocks_curved_lane_centerline() -> None:
    centerline = [(170, 230), (190, 190), (220, 150), (260, 110), (300, 70)]
    objects = attach_roi_bboxes(
        [DetectedObject("Car", 0.9, bbox_frame=(260, 210, 360, 290))],
        ROI_RECT,
        ROI_WIDTH,
        ROI_HEIGHT,
    )
    result = BlockingAnalyzer({"too_close_y_ratio": 0.98}).analyze(
        objects,
        centerline,
        ROI_WIDTH,
        ROI_HEIGHT,
    )
    assert result.need_avoid
    assert result.blocking_object is not None
    assert result.blocking_object.bbox_roi is not None
    assert result.lane_center_x_at_obstacle != ROI_WIDTH / 2


def test_frame_center_bbox_does_not_block_lane_if_not_on_lane_corridor() -> None:
    centerline = [(80, 230), (90, 190), (100, 150), (120, 110), (140, 70)]
    objects = attach_roi_bboxes(
        [DetectedObject("Human", 0.9, bbox_frame=(270, 190, 330, 270))],
        ROI_RECT,
        ROI_WIDTH,
        ROI_HEIGHT,
    )
    result = BlockingAnalyzer({}).analyze(objects, centerline, ROI_WIDTH, ROI_HEIGHT)
    assert not result.need_avoid


def test_avoidance_final_error_is_packet_error_source() -> None:
    centerline = [(170, 230), (190, 190), (220, 150), (260, 110), (300, 70)]
    target_selector = TargetSelector({})
    normal_target = target_selector.select(centerline, ROI_WIDTH, ROI_HEIGHT, 0.85, 0.001)
    objects = attach_roi_bboxes(
        [DetectedObject("Car", 0.9, bbox_frame=(260, 190, 360, 260))],
        ROI_RECT,
        ROI_WIDTH,
        ROI_HEIGHT,
    )
    blocking = BlockingAnalyzer({"too_close_y_ratio": 0.98}).analyze(
        objects,
        centerline,
        ROI_WIDTH,
        ROI_HEIGHT,
    )
    avoidance = AvoidanceTargetPlanner({}, target_selector).plan(
        centerline_points=centerline,
        normal_target=normal_target,
        blocking_result=blocking,
        roi_width=ROI_WIDTH,
        roi_height=ROI_HEIGHT,
        lane_confidence=0.85,
        curvature=0.001,
    )
    packet = build_packet(
        {
            "lateral_error_px": avoidance.final_lateral_error_px,
            "steer_deg": 3.0,
        }
    )
    parsed = parse_packet(packet)
    assert parsed["lateral_error_px"] == float(int(avoidance.final_lateral_error_px))
    assert parsed["lateral_error_px"] != 12.0


def test_gold_target_overrides_to_gold_center() -> None:
    planner = GoldTargetPlanner({"min_confidence": 0.2})
    result = planner.plan(
        objects=[
            DetectedObject("coin", 0.91, bbox_frame=(260, 160, 340, 240)),
        ],
        roi_rect=ROI_RECT,
        roi_width=ROI_WIDTH,
        roi_height=ROI_HEIGHT,
    )
    assert result.active
    assert result.target_object is not None
    assert result.target_object.class_name == "coin"
    assert result.target_point_roi[0] == 200.0
    assert result.final_lateral_error_px == 0.0


def test_gold_target_holds_briefly_then_releases() -> None:
    planner = GoldTargetPlanner({"min_confidence": 0.2, "hold_frames": 1})
    first = planner.plan(
        objects=[DetectedObject("coin", 0.8, bbox_frame=(300, 180, 360, 250))],
        roi_rect=ROI_RECT,
        roi_width=ROI_WIDTH,
        roi_height=ROI_HEIGHT,
    )
    assert first.active
    held = planner.plan([], ROI_RECT, ROI_WIDTH, ROI_HEIGHT)
    assert held.active
    assert held.using_hold
    released = planner.plan([], ROI_RECT, ROI_WIDTH, ROI_HEIGHT)
    assert not released.active
