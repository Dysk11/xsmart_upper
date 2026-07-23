from __future__ import annotations

import pytest

from core.io.protocol import resolve_configured_speed_state
from core.object.blocking import DetectedObject
from core.planning.car_avoidance import CarAvoidancePlanner
from core.planning.high_level import HighLevelPlanner, build_car_avoidance_hint
from core.planning.target_selector import TargetSelector
from tests.test_stop_policy import make_tracked_state


ROI_RECT = (0, 0, 200, 200)
CENTERLINE = [(100.0, float(y)) for y in range(196, 0, -4)]


def make_planner(
    *,
    perspective_top: float = 20.0,
    perspective_bottom: float = 40.0,
    **overrides: object,
) -> CarAvoidancePlanner:
    config: dict[str, object] = {
        "enabled": True,
        "clearance_px": 1,
        "transition_margin_px": 40,
        "edge_slow_margin_px": 20,
    }
    config.update(overrides)
    return CarAvoidancePlanner(
        config,
        TargetSelector({"fixed_target_y": 80}),
        {
            "perspective_width_top_px": perspective_top,
            "perspective_width_bottom_px": perspective_bottom,
        },
    )


def car(bbox: tuple[int, int, int, int]) -> DetectedObject:
    return DetectedObject("car", 0.9, bbox)


def normal_target(points: list[tuple[float, float]] = CENTERLINE):
    return TargetSelector({"fixed_target_y": 80}).select(points, 200, 200, 0.9)


def plan(
    objects: list[DetectedObject],
    *,
    target_x: float,
    centerline: list[tuple[float, float]] = CENTERLINE,
    candidate_route: list[tuple[float, float]] | None = None,
):
    target = normal_target(centerline)
    return make_planner().plan(
        objects=objects,
        centerline_points=centerline,
        candidate_route_points=candidate_route or centerline,
        candidate_target_roi=(target_x, 80.0),
        roi_rect=ROI_RECT,
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
        normal_target=target,
    )


def assert_route_clear(result) -> None:
    for zone in result.warning_zones:
        assert not CarAvoidancePlanner.polyline_intersects_polygon(
            result.shifted_centerline_points,
            zone.polygon_roi,
        )


def test_warning_polygon_uses_top_and_bottom_perspective_half_widths() -> None:
    result = plan([car((80, 60, 120, 100))], target_x=90.0)

    assert result.active
    top_offset = 0.5 * (20.0 + 20.0 * 60.0 / 199.0)
    bottom_offset = 0.5 * (20.0 + 20.0 * 100.0 / 199.0)
    expected = (
        (80.0 - top_offset, 60.0 - top_offset),
        (120.0 + top_offset, 60.0 - top_offset),
        (120.0 + bottom_offset, 100.0 + bottom_offset),
        (80.0 - bottom_offset, 100.0 + bottom_offset),
    )
    for actual, expected_point in zip(
        result.warning_zones[0].polygon_frame,
        expected,
    ):
        assert actual == pytest.approx(expected_point)
    for actual, expected_point in zip(
        result.warning_zones[0].polygon_roi,
        expected,
    ):
        assert actual == pytest.approx(expected_point)


def test_perspective_width_clamps_box_rows_outside_roi() -> None:
    planner = make_planner(perspective_top=20.0, perspective_bottom=60.0)
    target = normal_target()
    result = planner.plan(
        objects=[car((80, -20, 120, 240))],
        centerline_points=CENTERLINE,
        candidate_route_points=CENTERLINE,
        candidate_target_roi=(90.0, 80.0),
        roi_rect=ROI_RECT,
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
        normal_target=target,
    )

    polygon = result.warning_zones[0].polygon_roi
    assert polygon[0] == pytest.approx((70.0, -30.0))
    assert polygon[2] == pytest.approx((150.0, 270.0))


def test_warning_polygon_converts_between_frame_and_nonzero_roi_coordinates() -> None:
    planner = make_planner()
    target = normal_target()
    result = planner.plan(
        objects=[car((90, 80, 130, 120))],
        centerline_points=CENTERLINE,
        candidate_route_points=CENTERLINE,
        candidate_target_roi=(90.0, 80.0),
        roi_rect=(10, 20, 210, 220),
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
        normal_target=target,
    )

    zone = result.warning_zones[0]
    for frame_point, roi_point in zip(zone.polygon_frame, zone.polygon_roi):
        assert frame_point == pytest.approx((roi_point[0] + 10.0, roi_point[1] + 20.0))


def test_near_car_has_larger_offsets_than_same_size_far_car() -> None:
    far = plan([car((80, 20, 120, 40))], target_x=90.0)
    near = plan([car((80, 140, 120, 160))], target_x=90.0)

    far_polygon = far.warning_zones[0].polygon_roi
    near_polygon = near.warning_zones[0].polygon_roi
    far_bottom_offset = far_polygon[2][0] - 120.0
    near_bottom_offset = near_polygon[2][0] - 120.0
    assert near_bottom_offset > far_bottom_offset


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ([(50.0, 0.0), (50.0, 100.0)], True),
        ([(25.0, 20.0), (25.0, 40.0)], False),
        ([(40.0, 20.0), (30.0, 50.0)], True),
        ([(10.0, 0.0), (10.0, 100.0)], False),
    ],
)
def test_polyline_collision_uses_trapezoid_not_bounding_rectangle(
    route: list[tuple[float, float]],
    expected: bool,
) -> None:
    polygon = (
        (40.0, 20.0),
        (60.0, 20.0),
        (80.0, 80.0),
        (20.0, 80.0),
    )

    assert CarAvoidancePlanner.polyline_intersects_polygon(route, polygon) is expected


@pytest.mark.parametrize(
    ("target_x", "side"),
    [
        (90.0, "left"),
        (110.0, "right"),
        (100.0, "left"),
    ],
)
def test_target_relative_to_car_center_selects_side(
    target_x: float,
    side: str,
) -> None:
    result = plan([car((80, 60, 120, 100))], target_x=target_x)

    assert result.warning_zones[0].avoid_side == side
    assert_route_clear(result)


def test_special_candidate_line_can_trigger_avoidance_when_lane_is_clear() -> None:
    lane = [(150.0, float(y)) for y in range(196, 0, -4)]
    result = plan(
        [car((70, 80, 90, 120))],
        target_x=50.0,
        centerline=lane,
        candidate_route=[(100.0, 199.0), (50.0, 40.0)],
    )

    assert result.active
    assert result.mode == "CAR_AVOID"
    assert result.warning_zones[0].avoid_side == "left"
    assert_route_clear(result)


def test_multiple_opposite_constraints_use_gap_between_zones() -> None:
    objects = [
        car((130, 70, 150, 110)),
        car((50, 70, 70, 110)),
    ]
    result = plan(
        objects,
        target_x=100.0,
        candidate_route=[
            (100.0, 196.0),
            (140.0, 100.0),
            (60.0, 80.0),
            (100.0, 0.0),
        ],
    )

    assert result.active
    assert not result.stop_required
    assert {zone.avoid_side for zone in result.warning_zones} == {"left", "right"}
    assert_route_clear(result)


def test_multiple_same_side_constraints_use_the_strictest_boundary() -> None:
    result = plan(
        [
            car((85, 70, 105, 110)),
            car((95, 70, 115, 110)),
        ],
        target_x=80.0,
    )

    assert result.active
    assert not result.stop_required
    assert {zone.avoid_side for zone in result.warning_zones} == {"left"}
    inside_x = [
        x
        for x, y in result.shifted_centerline_points
        if 60.0 <= y <= 120.0
    ]
    assert max(inside_x) <= 79.0
    assert_route_clear(result)


def test_overlapping_opposite_constraints_stop() -> None:
    result = plan(
        [
            car((95, 70, 125, 110)),
            car((75, 70, 105, 110)),
        ],
        target_x=100.0,
        candidate_route=[
            (100.0, 196.0),
            (110.0, 100.0),
            (90.0, 80.0),
            (100.0, 0.0),
        ],
    )

    assert result.active
    assert result.stop_required
    assert result.mode == "CAR_AVOID_STOP"
    assert result.shifted_centerline_points == []


def test_edge_route_is_slow_mode_and_remains_outside_zone() -> None:
    result = plan(
        [car((20, 70, 60, 110))],
        target_x=20.0,
        candidate_route=[(100.0, 199.0), (20.0, 80.0)],
    )

    assert result.active
    assert result.edge_limited
    assert result.mode == "CAR_AVOID_EDGE"
    assert_route_clear(result)


def test_selected_side_fully_blocked_by_roi_edge_stops() -> None:
    result = plan(
        [car((0, 70, 40, 110))],
        target_x=0.0,
        candidate_route=[(100.0, 199.0), (0.0, 80.0)],
    )

    assert result.active
    assert result.stop_required
    assert result.mode == "CAR_AVOID_STOP"
    assert result.shifted_centerline_points == []


def test_edge_route_is_limited_to_planner_minimum_speed() -> None:
    result = plan(
        [car((20, 70, 60, 110))],
        target_x=20.0,
        candidate_route=[(100.0, 199.0), (20.0, 80.0)],
    )
    hint = build_car_avoidance_hint(result, min_speed=0.45)
    command = HighLevelPlanner({}).plan(make_tracked_state(), hint)

    assert hint is not None
    assert hint.speed_limit == pytest.approx(0.45)
    assert command.mode == "CAR_AVOID_EDGE"
    assert command.target_speed == pytest.approx(0.45)


def test_infeasible_route_stops_and_maps_to_zero_protocol_state() -> None:
    result = plan(
        [car((0, 70, 40, 110))],
        target_x=0.0,
        candidate_route=[(100.0, 199.0), (0.0, 80.0)],
    )
    hint = build_car_avoidance_hint(result, min_speed=0.45)
    command = HighLevelPlanner({}).plan(make_tracked_state(), hint)

    assert hint is not None and hint.stop
    assert command.mode == "CAR_AVOID_STOP"
    assert command.target_speed == 0.0
    assert resolve_configured_speed_state(command.target_speed, 2) == 0


def test_transition_anchors_are_smooth_and_fixed_target_height_is_preserved() -> None:
    result = plan([car((80, 60, 120, 100))], target_x=90.0)
    route = {round(y, 6): x for x, y in result.shifted_centerline_points}
    polygon = result.warning_zones[0].polygon_roi
    top = min(point[1] for point in polygon)
    bottom = max(point[1] for point in polygon)

    assert route[round(max(0.0, top - 40.0), 6)] == pytest.approx(100.0)
    assert route[round(min(199.0, bottom + 40.0), 6)] == pytest.approx(100.0)
    assert result.target_result.target_point_roi[1] == pytest.approx(80.0)
    assert_route_clear(result)


def test_non_car_and_clear_car_do_not_activate() -> None:
    human = DetectedObject("human", 0.9, (80, 60, 120, 100))
    result = plan([human, car((150, 60, 180, 100))], target_x=90.0)

    assert not result.active
    assert len(result.warning_zones) == 1


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"clearance_px": -1}, "clearance_px"),
        ({"clearance_px": 0.5}, "clearance_px"),
        ({"transition_margin_px": 0}, "transition_margin_px"),
        ({"edge_slow_margin_px": -1}, "edge_slow_margin_px"),
    ],
)
def test_config_is_validated(config: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_planner(**config)


@pytest.mark.parametrize(
    ("top", "bottom", "message"),
    [
        (0.0, 40.0, "perspective_width_top_px"),
        (20.0, 0.0, "perspective_width_bottom_px"),
        (40.0, 20.0, "greater than or equal"),
    ],
)
def test_shared_perspective_config_is_validated(
    top: float,
    bottom: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_planner(perspective_top=top, perspective_bottom=bottom)
