from __future__ import annotations

import pytest

from core.io.protocol import resolve_configured_speed_state
from core.lane.detector import LaneBoundaryRow
from core.object.blocking import DetectedObject
from core.planning.car_avoidance import CarAvoidancePlanner
from core.planning.high_level import HighLevelPlanner, build_car_avoidance_hint
from core.planning.target_selector import TargetSelector
from tests.test_stop_policy import make_tracked_state


ROI_RECT = (0, 0, 200, 200)
CENTERLINE = [(90.0, float(y)) for y in range(199, -1, -4)]


def make_planner(**overrides: object) -> CarAvoidancePlanner:
    config: dict[str, object] = {
        "enabled": True,
        "box_scale": 1.5,
        "transition_margin_px": 40,
        "edge_slow_margin_px": 20,
        "release_duration_s": 1.0,
    }
    config.update(overrides)
    return CarAvoidancePlanner(
        config,
        TargetSelector({"fixed_target_y": 80}),
        max_boundary_gap_rows=12,
    )


def boundary_rows(
    left_x: int = 40,
    right_x: int = 160,
    invalid_ys: set[int] | None = None,
) -> list[LaneBoundaryRow]:
    invalid_ys = invalid_ys or set()
    return [
        LaneBoundaryRow(
            y=y,
            left_x=left_x,
            right_x=right_x,
            left_valid=y not in invalid_ys,
            right_valid=y not in invalid_ys,
        )
        for y in range(199, -1, -1)
    ]


def car(bbox: tuple[int, int, int, int]) -> DetectedObject:
    return DetectedObject("car", 0.9, bbox)


def normal_target(points: list[tuple[float, float]] = CENTERLINE):
    return TargetSelector({"fixed_target_y": 80}).select(points, 200, 200, 0.9)


def plan(
    objects: list[DetectedObject],
    *,
    planner: CarAvoidancePlanner | None = None,
    centerline: list[tuple[float, float]] = CENTERLINE,
    candidate_route: list[tuple[float, float]] | None = None,
    boundaries: list[LaneBoundaryRow] | None = None,
    detection_id: int = 1,
    now: float = 0.0,
):
    planner = planner or make_planner()
    target = normal_target(centerline)
    return planner.plan(
        objects=objects,
        centerline_points=centerline,
        candidate_route_points=candidate_route or centerline,
        candidate_target_roi=target.target_point_roi,
        track_boundary_rows=boundaries or boundary_rows(),
        detection_result_id=detection_id,
        now_monotonic=now,
        roi_rect=ROI_RECT,
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
        normal_target=target,
    )


def assert_route_clear(result) -> None:
    for zone in result.warning_zones:
        assert not CarAvoidancePlanner.polyline_intersects_rect(
            result.shifted_centerline_points,
            zone.bbox_roi,
        )


def test_warning_box_scales_width_and_height_about_center() -> None:
    result = plan([car((80, 60, 120, 100))])

    assert result.active
    assert result.warning_zones[0].bbox_frame == pytest.approx((70, 50, 130, 110))
    assert result.warning_zones[0].bbox_roi == pytest.approx((70, 50, 130, 110))


@pytest.mark.parametrize(
    ("center_x", "side", "expected_boundary"),
    [
        (90.0, "left", 40.0),
        (110.0, "right", 160.0),
        (100.0, "left", 40.0),
    ],
)
def test_centerline_at_car_height_selects_boundary_side(
    center_x: float,
    side: str,
    expected_boundary: float,
) -> None:
    centerline = [(center_x, float(y)) for y in range(199, -1, -4)]
    result = plan([car((80, 60, 120, 100))], centerline=centerline)

    assert result.warning_zones[0].avoid_side == side
    core = {
        round(y): x
        for x, y in result.shifted_centerline_points
        if 50.0 <= y <= 110.0
    }
    assert core
    assert all(x == pytest.approx(expected_boundary) for x in core.values())
    assert_route_clear(result)


def test_candidate_target_x_does_not_choose_avoidance_side() -> None:
    planner = make_planner()
    target = normal_target()
    result = planner.plan(
        objects=[car((80, 60, 120, 100))],
        centerline_points=CENTERLINE,
        candidate_route_points=CENTERLINE,
        candidate_target_roi=(190.0, 80.0),
        track_boundary_rows=boundary_rows(),
        detection_result_id=1,
        now_monotonic=0.0,
        roi_rect=ROI_RECT,
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
        normal_target=target,
    )

    assert result.warning_zones[0].avoid_side == "left"


def test_special_candidate_line_can_trigger_when_lane_centerline_is_clear() -> None:
    lane = [(150.0, float(y)) for y in range(199, -1, -4)]
    result = plan(
        [car((70, 80, 90, 120))],
        centerline=lane,
        candidate_route=[(100.0, 199.0), (50.0, 40.0)],
        boundaries=boundary_rows(left_x=20, right_x=180),
    )

    assert result.active
    assert result.warning_zones[0].avoid_side == "right"
    assert all(
        x == pytest.approx(180.0)
        for x, y in result.shifted_centerline_points
        if 70.0 <= y <= 130.0
    )
    assert_route_clear(result)


def test_candidate_route_blocked_without_centerline_stops() -> None:
    result = plan(
        [car((80, 60, 120, 100))],
        centerline=[],
        candidate_route=[(100.0, 199.0), (100.0, 0.0)],
    )

    assert result.stop_required
    assert result.mode == "CAR_AVOID_STOP"


def test_warning_rows_use_exact_track_boundary_and_smoothstep_transition() -> None:
    result = plan([car((80, 60, 120, 100))])
    route = {round(y, 6): x for x, y in result.shifted_centerline_points}

    assert route[10.0] == pytest.approx(90.0)
    assert route[30.0] == pytest.approx(65.0)
    assert route[50.0] == pytest.approx(40.0)
    assert route[110.0] == pytest.approx(40.0)
    assert route[130.0] == pytest.approx(65.0)
    assert route[150.0] == pytest.approx(90.0)
    assert abs(route[11.0] - route[10.0]) < abs(route[30.0] - route[29.0])
    assert result.target_result.target_point_roi[1] == pytest.approx(80.0)
    assert_route_clear(result)


def test_short_invalid_boundary_gap_is_interpolated() -> None:
    result = plan(
        [car((80, 60, 120, 100))],
        boundaries=boundary_rows(invalid_ys=set(range(78, 84))),
    )

    assert result.active
    assert not result.stop_required
    route = {round(y): x for x, y in result.shifted_centerline_points}
    assert route[80] == pytest.approx(40.0)
    assert_route_clear(result)


@pytest.mark.parametrize(
    "boundaries",
    [
        boundary_rows(invalid_ys=set(range(70, 91))),
        [
            LaneBoundaryRow(y, 40, 160, y <= 80, y <= 80)
            for y in range(199, -1, -1)
        ],
    ],
)
def test_unbounded_or_long_boundary_gap_stops(
    boundaries: list[LaneBoundaryRow],
) -> None:
    result = plan([car((80, 60, 120, 100))], boundaries=boundaries)

    assert result.stop_required
    assert result.mode == "CAR_AVOID_STOP"
    assert result.shifted_centerline_points == []


def test_measured_roi_edge_boundary_is_valid_and_speed_limited() -> None:
    centerline = [(20.0, float(y)) for y in range(199, -1, -4)]
    result = plan(
        [car((20, 70, 60, 110))],
        centerline=centerline,
        boundaries=boundary_rows(left_x=0, right_x=160),
    )

    assert result.active
    assert not result.stop_required
    assert result.edge_limited
    assert result.mode == "CAR_AVOID_EDGE"
    assert_route_clear(result)

    hint = build_car_avoidance_hint(result, min_speed=0.45)
    command = HighLevelPlanner({}).plan(make_tracked_state(), hint)
    assert hint is not None
    assert hint.speed_limit == pytest.approx(0.45)
    assert command.target_speed == pytest.approx(0.45)


def test_boundary_that_still_crosses_warning_zone_stops() -> None:
    result = plan(
        [car((80, 60, 120, 100))],
        boundaries=boundary_rows(left_x=80, right_x=160),
    )

    assert result.stop_required
    assert result.mode == "CAR_AVOID_STOP"
    hint = build_car_avoidance_hint(result, min_speed=0.45)
    command = HighLevelPlanner({}).plan(make_tracked_state(), hint)
    assert hint is not None and hint.stop
    assert command.target_speed == 0.0
    assert resolve_configured_speed_state(command.target_speed, 2) == 0


def test_multiple_same_side_cars_share_track_boundary() -> None:
    result = plan(
        [
            car((80, 70, 120, 110)),
            car((90, 70, 130, 110)),
        ],
    )

    assert result.active
    assert not result.stop_required
    assert len(result.warning_zones) == 2
    assert {zone.avoid_side for zone in result.warning_zones} == {"left"}
    assert_route_clear(result)


def test_overlapping_opposite_boundary_requirements_stop() -> None:
    result = plan(
        [
            car((110, 70, 130, 110)),
            car((70, 70, 90, 110)),
        ],
        centerline=[(100.0, float(y)) for y in range(199, -1, -4)],
        candidate_route=[
            (100.0, 199.0),
            (120.0, 100.0),
            (80.0, 80.0),
            (100.0, 0.0),
        ],
    )

    assert result.stop_required
    assert {zone.avoid_side for zone in result.warning_zones} == {"left", "right"}


def test_vertically_separated_opposite_sides_remain_feasible() -> None:
    result = plan(
        [
            car((110, 35, 130, 55)),
            car((70, 125, 90, 145)),
        ],
        centerline=[(100.0, float(y)) for y in range(199, -1, -4)],
        candidate_route=[
            (100.0, 199.0),
            (80.0, 135.0),
            (100.0, 100.0),
            (120.0, 45.0),
            (100.0, 0.0),
        ],
    )

    assert result.active
    assert not result.stop_required
    assert {zone.avoid_side for zone in result.warning_zones} == {"left", "right"}
    assert_route_clear(result)


def test_cached_detection_cannot_start_recovery_and_new_clear_result_can() -> None:
    planner = make_planner()
    active = plan([car((80, 60, 120, 100))], planner=planner, detection_id=10, now=5.0)
    cached = plan([], planner=planner, detection_id=10, now=5.5)
    started = plan([], planner=planner, detection_id=11, now=6.0)

    assert active.mode == "CAR_AVOID"
    assert cached.mode == "CAR_AVOID"
    assert started.mode == "CAR_AVOID_RECOVERY"
    assert started.recovery_progress == pytest.approx(0.0)


def test_recovery_smoothstep_returns_to_current_centerline_in_one_second() -> None:
    planner = make_planner()
    active = plan([car((80, 60, 120, 100))], planner=planner, detection_id=1, now=10.0)
    started = plan([], planner=planner, detection_id=2, now=11.0)
    halfway = plan([], planner=planner, detection_id=2, now=11.5)
    complete = plan([], planner=planner, detection_id=2, now=12.0)

    active_route = {round(y): x for x, y in active.shifted_centerline_points}
    started_route = {round(y): x for x, y in started.shifted_centerline_points}
    halfway_route = {round(y): x for x, y in halfway.shifted_centerline_points}
    assert started_route[80] == pytest.approx(active_route[80])
    assert halfway_route[80] == pytest.approx(65.0)
    assert halfway.recovery_progress == pytest.approx(0.5)
    assert halfway.target_result.target_point_roi[1] == pytest.approx(80.0)
    assert not complete.active
    assert complete.mode == "LANE_FOLLOW"


def test_car_reappearing_during_recovery_cancels_release() -> None:
    planner = make_planner()
    plan([car((80, 60, 120, 100))], planner=planner, detection_id=1, now=1.0)
    plan([], planner=planner, detection_id=2, now=2.0)
    recovered = plan(
        [car((80, 60, 120, 100))],
        planner=planner,
        detection_id=3,
        now=2.5,
    )

    assert recovered.mode == "CAR_AVOID"
    assert recovered.recovery_progress == pytest.approx(0.0)


def test_non_car_and_clear_car_do_not_activate() -> None:
    human = DetectedObject("human", 0.9, (80, 60, 120, 100))
    result = plan([human, car((150, 60, 180, 100))])

    assert not result.active
    assert len(result.warning_zones) == 1


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"box_scale": 0.9}, "box_scale"),
        ({"transition_margin_px": 0}, "transition_margin_px"),
        ({"edge_slow_margin_px": -1}, "edge_slow_margin_px"),
        ({"release_duration_s": 0}, "release_duration_s"),
    ],
)
def test_config_is_validated(config: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_planner(**config)
