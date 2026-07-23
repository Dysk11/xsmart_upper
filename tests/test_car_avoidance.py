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
        "entry_duration_s": 1.0,
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


def car(
    bbox: tuple[int, int, int, int],
    confidence: float = 0.9,
) -> DetectedObject:
    return DetectedObject("car", confidence, bbox)


def normal_target(points: list[tuple[float, float]] = CENTERLINE):
    return TargetSelector({"fixed_target_y": 80}).select(points, 200, 200, 0.9)


def plan(
    objects: list[DetectedObject],
    *,
    planner: CarAvoidancePlanner | None = None,
    centerline: list[tuple[float, float]] = CENTERLINE,
    boundaries: list[LaneBoundaryRow] | None = None,
    detection_id: int = 1,
    now: float = 0.0,
):
    planner = planner or make_planner()
    return planner.plan(
        objects=objects,
        centerline_points=centerline,
        track_boundary_rows=boundaries or boundary_rows(),
        detection_result_id=detection_id,
        now_monotonic=now,
        roi_rect=ROI_RECT,
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
        normal_target=normal_target(centerline),
    )


def fully_entered(
    objects: list[DetectedObject],
    *,
    planner: CarAvoidancePlanner | None = None,
    centerline: list[tuple[float, float]] = CENTERLINE,
    boundaries: list[LaneBoundaryRow] | None = None,
    detection_id: int = 1,
):
    planner = planner or make_planner()
    plan(
        objects,
        planner=planner,
        centerline=centerline,
        boundaries=boundaries,
        detection_id=detection_id,
        now=0.0,
    )
    return plan(
        objects,
        planner=planner,
        centerline=centerline,
        boundaries=boundaries,
        detection_id=detection_id,
        now=1.0,
    )


def route_x(result, y: float = 80.0) -> float:
    return CarAvoidancePlanner._interpolate_x(
        result.shifted_centerline_points,
        y,
    )


def assert_route_clear(result) -> None:
    for zone in result.warning_zones:
        assert not CarAvoidancePlanner.polyline_intersects_rect(
            result.shifted_centerline_points,
            zone.bbox_roi,
        )


def test_original_box_is_used_without_expansion() -> None:
    result = fully_entered([car((80, 60, 120, 100))])

    assert result.warning_zones[0].bbox_frame == pytest.approx((80, 60, 120, 100))
    assert result.warning_zones[0].bbox_roi == pytest.approx((80, 60, 120, 100))


def test_box_that_only_old_expansion_would_put_in_roi_does_not_trigger() -> None:
    result = plan([car((202, 60, 222, 100))])

    assert not result.active
    assert result.warning_zones == []


@pytest.mark.parametrize(
    "bbox",
    [
        (200, 60, 220, 100),
        (-20, 60, 0, 100),
        (80, 200, 120, 220),
        (80, -20, 120, 0),
        (190, 60, 220, 100),
        (80, 60, 120, 100),
    ],
)
def test_original_box_contact_or_overlap_triggers(
    bbox: tuple[int, int, int, int],
) -> None:
    result = plan([car(bbox)])

    assert result.active
    assert len(result.warning_zones) == 1


@pytest.mark.parametrize(
    ("center_x", "side", "expected_boundary"),
    [
        (90.0, "left", 40.0),
        (110.0, "right", 160.0),
        (100.0, "left", 40.0),
    ],
)
def test_centerline_at_car_height_selects_and_locks_side(
    center_x: float,
    side: str,
    expected_boundary: float,
) -> None:
    centerline = [(center_x, float(y)) for y in range(199, -1, -4)]
    result = fully_entered(
        [car((80, 60, 120, 100))],
        centerline=centerline,
    )

    assert result.locked_side == side
    assert {zone.avoid_side for zone in result.warning_zones} == {side}
    assert route_x(result) == pytest.approx(expected_boundary)
    assert result.target_result.target_point_roi[1] == pytest.approx(80.0)
    assert_route_clear(result)


def test_side_comparison_maps_nonzero_frame_roi_to_roi_coordinates() -> None:
    planner = make_planner()
    centerline = [(70.0, float(y)) for y in range(199, -1, -4)]
    obj = car((80, 90, 120, 130))
    target = normal_target(centerline)

    planner.plan(
        objects=[obj],
        centerline_points=centerline,
        track_boundary_rows=boundary_rows(),
        detection_result_id=1,
        now_monotonic=0.0,
        roi_rect=(20, 50, 220, 250),
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
        normal_target=target,
    )
    result = planner.plan(
        objects=[obj],
        centerline_points=centerline,
        track_boundary_rows=boundary_rows(),
        detection_result_id=1,
        now_monotonic=1.0,
        roi_rect=(20, 50, 220, 250),
        roi_width=200,
        roi_height=200,
        lane_confidence=0.9,
        normal_target=target,
    )

    # ROI x=70 maps to frame x=90, which is left of car center x=100.
    assert result.locked_side == "left"
    assert result.warning_zones[0].bbox_roi == pytest.approx((60, 40, 100, 80))


def test_side_remains_locked_when_centerline_moves_across_car() -> None:
    planner = make_planner()
    first_centerline = [(90.0, float(y)) for y in range(199, -1, -4)]
    moved_centerline = [(180.0, float(y)) for y in range(199, -1, -4)]

    first = plan(
        [car((80, 60, 120, 100))],
        planner=planner,
        centerline=first_centerline,
        detection_id=1,
        now=0.0,
    )
    moved = plan(
        [car((80, 60, 120, 100))],
        planner=planner,
        centerline=moved_centerline,
        detection_id=2,
        now=0.5,
    )
    complete = plan(
        [car((80, 60, 120, 100))],
        planner=planner,
        centerline=moved_centerline,
        detection_id=3,
        now=1.0,
    )

    assert first.locked_side == "left"
    assert moved.locked_side == "left"
    assert complete.locked_side == "left"
    assert route_x(complete) == pytest.approx(40.0)


def test_primary_car_is_lowest_then_highest_confidence() -> None:
    centerline = [(100.0, float(y)) for y in range(199, -1, -4)]
    lower_wins = fully_entered(
        [
            car((70, 30, 90, 70), confidence=0.99),
            car((110, 90, 130, 130), confidence=0.60),
        ],
        centerline=centerline,
    )
    confidence_wins = fully_entered(
        [
            car((70, 90, 90, 130), confidence=0.60),
            car((110, 90, 130, 130), confidence=0.99),
        ],
        centerline=centerline,
    )

    assert lower_wins.locked_side == "left"
    assert confidence_wins.locked_side == "left"


def test_entry_uses_one_second_smoothstep_and_stops_while_route_collides() -> None:
    planner = make_planner()
    obj = car((80, 60, 120, 100))

    started = plan([obj], planner=planner, now=10.0)
    halfway = plan([obj], planner=planner, now=10.5)
    complete = plan([obj], planner=planner, now=11.0)

    assert route_x(started) == pytest.approx(90.0)
    assert started.transition_phase == "entry"
    assert started.transition_progress == pytest.approx(0.0)
    assert started.mode == "CAR_AVOID_STOP"
    assert route_x(halfway) == pytest.approx(65.0)
    assert halfway.transition_progress == pytest.approx(0.5)
    assert halfway.mode == "CAR_AVOID"
    assert route_x(complete) == pytest.approx(40.0)
    assert complete.transition_phase == "hold"
    assert complete.transition_progress == pytest.approx(1.0)
    assert complete.target_result.target_point_roi[1] == pytest.approx(80.0)


def test_side_car_can_move_during_entry_when_blended_route_is_clear() -> None:
    lane = [(90.0, float(y)) for y in range(199, -1, -4)]
    result = plan(
        [car((150, 60, 180, 100))],
        centerline=lane,
    )

    assert result.active
    assert not result.stop_required
    assert result.locked_side == "left"
    assert result.mode == "CAR_AVOID"


def test_short_invalid_boundary_gap_is_interpolated() -> None:
    result = fully_entered(
        [car((80, 60, 120, 100))],
        boundaries=boundary_rows(invalid_ys=set(range(78, 84))),
    )

    assert result.active
    assert not result.stop_required
    assert route_x(result) == pytest.approx(40.0)


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
    result = plan(
        [car((80, 60, 120, 100))],
        boundaries=boundaries,
    )

    assert result.stop_required
    assert result.mode == "CAR_AVOID_STOP"
    assert result.shifted_centerline_points == []


def test_measured_roi_edge_boundary_is_valid_and_speed_limited() -> None:
    centerline = [(20.0, float(y)) for y in range(199, -1, -4)]
    result = fully_entered(
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


def test_boundary_that_still_crosses_original_car_box_stops() -> None:
    result = fully_entered(
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


def test_any_secondary_car_intersecting_locked_route_stops() -> None:
    result = fully_entered(
        [
            car((80, 80, 120, 130), confidence=0.9),
            car((30, 30, 50, 70), confidence=0.8),
        ],
    )

    assert result.locked_side == "left"
    assert result.stop_required
    assert result.mode == "CAR_AVOID_STOP"


def test_multiple_cars_clear_of_locked_route_remain_feasible() -> None:
    result = fully_entered(
        [
            car((80, 80, 120, 130)),
            car((90, 30, 130, 70)),
        ],
    )

    assert result.locked_side == "left"
    assert not result.stop_required
    assert_route_clear(result)


def test_cached_detection_cannot_start_recovery_and_new_clear_result_can() -> None:
    planner = make_planner()
    obj = car((80, 60, 120, 100))
    plan([obj], planner=planner, detection_id=10, now=5.0)
    active = plan([obj], planner=planner, detection_id=10, now=6.0)
    cached = plan([], planner=planner, detection_id=10, now=6.5)
    started = plan([], planner=planner, detection_id=11, now=7.0)

    assert active.transition_phase == "hold"
    assert cached.transition_phase == "hold"
    assert started.mode == "CAR_AVOID_RECOVERY"
    assert started.transition_progress == pytest.approx(0.0)


def test_recovery_smoothstep_returns_to_current_centerline_in_one_second() -> None:
    planner = make_planner()
    obj = car((80, 60, 120, 100))
    plan([obj], planner=planner, detection_id=1, now=10.0)
    active = plan([obj], planner=planner, detection_id=1, now=11.0)
    started = plan([], planner=planner, detection_id=2, now=12.0)
    halfway = plan([], planner=planner, detection_id=2, now=12.5)
    complete = plan([], planner=planner, detection_id=2, now=13.0)

    assert route_x(started) == pytest.approx(route_x(active))
    assert route_x(halfway) == pytest.approx(65.0)
    assert halfway.transition_phase == "recovery"
    assert halfway.transition_progress == pytest.approx(0.5)
    assert halfway.target_result.target_point_roi[1] == pytest.approx(80.0)
    assert not complete.active
    assert complete.mode == "LANE_FOLLOW"
    assert complete.locked_side is None


def test_car_reappearing_during_recovery_has_no_route_jump() -> None:
    planner = make_planner()
    obj = car((80, 60, 120, 100))
    plan([obj], planner=planner, detection_id=1, now=1.0)
    plan([obj], planner=planner, detection_id=1, now=2.0)
    plan([], planner=planner, detection_id=2, now=3.0)
    halfway = plan([], planner=planner, detection_id=2, now=3.5)
    reappeared = plan([obj], planner=planner, detection_id=3, now=3.5)
    resumed = plan([obj], planner=planner, detection_id=3, now=4.0)

    assert route_x(reappeared) == pytest.approx(route_x(halfway))
    assert reappeared.transition_phase == "entry"
    assert reappeared.transition_progress == pytest.approx(0.0)
    assert reappeared.locked_side == "left"
    assert route_x(resumed) == pytest.approx(52.5)


def test_car_in_roi_without_centerline_stops() -> None:
    result = plan(
        [car((80, 60, 120, 100))],
        centerline=[],
    )

    assert result.stop_required
    assert result.mode == "CAR_AVOID_STOP"


def test_non_car_objects_do_not_activate_avoidance() -> None:
    result = plan([DetectedObject("human", 0.9, (80, 60, 120, 100))])

    assert not result.active


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"entry_duration_s": 0}, "entry_duration_s"),
        ({"edge_slow_margin_px": -1}, "edge_slow_margin_px"),
        ({"release_duration_s": 0}, "release_duration_s"),
    ],
)
def test_config_is_validated(config: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_planner(**config)
