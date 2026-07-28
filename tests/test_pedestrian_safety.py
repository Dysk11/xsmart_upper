from __future__ import annotations

import pytest

from core.io.protocol import resolve_configured_speed_state
from core.object.blocking import DetectedObject
from core.object.pedestrian_safety import (
    PedestrianSafetyAnalyzer,
    PedestrianSafetyResult,
)
from core.planning.high_level import HighLevelPlanner, build_safety_stop_hint
from tests.test_stop_policy import make_tracked_state


ROI_RECT = (10, 20, 210, 120)
CENTER_REGION = (70.0, 20.0, 150.0, 120.0)


def make_analyzer(**overrides: object) -> PedestrianSafetyAnalyzer:
    config: dict[str, object] = {
        "enabled": True,
        "slowdown_min_box_area_px": 0,
        "stop_min_box_area_px": 600,
        "approach_speed_state": 2,
        "rearm_cooldown_sec": 3.0,
        "target_stability_threshold_px": 20,
        "target_stability_confirm_frames": 2,
        "crossing_confirm_frames": 3,
        "moving_away_min_delta_px": 3,
        "moving_away_confirm_frames": 2,
        "center_region": {
            "left_ratio": 0.30,
            "right_ratio": 0.70,
        },
    }
    config.update(overrides)
    return PedestrianSafetyAnalyzer(config)


def detected_center(
    center_x: int,
    center_y: int = 60,
    width: int = 30,
    height: int = 30,
    *,
    class_name: str = "human",
    confidence: float = 0.9,
) -> DetectedObject:
    half_width = width // 2
    half_height = height // 2
    return DetectedObject(
        class_name=class_name,
        confidence=confidence,
        bbox_frame=(
            center_x - half_width,
            center_y - half_height,
            center_x - half_width + width,
            center_y - half_height + height,
        ),
    )


def analyze(
    analyzer: PedestrianSafetyAnalyzer,
    objects: list[DetectedObject],
    *,
    target_x_roi: float = 90.0,
    result_id: int,
    now: float,
) -> PedestrianSafetyResult:
    return analyzer.analyze(
        objects=objects,
        avoidance_roi_rect=ROI_RECT,
        target_x_frame=float(ROI_RECT[0]) + target_x_roi,
        detection_result_id=result_id,
        now_monotonic=now,
    )


def lock_target(
    analyzer: PedestrianSafetyAnalyzer,
    objects: list[DetectedObject],
    *,
    target_x_roi: float = 90.0,
    result_id: int,
    now: float,
) -> PedestrianSafetyResult:
    result = analyze(
        analyzer,
        objects,
        target_x_roi=target_x_roi,
        result_id=result_id,
        now=now,
    )
    result = analyze(
        analyzer,
        objects,
        target_x_roi=target_x_roi,
        result_id=result_id,
        now=now + 0.01,
    )
    return result


def establish_crossing_baseline(
    analyzer: PedestrianSafetyAnalyzer,
    objects: list[DetectedObject],
    *,
    target_x_roi: float = 90.0,
    result_id: int,
    now: float,
) -> PedestrianSafetyResult:
    return analyze(
        analyzer,
        objects,
        target_x_roi=target_x_roi,
        result_id=result_id,
        now=now,
    )


def trigger_lock_and_establish_baseline(
    analyzer: PedestrianSafetyAnalyzer,
    start_x: int,
    *,
    target_x_roi: float = 90.0,
) -> PedestrianSafetyResult:
    objects = [detected_center(start_x)]
    analyze(
        analyzer,
        objects,
        target_x_roi=target_x_roi,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        objects,
        target_x_roi=target_x_roi,
        result_id=1,
        now=0.01,
    )
    return establish_crossing_baseline(
        analyzer,
        objects,
        target_x_roi=target_x_roi,
        result_id=2,
        now=0.1,
    )


def stop_result() -> PedestrianSafetyResult:
    return PedestrianSafetyResult(
        stop_required=True,
        latched=True,
        armed=False,
        center_region_frame=CENTER_REGION,
        frozen_target_x_frame=100.0,
        target_region="center",
        tracked_center_frame=(80.0, 60.0),
        human_count=1,
        cooldown_remaining_sec=0.0,
        reason="test pedestrian stop",
    )


def test_center_region_and_cooldown_are_strictly_validated() -> None:
    with pytest.raises(ValueError, match="left_ratio"):
        make_analyzer(
            center_region={
                "left_ratio": 0.7,
                "right_ratio": 0.3,
            }
        )
    with pytest.raises(ValueError, match="rearm_cooldown_sec"):
        make_analyzer(rearm_cooldown_sec=-1)
    with pytest.raises(ValueError, match="target_stability_threshold_px"):
        make_analyzer(target_stability_threshold_px=float("inf"))
    with pytest.raises(ValueError, match="target_stability_confirm_frames"):
        make_analyzer(target_stability_confirm_frames=1.5)
    with pytest.raises(ValueError, match="crossing_confirm_frames"):
        make_analyzer(crossing_confirm_frames=0)
    with pytest.raises(ValueError, match="crossing_confirm_frames"):
        make_analyzer(crossing_confirm_frames=-1)
    with pytest.raises(ValueError, match="crossing_confirm_frames"):
        make_analyzer(crossing_confirm_frames=True)
    with pytest.raises(ValueError, match="crossing_confirm_frames"):
        make_analyzer(crossing_confirm_frames=1.5)
    with pytest.raises(ValueError, match="moving_away_min_delta_px"):
        make_analyzer(moving_away_min_delta_px=float("inf"))
    with pytest.raises(ValueError, match="moving_away_min_delta_px"):
        make_analyzer(moving_away_min_delta_px=-1)
    with pytest.raises(ValueError, match="moving_away_confirm_frames"):
        make_analyzer(moving_away_confirm_frames=1.5)
    with pytest.raises(ValueError, match="slowdown_min_box_area_px"):
        make_analyzer(slowdown_min_box_area_px=float("inf"))
    with pytest.raises(ValueError, match="stop_min_box_area_px"):
        make_analyzer(stop_min_box_area_px=-1)
    with pytest.raises(ValueError, match="must not exceed"):
        make_analyzer(
            slowdown_min_box_area_px=601,
            stop_min_box_area_px=600,
        )
    with pytest.raises(ValueError, match="approach_speed_state"):
        make_analyzer(approach_speed_state=0)


def test_crossing_confirmation_defaults_to_three_frames() -> None:
    analyzer = PedestrianSafetyAnalyzer({})

    assert analyzer.crossing_confirm_frames == 3


@pytest.mark.parametrize(
    "objects",
    [
        [detected_center(100, width=10, height=10)],
        [detected_center(210, width=40, height=40)],
        [detected_center(100, class_name="car")],
    ],
)
def test_non_qualifying_objects_do_not_trigger(
    objects: list[DetectedObject],
) -> None:
    result = analyze(
        make_analyzer(),
        objects,
        result_id=1,
        now=0.0,
    )

    assert result.center_region_frame == CENTER_REGION
    assert result.armed
    assert not result.stop_required


def test_exact_stop_area_threshold_does_not_trigger() -> None:
    result = analyze(
        make_analyzer(),
        [detected_center(100, width=20, height=30)],
        result_id=1,
        now=0.0,
    )

    assert result.armed
    assert not result.stop_required
    assert result.tracked_center_frame is None


def test_area_strictly_above_stop_threshold_triggers_inside_roi() -> None:
    result = analyze(
        make_analyzer(),
        [detected_center(100, width=21, height=30)],
        result_id=1,
        now=0.0,
    )

    assert result.stop_required
    assert result.frozen_target_x_frame is None
    assert result.target_region == "none"
    assert result.tracked_center_frame == (100.5, 60.0)


def test_configured_1600_area_boundary_is_strict() -> None:
    exact = analyze(
        make_analyzer(stop_min_box_area_px=1600),
        [detected_center(100, width=40, height=40)],
        result_id=1,
        now=0.0,
    )
    above = analyze(
        make_analyzer(stop_min_box_area_px=1600),
        [detected_center(100, width=41, height=40)],
        result_id=1,
        now=0.0,
    )

    assert not exact.stop_required
    assert above.stop_required


def test_legacy_min_box_area_config_is_used_as_stop_threshold() -> None:
    analyzer = PedestrianSafetyAnalyzer(
        {
            "min_box_area_px": 600,
            "slowdown_min_box_area_px": 0,
        }
    )

    exact = analyze(
        analyzer,
        [detected_center(100, width=20, height=30)],
        result_id=1,
        now=0.0,
    )
    above = analyze(
        analyzer,
        [detected_center(100, width=21, height=30)],
        result_id=2,
        now=0.1,
    )

    assert not exact.stop_required
    assert above.stop_required


def test_trigger_and_regions_use_dedicated_avoidance_roi() -> None:
    analyzer = make_analyzer()
    result = analyzer.analyze(
        objects=[detected_center(100, center_y=60)],
        avoidance_roi_rect=(50, 20, 150, 120),
        target_x_frame=170.0,
        detection_result_id=1,
        now_monotonic=0.0,
    )

    assert result.stop_required
    assert result.center_region_frame == pytest.approx((80.0, 20.0, 120.0, 120.0))
    assert result.frozen_target_x_frame is None
    for now in (0.01, 0.02):
        result = analyzer.analyze(
            objects=[detected_center(100, center_y=60)],
            avoidance_roi_rect=(50, 20, 150, 120),
            target_x_frame=170.0,
            detection_result_id=1,
            now_monotonic=now,
        )
    assert result.frozen_target_x_frame == pytest.approx(170.0)
    assert result.target_region == "right"


def test_human_inside_lane_roi_but_outside_avoidance_roi_does_not_trigger() -> None:
    analyzer = make_analyzer()
    result = analyzer.analyze(
        objects=[detected_center(100, center_y=80)],
        avoidance_roi_rect=(0, 0, 60, 60),
        target_x_frame=100.0,
        detection_result_id=1,
        now_monotonic=0.0,
    )

    assert result.armed
    assert not result.stop_required


def test_largest_qualifying_human_is_selected_deterministically() -> None:
    result = analyze(
        make_analyzer(),
        [
            detected_center(80, width=30, height=30, confidence=0.99),
            detected_center(120, width=40, height=40, confidence=0.80),
        ],
        result_id=1,
        now=0.0,
    )

    assert result.stop_required
    assert result.human_count == 2
    assert result.tracked_center_frame == (120.0, 60.0)


@pytest.mark.parametrize(
    ("target_x_roi", "expected_region"),
    [
        (59.0, "left"),
        (60.0, "center"),
        (140.0, "center"),
        (141.0, "right"),
    ],
)
def test_target_region_uses_fixed_center_boundaries(
    target_x_roi: float,
    expected_region: str,
) -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(100)],
        target_x_roi=target_x_roi,
        result_id=1,
        now=0.0,
    )
    result = lock_target(
        analyzer,
        [detected_center(100)],
        target_x_roi=target_x_roi,
        result_id=1,
        now=0.01,
    )

    assert result.target_region == expected_region
    assert result.frozen_target_x_frame == pytest.approx(10.0 + target_x_roi)


def test_cached_lane_frames_lock_after_two_strictly_small_target_jumps() -> None:
    analyzer = make_analyzer()
    trigger = analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.0,
    )
    first_stable = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=109.0,
        result_id=1,
        now=0.01,
    )
    locked = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=128.0,
        result_id=1,
        now=0.02,
    )

    assert trigger.stop_required
    assert trigger.frozen_target_x_frame is None
    assert first_stable.frozen_target_x_frame is None
    assert "target stabilizing 1/2" in first_stable.reason
    assert locked.frozen_target_x_frame == pytest.approx(138.0)
    assert locked.tracked_center_frame == (80.0, 60.0)


def test_exact_threshold_resets_target_stability_count() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.0,
    )
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=109.0,
        result_id=1,
        now=0.01,
    )
    exact_threshold = analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=129.0,
        result_id=1,
        now=0.02,
    )
    first_after_reset = analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=148.0,
        result_id=1,
        now=0.03,
    )
    locked = analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=167.0,
        result_id=1,
        now=0.04,
    )

    assert exact_threshold.frozen_target_x_frame is None
    assert "target stabilizing 0/2" in exact_threshold.reason
    assert "target stabilizing 1/2" in first_after_reset.reason
    assert locked.frozen_target_x_frame == pytest.approx(177.0)


def test_movement_before_target_lock_cannot_release_pedestrian_wait() -> None:
    analyzer = make_analyzer(crossing_confirm_frames=1)
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.0,
    )
    moved_before_lock = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=120.0,
        result_id=2,
        now=0.1,
    )
    analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=90.0,
        result_id=2,
        now=0.11,
    )
    analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=90.0,
        result_id=2,
        now=0.12,
    )
    locked = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=90.0,
        result_id=2,
        now=0.13,
    )
    baseline = analyze(
        analyzer,
        [detected_center(80)],
        result_id=3,
        now=0.2,
    )
    released = analyze(
        analyzer,
        [detected_center(120)],
        result_id=4,
        now=0.3,
    )

    assert moved_before_lock.stop_required
    assert moved_before_lock.frozen_target_x_frame is None
    assert locked.frozen_target_x_frame == pytest.approx(100.0)
    assert baseline.stop_required
    assert "crossing baseline established" in baseline.reason
    assert not released.stop_required


def test_locked_target_offset_below_threshold_keeps_frozen_line() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=2,
        now=0.1,
    )

    result = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=109.0,
        result_id=2,
        now=0.2,
    )

    assert result.frozen_target_x_frame == pytest.approx(100.0)
    assert result.target_region == "center"
    assert result.tracked_center_frame == (80.0, 60.0)
    assert "offset=19.0px" in result.reason


def test_exact_locked_target_offset_relocks_on_next_cached_lane_frame() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.01,
    )

    invalidated = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=110.0,
        result_id=1,
        now=0.1,
    )
    relocked = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=170.0,
        result_id=1,
        now=0.2,
    )

    assert invalidated.stop_required
    assert invalidated.frozen_target_x_frame is None
    assert invalidated.target_region == "none"
    assert invalidated.tracked_center_frame == (80.0, 60.0)
    assert "target relock pending" in invalidated.reason
    assert "offset=20.0px" in invalidated.reason
    assert relocked.frozen_target_x_frame == pytest.approx(180.0)
    assert relocked.target_region == "right"
    assert relocked.tracked_center_frame == (80.0, 60.0)
    assert "target relocked" in relocked.reason


def test_relock_discards_old_crossing_and_requires_new_baseline() -> None:
    analyzer = make_analyzer(crossing_confirm_frames=1)
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=2,
        now=0.1,
    )

    crossed_old_line = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=110.0,
        result_id=3,
        now=0.2,
    )
    relocked = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=150.0,
        result_id=3,
        now=0.21,
    )
    baseline = analyze(
        analyzer,
        [detected_center(140)],
        target_x_roi=150.0,
        result_id=4,
        now=0.3,
    )
    released = analyze(
        analyzer,
        [detected_center(180)],
        target_x_roi=150.0,
        result_id=5,
        now=0.4,
    )

    assert crossed_old_line.stop_required
    assert crossed_old_line.frozen_target_x_frame is None
    assert relocked.frozen_target_x_frame == pytest.approx(160.0)
    assert baseline.stop_required
    assert "crossing baseline established" in baseline.reason
    assert not released.stop_required


def test_relock_waits_for_finite_target_on_following_lane_frames() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(80)],
        target_x_roi=90.0,
        result_id=1,
        now=0.01,
    )
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=110.0,
        result_id=1,
        now=0.1,
    )

    still_pending = analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=float("nan"),
        result_id=1,
        now=0.2,
    )
    relocked = analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=130.0,
        result_id=1,
        now=0.3,
    )

    assert still_pending.frozen_target_x_frame is None
    assert "target relock pending" in still_pending.reason
    assert relocked.frozen_target_x_frame == pytest.approx(140.0)


@pytest.mark.parametrize(
    ("start_x", "end_x"),
    [
        (80, 120),
        (120, 80),
    ],
)
def test_center_region_releases_on_strict_crossing(
    start_x: int,
    end_x: int,
) -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(start_x)],
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(start_x)],
        result_id=1,
        now=0.01,
    )
    baseline = establish_crossing_baseline(
        analyzer,
        [detected_center(start_x)],
        result_id=2,
        now=0.1,
    )

    first_crossing = analyze(
        analyzer,
        [detected_center(end_x)],
        result_id=3,
        now=1.0,
    )
    second_crossing = analyze(
        analyzer,
        [detected_center(end_x)],
        result_id=4,
        now=1.1,
    )
    released = analyze(
        analyzer,
        [detected_center(end_x)],
        result_id=5,
        now=1.2,
    )

    assert baseline.stop_required
    assert first_crossing.stop_required
    assert "crossing=1/3" in first_crossing.reason
    assert second_crossing.stop_required
    assert "crossing=2/3" in second_crossing.reason
    assert not released.stop_required
    assert "crossed center target 3/3" in released.reason
    assert released.cooldown_remaining_sec == pytest.approx(3.0)


def test_center_region_online_start_requires_a_side_then_opposite_side() -> None:
    analyzer = make_analyzer()
    analyze(analyzer, [detected_center(100)], result_id=1, now=0.0)
    lock_target(
        analyzer,
        [detected_center(100)],
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(100)],
        result_id=2,
        now=0.1,
    )

    first_side = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=0.5,
    )
    first_crossing = analyze(
        analyzer,
        [detected_center(80)],
        result_id=4,
        now=1.0,
    )
    second_crossing = analyze(
        analyzer,
        [detected_center(80)],
        result_id=5,
        now=1.1,
    )
    released = analyze(
        analyzer,
        [detected_center(80)],
        result_id=6,
        now=1.2,
    )

    assert first_side.stop_required
    assert "moving_away=disabled(center crossing-only)" in first_side.reason
    assert "crossing=1/3" in first_crossing.reason
    assert "crossing=2/3" in second_crossing.reason
    assert not released.stop_required


def test_left_region_requires_right_to_left_crossing() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(30)],
        target_x_roi=40.0,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(30)],
        target_x_roi=40.0,
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(30)],
        target_x_roi=40.0,
        result_id=2,
        now=0.1,
    )
    wrong_direction = analyze(
        analyzer,
        [detected_center(70)],
        target_x_roi=40.0,
        result_id=3,
        now=0.5,
    )
    first_crossing = analyze(
        analyzer,
        [detected_center(40)],
        target_x_roi=40.0,
        result_id=4,
        now=1.0,
    )
    second_crossing = analyze(
        analyzer,
        [detected_center(40)],
        target_x_roi=40.0,
        result_id=5,
        now=1.1,
    )
    released = analyze(
        analyzer,
        [detected_center(40)],
        target_x_roi=40.0,
        result_id=6,
        now=1.2,
    )

    assert wrong_direction.stop_required
    assert wrong_direction.frozen_target_x_frame == 50.0
    assert wrong_direction.target_region == "left"
    assert "crossing=1/3" in first_crossing.reason
    assert "crossing=2/3" in second_crossing.reason
    assert not released.stop_required


def test_left_region_releases_when_starting_on_target_line() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(50)],
        target_x_roi=40.0,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(50)],
        target_x_roi=40.0,
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(50)],
        target_x_roi=40.0,
        result_id=2,
        now=0.1,
    )

    first_crossing = analyze(
        analyzer,
        [detected_center(40)],
        target_x_roi=40.0,
        result_id=3,
        now=1.0,
    )
    second_crossing = analyze(
        analyzer,
        [detected_center(40)],
        target_x_roi=40.0,
        result_id=4,
        now=1.1,
    )
    released = analyze(
        analyzer,
        [detected_center(40)],
        target_x_roi=40.0,
        result_id=5,
        now=1.2,
    )

    assert first_crossing.stop_required
    assert second_crossing.stop_required
    assert not released.stop_required


def test_right_region_requires_left_to_right_crossing() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=2,
        now=0.1,
    )
    wrong_direction = analyze(
        analyzer,
        [detected_center(150)],
        target_x_roi=170.0,
        result_id=3,
        now=0.5,
    )
    first_crossing = analyze(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=4,
        now=1.0,
    )
    second_crossing = analyze(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=5,
        now=1.1,
    )
    released = analyze(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=6,
        now=1.2,
    )

    assert wrong_direction.stop_required
    assert wrong_direction.frozen_target_x_frame == 180.0
    assert wrong_direction.target_region == "right"
    assert "crossing=1/3" in first_crossing.reason
    assert "crossing=2/3" in second_crossing.reason
    assert not released.stop_required


def test_right_region_releases_when_starting_on_target_line() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(180)],
        target_x_roi=170.0,
        result_id=1,
        now=0.0,
    )
    lock_target(
        analyzer,
        [detected_center(180)],
        target_x_roi=170.0,
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(180)],
        target_x_roi=170.0,
        result_id=2,
        now=0.1,
    )

    first_crossing = analyze(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=3,
        now=1.0,
    )
    second_crossing = analyze(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=4,
        now=1.1,
    )
    released = analyze(
        analyzer,
        [detected_center(190)],
        target_x_roi=170.0,
        result_id=5,
        now=1.2,
    )

    assert first_crossing.stop_required
    assert second_crossing.stop_required
    assert not released.stop_required


def test_cached_ai_result_does_not_advance_crossing_confirmation() -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, 80)

    first_crossing = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=0.5,
    )
    cached = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=0.6,
    )
    second_crossing = analyze(
        analyzer,
        [detected_center(120)],
        result_id=4,
        now=0.7,
    )
    released = analyze(
        analyzer,
        [detected_center(120)],
        result_id=5,
        now=0.8,
    )

    assert "crossing=1/3" in first_crossing.reason
    assert cached.stop_required
    assert "crossing=1/3" in cached.reason
    assert "crossing=2/3" in second_crossing.reason
    assert not released.stop_required


@pytest.mark.parametrize(
    "reset_objects",
    [
        [detected_center(100, width=10, height=10)],
        [detected_center(80, width=10, height=10)],
    ],
    ids=["on-target-line", "returned-to-source-side"],
)
def test_line_or_source_side_resets_crossing_confirmation(
    reset_objects: list[DetectedObject],
) -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, 80)
    first_crossing = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=0.5,
    )
    reset = analyze(
        analyzer,
        reset_objects,
        result_id=4,
        now=0.6,
    )

    assert "crossing=1/3" in first_crossing.reason
    assert reset.stop_required
    assert "crossing=" not in reset.reason
    assert analyzer.crossing_count == 0
    assert analyzer.crossing_destination_side is None


@pytest.mark.parametrize(
    "missing_objects",
    [
        [],
        [detected_center(220, width=10, height=10)],
    ],
    ids=["missing", "outside-avoidance-roi"],
)
def test_missing_or_roi_exit_resets_crossing_confirmation(
    missing_objects: list[DetectedObject],
) -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, 80)
    first_crossing = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=0.5,
    )
    missing = analyze(
        analyzer,
        missing_objects,
        result_id=4,
        now=0.6,
    )
    reappeared = analyze(
        analyzer,
        [detected_center(120, width=10, height=10)],
        result_id=5,
        now=0.7,
    )

    assert "crossing=1/3" in first_crossing.reason
    assert missing.stop_required
    assert "triggering pedestrian missing" in missing.reason
    assert analyzer.crossing_count == 0
    assert reappeared.stop_required
    assert "crossing baseline established" in reappeared.reason


def test_target_relock_resets_crossing_confirmation() -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, 80)
    first_crossing = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=0.5,
    )
    invalidated = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=110.0,
        result_id=3,
        now=0.6,
    )
    relocked = analyze(
        analyzer,
        [detected_center(120)],
        target_x_roi=150.0,
        result_id=3,
        now=0.7,
    )

    assert "crossing=1/3" in first_crossing.reason
    assert invalidated.stop_required
    assert "target relock pending" in invalidated.reason
    assert analyzer.crossing_count == 0
    assert relocked.stop_required
    assert relocked.frozen_target_x_frame == pytest.approx(160.0)


def test_crossing_confirmation_suppresses_moving_away_confirmation() -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, 70, target_x_roi=40.0)

    first_crossing = analyze(
        analyzer,
        [detected_center(40)],
        target_x_roi=40.0,
        result_id=3,
        now=0.5,
    )
    second_crossing = analyze(
        analyzer,
        [detected_center(34)],
        target_x_roi=40.0,
        result_id=4,
        now=0.6,
    )
    released = analyze(
        analyzer,
        [detected_center(28)],
        target_x_roi=40.0,
        result_id=5,
        now=0.7,
    )

    assert "crossing=1/3" in first_crossing.reason
    assert "crossing=2/3" in second_crossing.reason
    assert "moving_away=" not in second_crossing.reason
    assert not released.stop_required
    assert "crossed left target 3/3" in released.reason


@pytest.mark.parametrize(
    ("start_x", "middle_x", "end_x", "target_x_roi", "expected_region"),
    [
        (40, 37, 34, 40.0, "left"),
        (190, 193, 196, 170.0, "right"),
    ],
)
def test_releases_after_two_confirmed_movements_away_from_target(
    start_x: int,
    middle_x: int,
    end_x: int,
    target_x_roi: float,
    expected_region: str,
) -> None:
    analyzer = make_analyzer()
    baseline = trigger_lock_and_establish_baseline(
        analyzer,
        start_x,
        target_x_roi=target_x_roi,
    )

    first_away = analyze(
        analyzer,
        [detected_center(middle_x)],
        target_x_roi=target_x_roi,
        result_id=3,
        now=0.5,
    )
    released = analyze(
        analyzer,
        [detected_center(end_x)],
        target_x_roi=target_x_roi,
        result_id=4,
        now=1.0,
    )

    assert baseline.target_region == expected_region
    assert first_away.stop_required
    assert "moving_away=1/2" in first_away.reason
    assert "delta=3.0px" in first_away.reason
    assert not released.stop_required
    assert f"moved away from {expected_region} target 2/2" in released.reason
    assert released.cooldown_remaining_sec == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("positions",),
    [
        ((90, 87, 84, 81),),
        ((110, 113, 116, 119),),
    ],
)
def test_center_region_never_releases_for_moving_away(
    positions: tuple[int, ...],
) -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, positions[0])

    result: PedestrianSafetyResult | None = None
    for result_id, center_x in enumerate(positions[1:], start=3):
        result = analyze(
            analyzer,
            [detected_center(center_x)],
            result_id=result_id,
            now=float(result_id),
        )
        assert result.stop_required
        assert result.latched

    assert result is not None
    assert "moving_away=disabled(center crossing-only)" in result.reason


@pytest.mark.parametrize(
    (
        "start_x",
        "middle_x",
        "end_x",
        "target_x_roi",
        "expected_count",
    ),
    [
        (60, 63, 66, 40.0, 0),
        (170, 167, 164, 170.0, 0),
    ],
)
def test_noise_approach_stall_and_wrong_side_do_not_release(
    start_x: int,
    middle_x: int,
    end_x: int,
    target_x_roi: float,
    expected_count: int,
) -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(
        analyzer,
        start_x,
        target_x_roi=target_x_roi,
    )

    analyze(
        analyzer,
        [detected_center(middle_x)],
        target_x_roi=target_x_roi,
        result_id=3,
        now=0.5,
    )
    result = analyze(
        analyzer,
        [detected_center(end_x)],
        target_x_roi=target_x_roi,
        result_id=4,
        now=1.0,
    )

    assert result.stop_required
    assert f"moving_away={expected_count}/2" in result.reason


def test_cached_ai_result_does_not_advance_moving_away_confirmation() -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, 40, target_x_roi=40.0)

    first_away = analyze(
        analyzer,
        [detected_center(37)],
        target_x_roi=40.0,
        result_id=3,
        now=0.5,
    )
    cached = analyze(
        analyzer,
        [detected_center(34)],
        target_x_roi=40.0,
        result_id=3,
        now=0.6,
    )
    released = analyze(
        analyzer,
        [detected_center(34)],
        target_x_roi=40.0,
        result_id=4,
        now=1.0,
    )

    assert "moving_away=1/2" in first_away.reason
    assert cached.stop_required
    assert cached.tracked_center_frame == (37.0, 60.0)
    assert "moving_away=1/2" in cached.reason
    assert "delta=3.0px" in cached.reason
    assert not released.stop_required


def test_missing_pedestrian_resets_moving_away_confirmation() -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, 40, target_x_roi=40.0)

    first_away = analyze(
        analyzer,
        [detected_center(37)],
        target_x_roi=40.0,
        result_id=3,
        now=0.5,
    )
    missing = analyze(
        analyzer,
        [],
        target_x_roi=40.0,
        result_id=4,
        now=0.7,
    )
    after_missing = analyze(
        analyzer,
        [detected_center(34)],
        target_x_roi=40.0,
        result_id=5,
        now=1.0,
    )
    first_after_missing = analyze(
        analyzer,
        [detected_center(31)],
        target_x_roi=40.0,
        result_id=6,
        now=1.5,
    )
    released = analyze(
        analyzer,
        [detected_center(28)],
        target_x_roi=40.0,
        result_id=7,
        now=2.0,
    )

    assert "moving_away=1/2" in first_away.reason
    assert missing.stop_required
    assert "triggering pedestrian missing" in missing.reason
    assert after_missing.stop_required
    assert "crossing baseline established" in after_missing.reason
    assert "moving_away=1/2" in first_after_missing.reason
    assert not released.stop_required


def test_target_relock_resets_moving_away_and_requires_a_new_baseline() -> None:
    analyzer = make_analyzer()
    trigger_lock_and_establish_baseline(analyzer, 40, target_x_roi=40.0)

    first_away = analyze(
        analyzer,
        [detected_center(37)],
        target_x_roi=40.0,
        result_id=3,
        now=0.5,
    )
    invalidated = analyze(
        analyzer,
        [detected_center(34)],
        target_x_roi=60.0,
        result_id=3,
        now=0.6,
    )
    relocked = analyze(
        analyzer,
        [detected_center(34)],
        target_x_roi=20.0,
        result_id=3,
        now=0.7,
    )
    baseline = analyze(
        analyzer,
        [detected_center(25)],
        target_x_roi=20.0,
        result_id=4,
        now=1.0,
    )
    after_relock_first_away = analyze(
        analyzer,
        [detected_center(22)],
        target_x_roi=20.0,
        result_id=5,
        now=1.5,
    )
    released = analyze(
        analyzer,
        [detected_center(19)],
        target_x_roi=20.0,
        result_id=6,
        now=2.0,
    )

    assert "moving_away=1/2" in first_away.reason
    assert invalidated.frozen_target_x_frame is None
    assert relocked.frozen_target_x_frame == pytest.approx(30.0)
    assert baseline.stop_required
    assert "crossing baseline established" in baseline.reason
    assert "moving_away=1/2" in after_relock_first_away.reason
    assert not released.stop_required


def test_nearest_human_is_associated_without_distance_limit() -> None:
    analyzer = make_analyzer()
    analyze(
        analyzer,
        [detected_center(80)],
        target_x_roi=40.0,
        result_id=1,
        now=0.0,
    )

    tracked = analyze(
        analyzer,
        [
            detected_center(500, width=80, height=80),
            detected_center(110, width=10, height=10),
        ],
        target_x_roi=40.0,
        result_id=2,
        now=1.0,
    )

    assert tracked.stop_required
    assert tracked.tracked_center_frame == (110.0, 60.0)


def test_missing_triggering_pedestrian_holds_stop_until_reappearance() -> None:
    analyzer = make_analyzer(crossing_confirm_frames=1)
    analyze(analyzer, [detected_center(80)], result_id=1, now=0.0)
    lock_target(
        analyzer,
        [detected_center(80)],
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(80)],
        result_id=2,
        now=0.1,
    )

    missing = analyze(analyzer, [], result_id=3, now=1.0)
    reappeared = analyze(
        analyzer,
        [detected_center(120, width=10, height=10)],
        result_id=4,
        now=2.0,
    )
    released = analyze(
        analyzer,
        [detected_center(80, width=10, height=10)],
        result_id=5,
        now=2.5,
    )

    assert missing.stop_required
    assert missing.tracked_center_frame == (80.0, 60.0)
    assert reappeared.stop_required
    assert "crossing baseline established" in reappeared.reason
    assert not released.stop_required


def test_roi_exit_resets_release_baseline_and_cannot_cross_outside_roi() -> None:
    analyzer = make_analyzer(crossing_confirm_frames=1)
    trigger_lock_and_establish_baseline(analyzer, 80)

    outside = analyze(
        analyzer,
        [detected_center(220, width=10, height=10)],
        result_id=3,
        now=1.0,
    )
    reentered_opposite_side = analyze(
        analyzer,
        [detected_center(120, width=10, height=10)],
        result_id=4,
        now=1.5,
    )
    held = analyze(
        analyzer,
        [detected_center(120, width=10, height=10)],
        result_id=5,
        now=2.0,
    )
    released = analyze(
        analyzer,
        [detected_center(80, width=10, height=10)],
        result_id=6,
        now=2.5,
    )

    assert outside.stop_required
    assert "triggering pedestrian missing" in outside.reason
    assert reentered_opposite_side.stop_required
    assert "crossing baseline established" in reentered_opposite_side.reason
    assert held.stop_required
    assert not released.stop_required


def test_cached_ai_result_cannot_update_or_release_track() -> None:
    analyzer = make_analyzer(crossing_confirm_frames=1)
    analyze(analyzer, [detected_center(80)], result_id=1, now=0.0)
    lock_target(
        analyzer,
        [detected_center(80)],
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(80)],
        result_id=2,
        now=0.1,
    )

    cached = analyze(
        analyzer,
        [detected_center(120)],
        result_id=2,
        now=0.5,
    )
    released = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=1.0,
    )

    assert cached.stop_required
    assert cached.tracked_center_frame == (80.0, 60.0)
    assert not released.stop_required


def test_three_second_cooldown_ignores_results_and_requires_new_result_after_expiry() -> None:
    analyzer = make_analyzer(crossing_confirm_frames=1)
    analyze(analyzer, [detected_center(80)], result_id=1, now=0.0)
    lock_target(
        analyzer,
        [detected_center(80)],
        result_id=1,
        now=0.01,
    )
    establish_crossing_baseline(
        analyzer,
        [detected_center(80)],
        result_id=2,
        now=0.1,
    )
    released = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=1.0,
    )
    during_cooldown = analyze(
        analyzer,
        [detected_center(120)],
        result_id=4,
        now=3.5,
    )
    expired_cached = analyze(
        analyzer,
        [detected_center(120)],
        result_id=4,
        now=4.1,
    )
    retriggered = analyze(
        analyzer,
        [detected_center(120)],
        result_id=5,
        now=4.1,
    )

    assert released.cooldown_remaining_sec == pytest.approx(3.0)
    assert not during_cooldown.stop_required
    assert not during_cooldown.armed
    assert not expired_cached.stop_required
    assert expired_cached.armed
    assert retriggered.stop_required


def test_pedestrian_wait_produces_zero_speed_and_stop_protocol_state() -> None:
    hint = build_safety_stop_hint(
        pedestrian_safety_result=stop_result(),
        road_sign_waiting=True,
    )
    assert hint is not None
    command = HighLevelPlanner({}).plan(make_tracked_state(), hint)

    assert hint.stop
    assert hint.force_mode == "PEDESTRIAN_WAIT"
    assert command.mode == "PEDESTRIAN_WAIT"
    assert command.target_speed == 0.0
    assert resolve_configured_speed_state(command.target_speed, 2) == 0


def test_pedestrian_wait_has_priority_over_line_loss_hold() -> None:
    hint = build_safety_stop_hint(
        pedestrian_safety_result=stop_result(),
        road_sign_waiting=True,
    )

    assert hint is not None
    assert hint.stop
    command = HighLevelPlanner({}).plan(
        make_tracked_state(),
        hint,
        line_lost=True,
        now_monotonic=1.0,
    )
    assert command.mode == "PEDESTRIAN_WAIT"
