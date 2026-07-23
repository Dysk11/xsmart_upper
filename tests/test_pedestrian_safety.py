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
        "min_box_area_px": 600,
        "rearm_cooldown_sec": 3.0,
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
        roi_rect=ROI_RECT,
        target_point_roi=(target_x_roi, 80.0),
        detection_result_id=result_id,
        now_monotonic=now,
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


def test_exact_area_threshold_triggers_when_center_is_inside_roi() -> None:
    result = analyze(
        make_analyzer(),
        [detected_center(100, width=20, height=30)],
        result_id=1,
        now=0.0,
    )

    assert result.stop_required
    assert result.tracked_center_frame == (100.0, 60.0)


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
    result = analyze(
        make_analyzer(),
        [detected_center(100)],
        target_x_roi=target_x_roi,
        result_id=1,
        now=0.0,
    )

    assert result.target_region == expected_region
    assert result.frozen_target_x_frame == pytest.approx(10.0 + target_x_roi)


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

    released = analyze(
        analyzer,
        [detected_center(end_x)],
        result_id=2,
        now=1.0,
    )

    assert not released.stop_required
    assert released.cooldown_remaining_sec == pytest.approx(3.0)


def test_center_region_online_start_requires_a_side_then_opposite_side() -> None:
    analyzer = make_analyzer()
    analyze(analyzer, [detected_center(100)], result_id=1, now=0.0)

    first_side = analyze(
        analyzer,
        [detected_center(120)],
        result_id=2,
        now=0.5,
    )
    released = analyze(
        analyzer,
        [detected_center(80)],
        result_id=3,
        now=1.0,
    )

    assert first_side.stop_required
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
    wrong_direction = analyze(
        analyzer,
        [detected_center(70)],
        target_x_roi=150.0,
        result_id=2,
        now=0.5,
    )
    released = analyze(
        analyzer,
        [detected_center(40)],
        target_x_roi=150.0,
        result_id=3,
        now=1.0,
    )

    assert wrong_direction.stop_required
    assert wrong_direction.frozen_target_x_frame == 50.0
    assert wrong_direction.target_region == "left"
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

    released = analyze(
        analyzer,
        [detected_center(40)],
        result_id=2,
        now=1.0,
    )

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
    wrong_direction = analyze(
        analyzer,
        [detected_center(150)],
        target_x_roi=20.0,
        result_id=2,
        now=0.5,
    )
    released = analyze(
        analyzer,
        [detected_center(190)],
        target_x_roi=20.0,
        result_id=3,
        now=1.0,
    )

    assert wrong_direction.stop_required
    assert wrong_direction.frozen_target_x_frame == 180.0
    assert wrong_direction.target_region == "right"
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

    released = analyze(
        analyzer,
        [detected_center(190)],
        result_id=2,
        now=1.0,
    )

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
    analyzer = make_analyzer()
    analyze(analyzer, [detected_center(80)], result_id=1, now=0.0)

    missing = analyze(analyzer, [], result_id=2, now=1.0)
    released = analyze(
        analyzer,
        [detected_center(120, width=10, height=10)],
        result_id=3,
        now=2.0,
    )

    assert missing.stop_required
    assert missing.tracked_center_frame == (80.0, 60.0)
    assert not released.stop_required


def test_cached_ai_result_cannot_update_or_release_track() -> None:
    analyzer = make_analyzer()
    analyze(analyzer, [detected_center(80)], result_id=1, now=0.0)

    cached = analyze(
        analyzer,
        [detected_center(120)],
        result_id=1,
        now=0.5,
    )
    released = analyze(
        analyzer,
        [detected_center(120)],
        result_id=2,
        now=1.0,
    )

    assert cached.stop_required
    assert cached.tracked_center_frame == (80.0, 60.0)
    assert not released.stop_required


def test_three_second_cooldown_ignores_results_and_requires_new_result_after_expiry() -> None:
    analyzer = make_analyzer()
    analyze(analyzer, [detected_center(80)], result_id=1, now=0.0)
    released = analyze(
        analyzer,
        [detected_center(120)],
        result_id=2,
        now=1.0,
    )
    during_cooldown = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=3.5,
    )
    expired_cached = analyze(
        analyzer,
        [detected_center(120)],
        result_id=3,
        now=4.1,
    )
    retriggered = analyze(
        analyzer,
        [detected_center(120)],
        result_id=4,
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
        track_mask_visible=True,
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


def test_off_track_stop_has_priority_over_pedestrian_wait() -> None:
    hint = build_safety_stop_hint(
        track_mask_visible=False,
        pedestrian_safety_result=stop_result(),
        road_sign_waiting=True,
    )

    assert hint is not None
    assert hint.stop
    assert hint.force_mode == "OFF_TRACK_STOP"
