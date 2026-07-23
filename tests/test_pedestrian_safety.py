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
DANGER_ZONE = (70.0, 20.0, 150.0, 120.0)


def make_analyzer(**overrides: object) -> PedestrianSafetyAnalyzer:
    config: dict[str, object] = {
        "enabled": True,
        "min_box_area_px": 600,
        "danger_zone": {
            "left_ratio": 0.30,
            "right_ratio": 0.70,
            "top_ratio": 0.00,
            "bottom_ratio": 1.00,
        },
    }
    config.update(overrides)
    return PedestrianSafetyAnalyzer(config)


def detected(
    bbox: tuple[int, int, int, int],
    class_name: str = "human",
) -> DetectedObject:
    return DetectedObject(
        class_name=class_name,
        confidence=0.9,
        bbox_frame=bbox,
    )


def test_zone_ratios_are_strictly_validated() -> None:
    with pytest.raises(ValueError, match="left_ratio"):
        make_analyzer(
            danger_zone={
                "left_ratio": 0.7,
                "right_ratio": 0.3,
                "top_ratio": 0.0,
                "bottom_ratio": 1.0,
            }
        )
    with pytest.raises(ValueError, match="top_ratio"):
        make_analyzer(
            danger_zone={
                "left_ratio": 0.3,
                "right_ratio": 0.7,
                "top_ratio": 1.0,
                "bottom_ratio": 1.0,
            }
        )


@pytest.mark.parametrize(
    ("objects", "overlapping_count"),
    [
        ([detected((20, 30, 50, 60))], 0),
        ([detected((60, 30, 70, 90))], 0),  # edge contact has zero area
        ([detected((80, 30, 90, 40))], 1),  # overlap, but only 100 px^2
        ([detected((80, 30, 120, 60), class_name="car")], 0),
    ],
)
def test_non_qualifying_objects_do_not_stop(
    objects: list[DetectedObject],
    overlapping_count: int,
) -> None:
    result = make_analyzer().analyze(objects, ROI_RECT)

    assert result.danger_zone_frame == DANGER_ZONE
    assert not result.stop_required
    assert result.overlapping_count == overlapping_count


def test_qualifying_human_overlap_triggers_latched_stop() -> None:
    result = make_analyzer().analyze(
        [detected((60, 30, 80, 60))],  # 600 px^2 with positive overlap
        ROI_RECT,
    )

    assert result.stop_required
    assert result.latched
    assert result.human_count == 1
    assert result.overlapping_count == 1


def test_any_overlapping_human_keeps_multiple_human_scene_stopped() -> None:
    analyzer = make_analyzer()
    analyzer.analyze([detected((80, 30, 120, 60))], ROI_RECT)

    result = analyzer.analyze(
        [
            detected((20, 30, 50, 60)),
            detected((100, 30, 110, 40)),  # small boxes count after latching
        ],
        ROI_RECT,
    )

    assert result.stop_required
    assert result.human_count == 2
    assert result.overlapping_count == 1


def test_latch_waits_through_empty_detection_then_releases_when_all_humans_clear() -> None:
    analyzer = make_analyzer()
    analyzer.analyze([detected((80, 30, 120, 60))], ROI_RECT)

    missing = analyzer.analyze([], ROI_RECT)
    still_overlapping = analyzer.analyze([detected((80, 30, 90, 40))], ROI_RECT)
    released = analyzer.analyze(
        [
            detected((20, 30, 50, 60)),
            detected((220, 30, 250, 60)),  # fully outside ROI still proves clear
        ],
        ROI_RECT,
    )

    assert missing.stop_required
    assert still_overlapping.stop_required
    assert not released.stop_required
    assert released.human_count == 2
    assert released.overlapping_count == 0


def test_pedestrian_wait_produces_zero_speed_and_stop_protocol_state() -> None:
    result = PedestrianSafetyResult(
        stop_required=True,
        latched=True,
        danger_zone_frame=DANGER_ZONE,
        human_count=1,
        overlapping_count=1,
        reason="test pedestrian stop",
    )
    hint = build_safety_stop_hint(
        track_mask_visible=True,
        pedestrian_safety_result=result,
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
    result = PedestrianSafetyResult(
        stop_required=True,
        latched=True,
        danger_zone_frame=DANGER_ZONE,
        human_count=1,
        overlapping_count=1,
        reason="test pedestrian stop",
    )
    hint = build_safety_stop_hint(
        track_mask_visible=False,
        pedestrian_safety_result=result,
        road_sign_waiting=True,
    )

    assert hint is not None
    assert hint.stop
    assert hint.force_mode == "OFF_TRACK_STOP"
