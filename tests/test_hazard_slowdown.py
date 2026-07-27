import pytest

from core.object.blocking import DetectedObject
from core.planning.hazard_slowdown import (
    HazardSlowdownController,
    RoadSignApproachController,
    detect_hazard_presence,
)


ROI = (100, 100, 300, 300)


def detected(
    class_name: str,
    bbox: tuple[int, int, int, int],
) -> DetectedObject:
    return DetectedObject(
        class_name=class_name,
        confidence=0.9,
        bbox_frame=bbox,
    )


def test_hazard_presence_uses_class_specific_roi_rules() -> None:
    presence = detect_hazard_presence(
        objects=[
            detected("human", (120, 120, 150, 150)),
            detected("car", (300, 180, 340, 240)),
            detected("road_sign", (5, 5, 25, 25)),
            detected("speed_limit", (10, 10, 40, 40)),
            detected("Stop", (20, 20, 50, 50)),
        ],
        avoidance_roi_rect=ROI,
        pedestrian_min_box_area_px=600,
    )

    assert presence.human
    assert presence.car
    assert presence.road_sign
    assert presence.any


def test_human_requires_center_and_area_while_car_requires_box_overlap() -> None:
    presence = detect_hazard_presence(
        objects=[
            detected("human", (90, 120, 110, 140)),
            detected("human", (120, 120, 130, 130)),
            detected("car", (40, 150, 99, 220)),
        ],
        avoidance_roi_rect=ROI,
        pedestrian_min_box_area_px=600,
    )

    assert not presence.human
    assert not presence.car
    assert not presence.road_sign
    assert not presence.any


def test_road_sign_does_not_use_generic_one_gear_slowdown() -> None:
    controller = HazardSlowdownController(hold_sec=1.0)
    road_sign = [detected("road_sign", (5, 5, 25, 25))]

    controller.observe_ai_result(road_sign, ROI, 600, 7, 10.0)
    assert not controller.active(10.1)


def test_human_still_uses_generic_one_gear_slowdown() -> None:
    controller = HazardSlowdownController(hold_sec=1.0)
    human = [detected("human", (120, 120, 150, 150))]

    controller.observe_ai_result(human, ROI, 600, 7, 10.0)
    assert controller.active(10.999)


def test_car_presence_is_reported_without_generic_slowdown_hold() -> None:
    controller = HazardSlowdownController(hold_sec=1.0)
    cars = [detected("car", (150, 150, 220, 240))]

    first = controller.observe_ai_result(cars, ROI, 600, 7, 10.0)
    duplicate = controller.observe_ai_result([], ROI, 600, 7, 10.5)
    cleared = controller.observe_ai_result([], ROI, 600, 8, 10.6)

    assert first.car
    assert duplicate.car
    assert not cleared.car
    assert not controller.active(10.1)


def test_stateful_hazard_holds_for_one_second_after_release() -> None:
    controller = HazardSlowdownController(hold_sec=1.0)

    controller.observe_stateful_hazards(20.0, pedestrian_active=True)
    controller.observe_stateful_hazards(21.5, pedestrian_active=True)
    controller.observe_stateful_hazards(22.0)

    assert controller.active(22.999)
    assert not controller.active(23.0)


def test_road_sign_approach_uses_independent_hold_and_rearm_cycle() -> None:
    controller = RoadSignApproachController(speed_state=2, hold_sec=1.0)

    controller.observe(True, detection_result_id=7, now_monotonic=10.0)
    assert controller.speed_state == 2
    assert controller.active(10.999)
    assert controller.merge_speed_state(None, 10.5) == 2
    assert controller.merge_speed_state(1, 10.5) == 1
    assert controller.merge_speed_state(3, 10.5) == 2

    controller.observe(True, detection_result_id=7, now_monotonic=10.8)
    assert not controller.active(11.0)

    controller.observe(True, detection_result_id=8, now_monotonic=11.1)
    controller.observe(False, detection_result_id=9, now_monotonic=11.2)
    assert controller.active(12.199)
    assert not controller.active(12.2)


def test_road_sign_approach_stays_blocked_until_absence_after_ocr() -> None:
    controller = RoadSignApproachController(speed_state=2, hold_sec=1.0)

    controller.observe(True, detection_result_id=1, now_monotonic=1.0)
    controller.mark_ocr_started()
    assert not controller.active(1.1)

    controller.observe(True, detection_result_id=2, now_monotonic=1.2)
    assert not controller.active(1.3)
    assert controller.merge_speed_state(None, 1.3) is None

    controller.observe(False, detection_result_id=3, now_monotonic=1.4)
    assert not controller.active(1.5)

    controller.observe(True, detection_result_id=4, now_monotonic=1.6)
    assert controller.active(2.599)


@pytest.mark.parametrize(
    "invalid_state",
    (0, -1, 4, 2.5, True, float("inf"), "bad"),
)
def test_road_sign_approach_rejects_invalid_speed_state(
    invalid_state: object,
) -> None:
    with pytest.raises(ValueError, match="ocr.approach_speed_state"):
        RoadSignApproachController(speed_state=invalid_state, hold_sec=1.0)
