from core.object.blocking import DetectedObject
from core.planning.hazard_slowdown import (
    HazardSlowdownController,
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


def test_new_ai_result_refreshes_hold_but_duplicate_result_does_not() -> None:
    controller = HazardSlowdownController(hold_sec=1.0)
    road_sign = [detected("road_sign", (5, 5, 25, 25))]

    controller.observe_ai_result(road_sign, ROI, 600, 7, 10.0)
    assert controller.active(10.999)
    controller.observe_ai_result(road_sign, ROI, 600, 7, 10.8)
    assert not controller.active(11.0)

    controller.observe_ai_result(road_sign, ROI, 600, 8, 11.1)
    assert controller.active(12.099)
    controller.observe_ai_result([], ROI, 600, 9, 11.2)
    assert controller.active(12.199)
    assert not controller.active(12.2)


def test_stateful_hazard_holds_for_one_second_after_release() -> None:
    controller = HazardSlowdownController(hold_sec=1.0)

    controller.observe_stateful_hazards(20.0, car_avoidance_active=True)
    controller.observe_stateful_hazards(21.5, car_avoidance_active=True)
    controller.observe_stateful_hazards(22.0)

    assert controller.active(22.999)
    assert not controller.active(23.0)
