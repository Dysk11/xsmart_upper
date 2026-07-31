from __future__ import annotations

import random

from core.lane.tracker import LaneTracker
from utils.math_utils import ema


def _make_tracker() -> LaneTracker:
    return LaneTracker(
        {
            "centerline_alpha": 0.25,
            "max_centerline_point_shift_px": 80.0,
        }
    )


def _reference_smooth(
    tracker: LaneTracker,
    previous_points: list[tuple[int, int]],
    current_points: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    smoothed: list[tuple[int, int]] = []
    for current_x, current_y in current_points:
        ordered = sorted(
            previous_points,
            key=lambda item: item[1],
            reverse=True,
        )
        if current_y >= ordered[0][1]:
            previous_x = float(ordered[0][0])
        elif current_y <= ordered[-1][1]:
            previous_x = float(ordered[-1][0])
        else:
            previous_x = float(ordered[-1][0])
            for index in range(len(ordered) - 1):
                x1, y1 = ordered[index]
                x2, y2 = ordered[index + 1]
                if y1 >= current_y >= y2 and y1 != y2:
                    ratio = float(current_y - y1) / float(y2 - y1)
                    previous_x = float(x1 + (x2 - x1) * ratio)
                    break
        delta_x = max(
            -tracker.max_centerline_point_shift_px,
            min(
                tracker.max_centerline_point_shift_px,
                float(current_x - previous_x),
            ),
        )
        clamped_x = previous_x + delta_x
        smoothed_x = ema(previous_x, clamped_x, tracker.centerline_alpha)
        smoothed.append((int(round(smoothed_x)), int(current_y)))
    return smoothed


def test_centerline_smoothing_preserves_previous_interpolation_semantics() -> None:
    tracker = _make_tracker()
    random_source = random.Random(20260731)
    previous = [
        (random_source.randrange(80, 560), y)
        for y in range(220, 420, 4)
    ]
    random_source.shuffle(previous)
    current = [
        (random_source.randrange(80, 560), y)
        for y in range(218, 424, 4)
    ]

    assert tracker._smooth_centerline_points(
        previous,
        current,
    ) == _reference_smooth(tracker, previous, current)


def test_centerline_smoothing_keeps_empty_input_behavior() -> None:
    tracker = _make_tracker()

    assert tracker._smooth_centerline_points([], [(10, 20)]) == [(10, 20)]
    assert tracker._smooth_centerline_points([(10, 20)], []) == []


def test_centerline_smoothing_preserves_duplicate_y_semantics() -> None:
    tracker = _make_tracker()
    previous = [
        (110, 100),
        (120, 100),
        (130, 80),
        (140, 60),
        (150, 60),
    ]
    current = [
        (200, 105),
        (200, 100),
        (200, 90),
        (200, 80),
        (200, 70),
        (200, 60),
        (200, 55),
    ]

    assert tracker._smooth_centerline_points(
        previous,
        current,
    ) == _reference_smooth(tracker, previous, current)
