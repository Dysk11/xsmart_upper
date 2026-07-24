from __future__ import annotations

import numpy as np
import pytest

from utils.roi import compute_normalized_roi_rect


def test_explicit_avoidance_roi_is_mapped_to_frame() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    rect = compute_normalized_roi_rect(
        frame,
        {
            "left_ratio": 0.10,
            "right_ratio": 0.80,
            "top_ratio": 0.20,
            "bottom_ratio": 0.90,
        },
        validate_name="avoidance_roi",
    )

    assert rect == (20, 20, 160, 90)


def test_missing_avoidance_roi_values_fall_back_to_lane_roi() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    lane_roi = {
        "left_ratio": 0.05,
        "right_ratio": 0.95,
        "top_ratio": 0.40,
        "bottom_ratio": 1.0,
    }

    lane_rect = compute_normalized_roi_rect(frame, lane_roi)
    avoidance_rect = compute_normalized_roi_rect(
        frame,
        {},
        fallback_config=lane_roi,
        validate_name="avoidance_roi",
    )

    assert avoidance_rect == lane_rect


@pytest.mark.parametrize(
    "config",
    [
        {
            "left_ratio": 0.8,
            "right_ratio": 0.2,
            "top_ratio": 0.2,
            "bottom_ratio": 0.9,
        },
        {
            "left_ratio": 0.1,
            "right_ratio": 0.9,
            "top_ratio": 0.9,
            "bottom_ratio": 0.2,
        },
        {
            "left_ratio": -0.1,
            "right_ratio": 0.9,
            "top_ratio": 0.2,
            "bottom_ratio": 0.9,
        },
        {
            "left_ratio": 0.1,
            "right_ratio": 1.1,
            "top_ratio": 0.2,
            "bottom_ratio": 0.9,
        },
    ],
)
def test_invalid_avoidance_roi_ratios_are_rejected(
    config: dict[str, float],
) -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="avoidance_roi"):
        compute_normalized_roi_rect(
            frame,
            config,
            validate_name="avoidance_roi",
        )
