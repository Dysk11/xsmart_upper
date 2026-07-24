"""Normalized frame-ROI conversion helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


_ROI_RATIO_DEFAULTS = {
    "left_ratio": 0.05,
    "right_ratio": 0.95,
    "top_ratio": 0.585,
    "bottom_ratio": 1.0,
}


def compute_normalized_roi_rect(
    frame: np.ndarray,
    roi_config: Mapping[str, Any],
    *,
    fallback_config: Mapping[str, Any] | None = None,
    validate_name: str | None = None,
) -> tuple[int, int, int, int]:
    """Convert normalized ROI ratios to a clipped frame rectangle."""

    fallback = fallback_config or {}
    ratios = {
        key: float(
            roi_config.get(
                key,
                fallback.get(key, default),
            )
        )
        for key, default in _ROI_RATIO_DEFAULTS.items()
    }
    if validate_name is not None:
        left = ratios["left_ratio"]
        right = ratios["right_ratio"]
        top = ratios["top_ratio"]
        bottom = ratios["bottom_ratio"]
        if not 0.0 <= left < right <= 1.0:
            raise ValueError(
                f"{validate_name} requires 0 <= left_ratio < right_ratio <= 1"
            )
        if not 0.0 <= top < bottom <= 1.0:
            raise ValueError(
                f"{validate_name} requires 0 <= top_ratio < bottom_ratio <= 1"
            )

    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(round(width * ratios["left_ratio"]))))
    x2 = max(x1 + 1, min(width, int(round(width * ratios["right_ratio"]))))
    y1 = max(0, min(height - 1, int(round(height * ratios["top_ratio"]))))
    y2 = max(y1 + 1, min(height, int(round(height * ratios["bottom_ratio"]))))
    return x1, y1, x2, y2
