"""Shared object-detection data types.

The former lane-corridor blocking analyzer was removed when pedestrian handling
changed from lateral avoidance to a frozen-target crossing stop policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


BBoxInt = tuple[int, int, int, int]


@dataclass
class DetectedObject:
    """An object detection in full-frame coordinates."""

    class_name: str
    confidence: float
    bbox_frame: BBoxInt
    bbox_roi: Optional[BBoxInt] = None
