"""Tests for Go/Stop path targeting and lane-gap connection."""

from __future__ import annotations

import unittest

from core.object.blocking import DetectedObject
from core.planning.path_marker_target import PathMarkerTargetPlanner


class PathMarkerTargetPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = PathMarkerTargetPlanner(
            {
                "enabled": True,
                "class_names": ["Go", "Stop"],
                "min_confidence": 0.20,
                "hold_frames": 4,
                "connection_margin_px": 12,
                "interpolation_step_px": 4,
                "release_y_ratio": 0.92,
            }
        )
        self.roi_rect = (10, 100, 210, 300)
        self.current_centerline = [
            (98, 199),
            (99, 160),
            (100, 120),
            (102, 40),
            (103, 20),
        ]

    def make_object(
        self,
        class_name: str = "Go",
        confidence: float = 0.90,
        bbox: tuple[int, int, int, int] = (80, 160, 140, 200),
    ) -> DetectedObject:
        return DetectedObject(
            class_name=class_name,
            confidence=confidence,
            bbox_frame=bbox,
        )

    def plan(self, objects, current=None, historical=None):
        return self.planner.plan(
            objects=objects,
            centerline_points=self.current_centerline if current is None else current,
            historical_centerline_points=[] if historical is None else historical,
            roi_rect=self.roi_rect,
            roi_width=200,
            roi_height=200,
        )

    def test_connects_lower_lane_through_box_center_to_upper_lane(self) -> None:
        result = self.plan([self.make_object()])

        self.assertTrue(result.active)
        self.assertEqual(result.target_class_name, "Go")
        self.assertEqual(result.target_point_roi, (100.0, 80.0))
        self.assertEqual(result.lower_anchor_roi, (100.0, 120.0))
        self.assertEqual(result.upper_anchor_roi, (102.0, 40.0))
        self.assertIn((100.0, 80.0), result.connected_centerline_points)
        rows = [point[1] for point in result.connected_centerline_points]
        self.assertEqual(rows, sorted(rows, reverse=True))
        self.assertEqual(len({round(row) for row in rows}), len(rows))

    def test_uses_ego_as_lower_anchor_and_history_as_upper_anchor(self) -> None:
        result = self.plan(
            [self.make_object(class_name="Stop")],
            current=[],
            historical=[(105, 40), (106, 20)],
        )

        self.assertTrue(result.active)
        self.assertEqual(result.lower_anchor_roi, (100.0, 199.0))
        self.assertEqual(result.upper_anchor_roi, (105.0, 40.0))
        self.assertTrue(result.using_historical_upper)
        self.assertLess(result.confidence, 0.90)

    def test_missing_upper_lane_ends_at_marker_and_reduces_confidence(self) -> None:
        result = self.plan(
            [self.make_object()],
            current=[(98, 199), (100, 120)],
            historical=[],
        )

        self.assertTrue(result.active)
        self.assertIsNone(result.upper_anchor_roi)
        self.assertEqual(result.connected_centerline_points[-1], (100.0, 80.0))
        self.assertAlmostEqual(result.confidence, 0.90 * 0.65)

    def test_selects_path_marker_instead_of_coin_and_scores_multiple_markers(self) -> None:
        coin = self.make_object(class_name="coin", confidence=0.99)
        far_go = self.make_object(class_name="Go", confidence=0.75, bbox=(70, 140, 120, 170))
        near_stop = self.make_object(class_name="Stop", confidence=0.90, bbox=(90, 170, 150, 220))

        result = self.plan([coin, far_go, near_stop])

        self.assertTrue(result.active)
        self.assertEqual(result.target_class_name, "Stop")
        self.assertEqual(result.target_object, near_stop)

    def test_holds_for_four_misses_then_releases(self) -> None:
        self.assertTrue(self.plan([self.make_object()]).active)
        for miss in range(1, 5):
            result = self.plan([])
            self.assertTrue(result.active, f"miss {miss} should still be held")
            self.assertTrue(result.using_hold)
        self.assertFalse(self.plan([]).active)

    def test_releases_marker_near_roi_bottom_without_hold(self) -> None:
        passed = self.make_object(bbox=(80, 270, 140, 300))
        result = self.plan([passed])

        self.assertFalse(result.active)
        self.assertIn("passed", result.reason)
        self.assertFalse(self.plan([]).active)


if __name__ == "__main__":
    unittest.main()
