"""Synthetic-mask regression coverage for the right-fork state machine."""

from __future__ import annotations

import cv2
import numpy as np

from core.lane.detector import LaneDetector


HEIGHT = 200
WIDTH = 300


def make_detector(**fork_overrides) -> LaneDetector:
    fork = {
        "confirm_frames": 2,
        "release_frames": 3,
        "recovery_confirm_frames": 5,
        "return_release_frames": 3,
        **fork_overrides,
    }
    return LaneDetector(
        {
            "fork": fork,
            "boundary": {"min_run_width_px": 4},
            "centerline": {
                "min_valid_points": 4,
                "scan_step": 2,
                "default_lane_width_px": 60,
            },
            "confidence": {"lost_threshold": 0.0},
        }
    )


def make_junction(kind: str) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (80, 0), (140, HEIGHT - 1), 255, -1)
    if kind == "split":
        cv2.line(mask, (140, 125), (270, 45), 255, 34)
    elif kind == "merge":
        cv2.line(mask, (140, 75), (270, 155), 255, 34)
    else:
        raise ValueError(kind)
    return mask


def make_corridor(left: int = 150, right: int = 210) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (left, 0), (right, HEIGHT - 1), 255, -1)
    return mask


def make_return_break() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(mask, (130, 100), (200, HEIGHT - 1), 255, -1)
    cv2.rectangle(mask, (220, 0), (285, 99), 255, -1)
    return mask


def observe(detector: LaneDetector, mask: np.ndarray):
    row_runs = detector._build_row_runs(mask)
    values = detector._extract_row_boundaries(mask, row_runs=row_runs)
    return detector._observe_right_fork(
        mask,
        row_runs,
        values[0],
        values[1],
        values[3],
        values[4],
        values[7],
    )


def enter_confirmed_junction(detector: LaneDetector, kind: str, route: str):
    result = None
    mask = make_junction(kind)
    for _ in range(detector.fork_confirm_frames):
        result = detector.detect_from_mask(mask, route_direction=route)
    assert result is not None
    return result


def test_ap_geometry_distinguishes_right_split_and_merge() -> None:
    detector = make_detector()

    split = observe(detector, make_junction("split"))
    merge = observe(detector, make_junction("merge"))

    assert split.a_point is not None and split.p_point is not None
    assert merge.a_point is not None and merge.p_point is not None
    assert split.junction_kind == "split"
    assert merge.junction_kind == "merge"
    assert split.p_point[0] > split.a_point[0]
    assert merge.p_point[0] > merge.a_point[0]


def test_right_request_is_ignored_at_merge_and_raw_mask_is_preserved() -> None:
    detector = make_detector()
    mask = make_junction("merge")
    original = mask.copy()

    result = enter_confirmed_junction(detector, "merge", "right")

    assert result.fork_result.state == "JUNCTION"
    assert result.fork_result.junction_kind == "merge"
    assert not result.fork_result.fork_detected
    assert result.fork_result.selected_direction == "left"
    assert result.fork_result.path_overridden
    assert result.fork_result.patch_line == (
        result.fork_result.a_point,
        result.fork_result.p_point,
    )
    assert np.array_equal(result.mask, original)


def test_split_defaults_left_but_latches_a_late_right_decision() -> None:
    detector = make_detector()
    result = enter_confirmed_junction(detector, "split", "left")

    assert result.fork_result.fork_detected
    assert result.fork_result.selected_direction == "left"
    assert result.fork_result.patch_line == (
        result.fork_result.a_point,
        result.fork_result.p_point,
    )

    result = detector.detect_from_mask(make_junction("split"), route_direction="right")
    assert result.fork_result.selected_direction == "right"
    assert result.fork_result.patch_line is not None
    bottom_left, p_point = result.fork_result.patch_line
    assert bottom_left[1] == HEIGHT - 1
    assert p_point == result.fork_result.p_point

    result = detector.detect_from_mask(make_junction("split"), route_direction="left")
    assert result.fork_result.selected_direction == "right"
    assert result.fork_result.path_overridden
    assert all(
        left[0] < right[0]
        for left, right in zip(result.left_boundary_points, result.right_boundary_points)
    )


def test_left_split_returns_main_instead_of_entering_branch() -> None:
    detector = make_detector()
    enter_confirmed_junction(detector, "split", "left")

    states = []
    for _ in range(detector.fork_release_frames):
        result = detector.detect_from_mask(make_corridor(), route_direction="left")
        states.append(result.fork_result.state)
        if result.fork_result.state == "JUNCTION":
            assert result.fork_result.path_overridden

    assert states == ["JUNCTION", "JUNCTION", "MAIN"]


def test_right_split_runs_strict_state_sequence_back_to_main() -> None:
    detector = make_detector()
    enter_confirmed_junction(detector, "split", "right")

    states = []
    for _ in range(detector.fork_release_frames):
        result = detector.detect_from_mask(make_corridor(), route_direction="left")
        states.append(result.fork_result.state)
    assert states == ["JUNCTION", "JUNCTION", "BRANCH"]

    for _ in range(detector.fork_confirm_frames):
        result = detector.detect_from_mask(make_return_break(), route_direction="left")
    assert result.fork_result.state == "RETURNING"
    assert result.fork_result.break_point == (130, 100)
    assert result.fork_result.patch_line == ((130, 100), (WIDTH - 1, 0))
    assert result.fork_result.path_overridden

    states = []
    for _ in range(detector.fork_return_release_frames):
        result = detector.detect_from_mask(make_corridor(100, 160), route_direction="right")
        states.append(result.fork_result.state)
    assert states == ["RETURNING", "RETURNING", "MAIN"]
    assert result.fork_result.patch_line is None


def test_main_can_recover_returning_state_after_mid_branch_restart() -> None:
    detector = make_detector()

    for _ in range(detector.fork_recovery_confirm_frames):
        result = detector.detect_from_mask(make_return_break(), route_direction="left")

    assert result.fork_result.state == "RETURNING"
    assert result.fork_result.patch_line == ((130, 100), (WIDTH - 1, 0))


def test_single_frame_ap_flicker_and_ordinary_lane_do_not_change_state() -> None:
    detector = make_detector()
    result = detector.detect_from_mask(make_junction("split"), route_direction="right")
    assert result.fork_result.state == "MAIN"

    ordinary = make_corridor(90, 150)
    cv2.circle(ordinary, (265, 40), 8, 255, -1)
    for _ in range(8):
        result = detector.detect_from_mask(ordinary, route_direction="right")

    assert result.fork_result.state == "MAIN"
    assert not result.fork_result.fork_detected
    assert result.fork_result.a_point is None
    assert result.fork_result.p_point is None
    assert result.fork_result.patch_line is None


def test_curve_edge_loss_and_internal_obstruction_are_not_ap_junctions() -> None:
    masks: list[np.ndarray] = []

    curve = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for y in range(HEIGHT):
        center = 120 + int(round(45.0 * ((HEIGHT - 1 - y) / (HEIGHT - 1)) ** 2))
        curve[y, center - 30 : center + 31] = 255
    masks.append(curve)

    edge_lane = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.rectangle(edge_lane, (0, 0), (60, HEIGHT - 1), 255, -1)
    masks.append(edge_lane)

    obstructed = make_corridor(90, 170)
    cv2.circle(obstructed, (135, 90), 14, 0, -1)
    masks.append(obstructed)

    for mask in masks:
        detector = make_detector()
        for _ in range(6):
            result = detector.detect_from_mask(mask, route_direction="right")
        assert result.fork_result.state == "MAIN"
        assert not result.fork_result.fork_detected
        assert result.fork_result.a_point is None
        assert result.fork_result.p_point is None
