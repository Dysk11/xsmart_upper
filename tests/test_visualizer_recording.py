"""Regression tests for raw-frame visualizer recording."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from core.visualization.visualizer import Visualizer


def render_once(visualizer: Visualizer, frame: np.ndarray) -> bool:
    """Call render with lightweight placeholders after canvas construction is mocked."""

    return visualizer.render(
        frame=frame,
        roi_rect=(0, 0, frame.shape[1], frame.shape[0]),
        detection_result=None,  # type: ignore[arg-type]
        tracked_state=None,  # type: ignore[arg-type]
        control_command=None,  # type: ignore[arg-type]
        fps_value=25.0,
    )


@pytest.mark.parametrize("config", [{"save_video": True}, {"save_video": True, "record_without_ui": False}])
def test_recording_defaults_to_annotated_canvas(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
) -> None:
    frame = np.full((4, 6, 3), 10, dtype=np.uint8)
    canvas = np.full((8, 12, 3), 20, dtype=np.uint8)
    recorded: list[np.ndarray] = []
    visualizer = Visualizer({**config, "show_window": False, "save_dir": "."})
    monkeypatch.setattr(visualizer, "_build_canvas", lambda **_: canvas)
    monkeypatch.setattr(visualizer, "_write_video", recorded.append)

    assert render_once(visualizer, frame)
    assert recorded == [canvas]


def test_raw_recording_keeps_annotated_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.full((4, 6, 3), 10, dtype=np.uint8)
    canvas = np.full((8, 12, 3), 20, dtype=np.uint8)
    recorded: list[np.ndarray] = []
    displayed: list[tuple[str, np.ndarray]] = []
    debug_panel = np.full((5, 12, 3), 30, dtype=np.uint8)
    visualizer = Visualizer(
        {
            "show_window": True,
            "save_video": True,
            "record_without_ui": True,
            "save_dir": ".",
        }
    )
    monkeypatch.setattr(visualizer, "_build_canvas", lambda **_: canvas)
    monkeypatch.setattr(visualizer, "_build_debug_panel", lambda **_: debug_panel)
    monkeypatch.setattr(visualizer, "_write_video", recorded.append)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: displayed.append((name, image)))
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)

    assert render_once(visualizer, frame)
    assert recorded == [frame]
    assert displayed == [
        ("X-SmartCar Upper", canvas),
        ("X-SmartCar Debug", debug_panel),
    ]


def test_raw_recording_continues_without_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.full((4, 6, 3), 10, dtype=np.uint8)
    canvas = np.full((8, 12, 3), 20, dtype=np.uint8)
    recorded: list[np.ndarray] = []
    visualizer = Visualizer(
        {
            "show_window": False,
            "save_video": True,
            "record_without_ui": True,
            "save_dir": ".",
        }
    )
    monkeypatch.setattr(visualizer, "_build_canvas", lambda **_: canvas)
    monkeypatch.setattr(visualizer, "_write_video", recorded.append)
    monkeypatch.setattr(cv2, "imshow", lambda *_: pytest.fail("window must stay hidden"))

    assert render_once(visualizer, frame)
    assert recorded == [frame]


def test_disabled_recording_does_not_write_video(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.full((4, 6, 3), 10, dtype=np.uint8)
    canvas = np.full((8, 12, 3), 20, dtype=np.uint8)
    visualizer = Visualizer(
        {
            "show_window": False,
            "save_video": False,
            "record_without_ui": True,
            "save_dir": ".",
        }
    )
    monkeypatch.setattr(visualizer, "_build_canvas", lambda **_: canvas)
    monkeypatch.setattr(visualizer, "_write_video", lambda *_: pytest.fail("video writing must stay disabled"))

    assert render_once(visualizer, frame)
    assert visualizer.video_writer is None
