from __future__ import annotations

import queue
import struct
from multiprocessing import shared_memory

import cv2
import numpy as np
import pytest

from core.io.camera import CameraReader
from core.lane.rknn_segmenter import RknnLaneSegmenter
from core.object.rknn_detector import RknnObjectDetector
from core.runtime.app import (
    SharedArrayPool,
    _share_ai_frames,
    _take_shared_ai_frames,
)


class _FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()


def test_video_read_keeps_bgr_and_converts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    bgr = np.array([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)
    reader = CameraReader({"mode": "video", "video_path": "unused.mp4"})
    reader.capture = _FakeCapture(bgr)  # type: ignore[assignment]

    original_cvt_color = cv2.cvtColor
    calls: list[int] = []

    def counted_cvt_color(source: np.ndarray, code: int, *args: object, **kwargs: object) -> np.ndarray:
        calls.append(code)
        return original_cvt_color(source, code, *args, **kwargs)

    monkeypatch.setattr(cv2, "cvtColor", counted_cvt_color)
    success, captured = reader.read()

    assert success and captured is not None
    assert np.array_equal(captured.bgr, bgr)
    assert np.array_equal(captured.rgb, bgr[..., ::-1])
    assert calls == [cv2.COLOR_BGR2RGB]
    assert reader.color_conversion_count == 1


def test_shared_memory_read_keeps_rgb_and_converts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ],
        dtype=np.uint8,
    )
    header = struct.Struct("@QII")
    shm = shared_memory.SharedMemory(create=True, size=header.size + rgb.nbytes)
    reader = CameraReader({"mode": "shared_memory"})
    reader._shared_memory = shm
    shm.buf[: header.size] = header.pack(17, rgb.shape[1], rgb.shape[0])
    shm.buf[header.size : header.size + rgb.nbytes] = rgb.tobytes()

    original_cvt_color = cv2.cvtColor
    calls: list[int] = []

    def counted_cvt_color(source: np.ndarray, code: int, *args: object, **kwargs: object) -> np.ndarray:
        calls.append(code)
        return original_cvt_color(source, code, *args, **kwargs)

    monkeypatch.setattr(cv2, "cvtColor", counted_cvt_color)
    try:
        success, captured = reader._read_shared_memory(reconnect_on_timeout=False)
        assert success and captured is not None
        assert captured.source_frame_id == 17
        assert np.array_equal(captured.rgb, rgb)
        assert np.array_equal(captured.bgr, rgb[..., ::-1])
        assert calls == [cv2.COLOR_RGB2BGR]
        assert reader.color_conversion_count == 1
    finally:
        reader._shared_memory = None
        shm.close()
        shm.unlink()


def test_model_preprocessing_accepts_rgb_without_color_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    frame_rgb[..., 0] = 11
    frame_rgb[..., 1] = 22
    frame_rgb[..., 2] = 33
    lane = RknnLaneSegmenter({"enable": False, "input_size": [6, 4]})
    objects = RknnObjectDetector(
        {
            "enable": False,
            "input_size": [6, 4],
            "input_color": "rgb",
            "input_layout": "nhwc",
        }
    )

    def unexpected_conversion(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("model preprocessing must not perform a color conversion")

    monkeypatch.setattr(cv2, "cvtColor", unexpected_conversion)
    lane_tensor, _ = lane._preprocess(frame_rgb)
    object_tensor, _ = objects._preprocess(frame_rgb)

    assert np.array_equal(lane_tensor[0], frame_rgb)
    assert np.array_equal(object_tensor[0], frame_rgb)


def test_object_detector_rejects_non_rgb_model_configuration() -> None:
    with pytest.raises(ValueError, match="input_color must be 'rgb'"):
        RknnObjectDetector({"enable": True, "input_color": "bgr"})


def test_ai_shared_packet_keeps_rgb_bgr_pair_on_same_frame() -> None:
    rgb_ack: queue.Queue[int] = queue.Queue()
    bgr_ack: queue.Queue[int] = queue.Queue()
    rgb_pool = SharedArrayPool("ai_rgb_frame", rgb_ack, slot_count=2)
    bgr_pool = SharedArrayPool("ai_bgr_frame", bgr_ack, slot_count=2)
    rgb = np.full((3, 4, 3), (1, 2, 3), dtype=np.uint8)
    bgr = np.full((3, 4, 3), (3, 2, 1), dtype=np.uint8)
    try:
        packet = _share_ai_frames(rgb, bgr, 23, 99, rgb_pool, bgr_pool)
        assert packet is not None
        restored_rgb, restored_bgr = _take_shared_ai_frames(
            packet,
            {
                "ai_rgb_frame": rgb_ack,
                "ai_bgr_frame": bgr_ack,
            },
            expected_frame_id=23,
        )
        assert np.array_equal(restored_rgb, rgb)
        assert np.array_equal(restored_bgr, bgr)
    finally:
        rgb_pool.close()
        bgr_pool.close()


def test_ai_shared_packet_rejects_cross_frame_use() -> None:
    rgb_ack: queue.Queue[int] = queue.Queue()
    bgr_ack: queue.Queue[int] = queue.Queue()
    rgb_pool = SharedArrayPool("ai_rgb_frame", rgb_ack, slot_count=2)
    bgr_pool = SharedArrayPool("ai_bgr_frame", bgr_ack, slot_count=2)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    try:
        packet = _share_ai_frames(frame, frame, 7, 70, rgb_pool, bgr_pool)
        assert packet is not None
        with pytest.raises(ValueError, match="packet frame mismatch"):
            _take_shared_ai_frames(
                packet,
                {
                    "ai_rgb_frame": rgb_ack,
                    "ai_bgr_frame": bgr_ack,
                },
                expected_frame_id=8,
            )
    finally:
        rgb_pool.close()
        bgr_pool.close()
