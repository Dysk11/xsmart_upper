from __future__ import annotations

import queue
import struct
import threading
from multiprocessing import shared_memory

import cv2
import numpy as np
import pytest

from core.io.camera import CameraReader
from core.lane.rknn_segmenter import RknnLaneSegmenter
from core.object.rknn_detector import RknnObjectDetector
import core.runtime.app as runtime_app
from core.runtime.app import (
    AI_FRAME_TRANSPORT_COPY,
    AI_FRAME_TRANSPORT_LEASE,
    SharedArrayPool,
    _ai_inference_worker,
    _put_latest,
    _release_shared_payload,
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
        restored = _take_shared_ai_frames(
            packet,
            {
                "ai_rgb_frame": rgb_ack,
                "ai_bgr_frame": bgr_ack,
            },
            expected_frame_id=23,
        )
        assert np.array_equal(restored.rgb, rgb)
        assert restored.bgr is not None
        assert np.array_equal(restored.bgr, bgr)
        assert restored.leases == []
        assert restored.timing["worker_copy_ms"] >= 0.0
        restored.close()
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


def test_ai_rgb_lease_holds_slot_until_explicit_close() -> None:
    rgb_ack: queue.Queue[int] = queue.Queue()
    bgr_ack: queue.Queue[int] = queue.Queue()
    rgb_pool = SharedArrayPool("ai_rgb_frame", rgb_ack, slot_count=2)
    bgr_pool = SharedArrayPool("ai_bgr_frame", bgr_ack, slot_count=2)
    first = np.full((2, 3, 3), 11, dtype=np.uint8)
    second = np.full((2, 3, 3), 22, dtype=np.uint8)
    third = np.full((2, 3, 3), 33, dtype=np.uint8)
    leases = []
    try:
        for frame_id, frame in ((1, first), (2, second)):
            packet = _share_ai_frames(
                frame,
                frame[..., ::-1],
                frame_id,
                frame_id,
                rgb_pool,
                bgr_pool,
                AI_FRAME_TRANSPORT_LEASE,
            )
            assert packet is not None
            payload = _take_shared_ai_frames(
                packet,
                {
                    "ai_rgb_frame": rgb_ack,
                    "ai_bgr_frame": bgr_ack,
                },
                expected_frame_id=frame_id,
            )
            assert payload.bgr is None
            assert len(payload.leases) == 1
            assert np.shares_memory(payload.rgb, payload.leases[0].array)
            leases.append(payload)

        assert (
            _share_ai_frames(
                third,
                third,
                3,
                3,
                rgb_pool,
                bgr_pool,
                AI_FRAME_TRANSPORT_LEASE,
            )
            is None
        )
        leases[0].close()
        replacement = _share_ai_frames(
            third,
            third,
            3,
            3,
            rgb_pool,
            bgr_pool,
            AI_FRAME_TRANSPORT_LEASE,
        )
        assert replacement is not None
        restored = _take_shared_ai_frames(
            replacement,
            {
                "ai_rgb_frame": rgb_ack,
                "ai_bgr_frame": bgr_ack,
            },
            expected_frame_id=3,
        )
        assert np.array_equal(restored.rgb, third)
        restored.close()
    finally:
        for payload in leases:
            payload.close()
        rgb_pool.close()
        bgr_pool.close()


def test_ai_copy_transport_remains_explicit_fallback() -> None:
    rgb_ack: queue.Queue[int] = queue.Queue()
    bgr_ack: queue.Queue[int] = queue.Queue()
    rgb_pool = SharedArrayPool("ai_rgb_frame", rgb_ack, slot_count=2)
    bgr_pool = SharedArrayPool("ai_bgr_frame", bgr_ack, slot_count=2)
    rgb = np.full((2, 2, 3), (4, 5, 6), dtype=np.uint8)
    try:
        packet = _share_ai_frames(
            rgb,
            rgb[..., ::-1],
            9,
            90,
            rgb_pool,
            bgr_pool,
            AI_FRAME_TRANSPORT_COPY,
        )
        assert packet is not None
        payload = _take_shared_ai_frames(
            packet,
            {
                "ai_rgb_frame": rgb_ack,
                "ai_bgr_frame": bgr_ack,
            },
            expected_frame_id=9,
        )
        assert payload.bgr is not None
        assert payload.leases == []
        assert np.array_equal(payload.bgr, rgb[..., ::-1])
    finally:
        rgb_pool.close()
        bgr_pool.close()


def test_ai_rgb_lease_reconstructs_exact_old_path_bgr_pixels() -> None:
    rgb_ack: queue.Queue[int] = queue.Queue()
    bgr_ack: queue.Queue[int] = queue.Queue()
    rgb_pool = SharedArrayPool("ai_rgb_frame", rgb_ack, slot_count=2)
    bgr_pool = SharedArrayPool("ai_bgr_frame", bgr_ack, slot_count=2)
    rgb = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    expected_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        packet = _share_ai_frames(
            rgb,
            expected_bgr,
            12,
            120,
            rgb_pool,
            bgr_pool,
            AI_FRAME_TRANSPORT_LEASE,
        )
        assert packet is not None
        payload = _take_shared_ai_frames(
            packet,
            {
                "ai_rgb_frame": rgb_ack,
                "ai_bgr_frame": bgr_ack,
            },
            expected_frame_id=12,
        )
        reconstructed_bgr = cv2.cvtColor(payload.rgb, cv2.COLOR_RGB2BGR)
        assert np.array_equal(payload.rgb, rgb)
        assert np.array_equal(reconstructed_bgr, expected_bgr)
        payload.close()
    finally:
        rgb_pool.close()
        bgr_pool.close()


def test_dropped_ai_queue_packet_releases_rgb_lease_slot() -> None:
    rgb_ack: queue.Queue[int] = queue.Queue()
    rgb_pool = SharedArrayPool("ai_rgb_frame", rgb_ack, slot_count=1)
    bgr_pool = SharedArrayPool(
        "ai_bgr_frame",
        queue.Queue(),
        slot_count=1,
    )
    work_queue: queue.Queue[object] = queue.Queue(maxsize=1)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    pools = {
        "ai_rgb_frame": rgb_pool,
        "ai_bgr_frame": bgr_pool,
    }
    try:
        packet = _share_ai_frames(
            frame,
            frame,
            1,
            1,
            rgb_pool,
            bgr_pool,
            AI_FRAME_TRANSPORT_LEASE,
        )
        assert packet is not None
        work_queue.put_nowait((1, 1.0, packet))

        _put_latest(
            work_queue,
            "replacement",
            release_func=lambda value: _release_shared_payload(value, pools),
        )

        replacement = _share_ai_frames(
            frame,
            frame,
            2,
            2,
            rgb_pool,
            bgr_pool,
            AI_FRAME_TRANSPORT_LEASE,
        )
        assert replacement is not None
        _release_shared_payload(replacement, pools)
    finally:
        rgb_pool.close()
        bgr_pool.close()


@pytest.mark.parametrize("failure_stage", ["detection", "ocr"])
def test_ai_worker_releases_lease_after_future_or_ocr_exception(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    class FakeDetector:
        runtime_backend = "lite2"
        core_mask_name = "NPU_CORE_2"
        pipeline_depth = 1

        def __init__(self, _config: dict[str, object]) -> None:
            pass

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        def detect_with_timing(
            self,
            _frame: np.ndarray,
            *,
            frame_id: int,
        ) -> tuple[list[object], dict[str, float]]:
            if failure_stage == "detection":
                raise RuntimeError("future failed")
            return [], {"frame_id": float(frame_id)}

    class FakeOcrSession:
        last_attempt = None

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def would_run_recognizer(
            self,
            _frame: np.ndarray,
            _detections: list[object],
        ) -> bool:
            return failure_stage == "ocr"

        def update(
            self,
            _frame: np.ndarray,
            _frame_id: int,
            _detections: list[object],
        ) -> None:
            if failure_stage == "ocr":
                raise RuntimeError("ocr failed")
            return None

        def close(self) -> None:
            pass

    monkeypatch.setattr(runtime_app, "RknnObjectDetector", FakeDetector)
    monkeypatch.setattr(runtime_app, "RoadSignOcrSession", FakeOcrSession)

    rgb_ack: queue.Queue[int] = queue.Queue()
    bgr_ack: queue.Queue[int] = queue.Queue()
    rgb_pool = SharedArrayPool("ai_rgb_frame", rgb_ack, slot_count=1)
    bgr_pool = SharedArrayPool("ai_bgr_frame", bgr_ack, slot_count=1)
    input_queue: queue.Queue[object] = queue.Queue()
    output_queue: queue.Queue[object] = queue.Queue()
    startup_queue: queue.Queue[object] = queue.Queue()
    frame = np.full((2, 2, 3), 17, dtype=np.uint8)
    try:
        packet = _share_ai_frames(
            frame,
            frame,
            5,
            5,
            rgb_pool,
            bgr_pool,
            AI_FRAME_TRANSPORT_LEASE,
        )
        assert packet is not None
        input_queue.put((5, 5.0, packet))
        input_queue.put(None)

        _ai_inference_worker(
            {
                "ai_frame_transport": AI_FRAME_TRANSPORT_LEASE,
            },
            {},
            ".",
            input_queue,
            output_queue,
            queue.Queue(),
            rgb_ack,
            bgr_ack,
            startup_queue,
            threading.Event(),
        )

        assert startup_queue.get_nowait()[0] == "ready"
        assert output_queue.get_nowait()[0] == 5
        replacement = _share_ai_frames(
            frame,
            frame,
            6,
            6,
            rgb_pool,
            bgr_pool,
            AI_FRAME_TRANSPORT_LEASE,
        )
        assert replacement is not None
        _release_shared_payload(
            replacement,
            {
                "ai_rgb_frame": rgb_pool,
                "ai_bgr_frame": bgr_pool,
            },
        )
    finally:
        rgb_pool.close()
        bgr_pool.close()
