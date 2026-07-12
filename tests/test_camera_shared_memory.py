from __future__ import annotations

import struct
import unittest
import uuid
from multiprocessing import shared_memory
from unittest import mock

import numpy as np

from core.camera import CameraReader


HEADER = struct.Struct("@QII")


def _name() -> str:
    return f"xsmart_test_{uuid.uuid4().hex}"


def _create_frame_shm(rgb: np.ndarray, frame_id: int = 1) -> shared_memory.SharedMemory:
    height, width = rgb.shape[:2]
    shm = shared_memory.SharedMemory(name=_name(), create=True, size=HEADER.size + rgb.nbytes)
    shm.buf[: HEADER.size] = HEADER.pack(frame_id, width, height)
    shm.buf[HEADER.size :] = rgb.tobytes()
    return shm


class SharedMemoryCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        # Producer and consumer live in this test process. Keep the producer's
        # tracker registration intact so cleanup can unlink it normally.
        self.tracker_patch = mock.patch("core.camera.resource_tracker.unregister")
        self.tracker_patch.start()

    def tearDown(self) -> None:
        self.tracker_patch.stop()

    def test_returns_bgr_and_does_not_unlink_owner_memory(self) -> None:
        rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        shm = _create_frame_shm(rgb)
        reader = CameraReader({"mode": "shared_memory", "shared_memory_name": shm.name})
        try:
            reader.open()
            success, frame = reader.read()
            self.assertTrue(success)
            self.assertIsNotNone(frame)
            np.testing.assert_array_equal(frame, rgb[:, :, ::-1])
            reader.release()

            still_present = shared_memory.SharedMemory(name=shm.name, create=False)
            still_present.close()
        finally:
            reader.release()
            shm.close()
            shm.unlink()

    def test_mirror_and_new_frame_id(self) -> None:
        rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        shm = _create_frame_shm(rgb, frame_id=7)
        reader = CameraReader({"mode": "shared_memory", "shared_memory_name": shm.name, "mirror": True})
        try:
            reader.open()
            success, first = reader.read()
            self.assertTrue(success)
            self.assertIsNotNone(first)
            np.testing.assert_array_equal(first, rgb[:, ::-1, ::-1])

            updated = np.array([[[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
            shm.buf[HEADER.size :] = updated.tobytes()
            shm.buf[: HEADER.size] = HEADER.pack(8, 2, 1)
            success, second = reader.read()
            self.assertTrue(success)
            self.assertIsNotNone(second)
            np.testing.assert_array_equal(second, updated[:, ::-1, ::-1])
        finally:
            reader.release()
            shm.close()
            shm.unlink()

    def test_same_frame_id_is_not_returned_twice(self) -> None:
        rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
        shm = _create_frame_shm(rgb, frame_id=9)
        reader = CameraReader(
            {"mode": "shared_memory", "shared_memory_name": shm.name, "max_reconnect_attempts": 1}
        )
        try:
            reader.open()
            self.assertTrue(reader.read()[0])
            ticks = iter((0.0, 2.0, 2.0, 4.0))
            with mock.patch("core.camera.time.sleep"), mock.patch(
                "core.camera.time.monotonic", side_effect=lambda: next(ticks, 4.0)
            ):
                self.assertEqual(reader.read(), (False, None))
        finally:
            reader.release()
            shm.close()
            shm.unlink()

    def test_frame_changed_during_copy_is_discarded(self) -> None:
        rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
        shm = _create_frame_shm(rgb, frame_id=11)
        reader = CameraReader(
            {"mode": "shared_memory", "shared_memory_name": shm.name, "max_reconnect_attempts": 1}
        )

        class MutatingView:
            def copy(self) -> np.ndarray:
                frame_id, width, height = HEADER.unpack(bytes(shm.buf[: HEADER.size]))
                shm.buf[: HEADER.size] = HEADER.pack(frame_id + 1, width, height)
                return rgb.copy()

        ticks = iter((0.0, 0.0, 2.0, 2.0, 2.0, 4.0))
        try:
            reader.open()
            with mock.patch("core.camera.np.ndarray", return_value=MutatingView()), mock.patch(
                "core.camera.time.sleep"
            ), mock.patch("core.camera.time.monotonic", side_effect=lambda: next(ticks, 4.0)):
                self.assertEqual(reader.read(), (False, None))
            self.assertEqual(reader._shared_memory_last_frame_id, 0)
        finally:
            reader.release()
            shm.close()
            shm.unlink()

    def test_rejects_frame_larger_than_mapping(self) -> None:
        shm = shared_memory.SharedMemory(name=_name(), create=True, size=HEADER.size + 3)
        shm.buf[: HEADER.size] = HEADER.pack(1, 640, 480)
        reader = CameraReader(
            {"mode": "shared_memory", "shared_memory_name": shm.name, "max_reconnect_attempts": 1}
        )
        ticks = iter((0.0, 2.0, 2.0, 4.0))
        try:
            reader.open()
            with mock.patch("core.camera.time.sleep"), mock.patch(
                "core.camera.time.monotonic", side_effect=lambda: next(ticks, 4.0)
            ):
                self.assertEqual(reader.read(), (False, None))
        finally:
            reader.release()
            shm.close()
            shm.unlink()

    def test_open_fails_when_mapping_is_missing(self) -> None:
        reader = CameraReader(
            {"mode": "shared_memory", "shared_memory_name": _name(), "max_reconnect_attempts": 2}
        )
        with mock.patch("core.camera.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "无法连接共享内存"):
                reader.open()

    def test_removed_stream_mode_is_rejected(self) -> None:
        reader = CameraReader({"mode": "stream"})
        with self.assertRaisesRegex(ValueError, "不支持的图像源模式"):
            reader.open()


if __name__ == "__main__":
    unittest.main()
