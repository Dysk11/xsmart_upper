"""Low-latency shared-memory client for the native RKNN lane backend."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
import os
import select
import struct
import subprocess
import time
from typing import Any
import uuid

import numpy as np

from core.lane.rknn_segmenter import SegmentationInstance, SegmentationResult


PROTOCOL_VERSION = 3
SLOT_COUNT = 2
GLOBAL_HEADER_SIZE = 64
INPUT_SLOT_HEADER_SIZE = 64
RESULT_SLOT_HEADER_SIZE = 288
INPUT_MAGIC = b"XSLNIN1\0"
RESULT_MAGIC = b"XSLNOT1\0"

GLOBAL_HEADER = struct.Struct("<8sIIIIQIIII16x")
INPUT_SLOT_HEADER = struct.Struct("<QQQQIIIII12x")
RESULT_BASE_HEADER = struct.Struct("<QQQQQQIIIIIIIdddddddddddQQQQQQQ")
RESULT_INSTANCE = struct.Struct("<iiiif")

BACKEND_STATE_INITIALIZING = 0
BACKEND_STATE_READY = 1
BACKEND_STATE_ERROR = 2

STATUS_NAMES = {
    0: "ok",
    1: "no_detection",
    2: "preprocess_error",
    3: "inference_error",
    4: "postprocess_error",
    5: "invalid_frame",
}

CORE_MASK_VALUES = {
    "NPU_CORE_0": 1,
    "NPU_CORE_1": 2,
    "NPU_CORE_2": 4,
}


@dataclass(frozen=True)
class NativeLaneResult:
    """One native result plus command-aligned timing and counters."""

    frame_id: int
    source_frame_id: int
    captured_at: float
    completed_at: float
    rknn_frame_id: int
    worker_index: int
    core_mask: int
    result: SegmentationResult
    timing: dict[str, float]
    counters: dict[str, int]


class LatestFrameSharedMemory:
    """Publish RGB frames to a double-buffered latest-value shared memory."""

    def __init__(self, width: int, height: int, *, name: str | None = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.payload_capacity = self.width * self.height * 3
        self.slot_size = INPUT_SLOT_HEADER_SIZE + self.payload_capacity
        self.name = name or _unique_name("xsmart_lane_input")
        self.shm = shared_memory.SharedMemory(
            name=self.name,
            create=True,
            size=GLOBAL_HEADER_SIZE + SLOT_COUNT * self.slot_size,
        )
        self.publish_seq = 0
        self.published_slot = 0
        self._closed = False
        self.shm.buf[:] = b"\0" * len(self.shm.buf)
        GLOBAL_HEADER.pack_into(
            self.shm.buf,
            0,
            INPUT_MAGIC,
            PROTOCOL_VERSION,
            GLOBAL_HEADER_SIZE,
            self.slot_size,
            self.payload_capacity,
            0,
            0,
            BACKEND_STATE_INITIALIZING,
            0,
            os.getpid(),
        )

    def publish(
        self,
        frame_rgb: np.ndarray,
        *,
        frame_id: int,
        source_frame_id: int,
        captured_at: float,
    ) -> int:
        frame = np.asarray(frame_rgb)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("native lane input must be an HxWx3 uint8 RGB array")
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        payload_bytes = int(frame.nbytes)
        if payload_bytes > self.payload_capacity:
            raise ValueError(
                f"lane frame {width}x{height} exceeds shared-memory capacity "
                f"{self.width}x{self.height}"
            )

        slot = 1 - self.published_slot
        offset = GLOBAL_HEADER_SIZE + slot * self.slot_size
        current_sequence = struct.unpack_from("<Q", self.shm.buf, offset)[0]
        writing_sequence = current_sequence + 1 if current_sequence % 2 == 0 else current_sequence + 2
        struct.pack_into("<Q", self.shm.buf, offset, writing_sequence)
        INPUT_SLOT_HEADER.pack_into(
            self.shm.buf,
            offset,
            writing_sequence,
            int(frame_id),
            int(source_frame_id),
            max(0, int(float(captured_at) * 1_000_000_000.0)),
            int(width),
            int(height),
            3,
            int(frame.strides[0]),
            payload_bytes,
        )
        payload_offset = offset + INPUT_SLOT_HEADER_SIZE
        self.shm.buf[payload_offset : payload_offset + payload_bytes] = frame.reshape(-1)
        completed_sequence = writing_sequence + 1
        struct.pack_into("<Q", self.shm.buf, offset, completed_sequence)

        self.publish_seq += 1
        self.published_slot = slot
        struct.pack_into("<I", self.shm.buf, 32, slot)
        struct.pack_into("<Q", self.shm.buf, 24, self.publish_seq)
        return self.publish_seq

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.shm.close()
        finally:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class LatestResultSharedMemory:
    """Read newest native lane result without queueing older completions."""

    def __init__(self, width: int, height: int, *, name: str | None = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.payload_capacity = (self.width * self.height + 7) // 8
        self.slot_size = RESULT_SLOT_HEADER_SIZE + self.payload_capacity
        self.name = name or _unique_name("xsmart_lane_result")
        self.shm = shared_memory.SharedMemory(
            name=self.name,
            create=True,
            size=GLOBAL_HEADER_SIZE + SLOT_COUNT * self.slot_size,
        )
        self.last_publish_seq = 0
        self._closed = False
        self.shm.buf[:] = b"\0" * len(self.shm.buf)
        GLOBAL_HEADER.pack_into(
            self.shm.buf,
            0,
            RESULT_MAGIC,
            PROTOCOL_VERSION,
            GLOBAL_HEADER_SIZE,
            self.slot_size,
            self.payload_capacity,
            0,
            0,
            BACKEND_STATE_INITIALIZING,
            0,
            os.getpid(),
        )

    @property
    def state(self) -> tuple[int, int]:
        values = GLOBAL_HEADER.unpack_from(self.shm.buf, 0)
        return int(values[7]), int(values[8])

    def read_latest(self) -> NativeLaneResult | None:
        header_before = GLOBAL_HEADER.unpack_from(self.shm.buf, 0)
        publish_seq = int(header_before[5])
        if publish_seq <= self.last_publish_seq:
            return None
        slot = int(header_before[6])
        if slot not in (0, 1):
            return None
        offset = GLOBAL_HEADER_SIZE + slot * self.slot_size
        sequence_before = struct.unpack_from("<Q", self.shm.buf, offset)[0]
        if sequence_before == 0 or sequence_before % 2:
            return None
        raw_header = bytes(self.shm.buf[offset : offset + RESULT_SLOT_HEADER_SIZE])
        values = RESULT_BASE_HEADER.unpack_from(raw_header, 0)
        sequence_after_header = int(values[0])
        if sequence_after_header != sequence_before:
            return None

        (
            _sequence,
            frame_id,
            source_frame_id,
            captured_ns,
            completed_ns,
            rknn_frame_id,
            worker_index,
            core_mask,
            status_code,
            instance_count,
            width,
            height,
            mask_bytes,
            preprocess_ms,
            input_sync_ms,
            inference_ms,
            output_sync_ms,
            postprocess_ms,
            decode_ms,
            prototype_ms,
            resize_union_ms,
            pack_ms,
            total_ms,
            publish_ms,
            claimed_count,
            completed_count,
            overwritten_count,
            out_of_order_count,
            error_count,
            notification_count,
            notification_error_count,
        ) = values
        if (
            width <= 0
            or height <= 0
            or mask_bytes > self.payload_capacity
            or mask_bytes != (int(width) * int(height) + 7) // 8
        ):
            return None

        payload_offset = offset + RESULT_SLOT_HEADER_SIZE
        packed = bytes(self.shm.buf[payload_offset : payload_offset + int(mask_bytes)])
        sequence_after = struct.unpack_from("<Q", self.shm.buf, offset)[0]
        header_after = GLOBAL_HEADER.unpack_from(self.shm.buf, 0)
        if (
            sequence_after != sequence_before
            or sequence_after % 2
            or int(header_after[5]) != publish_seq
            or int(header_after[6]) != slot
        ):
            return None

        mask = np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8),
            count=int(width) * int(height),
            bitorder="little",
        ).reshape((int(height), int(width)))
        mask = mask.astype(np.uint8, copy=False) * np.uint8(255)
        instances: list[SegmentationInstance] = []
        instance_count = min(3, int(instance_count))
        instance_offset = RESULT_BASE_HEADER.size
        for index in range(instance_count):
            x1, y1, x2, y2, confidence = RESULT_INSTANCE.unpack_from(
                raw_header,
                instance_offset + index * RESULT_INSTANCE.size,
            )
            instances.append(
                SegmentationInstance(
                    (int(x1), int(y1), int(x2), int(y2)),
                    float(confidence),
                )
            )
        status = STATUS_NAMES.get(int(status_code), "backend_error")
        confidence = instances[0].confidence if instances else 0.0
        self.last_publish_seq = publish_seq
        return NativeLaneResult(
            frame_id=int(frame_id),
            source_frame_id=int(source_frame_id),
            captured_at=float(captured_ns) / 1_000_000_000.0,
            completed_at=float(completed_ns) / 1_000_000_000.0,
            rknn_frame_id=int(rknn_frame_id),
            worker_index=int(worker_index),
            core_mask=int(core_mask),
            result=SegmentationResult(mask, instances, confidence, status),
            timing={
                "preprocess_ms": float(preprocess_ms),
                "input_sync_ms": float(input_sync_ms),
                "inference_ms": float(inference_ms),
                "output_sync_ms": float(output_sync_ms),
                "postprocess_queue_ms": 0.0,
                "postprocess_ms": float(postprocess_ms),
                "decode_ms": float(decode_ms),
                "prototype_ms": float(prototype_ms),
                "resize_union_ms": float(resize_union_ms),
                "pack_ms": float(pack_ms),
                "total_ms": float(total_ms),
                "publish_ms": float(publish_ms),
            },
            counters={
                "claimed_count": int(claimed_count),
                "completed_count": int(completed_count),
                "overwritten_count": int(overwritten_count),
                "out_of_order_count": int(out_of_order_count),
                "error_count": int(error_count),
                "notification_count": int(notification_count),
                "notification_error_count": int(notification_error_count),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.shm.close()
        finally:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class CapiLaneBackend:
    """Own shared-memory transport and the native backend subprocess."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        input_size = config.get("input_size", [640, 480])
        self.width = int(input_size[0])
        self.height = int(input_size[1])
        self.model_path = Path(str(config.get("model_path", ""))).expanduser()
        raw_binary = str(
            config.get(
                "c_api_binary",
                "native/lane_rknn_backend/build/lane_rknn_backend",
            )
        )
        binary_path = Path(raw_binary).expanduser()
        self.binary_path = (
            binary_path if binary_path.is_absolute() else project_root / binary_path
        )
        self.preprocess_backend = str(config.get("preprocess_backend", "auto")).lower()
        self.output_mode = str(config.get("output_mode", "float")).lower()
        if self.output_mode not in {"float", "native"}:
            raise ValueError("C API output_mode must be float or native")
        self.postprocess_backend = str(
            config.get("postprocess_backend", "reference")
        ).lower()
        if self.postprocess_backend not in {"reference", "neon_exact"}:
            raise ValueError(
                "C API postprocess_backend must be reference or neon_exact"
            )
        self.validate_postprocess_exact = bool(
            config.get("validate_postprocess_exact", False)
        )
        self.startup_timeout_sec = max(0.1, float(config.get("startup_timeout_sec", 8.0)))
        core_names = list(config.get("worker_core_masks", ["NPU_CORE_0", "NPU_CORE_1"]))
        if len(core_names) != 2:
            raise ValueError("C API lane backend requires exactly two worker_core_masks")
        try:
            self.core_masks = [CORE_MASK_VALUES[str(name)] for name in core_names]
        except KeyError as error:
            raise ValueError(f"unsupported C API NPU core mask: {error.args[0]}") from error
        self.cpu_cores = [int(value) for value in config.get("worker_cpu_cores", [6, 7])]
        if len(self.cpu_cores) != 2 or any(value < 0 for value in self.cpu_cores):
            raise ValueError("C API lane backend requires two non-negative worker_cpu_cores")
        self.score_threshold = float(config.get("score_threshold", 0.3))
        self.nms_threshold = float(config.get("nms_threshold", 0.45))
        self.mask_threshold = float(config.get("mask_threshold", 0.5))
        self.max_instances = min(3, max(1, int(config.get("max_instances", 3))))
        self.result_notification = str(
            config.get("result_notification", "eventfd")
        ).lower()
        if self.result_notification != "eventfd":
            raise ValueError("C API result_notification must be eventfd")
        self.input_transport: LatestFrameSharedMemory | None = None
        self.result_transport: LatestResultSharedMemory | None = None
        self.process: subprocess.Popen[str] | None = None
        self.result_event_fd: int | None = None
        self.wait_count = 0
        self.wait_timeout_count = 0
        self.wakeup_count = 0
        self.coalesced_notification_count = 0
        self.notification_read_error_count = 0
        self.actual_backend = "unstarted"

    def start(self) -> None:
        if self.process is not None:
            return
        if not self.binary_path.is_file():
            raise RuntimeError(f"native lane backend binary not found: {self.binary_path}")
        if not self.model_path.is_file():
            raise RuntimeError(f"native lane model not found: {self.model_path}")
        if not hasattr(os, "eventfd"):
            raise RuntimeError("eventfd result notification requires Linux/Python 3.10+")
        self.input_transport = LatestFrameSharedMemory(self.width, self.height)
        self.result_transport = LatestResultSharedMemory(self.width, self.height)
        try:
            self.result_event_fd = os.eventfd(
                0,
                os.EFD_NONBLOCK | os.EFD_CLOEXEC,
            )
        except Exception:
            self.close()
            raise
        command = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--input-shm",
            self.input_transport.name,
            "--result-shm",
            self.result_transport.name,
            "--result-event-fd",
            str(self.result_event_fd),
            "--core-masks",
            f"{self.core_masks[0]},{self.core_masks[1]}",
            "--cpu-cores",
            f"{self.cpu_cores[0]},{self.cpu_cores[1]}",
            "--preprocess",
            self.preprocess_backend,
            "--output-mode",
            self.output_mode,
            "--postprocess-backend",
            self.postprocess_backend,
            "--validate-postprocess-exact",
            "1" if self.validate_postprocess_exact else "0",
            "--score-threshold",
            str(self.score_threshold),
            "--nms-threshold",
            str(self.nms_threshold),
            "--mask-threshold",
            str(self.mask_threshold),
            "--max-instances",
            str(self.max_instances),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                text=True,
                pass_fds=(self.result_event_fd,),
            )
            deadline = time.monotonic() + self.startup_timeout_sec
            while time.monotonic() < deadline:
                state, error_code = self.result_transport.state
                if state == BACKEND_STATE_READY:
                    self.actual_backend = "c_api"
                    return
                if state == BACKEND_STATE_ERROR:
                    raise RuntimeError(
                        f"native lane backend initialization failed, code={error_code}"
                    )
                return_code = self.process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"native lane backend exited during startup, code={return_code}"
                    )
                time.sleep(0.01)
            raise TimeoutError(
                f"native lane backend did not become ready within "
                f"{self.startup_timeout_sec:.1f}s"
            )
        except Exception:
            self.close()
            raise

    def publish(
        self,
        frame_rgb: np.ndarray,
        *,
        frame_id: int,
        source_frame_id: int,
        captured_at: float,
    ) -> int:
        if self.input_transport is None:
            raise RuntimeError("native lane backend is not started")
        return self.input_transport.publish(
            frame_rgb,
            frame_id=frame_id,
            source_frame_id=source_frame_id,
            captured_at=captured_at,
        )

    def read_latest(self) -> NativeLaneResult | None:
        if self.result_transport is None:
            return None
        return self.result_transport.read_latest()

    def wait_for_result(self, timeout_sec: float) -> tuple[bool, int, float]:
        """Wait for a committed result notification without polling shared memory."""

        if self.result_event_fd is None:
            raise RuntimeError("native lane result eventfd is not available")
        timeout = max(0.0, float(timeout_sec))
        started = time.perf_counter()
        self.wait_count += 1
        try:
            readable, _, _ = select.select(
                [self.result_event_fd],
                [],
                [],
                timeout,
            )
        except (OSError, ValueError):
            self.notification_read_error_count += 1
            return False, 0, (time.perf_counter() - started) * 1000.0
        if not readable:
            self.wait_timeout_count += 1
            return False, 0, (time.perf_counter() - started) * 1000.0
        try:
            notification_count = int(os.eventfd_read(self.result_event_fd))
        except BlockingIOError:
            notification_count = 0
        except OSError:
            self.notification_read_error_count += 1
            return False, 0, (time.perf_counter() - started) * 1000.0
        if notification_count > 0:
            self.wakeup_count += 1
            self.coalesced_notification_count += max(0, notification_count - 1)
        return (
            notification_count > 0,
            notification_count,
            (time.perf_counter() - started) * 1000.0,
        )

    @property
    def notification_metrics(self) -> dict[str, int]:
        return {
            "wait_count": self.wait_count,
            "wait_timeout_count": self.wait_timeout_count,
            "wakeup_count": self.wakeup_count,
            "coalesced_notification_count": self.coalesced_notification_count,
            "notification_read_error_count": self.notification_read_error_count,
        }

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if self.input_transport is not None:
            self.input_transport.close()
            self.input_transport = None
        if self.result_transport is not None:
            self.result_transport.close()
            self.result_transport = None
        if self.result_event_fd is not None:
            try:
                os.close(self.result_event_fd)
            except OSError:
                pass
            self.result_event_fd = None
        self.actual_backend = "closed"


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
