"""摄像头与视频读取模块。"""

from __future__ import annotations

import threading
import time
import struct
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    """One captured image exposed in both model-RGB and OpenCV-BGR order."""

    rgb: np.ndarray
    bgr: np.ndarray
    captured_at: float
    source_frame_id: int
    color_conversion_ms: float


class CameraReader:
    """统一封装实时摄像头与本地视频读取。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """根据配置初始化图像源读取器。

        输入:
            config: 摄像头相关配置字典，包含模式、分辨率、帧率、镜像等参数。

        输出:
            无返回值，内部保存配置并等待 open() 真正打开设备。
        """

        self.config = config
        self.mode = str(config.get("mode", "camera")).lower()
        self.device_id = int(config.get("device_id", 0))
        self.video_path = str(config.get("video_path", ""))
        self.shared_memory_name = str(config.get("shared_memory_name", "shm_ar_video"))
        self.loop_video = bool(config.get("loop_video", False))
        self.width = int(config.get("width", 1280))
        self.height = int(config.get("height", 720))
        self.fps = int(config.get("fps", 30))
        self.mirror = bool(config.get("mirror", False))
        self.reconnect_interval_sec = float(config.get("reconnect_interval_sec", 0.5))
        self.max_reconnect_attempts = int(config.get("max_reconnect_attempts", 5))
        self.capture: Optional[cv2.VideoCapture] = None
        self._shared_memory: Optional[shared_memory.SharedMemory] = None
        self._shared_memory_header = struct.Struct("@QII")
        self._shared_memory_last_frame_id = 0
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = 0
        self._last_returned_frame_id = 0
        self._reader_failed = False
        self._captured_frame_id = 0
        self.color_conversion_count = 0
        self.color_conversion_total_ms = 0.0
        self.last_color_conversion_ms = 0.0

    def open(self) -> None:
        """打开摄像头或视频文件，并设置基础属性。

        输入:
            无。

        输出:
            无返回值；若打开失败则抛出 RuntimeError。
        """

        self.release()
        if self.mode == "shared_memory":
            if not self._connect_shared_memory_with_retries():
                raise RuntimeError(f"无法连接共享内存: {self.shared_memory_name}")
            return
        source = self._build_source()
        self.capture = cv2.VideoCapture(source)
        if not self.capture or not self.capture.isOpened():
            raise RuntimeError(f"无法打开图像源: {source}")

        self._apply_capture_properties()
        if self._uses_latest_frame_worker():
            self._start_latest_frame_worker()

    def read(self) -> Tuple[bool, Optional[CapturedFrame]]:
        """读取一帧图像，并在失败时自动尝试重连。

        输入:
            无。

        输出:
            返回二元组 (success, frame)。
            当 success 为 True 时，frame 同时包含对齐的 RGB 与 BGR 图像；
            当 success 为 False 时，frame 为 None。
        """

        if self.mode == "shared_memory":
            return self._read_shared_memory()

        if self._uses_latest_frame_worker():
            return self._read_latest_frame()

        if self.capture is None or not self.capture.isOpened():
            if not self._attempt_reconnect():
                return False, None

        assert self.capture is not None
        success, frame = self.capture.read()
        if success and frame is not None:
            if self.mirror:
                frame = cv2.flip(frame, 1)
            return True, self._from_bgr(frame)

        if self.mode == "video" and self.loop_video and self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.capture.read()
            if success and frame is not None:
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                return True, self._from_bgr(frame)
            return False, None

        if self.mode == "video" and not self.loop_video:
            return False, None

        if not self._attempt_reconnect():
            return False, None

        assert self.capture is not None
        success, frame = self.capture.read()
        if success and frame is not None and self.mirror:
            frame = cv2.flip(frame, 1)
        return (True, self._from_bgr(frame)) if success and frame is not None else (False, None)

    def release(self) -> None:
        """释放当前图像源资源。

        输入:
            无。

        输出:
            无返回值，若设备已打开则安全关闭。
        """

        self._stop_latest_frame_worker()
        self._close_shared_memory()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._last_returned_frame_id = 0
            self._reader_failed = False
        self._shared_memory_last_frame_id = 0
        self._captured_frame_id = 0
        self.color_conversion_count = 0
        self.color_conversion_total_ms = 0.0
        self.last_color_conversion_ms = 0.0

    def _build_source(self) -> Any:
        """根据模式构建 OpenCV VideoCapture 的输入源。

        输入:
            无。

        输出:
            返回设备号或视频文件路径。
        """

        if self.mode == "video":
            if not self.video_path:
                raise ValueError("视频模式下未配置 video_path。")
            return str(Path(self.video_path))
        elif self.mode == "shared_memory":
            raise RuntimeError("共享内存模式不使用 OpenCV VideoCapture。")
        elif self.mode != "camera":
            raise ValueError(f"不支持的图像源模式: {self.mode}")
        return self.device_id

    def _apply_capture_properties(self) -> None:
        """为摄像头或视频对象设置基础读取参数。

        输入:
            无。

        输出:
            无返回值，内部修改 VideoCapture 的属性。
        """

        if self.capture is None:
            return

        # 减小摄像头缓存区，确保实时模式优先返回最新帧。
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def _uses_latest_frame_worker(self) -> bool:
        """Return True for realtime sources that should drop stale frames."""

        return self.mode == "camera"

    def _open_shared_memory(self) -> None:
        """连接由 AR 系统创建的共享内存，但不取得其所有权。"""

        if not self.shared_memory_name:
            raise ValueError("共享内存模式下未配置 shared_memory_name。")
        try:
            shm = shared_memory.SharedMemory(name=self.shared_memory_name, create=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f"共享内存不存在: {self.shared_memory_name}") from exc

        if shm.size < self._shared_memory_header.size:
            shm.close()
            raise RuntimeError(
                f"共享内存 {self.shared_memory_name} 小于 {self._shared_memory_header.size} 字节头部。"
            )

        # 接收端不能在退出时删除 AR 引擎创建的共享内存。
        try:
            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except (AttributeError, KeyError, ValueError):
            pass

        self._shared_memory = shm

    def _connect_shared_memory_with_retries(self) -> bool:
        """按配置的重连次数等待 AR 共享内存出现。"""

        attempts = max(1, self.max_reconnect_attempts)
        for attempt in range(attempts):
            self._close_shared_memory()
            try:
                self._open_shared_memory()
                return True
            except (RuntimeError, ValueError):
                if attempt + 1 < attempts:
                    time.sleep(self.reconnect_interval_sec)
        return False

    def _close_shared_memory(self) -> None:
        """关闭接收端句柄，不删除 AR 系统拥有的共享内存。"""

        if self._shared_memory is None:
            return
        try:
            self._shared_memory.close()
        finally:
            self._shared_memory = None

    def _read_shared_memory(self, reconnect_on_timeout: bool = True) -> Tuple[bool, Optional[CapturedFrame]]:
        """读取一致的 RGB888 新帧并生成一次可复用的 BGR 对应帧。"""

        if self._shared_memory is None:
            if not self._attempt_reconnect():
                return False, None

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            shm = self._shared_memory
            if shm is None:
                break
            try:
                header = bytes(shm.buf[: self._shared_memory_header.size])
                frame_id, width, height = self._shared_memory_header.unpack(header)
                if frame_id == 0 or frame_id == self._shared_memory_last_frame_id:
                    time.sleep(0.002)
                    continue

                if width <= 0 or height <= 0:
                    time.sleep(0.002)
                    continue
                frame_size = width * height * 3
                required_size = self._shared_memory_header.size + frame_size
                if required_size > shm.size:
                    time.sleep(0.002)
                    continue

                image_view = np.ndarray(
                    (height, width, 3),
                    dtype=np.uint8,
                    buffer=shm.buf,
                    offset=self._shared_memory_header.size,
                )
                rgb_frame = image_view.copy()
                del image_view

                header_after = bytes(shm.buf[: self._shared_memory_header.size])
                frame_id_after, width_after, height_after = self._shared_memory_header.unpack(header_after)
                if (frame_id_after, width_after, height_after) != (frame_id, width, height):
                    continue

                self._shared_memory_last_frame_id = frame_id
                if self.mirror:
                    rgb_frame = cv2.flip(rgb_frame, 1)
                return True, self._from_rgb(rgb_frame, source_frame_id=int(frame_id))
            except (BufferError, TypeError, ValueError, struct.error):
                break

        if reconnect_on_timeout and self._connect_shared_memory_with_retries():
            return self._read_shared_memory(reconnect_on_timeout=False)
        return False, None

    def _start_latest_frame_worker(self) -> None:
        """Continuously read realtime sources so callers receive the newest frame."""

        self._stop_latest_frame_worker()
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._last_returned_frame_id = 0
            self._reader_failed = False
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._latest_frame_worker,
            name="xsmart-camera-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _stop_latest_frame_worker(self) -> None:
        """Stop the realtime capture worker if it is running."""

        if self._reader_thread is None:
            return
        self._reader_stop.set()
        self._reader_thread.join(timeout=1.0)
        self._reader_thread = None

    def _latest_frame_worker(self) -> None:
        """Keep draining OpenCV capture and cache only the newest frame."""

        while not self._reader_stop.is_set():
            capture = self.capture
            if capture is None or not capture.isOpened():
                with self._frame_lock:
                    self._reader_failed = True
                break

            success, frame = capture.read()
            if not success or frame is None:
                with self._frame_lock:
                    self._reader_failed = True
                break
            if self._reader_stop.is_set() or capture is not self.capture:
                break

            if self.mirror:
                frame = cv2.flip(frame, 1)

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_frame_id += 1

    def _read_latest_frame(self) -> Tuple[bool, Optional[CapturedFrame]]:
        """Return the next new cached realtime frame, reconnecting if capture failed."""

        if self.capture is None or not self.capture.isOpened():
            if not self._attempt_reconnect():
                return False, None

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with self._frame_lock:
                has_new_frame = (
                    self._latest_frame is not None
                    and self._latest_frame_id > self._last_returned_frame_id
                )
                if has_new_frame:
                    frame = self._latest_frame.copy()
                    source_frame_id = self._latest_frame_id
                    self._last_returned_frame_id = source_frame_id
                else:
                    frame = None
                    source_frame_id = 0
                reader_failed = self._reader_failed

            if reader_failed:
                if not self._attempt_reconnect():
                    return False, None
                continue
            if frame is not None:
                return True, self._from_bgr(frame, source_frame_id=source_frame_id)
            time.sleep(0.001)

        return False, None

    def _from_rgb(
        self,
        frame_rgb: np.ndarray,
        source_frame_id: int | None = None,
    ) -> CapturedFrame:
        """Keep the source RGB frame and create its single reusable BGR peer."""

        captured_at = time.perf_counter()
        started = time.perf_counter()
        frame_rgb = np.ascontiguousarray(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        conversion_ms = (time.perf_counter() - started) * 1000.0
        return self._finish_capture(
            frame_rgb,
            frame_bgr,
            captured_at,
            source_frame_id,
            conversion_ms,
        )

    def _from_bgr(
        self,
        frame_bgr: np.ndarray,
        source_frame_id: int | None = None,
    ) -> CapturedFrame:
        """Keep the OpenCV BGR frame and create its single reusable RGB peer."""

        captured_at = time.perf_counter()
        started = time.perf_counter()
        frame_bgr = np.ascontiguousarray(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        conversion_ms = (time.perf_counter() - started) * 1000.0
        return self._finish_capture(
            frame_rgb,
            frame_bgr,
            captured_at,
            source_frame_id,
            conversion_ms,
        )

    def _finish_capture(
        self,
        frame_rgb: np.ndarray,
        frame_bgr: np.ndarray,
        captured_at: float,
        source_frame_id: int | None,
        conversion_ms: float,
    ) -> CapturedFrame:
        if source_frame_id is None:
            self._captured_frame_id += 1
            source_frame_id = self._captured_frame_id
        else:
            self._captured_frame_id = max(self._captured_frame_id, int(source_frame_id))
        self.color_conversion_count += 1
        self.color_conversion_total_ms += conversion_ms
        self.last_color_conversion_ms = conversion_ms
        return CapturedFrame(
            rgb=frame_rgb,
            bgr=frame_bgr,
            captured_at=captured_at,
            source_frame_id=int(source_frame_id),
            color_conversion_ms=conversion_ms,
        )

    def _attempt_reconnect(self) -> bool:
        """在读取失败后尝试重新连接图像源。

        输入:
            无。

        输出:
            成功重连时返回 True，失败返回 False。
        """

        if self.mode == "shared_memory":
            return self._connect_shared_memory_with_retries()

        for _ in range(self.max_reconnect_attempts):
            try:
                time.sleep(self.reconnect_interval_sec)
                self.open()
                return True
            except Exception:
                continue
        return False
