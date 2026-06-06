"""摄像头与视频读取模块。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


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
        self.stream_url = str(config.get("stream_url", ""))
        self.loop_video = bool(config.get("loop_video", False))
        self.width = int(config.get("width", 1280))
        self.height = int(config.get("height", 720))
        self.fps = int(config.get("fps", 30))
        self.mirror = bool(config.get("mirror", False))
        self.reconnect_interval_sec = float(config.get("reconnect_interval_sec", 0.5))
        self.max_reconnect_attempts = int(config.get("max_reconnect_attempts", 5))
        self.capture: Optional[cv2.VideoCapture] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id = 0
        self._last_returned_frame_id = 0
        self._reader_failed = False

    def open(self) -> None:
        """打开摄像头或视频文件，并设置基础属性。

        输入:
            无。

        输出:
            无返回值；若打开失败则抛出 RuntimeError。
        """

        self.release()
        source = self._build_source()
        self.capture = cv2.VideoCapture(source)
        if not self.capture or not self.capture.isOpened():
            raise RuntimeError(f"无法打开图像源: {source}")

        self._apply_capture_properties()
        if self._uses_latest_frame_worker():
            self._start_latest_frame_worker()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧图像，并在失败时自动尝试重连。

        输入:
            无。

        输出:
            返回二元组 (success, frame)。
            当 success 为 True 时，frame 为一帧 BGR 图像；
            当 success 为 False 时，frame 为 None。
        """

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
            return True, frame

        if self.mode == "video" and self.loop_video and self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.capture.read()
            if success and frame is not None:
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                return True, frame
            return False, None

        if self.mode == "video" and not self.loop_video:
            return False, None

        if not self._attempt_reconnect():
            return False, None

        assert self.capture is not None
        success, frame = self.capture.read()
        if success and frame is not None and self.mirror:
            frame = cv2.flip(frame, 1)
        return success, frame

    def release(self) -> None:
        """释放当前图像源资源。

        输入:
            无。

        输出:
            无返回值，若设备已打开则安全关闭。
        """

        self._stop_latest_frame_worker()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._last_returned_frame_id = 0
            self._reader_failed = False

    def _build_source(self) -> Any:
        """根据模式构建 OpenCV VideoCapture 的输入源。

        输入:
            无。

        输出:
            返回设备号、视频文件路径或网络流地址。
        """

        if self.mode == "video":
            if not self.video_path:
                raise ValueError("视频模式下未配置 video_path。")
            return str(Path(self.video_path))
        elif self.mode in ("stream", "http"):
            if not self.stream_url:
                raise ValueError("网络流模式下未配置 stream_url。")
            return self.stream_url
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

        # 减小缓存区大小，这对降低网络流的延迟非常关键
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def _uses_latest_frame_worker(self) -> bool:
        """Return True for realtime sources that should drop stale frames."""

        return self.mode in ("camera", "stream", "http")

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

    def _read_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
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
                    self._last_returned_frame_id = self._latest_frame_id
                else:
                    frame = None
                reader_failed = self._reader_failed

            if reader_failed:
                if not self._attempt_reconnect():
                    return False, None
                continue
            if frame is not None:
                return True, frame
            time.sleep(0.001)

        return False, None

    def _attempt_reconnect(self) -> bool:
        """在读取失败后尝试重新连接图像源。

        输入:
            无。

        输出:
            成功重连时返回 True，失败返回 False。
        """

        for _ in range(self.max_reconnect_attempts):
            try:
                time.sleep(self.reconnect_interval_sec)
                self.open()
                return True
            except Exception:
                continue
        return False
