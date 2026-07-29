"""ctypes wrapper for the RKNN C API PP-YOLOE object detector."""

from __future__ import annotations

import ctypes
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from core.object.blocking import DetectedObject


ABI_VERSION = 1
MAX_NATIVE_DETECTIONS = 64
NPU_CORE_2_MASK = 4
ERROR_CAPACITY = 512


class NativeConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("class_count", ctypes.c_uint32),
        ("max_detections", ctypes.c_uint32),
        ("pipeline_depth", ctypes.c_uint32),
        ("core_mask", ctypes.c_uint32),
        ("score_threshold", ctypes.c_float),
        ("nms_threshold", ctypes.c_float),
        ("class_agnostic_nms", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 7),
    ]


class NativeDetection(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int32),
        ("confidence", ctypes.c_float),
        ("x1", ctypes.c_int32),
        ("y1", ctypes.c_int32),
        ("x2", ctypes.c_int32),
        ("y2", ctypes.c_int32),
    ]


class NativeResult(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("frame_id", ctypes.c_uint64),
        ("context_index", ctypes.c_uint32),
        ("core_mask", ctypes.c_uint32),
        ("status_code", ctypes.c_int32),
        ("detection_count", ctypes.c_uint32),
        ("preprocess_ms", ctypes.c_float),
        ("input_sync_ms", ctypes.c_float),
        ("inference_ms", ctypes.c_float),
        ("npu_run_ms", ctypes.c_float),
        ("output_sync_ms", ctypes.c_float),
        ("postprocess_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
        ("detections", NativeDetection * MAX_NATIVE_DETECTIONS),
    ]


def validate_capi_object_config(config: Mapping[str, Any]) -> None:
    """Reject configurations that could move object inference away from NPU2."""

    if str(config.get("runtime_backend", "c_api")).lower() != "c_api":
        raise ValueError(
            "rknn_object_detector.runtime_backend must be 'c_api' in production"
        )
    if str(config.get("core_mask", "NPU_CORE_2")) != "NPU_CORE_2":
        raise ValueError(
            "rknn_object_detector.core_mask must be exactly 'NPU_CORE_2'"
        )
    if int(config.get("pipeline_depth", 2)) != 2:
        raise ValueError(
            "rknn_object_detector.pipeline_depth must be exactly 2"
        )
    input_size = list(config.get("input_size", [640, 480]))
    if input_size != [640, 480]:
        raise ValueError(
            "rknn_object_detector.input_size must be [640, 480]"
        )
    if str(config.get("input_layout", "nhwc")).lower() != "nhwc":
        raise ValueError("rknn_object_detector.input_layout must be 'nhwc'")
    if str(config.get("input_color", "rgb")).lower() != "rgb":
        raise ValueError("rknn_object_detector.input_color must be 'rgb'")
    if str(config.get("input_dtype", "uint8")).lower() != "uint8":
        raise ValueError("rknn_object_detector.input_dtype must be 'uint8'")


class CapiObjectBackend:
    """Own two preallocated RKNN C API contexts, both fixed to NPU2."""

    def __init__(
        self,
        config: Mapping[str, Any],
        class_names: Sequence[str],
        *,
        library_loader: Any = ctypes.CDLL,
    ) -> None:
        validate_capi_object_config(config)
        self.model_path = Path(str(config.get("model_path", ""))).expanduser()
        self.library_path = Path(
            str(
                config.get(
                    "c_api_library",
                    "native/object_rknn_backend/build/libxsmart_object_rknn.so",
                )
            )
        ).expanduser()
        self.class_names = [str(name) for name in class_names]
        self.max_detections = min(
            MAX_NATIVE_DETECTIONS,
            max(1, int(config.get("max_detections", 30))),
        )
        self.pipeline_depth = 2
        self.core_mask = NPU_CORE_2_MASK
        self.last_timing: dict[str, float] = {}
        self.last_context_index = -1
        self.last_frame_id = -1
        self._timing_by_frame: dict[int, dict[str, float]] = {}
        self._state_lock = threading.Lock()
        self._handle = ctypes.c_void_p()
        self._closed = False

        if not self.model_path.is_file():
            raise RuntimeError(f"native object model not found: {self.model_path}")
        if not self.library_path.is_file():
            raise RuntimeError(
                f"native object backend library not found: {self.library_path}"
            )
        self._library = library_loader(str(self.library_path))
        self._configure_signatures()
        native_config = NativeConfig(
            abi_version=ABI_VERSION,
            struct_size=ctypes.sizeof(NativeConfig),
            class_count=len(self.class_names),
            max_detections=self.max_detections,
            pipeline_depth=self.pipeline_depth,
            core_mask=self.core_mask,
            score_threshold=float(config.get("score_threshold", 0.40)),
            nms_threshold=float(config.get("nms_threshold", 0.45)),
            class_agnostic_nms=int(
                bool(config.get("class_agnostic_nms", False))
            ),
        )
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        ret = self._library.xsmart_object_create(
            str(self.model_path).encode(),
            ctypes.byref(native_config),
            ctypes.byref(self._handle),
            error,
            ERROR_CAPACITY,
        )
        if ret != 0 or not self._handle.value:
            raise RuntimeError(
                "native object backend initialization failed: "
                + self._error_text(error)
            )

    def _configure_signatures(self) -> None:
        self._library.xsmart_object_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(NativeConfig),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self._library.xsmart_object_create.restype = ctypes.c_int
        self._library.xsmart_object_detect.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.POINTER(NativeResult),
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self._library.xsmart_object_detect.restype = ctypes.c_int
        self._library.xsmart_object_destroy.argtypes = [ctypes.c_void_p]
        self._library.xsmart_object_destroy.restype = None

    def detect(
        self,
        frame_rgb: np.ndarray,
        *,
        frame_id: int = 0,
    ) -> list[DetectedObject]:
        if self._closed or not self._handle.value:
            raise RuntimeError("native object backend is closed")
        frame = np.asarray(frame_rgb)
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            raise ValueError("native object input must be HxWx3 uint8 RGB")
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        result = NativeResult()
        error = ctypes.create_string_buffer(ERROR_CAPACITY)
        ret = self._library.xsmart_object_detect(
            self._handle,
            frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            width,
            height,
            int(frame.strides[0]),
            max(0, int(frame_id)),
            ctypes.byref(result),
            error,
            ERROR_CAPACITY,
        )
        if ret != 0:
            raise RuntimeError(
                "native object inference failed: " + self._error_text(error)
            )
        if result.abi_version != ABI_VERSION:
            raise RuntimeError("native object result ABI version mismatch")
        if result.core_mask != NPU_CORE_2_MASK:
            raise RuntimeError(
                f"native object context escaped NPU2: mask={result.core_mask}"
            )
        if result.context_index >= self.pipeline_depth:
            raise RuntimeError(
                f"native object context index is invalid: {result.context_index}"
            )
        timing = {
            "preprocess_ms": float(result.preprocess_ms),
            "input_sync_ms": float(result.input_sync_ms),
            "inference_ms": float(result.inference_ms),
            "npu_run_ms": float(result.npu_run_ms),
            "output_sync_ms": float(result.output_sync_ms),
            "postprocess_ms": float(result.postprocess_ms),
            "total_ms": float(result.total_ms),
            "context_index": float(result.context_index),
            "core_mask": float(result.core_mask),
        }
        with self._state_lock:
            self.last_frame_id = max(self.last_frame_id, int(result.frame_id))
            self.last_context_index = int(result.context_index)
            self.last_timing = timing
            self._timing_by_frame[int(result.frame_id)] = timing
        detections: list[DetectedObject] = []
        count = min(int(result.detection_count), self.max_detections)
        for index in range(count):
            native = result.detections[index]
            class_id = int(native.class_id)
            class_name = (
                self.class_names[class_id]
                if 0 <= class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            if native.x2 <= native.x1 or native.y2 <= native.y1:
                continue
            detections.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=float(native.confidence),
                    bbox_frame=(
                        int(native.x1),
                        int(native.y1),
                        int(native.x2),
                        int(native.y2),
                    ),
                )
            )
        return detections

    def take_timing(self, frame_id: int) -> dict[str, float]:
        with self._state_lock:
            return dict(self._timing_by_frame.pop(int(frame_id), {}))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle.value:
            self._library.xsmart_object_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    @staticmethod
    def _error_text(buffer: ctypes.Array[Any]) -> str:
        text = bytes(buffer.value).decode(errors="replace").strip()
        return text or "unknown native error"
