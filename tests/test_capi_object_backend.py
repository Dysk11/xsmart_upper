from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from core.object.capi_backend import (
    ABI_VERSION,
    CapiObjectBackend,
    NativeConfig,
    NativeResult,
    NPU_CORE_2_MASK,
    validate_capi_object_config,
)
from core.object.rknn_detector import RknnObjectDetector


def object_config(tmp_path: Path, **overrides: object) -> dict[str, object]:
    model = tmp_path / "object.rknn"
    library = tmp_path / "libxsmart_object_rknn.so"
    model.write_bytes(b"rknn")
    library.write_bytes(b"elf")
    config: dict[str, object] = {
        "enable": True,
        "runtime_backend": "c_api",
        "model_path": str(model),
        "c_api_library": str(library),
        "input_size": [640, 480],
        "input_layout": "nhwc",
        "input_color": "rgb",
        "input_dtype": "uint8",
        "score_threshold": 0.4,
        "nms_threshold": 0.45,
        "max_detections": 30,
        "class_names": ["car", "human"],
        "core_mask": "NPU_CORE_2",
        "pipeline_depth": 2,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "core_mask",
    [
        "NPU_CORE_AUTO",
        "NPU_CORE_0",
        "NPU_CORE_1",
        "NPU_CORE_0_1",
        "NPU_CORE_0_1_2",
    ],
)
def test_object_capi_rejects_every_non_npu2_mask(
    tmp_path: Path,
    core_mask: str,
) -> None:
    with pytest.raises(ValueError, match="must be exactly 'NPU_CORE_2'"):
        validate_capi_object_config(
            object_config(tmp_path, core_mask=core_mask)
        )


def test_object_detector_validates_npu2_before_runtime_load(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be exactly 'NPU_CORE_2'"):
        RknnObjectDetector(
            object_config(tmp_path, core_mask="NPU_CORE_0_1_2")
        )


def test_object_capi_requires_two_contexts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pipeline_depth must be exactly 2"):
        validate_capi_object_config(
            object_config(tmp_path, pipeline_depth=1)
        )


class FakeFunction:
    def __init__(self, callback: object) -> None:
        self.callback = callback
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: object) -> object:
        return self.callback(*args)  # type: ignore[operator]


class FakeNativeLibrary:
    def __init__(self) -> None:
        self.create_config: NativeConfig | None = None
        self.detect_calls = 0
        self.destroy_calls = 0
        self.xsmart_object_create = FakeFunction(self._create)
        self.xsmart_object_detect = FakeFunction(self._detect)
        self.xsmart_object_destroy = FakeFunction(self._destroy)

    def _create(
        self,
        _model_path: object,
        config_pointer: object,
        handle_pointer: object,
        _error: object,
        _capacity: object,
    ) -> int:
        config = ctypes.cast(
            config_pointer,
            ctypes.POINTER(NativeConfig),
        ).contents
        self.create_config = NativeConfig.from_buffer_copy(config)
        ctypes.cast(
            handle_pointer,
            ctypes.POINTER(ctypes.c_void_p),
        )[0] = ctypes.c_void_p(123)
        return 0

    def _detect(
        self,
        _handle: object,
        _rgb: object,
        _width: object,
        _height: object,
        _stride: object,
        frame_id: object,
        result_pointer: object,
        _error: object,
        _capacity: object,
    ) -> int:
        result = ctypes.cast(
            result_pointer,
            ctypes.POINTER(NativeResult),
        ).contents
        result.abi_version = ABI_VERSION
        result.struct_size = ctypes.sizeof(NativeResult)
        result.frame_id = int(frame_id)  # type: ignore[arg-type]
        result.context_index = self.detect_calls % 2
        result.core_mask = NPU_CORE_2_MASK
        result.detection_count = 1
        result.preprocess_ms = 0.2
        result.input_sync_ms = 0.1
        result.inference_ms = 8.0
        result.npu_run_ms = 7.5
        result.output_sync_ms = 0.1
        result.postprocess_ms = 0.3
        result.total_ms = 8.7
        result.detections[0].class_id = 1
        result.detections[0].confidence = 0.91
        result.detections[0].x1 = 10
        result.detections[0].y1 = 20
        result.detections[0].x2 = 30
        result.detections[0].y2 = 40
        self.detect_calls += 1
        return 0

    def _destroy(self, _handle: object) -> None:
        self.destroy_calls += 1


def test_object_capi_maps_results_and_reports_npu2(
    tmp_path: Path,
) -> None:
    library = FakeNativeLibrary()
    backend = CapiObjectBackend(
        object_config(tmp_path),
        ["car", "human"],
        library_loader=lambda _path: library,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        first = backend.detect(frame, frame_id=10)
        second = backend.detect(frame, frame_id=11)
    finally:
        backend.close()

    assert library.create_config is not None
    assert library.create_config.core_mask == NPU_CORE_2_MASK
    assert library.create_config.pipeline_depth == 2
    assert first[0].class_name == "human"
    assert first[0].bbox_frame == (10, 20, 30, 40)
    assert second[0].class_name == "human"
    assert backend.last_context_index == 1
    assert backend.last_timing["core_mask"] == 4.0
    assert backend.last_timing["npu_run_ms"] == pytest.approx(7.5)
    assert library.destroy_calls == 1


def test_object_capi_keeps_high_watermark_for_out_of_order_contexts(
    tmp_path: Path,
) -> None:
    library = FakeNativeLibrary()
    backend = CapiObjectBackend(
        object_config(tmp_path),
        ["car", "human"],
        library_loader=lambda _path: library,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        backend.detect(frame, frame_id=9)
        backend.detect(frame, frame_id=8)
        assert backend.last_frame_id == 9
        assert backend.take_timing(8)["core_mask"] == 4.0
    finally:
        backend.close()
