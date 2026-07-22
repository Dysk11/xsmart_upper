"""Typed adapter between ``xsmart_rknn_native`` and the Python planner/UI."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from core.lane.rknn_segmenter import SegmentationInstance, SegmentationResult
from core.object.blocking import DetectedObject
from core.ocr.recognizer import OcrResult, OcrTextItem


class NativePerceptionBackend:
    """Own native capture and all four RKNN contexts in one extension instance."""

    def __init__(
        self,
        config: dict[str, Any],
        project_root: Path,
        *,
        want_bgr: bool,
        module: Any | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.mode = str(config.get("camera", {}).get("mode", "shared_memory"))
        self.loop_video = bool(config.get("camera", {}).get("loop_video", False))
        self.want_bgr = bool(want_bgr)
        self._module = module
        self._engine: Any | None = None
        self._placeholder: np.ndarray | None = None
        self.last_packet: dict[str, Any] | None = None
        self.last_segmentation = SegmentationResult(
            np.zeros((1, 1), dtype=np.uint8), [], 0.0, "not_started"
        )
        self.last_detections: list[DetectedObject] = []
        self.last_timing: dict[str, Any] = {}

    def open(self) -> None:
        if self._engine is not None:
            return
        module = self._module
        if module is None:
            try:
                module = importlib.import_module("xsmart_rknn_native")
            except ImportError as exc:
                raise RuntimeError(
                    "xsmart_rknn_native is required; build it with "
                    "`PYTHONPATH=.build-deps python3 setup.py build_ext --inplace` "
                    "on RK3588"
                ) from exc
        self._engine = module.NativePerception(self._flatten_config())
        try:
            self._engine.open()
        except Exception:
            self._engine = None
            raise

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._engine is None:
            raise RuntimeError("native perception backend is not open")
        packet = self._engine.next_frame(want_bgr=self.want_bgr)
        self.last_packet = packet
        if not bool(packet.get("ok", False)):
            return False, None

        lane = packet["lane"]
        self.last_segmentation = SegmentationResult(
            mask=np.asarray(lane["mask"], dtype=np.uint8),
            instances=[
                SegmentationInstance(
                    bbox_frame=tuple(int(value) for value in item["bbox"]),
                    confidence=float(item["confidence"]),
                )
                for item in lane.get("instances", [])
            ],
            confidence=float(lane.get("confidence", 0.0)),
            status=str(lane.get("status", "inference_error")),
        )
        self.last_detections = [
            DetectedObject(
                class_name=str(item["class_name"]),
                confidence=float(item["confidence"]),
                bbox_frame=tuple(int(value) for value in item["bbox"]),
            )
            for item in packet.get("detections", [])
        ]
        self.last_timing = {
            "lane": dict(lane.get("timing", {})),
            "object": dict(packet.get("object_timing", {})),
            "total_ms": float(packet.get("total_ms", 0.0)),
        }
        frame = packet.get("frame_bgr")
        if frame is None:
            height = int(packet["height"])
            width = int(packet["width"])
            if self._placeholder is None or self._placeholder.shape[:2] != (height, width):
                self._placeholder = np.zeros((height, width, 3), dtype=np.uint8)
            frame = self._placeholder
        return True, np.asarray(frame, dtype=np.uint8)

    @property
    def frame_id(self) -> int:
        return int(self.last_packet.get("frame_id", 0)) if self.last_packet else 0

    @property
    def captured_at(self) -> float:
        return float(self.last_packet.get("captured_at", 0.0)) if self.last_packet else 0.0

    def submit_ocr(self, trigger_id: int, frame_id: int, bbox: tuple[int, int, int, int]) -> bool:
        if self._engine is None:
            raise RuntimeError("native perception backend is not open")
        return bool(self._engine.submit_ocr(int(trigger_id), int(frame_id), tuple(bbox)))

    def poll_ocr(self) -> OcrResult | None:
        if self._engine is None:
            return None
        raw = self._engine.poll_ocr()
        if raw is None:
            return None
        items = tuple(
            OcrTextItem(
                text=str(item["text"]),
                score=float(item["score"]),
                box=tuple(
                    (float(item["box"][index]), float(item["box"][index + 1]))
                    for index in range(0, 8, 2)
                ),
            )
            for item in raw.get("items", [])
        )
        return OcrResult(
            text=str(raw.get("text", "")),
            items=items,
            frame_id=int(raw.get("frame_id", 0)),
            error=raw.get("error"),
            confidence=float(raw.get("confidence", 0.0)),
            source_bbox=tuple(int(value) for value in raw["source_bbox"]),
            inference_ms=float(raw.get("inference_ms", 0.0)),
            trigger_id=int(raw.get("trigger_id", 0)),
        )

    def release(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = None

    close = release

    def _resolve(self, value: str | Path) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else self.project_root / path)

    def _flatten_config(self) -> dict[str, Any]:
        camera = self.config.get("camera", {})
        lane = self.config.get("rknn_lane_segmenter", {})
        objects = self.config.get("rknn_object_detector", {})
        ocr = self.config.get("ocr", {})
        native = self.config.get("native_perception", {})
        lane_size = lane.get("input_size", [640, 480])
        object_size = objects.get("input_size", [640, 480])
        return {
            "camera_mode": str(camera.get("mode", "shared_memory")),
            "camera_device_id": int(camera.get("device_id", 0)),
            "video_path": self._resolve(camera.get("video_path", "")),
            "shared_memory_name": str(camera.get("shared_memory_name", "shm_ar_video")),
            "loop_video": bool(camera.get("loop_video", False)),
            "mirror": bool(camera.get("mirror", False)),
            "camera_width": int(camera.get("width", 640)),
            "camera_height": int(camera.get("height", 480)),
            "camera_fps": int(camera.get("fps", 60)),
            "reconnect_attempts": int(camera.get("max_reconnect_attempts", 5)),
            "reconnect_interval_sec": float(camera.get("reconnect_interval_sec", 0.5)),
            "lane_model_path": self._resolve(lane["model_path"]),
            "lane_input_width": int(lane_size[0]),
            "lane_input_height": int(lane_size[1]),
            "lane_score_threshold": float(lane.get("score_threshold", 0.30)),
            "lane_nms_threshold": float(lane.get("nms_threshold", 0.45)),
            "lane_mask_threshold": float(lane.get("mask_threshold", 0.50)),
            "lane_max_instances": int(lane.get("max_instances", 3)),
            "object_model_path": self._resolve(objects["model_path"]),
            "object_input_width": int(object_size[0]),
            "object_input_height": int(object_size[1]),
            "object_score_threshold": float(objects.get("score_threshold", 0.50)),
            "object_nms_threshold": float(objects.get("nms_threshold", 0.45)),
            "object_max_detections": int(objects.get("max_detections", 30)),
            "object_class_agnostic_nms": bool(objects.get("class_agnostic_nms", False)),
            "object_class_names": [str(name) for name in objects.get("class_names", [])],
            "ocr_det_model_path": self._resolve(ocr["det_model_path"]),
            "ocr_rec_model_path": self._resolve(ocr["rec_model_path"]),
            "ocr_character_dict_path": self._resolve(
                ocr.get("character_dict_path", "models/ocr/ppocr_keys_v1.txt")
            ),
            "ocr_det_width": int(ocr.get("input_width", 480)),
            "ocr_det_height": int(ocr.get("input_height", 480)),
            "ocr_rec_width": int(ocr.get("rec_input_width", 320)),
            "ocr_rec_height": int(ocr.get("rec_input_height", 48)),
            "ocr_det_threshold": float(ocr.get("det_threshold", 0.30)),
            "ocr_box_threshold": float(ocr.get("box_threshold", 0.60)),
            "ocr_unclip_ratio": float(ocr.get("unclip_ratio", 1.50)),
            "ocr_min_score": float(ocr.get("min_score", 0.50)),
            "ring_buffer_size": max(3, int(native.get("ring_buffer_size", 3))),
        }
