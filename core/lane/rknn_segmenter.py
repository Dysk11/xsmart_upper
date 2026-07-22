"""RKNNLite2 YOLOv5-seg lane segmentation for RK3588."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


STRIDES = np.asarray([8.0, 16.0, 32.0], dtype=np.float32)
ANCHORS = np.asarray(
    [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ],
    dtype=np.float32,
)
EXPECTED_OUTPUT_SHAPES = (
    (1, 18, 60, 80),
    (1, 96, 60, 80),
    (1, 18, 30, 40),
    (1, 96, 30, 40),
    (1, 18, 15, 20),
    (1, 96, 15, 20),
    (1, 32, 120, 160),
)


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    resized_width: int
    resized_height: int
    source_width: int
    source_height: int


@dataclass(frozen=True)
class SegmentationInstance:
    bbox_frame: tuple[int, int, int, int]
    confidence: float


@dataclass
class SegmentationResult:
    mask: np.ndarray
    instances: list[SegmentationInstance]
    confidence: float
    status: str


@dataclass
class LaneInference:
    """RKNN outputs and timing state awaiting CPU post-processing."""

    source_shape: tuple[int, int]
    started_at: float
    preprocessed_at: float
    inferred_at: float
    outputs: list[np.ndarray] | None = None
    letterbox: LetterboxInfo | None = None
    status: str = "ok"


class RknnLaneSegmenter:
    """Load a split-head YOLOv5n-seg RKNN model and reconstruct track masks."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enable", True))
        self.model_path = Path(str(config.get("model_path", ""))).expanduser()
        input_size = config.get("input_size", [640, 480])
        self.input_width = int(input_size[0])
        self.input_height = int(input_size[1])
        self.score_threshold = float(config.get("score_threshold", 0.25))
        self.nms_threshold = float(config.get("nms_threshold", 0.45))
        self.mask_threshold = float(config.get("mask_threshold", 0.5))
        self.max_instances = int(config.get("max_instances", 20))
        self.runtime_backend = str(config.get("runtime_backend", "lite2")).lower()
        self.core_mask_name = str(config.get("core_mask", "NPU_CORE_0"))
        self._rknn: Any = None
        self._runtime_ready = False
        self._warned: set[str] = set()
        self.last_timing: dict[str, float] = {}
        self._canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        self._grids: dict[tuple[int, int], np.ndarray] = {}

    def segment(self, frame_rgb: np.ndarray) -> SegmentationResult:
        inference = self.infer(frame_rgb)
        result, timing = self.complete(inference)
        self.last_timing = timing
        return result

    def infer(self, frame_rgb: np.ndarray) -> LaneInference:
        """Run preprocessing and synchronous RKNN inference only."""

        started = time.perf_counter()
        shape = tuple(frame_rgb.shape[:2]) if frame_rgb.ndim >= 2 else (1, 1)
        if not self.enabled:
            return LaneInference(shape, started, started, started, status="disabled")
        if frame_rgb.size == 0:
            return LaneInference(shape, started, started, started, status="empty_frame")
        if not self._ensure_runtime():
            return LaneInference(shape, started, started, started, status="runtime_unavailable")

        try:
            input_tensor, letterbox = self._preprocess(frame_rgb)
            preprocessed = time.perf_counter()
            outputs = self._rknn.inference(inputs=[input_tensor])
            inferred = time.perf_counter()
            if outputs is None:
                raise RuntimeError("RKNN inference returned no outputs")
            return LaneInference(
                source_shape=shape,
                started_at=started,
                preprocessed_at=preprocessed,
                inferred_at=inferred,
                outputs=list(outputs),
                letterbox=letterbox,
            )
        except Exception as error:
            self._warn_once("inference", f"RKNN lane segmentation failed: {error}")
            failed_at = time.perf_counter()
            return LaneInference(shape, started, failed_at, failed_at, status="inference_error")

    def complete(self, inference: LaneInference) -> tuple[SegmentationResult, dict[str, float]]:
        """Post-process one completed inference without touching the RKNN runtime."""

        postprocess_started = time.perf_counter()
        if inference.outputs is None or inference.letterbox is None:
            result = self._empty(inference.source_shape, inference.status)
            finished = time.perf_counter()
        else:
            try:
                result = self.postprocess(inference.outputs, inference.letterbox)
            except Exception as error:
                self._warn_once("postprocess", f"RKNN lane post-processing failed: {error}")
                result = self._empty(inference.source_shape, "postprocess_error")
            finished = time.perf_counter()

        timing = {
            "preprocess_ms": (inference.preprocessed_at - inference.started_at) * 1000.0,
            "inference_ms": (inference.inferred_at - inference.preprocessed_at) * 1000.0,
            "postprocess_queue_ms": (postprocess_started - inference.inferred_at) * 1000.0,
            "postprocess_ms": (finished - postprocess_started) * 1000.0,
            "total_ms": (finished - inference.started_at) * 1000.0,
        }
        self.last_timing = timing
        return result, timing

    def close(self) -> None:
        if self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:
                pass
        self._rknn = None
        self._runtime_ready = False

    def _ensure_runtime(self) -> bool:
        if self._runtime_ready:
            return True
        if self.runtime_backend != "lite2":
            self._warn_once("backend", f"Unsupported RKNN runtime backend: {self.runtime_backend}")
            return False
        try:
            from rknnlite.api import RKNNLite  # type: ignore
        except ImportError:
            self._warn_once("import", "rknnlite.api is not installed; RKNN lane segmentation is unavailable.")
            return False
        if not self.model_path.is_file():
            self._warn_once("model", f"RKNN lane model not found: {self.model_path}")
            return False
        rknn = RKNNLite()
        ret = rknn.load_rknn(str(self.model_path))
        if ret != 0:
            self._warn_once("load", f"Failed to load RKNN lane model, ret={ret}")
            return False
        core_mask = getattr(RKNNLite, self.core_mask_name, None)
        try:
            ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime()
        except TypeError:
            ret = rknn.init_runtime()
        if ret != 0:
            rknn.release()
            self._warn_once("runtime", f"Failed to initialize RKNN lane runtime, ret={ret}")
            return False
        self._rknn = rknn
        self._runtime_ready = True
        print(f"RKNN lane segmenter loaded: {self.model_path}")
        return True

    def _preprocess(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        height, width = frame_rgb.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(frame_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = self._canvas
        canvas.fill(114)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas[None, ...], LetterboxInfo(
            scale, pad_x, pad_y, resized_width, resized_height, width, height
        )

    def decode(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        arrays = [np.asarray(output) for output in outputs]
        shapes = tuple(tuple(array.shape) for array in arrays)
        if shapes != EXPECTED_OUTPUT_SHAPES:
            raise ValueError(f"unexpected RKNN lane output shapes: {shapes}")
        predictions: list[np.ndarray] = []
        for level in range(3):
            box_cls, coeff = arrays[level * 2 : level * 2 + 2]
            batch, _, height, width = box_cls.shape
            box_cls = box_cls.reshape(batch, 3, 6, height, width).transpose(0, 1, 3, 4, 2)
            coeff = coeff.reshape(batch, 3, 32, height, width).transpose(0, 1, 3, 4, 2)
            grid_key = (height, width)
            grid = self._grids.get(grid_key)
            if grid is None:
                gy, gx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
                grid = np.stack((gx, gy), axis=-1)[None, None].astype(np.float32) - 0.5
                self._grids[grid_key] = grid
            xy = (box_cls[..., :2] * 2.0 + grid) * STRIDES[level]
            wh = (box_cls[..., 2:4] * 2.0) ** 2 * ANCHORS[level][None, :, None, None]
            values = np.concatenate((xy, wh, box_cls[..., 4:6], coeff), axis=-1)
            predictions.append(values.reshape(batch, -1, 38))
        return np.concatenate(predictions, axis=1), arrays[6]

    def postprocess(self, outputs: list[np.ndarray], info: LetterboxInfo) -> SegmentationResult:
        predictions, prototypes = self.decode(outputs)
        candidates = predictions[0]
        scores = candidates[:, 4] * candidates[:, 5]
        keep = np.flatnonzero(scores >= self.score_threshold)
        if keep.size == 0:
            return self._empty((info.source_height, info.source_width), "no_detection")
        candidates = candidates[keep]
        scores = scores[keep]
        boxes = self._xywh_to_xyxy(candidates[:, :4])
        selected = self._nms(boxes, scores)[: self.max_instances]
        union_input = np.zeros((self.input_height, self.input_width), dtype=np.uint8)
        proto = prototypes[0].reshape(32, -1)
        threshold = min(max(self.mask_threshold, 1e-6), 1.0 - 1e-6)
        logit_threshold = float(np.log(threshold / (1.0 - threshold)))
        selected_array = np.asarray(selected, dtype=np.intp)
        selected_logits = candidates[selected_array, 6:] @ proto
        selected_logits = selected_logits.reshape(
            len(selected), prototypes.shape[2], prototypes.shape[3]
        )
        resized_logits = cv2.resize(
            np.moveaxis(selected_logits, 0, -1),
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if resized_logits.ndim == 2:
            resized_logits = resized_logits[:, :, None]
        selected_masks = resized_logits >= logit_threshold
        instances: list[SegmentationInstance] = []
        for mask_index, index in enumerate(selected):
            box = boxes[index]
            x1, y1, x2, y2 = self._clip_box(box)
            if x2 > x1 and y2 > y1:
                crop = selected_masks[y1:y2, x1:x2, mask_index]
                union_input[y1:y2, x1:x2][crop] = 255
            instances.append(
                SegmentationInstance(self._restore_box(box, info), float(scores[index]))
            )
        restored = self._restore_mask(union_input, info)
        return SegmentationResult(restored, instances, float(scores[selected[0]]), "ok")

    def _restore_mask(self, mask: np.ndarray, info: LetterboxInfo) -> np.ndarray:
        cropped = mask[
            info.pad_y : info.pad_y + info.resized_height,
            info.pad_x : info.pad_x + info.resized_width,
        ]
        return cv2.resize(cropped, (info.source_width, info.source_height), interpolation=cv2.INTER_NEAREST)

    def _restore_box(self, box: np.ndarray, info: LetterboxInfo) -> tuple[int, int, int, int]:
        x1 = int(round((box[0] - info.pad_x) / info.scale))
        y1 = int(round((box[1] - info.pad_y) / info.scale))
        x2 = int(round((box[2] - info.pad_x) / info.scale))
        y2 = int(round((box[3] - info.pad_y) / info.scale))
        return (
            max(0, min(info.source_width, x1)), max(0, min(info.source_height, y1)),
            max(0, min(info.source_width, x2)), max(0, min(info.source_height, y2)),
        )

    def _clip_box(self, box: np.ndarray) -> tuple[int, int, int, int]:
        return (
            max(0, min(self.input_width, int(np.floor(box[0])))),
            max(0, min(self.input_height, int(np.floor(box[1])))),
            max(0, min(self.input_width, int(np.ceil(box[2])))),
            max(0, min(self.input_height, int(np.ceil(box[3])))),
        )

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        result = boxes.copy()
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return result

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        order = scores.argsort()[::-1]
        selected: list[int] = []
        while order.size:
            current = int(order[0])
            selected.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(0.0, boxes[current, 3] - boxes[current, 1])
            areas_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
            iou = intersection / np.maximum(area_current + areas_rest - intersection, 1e-6)
            order = rest[iou <= self.nms_threshold]
        return selected

    @staticmethod
    def _empty(shape: tuple[int, int], status: str) -> SegmentationResult:
        return SegmentationResult(np.zeros(shape, dtype=np.uint8), [], 0.0, status)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            print(f"[RknnLaneSegmenter] {message}")
            self._warned.add(key)
