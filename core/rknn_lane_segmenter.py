"""RKNNLite2 YOLOv5-seg lane segmentation for RK3588."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
        self.core_mask_name = str(config.get("core_mask", "NPU_CORE_0"))
        self._rknn: Any = None
        self._runtime_ready = False
        self._warned: set[str] = set()

    def segment(self, frame_bgr: np.ndarray) -> SegmentationResult:
        shape = frame_bgr.shape[:2] if frame_bgr.ndim >= 2 else (1, 1)
        if not self.enabled:
            return self._empty(shape, "disabled")
        if frame_bgr.size == 0:
            return self._empty(shape, "empty_frame")
        if not self._ensure_runtime():
            return self._empty(shape, "runtime_unavailable")

        input_tensor, letterbox = self._preprocess(frame_bgr)
        try:
            outputs = self._rknn.inference(inputs=[input_tensor])
            if outputs is None:
                raise RuntimeError("RKNN inference returned no outputs")
            return self.postprocess(outputs, letterbox)
        except Exception as error:
            self._warn_once("inference", f"RKNN lane segmentation failed: {error}")
            return self._empty(shape, "inference_error")

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

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        height, width = frame_bgr.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb[None, ...], LetterboxInfo(
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
            gy, gx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            grid = np.stack((gx, gy), axis=-1)[None, None].astype(np.float32) - 0.5
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
        instances: list[SegmentationInstance] = []
        for index in selected:
            box = boxes[index]
            logits = candidates[index, 6:] @ proto
            logits = logits.reshape(prototypes.shape[2], prototypes.shape[3])
            logits = cv2.resize(logits, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            instance_mask = logits >= logit_threshold
            x1, y1, x2, y2 = self._clip_box(box)
            cropped = np.zeros_like(instance_mask)
            if x2 > x1 and y2 > y1:
                cropped[y1:y2, x1:x2] = instance_mask[y1:y2, x1:x2]
            union_input[cropped] = 255
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
