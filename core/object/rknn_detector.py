"""RKNN object detector wrapper for RK3588/Orange Pi runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Sequence

import cv2
import numpy as np

from core.object.blocking import DetectedObject


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    output_width: int
    output_height: int


class RknnObjectDetector:
    """Loads a YOLO-style RKNN model and returns detections in frame coordinates."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enable", False))
        self.model_path = Path(str(config.get("model_path", ""))).expanduser()
        input_size = config.get("input_size", [640, 640])
        self.input_width = int(input_size[0])
        self.input_height = int(input_size[1])
        self.input_layout = str(config.get("input_layout", "nhwc")).lower()
        self.input_color = str(config.get("input_color", "rgb")).lower()
        self.input_dtype = str(config.get("input_dtype", "uint8")).lower()
        self.score_threshold = float(config.get("score_threshold", 0.35))
        self.nms_threshold = float(config.get("nms_threshold", 0.45))
        self.max_detections = int(config.get("max_detections", 50))
        self.class_names = [
            str(name)
            for name in config.get(
                "class_names",
                ["light", "speed_sign", "dir_sign", "human", "car", "coin", "Stop", "Go", "arch"],
            )
        ]
        self.class_agnostic_nms = bool(config.get("class_agnostic_nms", False))
        self.runtime_backend = str(config.get("runtime_backend", "lite2")).lower()
        self.core_mask_name = str(config.get("core_mask", "NPU_CORE_0_1_2"))

        self._rknn: Any = None
        self._rknn_cls: Any = None
        self._runtime_ready = False
        self._warned_unavailable = False
        self._warned_postprocess = False
        self._warned_class_count = False
        self.last_timing: dict[str, float] = {}
        self._canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        self._rgb_canvas = np.empty_like(self._canvas)

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        if not self.enabled:
            return []
        if frame_bgr.size == 0:
            return []
        if not self._ensure_runtime():
            return []

        started = time.perf_counter()
        input_tensor, letterbox = self._preprocess(frame_bgr)
        preprocessed = time.perf_counter()
        outputs = self._rknn.inference(inputs=[input_tensor])
        inferred = time.perf_counter()
        if outputs is None:
            return []

        detections = self._postprocess(outputs, letterbox, frame_bgr.shape[:2])
        finished = time.perf_counter()
        self.last_timing = {
            "preprocess_ms": (preprocessed - started) * 1000.0,
            "inference_ms": (inferred - preprocessed) * 1000.0,
            "postprocess_ms": (finished - inferred) * 1000.0,
            "total_ms": (finished - started) * 1000.0,
        }
        if len(detections) > self.max_detections:
            detections = detections[: self.max_detections]
        return detections

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
            self._warn_once(f"Unsupported RKNN runtime backend: {self.runtime_backend}")
            return False
        try:
            from rknnlite.api import RKNNLite  # type: ignore
        except ImportError:
            self._warn_once(
                "rknnlite.api is not installed; RKNN detection is disabled in this environment."
            )
            return False

        if not self.model_path.exists():
            self._warn_once(f"RKNN model not found: {self.model_path}")
            return False

        self._rknn_cls = RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(str(self.model_path))
        if ret != 0:
            self._warn_once(f"Failed to load RKNN model {self.model_path}, ret={ret}")
            return False

        core_mask = getattr(RKNNLite, self.core_mask_name, None)
        try:
            ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime()
        except TypeError:
            ret = rknn.init_runtime()

        if ret != 0:
            self._warn_once(f"Failed to init RKNN runtime, ret={ret}")
            return False

        self._rknn = rknn
        self._runtime_ready = True
        print(f"RKNN detector loaded: {self.model_path}")
        return True

    def _warn_once(self, message: str) -> None:
        if not self._warned_unavailable:
            print(f"[RknnObjectDetector] {message}")
            self._warned_unavailable = True

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        resized, info = self._letterbox(frame_bgr)
        if self.input_color == "rgb":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=self._rgb_canvas)

        if self.input_dtype == "float32":
            tensor = resized.astype(np.float32) / 255.0
        else:
            tensor = resized if resized.dtype == np.uint8 else resized.astype(np.uint8)

        if self.input_layout == "nchw":
            tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, info

    def _letterbox(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        frame_height, frame_width = frame_bgr.shape[:2]
        scale = min(self.input_width / frame_width, self.input_height / frame_height)
        new_width = int(round(frame_width * scale))
        new_height = int(round(frame_height * scale))
        resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        canvas = self._canvas
        canvas.fill(114)
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2
        canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
        return canvas, LetterboxInfo(
            scale=float(scale),
            pad_x=float(pad_x),
            pad_y=float(pad_y),
            output_width=self.input_width,
            output_height=self.input_height,
        )

    def _postprocess(
        self,
        outputs: Sequence[Any],
        letterbox: LetterboxInfo,
        frame_shape: tuple[int, int],
    ) -> list[DetectedObject]:
        arrays = [np.asarray(output) for output in outputs]
        if len(arrays) == 9:
            return self._postprocess_yolo_dfl(arrays, letterbox, frame_shape)
        if len(arrays) == 2:
            return self._postprocess_ppyoloe(arrays, letterbox, frame_shape)
        if len(arrays) == 1:
            return self._postprocess_flat_output(arrays[0], letterbox, frame_shape)

        if not self._warned_postprocess:
            print(f"[RknnObjectDetector] Unsupported RKNN output count: {len(arrays)}")
            self._warned_postprocess = True
        return []

    def _postprocess_ppyoloe(
        self,
        outputs: Sequence[np.ndarray],
        letterbox: LetterboxInfo,
        frame_shape: tuple[int, int],
    ) -> list[DetectedObject]:
        """Decode PP-YOLOE boxes [1,N,4] and class scores [1,C,N]."""
        boxes = np.squeeze(np.asarray(outputs[0], dtype=np.float32))
        class_scores = np.squeeze(np.asarray(outputs[1], dtype=np.float32))
        if boxes.ndim != 2 or class_scores.ndim != 2:
            return []
        if boxes.shape[1] != 4 and boxes.shape[0] == 4:
            boxes = boxes.transpose(1, 0)
        if boxes.shape[1] != 4:
            return []

        if class_scores.shape[0] == len(self.class_names):
            class_scores = class_scores.transpose(1, 0)
        elif class_scores.shape[1] != len(self.class_names):
            self._warn_class_count(min(class_scores.shape))
            return []
        if class_scores.shape[0] != boxes.shape[0]:
            return []

        class_scores = self._ensure_probability(class_scores)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        selected = np.flatnonzero(confidences >= self.score_threshold)
        candidates = [
            (
                float(confidences[index]),
                int(class_ids[index]),
                tuple(float(value) for value in boxes[index]),
            )
            for index in selected.tolist()
        ]
        return self._build_detections(candidates, letterbox, frame_shape)

    def _postprocess_yolo_dfl(
        self,
        outputs: Sequence[np.ndarray],
        letterbox: LetterboxInfo,
        frame_shape: tuple[int, int],
    ) -> list[DetectedObject]:
        candidates: list[tuple[float, int, tuple[float, float, float, float]]] = []
        for index in range(0, len(outputs), 3):
            reg = self._to_chw(outputs[index], expected_channels=None)
            cls = self._to_chw(outputs[index + 1], expected_channels=len(self.class_names))
            obj = self._to_chw(outputs[index + 2], expected_channels=1)
            if reg.ndim != 3 or cls.ndim != 3 or obj.ndim != 3:
                continue
            if cls.shape[0] != len(self.class_names):
                self._warn_class_count(cls.shape[0])
                continue

            channels, grid_h, grid_w = reg.shape
            if channels % 4 != 0:
                continue
            reg_max = channels // 4 - 1
            if reg_max <= 0:
                continue

            stride_x = letterbox.output_width / float(grid_w)
            stride_y = letterbox.output_height / float(grid_h)
            distances = self._decode_dfl_distances(reg, reg_max)
            class_probs = self._ensure_probability(cls)
            object_probs = self._ensure_probability(obj)[0]

            class_ids = np.argmax(class_probs, axis=0)
            class_scores = np.max(class_probs, axis=0)
            scores = class_scores * object_probs
            ys, xs = np.where(scores >= self.score_threshold)
            for y, x in zip(ys.tolist(), xs.tolist()):
                score = float(scores[y, x])
                class_id = int(class_ids[y, x])
                anchor_x = (float(x) + 0.5) * stride_x
                anchor_y = (float(y) + 0.5) * stride_y
                left = float(distances[0, y, x]) * stride_x
                top = float(distances[1, y, x]) * stride_y
                right = float(distances[2, y, x]) * stride_x
                bottom = float(distances[3, y, x]) * stride_y
                candidates.append(
                    (
                        score,
                        class_id,
                        (
                            anchor_x - left,
                            anchor_y - top,
                            anchor_x + right,
                            anchor_y + bottom,
                        ),
                    )
                )

        return self._build_detections(candidates, letterbox, frame_shape)

    def _postprocess_flat_output(
        self,
        output: np.ndarray,
        letterbox: LetterboxInfo,
        frame_shape: tuple[int, int],
    ) -> list[DetectedObject]:
        output = np.asarray(output)
        output = np.squeeze(output)
        if output.ndim != 2:
            return []
        if output.shape[0] < output.shape[1] and output.shape[0] <= 128:
            output = output.transpose(1, 0)
        if output.shape[1] < 5:
            return []

        candidates: list[tuple[float, int, tuple[float, float, float, float]]] = []
        class_count = max(1, output.shape[1] - 4)
        for row in output:
            box = row[:4].astype(np.float32)
            scores = self._ensure_probability(row[4 : 4 + class_count].reshape(class_count, 1, 1)).reshape(-1)
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            if score < self.score_threshold:
                continue
            cx, cy, width, height = [float(value) for value in box]
            candidates.append(
                (
                    score,
                    class_id,
                    (
                        cx - width * 0.5,
                        cy - height * 0.5,
                        cx + width * 0.5,
                        cy + height * 0.5,
                    ),
                )
            )
        return self._build_detections(candidates, letterbox, frame_shape)

    def _build_detections(
        self,
        candidates: Sequence[tuple[float, int, tuple[float, float, float, float]]],
        letterbox: LetterboxInfo,
        frame_shape: tuple[int, int],
    ) -> list[DetectedObject]:
        if not candidates:
            return []

        boxes = np.asarray([candidate[2] for candidate in candidates], dtype=np.float32)
        scores = np.asarray([candidate[0] for candidate in candidates], dtype=np.float32)
        class_ids = np.asarray([candidate[1] for candidate in candidates], dtype=np.int32)
        keep = self._nms(boxes, scores, class_ids)

        frame_height, frame_width = frame_shape
        detections: list[DetectedObject] = []
        for idx in keep[: self.max_detections]:
            x1, y1, x2, y2 = boxes[idx]
            x1 = (x1 - letterbox.pad_x) / max(letterbox.scale, 1e-6)
            x2 = (x2 - letterbox.pad_x) / max(letterbox.scale, 1e-6)
            y1 = (y1 - letterbox.pad_y) / max(letterbox.scale, 1e-6)
            y2 = (y2 - letterbox.pad_y) / max(letterbox.scale, 1e-6)
            x1 = max(0.0, min(float(frame_width - 1), x1))
            x2 = max(0.0, min(float(frame_width - 1), x2))
            y1 = max(0.0, min(float(frame_height - 1), y1))
            y2 = max(0.0, min(float(frame_height - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[idx])
            detections.append(
                DetectedObject(
                    class_name=self._class_name(class_id),
                    confidence=float(scores[idx]),
                    bbox_frame=(
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ),
                )
            )
        return detections

    def _to_chw(self, array: np.ndarray, expected_channels: int | None) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            return array
        if expected_channels is not None:
            if array.shape[0] == expected_channels:
                return array.astype(np.float32)
            if array.shape[-1] == expected_channels:
                return np.transpose(array, (2, 0, 1)).astype(np.float32)
        if array.shape[0] % 4 == 0 and array.shape[0] >= 16:
            return array.astype(np.float32)
        if array.shape[-1] % 4 == 0 and array.shape[-1] >= 16:
            return np.transpose(array, (2, 0, 1)).astype(np.float32)
        if array.shape[0] <= array.shape[-1]:
            return array.astype(np.float32)
        return np.transpose(array, (2, 0, 1)).astype(np.float32)

    def _decode_dfl_distances(self, reg: np.ndarray, reg_max: int) -> np.ndarray:
        _, grid_h, grid_w = reg.shape
        reg = reg.reshape(4, reg_max + 1, grid_h, grid_w)
        reg = reg - np.max(reg, axis=1, keepdims=True)
        probs = np.exp(reg)
        probs /= np.sum(probs, axis=1, keepdims=True) + 1e-9
        bins = np.arange(reg_max + 1, dtype=np.float32).reshape(1, reg_max + 1, 1, 1)
        return np.sum(probs * bins, axis=1)

    def _ensure_probability(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        if values.size == 0:
            return values
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if min_value < 0.0 or max_value > 1.0:
            return 1.0 / (1.0 + np.exp(-values))
        return values

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> list[int]:
        order = np.argsort(-scores)
        keep: list[int] = []
        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break

            rest = order[1:]
            if self.class_agnostic_nms:
                comparable = np.ones(rest.shape, dtype=bool)
            else:
                comparable = class_ids[rest] == class_ids[current]

            ious = np.zeros(rest.shape, dtype=np.float32)
            if np.any(comparable):
                ious[comparable] = self._iou(boxes[current], boxes[rest[comparable]])
            order = rest[ious <= self.nms_threshold]
        return keep

    def _iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_a = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
        area_b = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        return inter / np.maximum(area_a + area_b - inter, 1e-6)

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def _warn_class_count(self, output_count: int) -> None:
        if self._warned_class_count:
            return
        print(
            "[RknnObjectDetector] Class output count "
            f"{output_count} does not match configured class_names count {len(self.class_names)}"
        )
        self._warned_class_count = True
