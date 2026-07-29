"""Compare native RKNN C API lane output with the existing Lite2 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lane.capi_backend import CapiLaneBackend  # noqa: E402
from core.lane.rknn_segmenter import RknnLaneSegmenter  # noqa: E402
from core.runtime.app import load_config, prepare_runtime_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate C API YOLOv5-seg output against RKNNLite2."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--frames",
        default="0,30,60",
        help="Comma-separated zero-based video frame indexes.",
    )
    parser.add_argument("--minimum-mask-iou", type=float, default=0.99)
    parser.add_argument(
        "--output-mode",
        choices=("float", "native"),
        help="Override the C API output path.",
    )
    return parser


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = np.asarray(left) > 0
    right_mask = np.asarray(right) > 0
    union = np.count_nonzero(left_mask | right_mask)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left_mask & right_mask)) / float(union)


def read_frames(path: Path, indexes: list[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    try:
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, bgr = capture.read()
            if not success or bgr is None:
                raise RuntimeError(f"failed to read video frame {index}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames


def main() -> int:
    args = build_parser().parse_args()
    indexes = [int(value) for value in args.frames.split(",") if value.strip()]
    config = prepare_runtime_config(
        load_config(Path(args.config).expanduser().resolve()),
        PROJECT_ROOT,
    )
    lane_config = dict(config["rknn_lane_segmenter"])
    if args.output_mode:
        lane_config["output_mode"] = args.output_mode
    frames = read_frames(Path(args.video).expanduser().resolve(), indexes)

    reference_config = dict(lane_config)
    reference_config["runtime_backend"] = "lite2"
    reference_config["core_mask"] = "NPU_CORE_0"
    reference = RknnLaneSegmenter(reference_config)
    try:
        expected = [reference.segment(frame) for frame in frames]
    finally:
        reference.close()

    native = CapiLaneBackend(lane_config, PROJECT_ROOT)
    actual = []
    try:
        native.start()
        for sequence, frame in enumerate(frames, start=1):
            captured_at = time.perf_counter()
            native.publish(
                frame,
                frame_id=sequence,
                source_frame_id=indexes[sequence - 1] + 1,
                captured_at=captured_at,
            )
            deadline = time.monotonic() + 2.0
            result = None
            while time.monotonic() < deadline:
                result = native.read_latest()
                if result is not None and result.frame_id == sequence:
                    break
                time.sleep(0.0005)
            if result is None or result.frame_id != sequence:
                raise TimeoutError(f"native result timed out for frame {sequence}")
            actual.append(result.result)
    finally:
        native.close()

    rows = []
    failed = False
    for index, expected_result, actual_result in zip(indexes, expected, actual):
        iou = mask_iou(expected_result.mask, actual_result.mask)
        bbox_errors = []
        for expected_instance, actual_instance in zip(
            expected_result.instances,
            actual_result.instances,
        ):
            bbox_errors.append(
                max(
                    abs(left - right)
                    for left, right in zip(
                        expected_instance.bbox_frame,
                        actual_instance.bbox_frame,
                    )
                )
            )
        row = {
            "video_frame": index,
            "mask_iou": iou,
            "reference_status": expected_result.status,
            "native_status": actual_result.status,
            "reference_instances": len(expected_result.instances),
            "native_instances": len(actual_result.instances),
            "max_bbox_error_px": max(bbox_errors, default=0),
            "reference_boxes": [
                {
                    "bbox": instance.bbox_frame,
                    "confidence": instance.confidence,
                }
                for instance in expected_result.instances
            ],
            "native_boxes": [
                {
                    "bbox": instance.bbox_frame,
                    "confidence": instance.confidence,
                }
                for instance in actual_result.instances
            ],
        }
        rows.append(row)
        if (
            iou < args.minimum_mask_iou
            or expected_result.status != actual_result.status
            or len(expected_result.instances) != len(actual_result.instances)
            or row["max_bbox_error_px"] > 1
        ):
            failed = True
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
