"""Compare C API PP-YOLOE detections with the retained Lite2 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.object.rknn_detector import RknnObjectDetector  # noqa: E402
from core.runtime.app import load_config, prepare_runtime_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate NPU2 C API PP-YOLOE output against RKNNLite2."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", default="0,30,60")
    parser.add_argument("--max-bbox-error-px", type=int, default=1)
    parser.add_argument("--max-confidence-error", type=float, default=1e-3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    indexes = [int(value) for value in args.frames.split(",") if value.strip()]
    config = prepare_runtime_config(
        load_config(Path(args.config).expanduser().resolve()),
        PROJECT_ROOT,
    )
    object_config = dict(config["rknn_object_detector"])
    reference_config = dict(object_config)
    reference_config["runtime_backend"] = "lite2"
    native_config = dict(object_config)
    native_config["runtime_backend"] = "c_api"
    native_config["core_mask"] = "NPU_CORE_2"
    native_config["pipeline_depth"] = 2

    capture = cv2.VideoCapture(str(Path(args.video).expanduser().resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    frames = []
    try:
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, bgr = capture.read()
            if not success or bgr is None:
                raise RuntimeError(f"failed to read video frame {index}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    reference = RknnObjectDetector(reference_config)
    native = RknnObjectDetector(native_config)
    rows = []
    failed = False
    try:
        for sequence, (index, frame) in enumerate(
            zip(indexes, frames),
            start=1,
        ):
            expected = sorted(
                reference.detect(frame, frame_id=sequence),
                key=lambda item: (
                    item.class_name,
                    item.bbox_frame,
                    -item.confidence,
                ),
            )
            actual = sorted(
                native.detect(frame, frame_id=sequence),
                key=lambda item: (
                    item.class_name,
                    item.bbox_frame,
                    -item.confidence,
                ),
            )
            comparisons = []
            for left, right in zip(expected, actual):
                bbox_error = max(
                    abs(a - b)
                    for a, b in zip(left.bbox_frame, right.bbox_frame)
                )
                confidence_error = abs(left.confidence - right.confidence)
                comparisons.append(
                    {
                        "class_matches": left.class_name == right.class_name,
                        "bbox_error_px": bbox_error,
                        "confidence_error": confidence_error,
                    }
                )
                if (
                    left.class_name != right.class_name
                    or bbox_error > args.max_bbox_error_px
                    or confidence_error > args.max_confidence_error
                ):
                    failed = True
            if len(expected) != len(actual):
                failed = True
            rows.append(
                {
                    "video_frame": index,
                    "reference_count": len(expected),
                    "native_count": len(actual),
                    "native_core_mask": native.last_timing.get("core_mask"),
                    "native_context_index": native.last_timing.get(
                        "context_index"
                    ),
                    "comparisons": comparisons,
                }
            )
    finally:
        reference.close()
        native.close()
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
