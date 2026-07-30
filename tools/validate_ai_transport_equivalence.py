"""Validate copy and lease AI transports on identical fixed-video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.object.rknn_detector import RknnObjectDetector  # noqa: E402
from core.ocr.road_sign import select_road_sign_crop  # noqa: E402
from core.runtime.app import (  # noqa: E402
    AI_FRAME_TRANSPORT_COPY,
    AI_FRAME_TRANSPORT_LEASE,
    SharedArrayPool,
    _share_ai_frames,
    _take_shared_ai_frames,
    load_config,
    prepare_runtime_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int, default=900)
    parser.add_argument("--frame-step", type=int, default=30)
    parser.add_argument("--require-road-sign-crop", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def detection_signature(detections: list[object]) -> list[tuple[object, ...]]:
    return [
        (
            detection.class_name,
            detection.confidence,
            detection.bbox_frame,
            detection.bbox_roi,
        )
        for detection in detections
    ]


def main() -> int:
    args = parse_args()
    if args.frame_step <= 0 or args.stop_frame <= args.start_frame:
        raise ValueError("frame range must be non-empty with a positive step")

    config = prepare_runtime_config(
        load_config(Path(args.config).expanduser().resolve()),
        PROJECT_ROOT,
    )
    object_config = dict(config["rknn_object_detector"])
    object_config["runtime_backend"] = "c_api"
    object_config["core_mask"] = "NPU_CORE_2"
    object_config["pipeline_depth"] = 2
    ocr_config = dict(config.get("ocr", {}))
    class_names = {
        str(name).casefold()
        for name in ocr_config.get("class_names", ["road_sign"])
    }

    rgb_ack: queue.Queue[int] = queue.Queue()
    bgr_ack: queue.Queue[int] = queue.Queue()
    rgb_pool = SharedArrayPool("ai_rgb_frame", rgb_ack)
    bgr_pool = SharedArrayPool("ai_bgr_frame", bgr_ack)
    ack_queues = {
        "ai_rgb_frame": rgb_ack,
        "ai_bgr_frame": bgr_ack,
    }
    detector = RknnObjectDetector(object_config)
    capture = cv2.VideoCapture(str(Path(args.video).expanduser().resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    crop_count = 0
    detector.open()
    try:
        for sequence, video_frame in enumerate(
            range(args.start_frame, args.stop_frame, args.frame_step),
            start=1,
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
            success, original_bgr = capture.read()
            if not success or original_bgr is None:
                break
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

            copy_packet = _share_ai_frames(
                original_rgb,
                original_bgr,
                sequence,
                video_frame,
                rgb_pool,
                bgr_pool,
                AI_FRAME_TRANSPORT_COPY,
            )
            if copy_packet is None:
                raise RuntimeError("copy transport exhausted its shared slots")
            copied = _take_shared_ai_frames(
                copy_packet,
                ack_queues,
                expected_frame_id=sequence,
            )
            assert copied.bgr is not None
            copy_detections = detector.detect(
                copied.rgb,
                frame_id=sequence * 2,
            )
            copy_crop = select_road_sign_crop(
                copied.bgr,
                copy_detections,
                class_names,
                float(ocr_config.get("bbox_min_confidence", 0.50)),
                int(ocr_config.get("bbox_min_width_px", 96)),
                int(ocr_config.get("bbox_min_height_px", 48)),
                float(ocr_config.get("bbox_padding_ratio", 0.10)),
            )

            lease_packet = _share_ai_frames(
                original_rgb,
                original_bgr,
                sequence,
                video_frame,
                rgb_pool,
                bgr_pool,
                AI_FRAME_TRANSPORT_LEASE,
            )
            if lease_packet is None:
                raise RuntimeError("lease transport exhausted its shared slots")
            leased = _take_shared_ai_frames(
                lease_packet,
                ack_queues,
                expected_frame_id=sequence,
            )
            lease_bgr = cv2.cvtColor(leased.rgb, cv2.COLOR_RGB2BGR)
            lease_detections = detector.detect(
                leased.rgb,
                frame_id=sequence * 2 + 1,
            )
            lease_crop = select_road_sign_crop(
                lease_bgr,
                lease_detections,
                class_names,
                float(ocr_config.get("bbox_min_confidence", 0.50)),
                int(ocr_config.get("bbox_min_width_px", 96)),
                int(ocr_config.get("bbox_min_height_px", 48)),
                float(ocr_config.get("bbox_padding_ratio", 0.10)),
            )

            rgb_equal = np.array_equal(copied.rgb, leased.rgb)
            bgr_equal = np.array_equal(original_bgr, lease_bgr)
            detections_equal = (
                detection_signature(copy_detections)
                == detection_signature(lease_detections)
            )
            crops_equal = (
                copy_crop is None
                and lease_crop is None
            ) or (
                copy_crop is not None
                and lease_crop is not None
                and copy_crop.bbox == lease_crop.bbox
                and np.array_equal(copy_crop.image, lease_crop.image)
            )
            if copy_crop is not None:
                crop_count += 1
            row = {
                "video_frame": video_frame,
                "detection_count": len(copy_detections),
                "rgb_equal": rgb_equal,
                "bgr_equal": bgr_equal,
                "detections_equal": detections_equal,
                "ocr_crop_equal": crops_equal,
                "has_ocr_crop": copy_crop is not None,
            }
            rows.append(row)
            if not all((rgb_equal, bgr_equal, detections_equal, crops_equal)):
                failures.append(row)
            copied.close()
            leased.close()
    finally:
        capture.release()
        detector.close()
        rgb_pool.close()
        bgr_pool.close()

    accepted = not failures and (
        crop_count > 0 or not args.require_road_sign_crop
    )
    report = {
        "frames_compared": len(rows),
        "road_sign_crop_frames": crop_count,
        "failures": failures,
        "accepted": accepted,
        "rows": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
