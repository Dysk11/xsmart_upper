"""Scan a video through the deployed detector and road-sign OCR session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rknn_object_detector import RknnObjectDetector
from core.road_sign_ocr import RoadSignOcrSession


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=3000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with _resolve(root, args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    detector_config = dict(config["rknn_object_detector"])
    detector_config["model_path"] = str(_resolve(root, detector_config["model_path"]))
    ocr_config = dict(config["extensions"]["ocr"])
    for key in ("det_model_path", "rec_model_path", "system_python_dir", "output_dir"):
        ocr_config[key] = str(_resolve(root, ocr_config[key]))

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    if args.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    detector = RknnObjectDetector(detector_config)
    session = RoadSignOcrSession(ocr_config, project_root=root)
    qualifying_frames = 0
    try:
        frame_id = max(0, args.start_frame)
        while frame_id < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % max(1, args.stride):
                continue
            detections = detector.detect(frame)
            road_signs = [item for item in detections if item.class_name.casefold() == "road_sign"]
            if not road_signs:
                continue
            qualifying_frames += 1
            result = session.update(frame, frame_id, detections)
            raw_result = session.recognizer.last_result
            print(json.dumps({
                "frame_id": frame_id,
                "road_signs": [
                    {"confidence": item.confidence, "bbox": item.bbox_frame}
                    for item in road_signs
                ],
                "ocr": None if result is None else {
                    "event_id": result.event_id,
                    "text": result.text,
                    "confidence": result.confidence,
                    "error": result.error,
                },
                "raw_ocr": None if raw_result.frame_id != frame_id else {
                    "text": raw_result.text,
                    "confidence": raw_result.confidence,
                    "inference_ms": raw_result.inference_ms,
                    "error": raw_result.error,
                },
            }, ensure_ascii=False))
            if result is not None and result.event_id > 0:
                break
    finally:
        session.close()
        detector.close()
        capture.release()
    print(json.dumps({"frames_scanned": frame_id, "road_sign_frames": qualifying_frames}))
    return 0


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    raise SystemExit(main())
