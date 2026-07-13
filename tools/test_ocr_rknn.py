"""Board-side smoke test for the PPOCR detector and recognizer RKNN models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--image", help="Optional image used by both OCR models")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = _resolve(project_root, args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["extensions"]["ocr"]

    system_python = _resolve(project_root, config["system_python_dir"])
    sys.path.insert(0, str(system_python))
    from ppocr_system import TextSystem

    model_args = SimpleNamespace(
        det_model_path=str(_resolve(project_root, config["det_model_path"])),
        rec_model_path=str(_resolve(project_root, config["rec_model_path"])),
        target=config.get("target", "rk3588"),
        device_id=config.get("device_id"),
        core_mask=config.get("core_mask", "NPU_CORE_1"),
    )
    image = cv2.imread(args.image) if args.image else None
    if image is None:
        image = np.full((480, 480, 3), 114, dtype=np.uint8)

    system = None
    try:
        load_started = time.perf_counter()
        system = TextSystem(model_args)
        load_ms = (time.perf_counter() - load_started) * 1000.0

        det_started = time.perf_counter()
        boxes = system.text_detector.run(cv2.resize(image, (480, 480)))
        det_ms = (time.perf_counter() - det_started) * 1000.0

        rec_started = time.perf_counter()
        recognition = system.text_recognizer.run([image])
        rec_ms = (time.perf_counter() - rec_started) * 1000.0

        system_started = time.perf_counter()
        system_boxes, system_recognition = system.run(cv2.resize(image, (480, 480)))
        system_ms = (time.perf_counter() - system_started) * 1000.0
        print(json.dumps({
            "core_mask": model_args.core_mask,
            "load_ms": round(load_ms, 3),
            "det_ms": round(det_ms, 3),
            "det_boxes": 0 if boxes is None else len(boxes),
            "rec_ms": round(rec_ms, 3),
            "rec_result": recognition,
            "system_ms": round(system_ms, 3),
            "system_boxes": 0 if system_boxes is None else len(system_boxes),
            "system_result": system_recognition,
        }, ensure_ascii=False, default=_json_default))
        return 0
    finally:
        if system is not None:
            for component in (system.text_detector, system.text_recognizer):
                component.model.release()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
