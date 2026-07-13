"""Export the exact square-padded 480x480 image passed to PPOCR."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ocr import OcrRecognizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--frame-id", type=int, required=True, help="One-based runtime frame id")
    parser.add_argument("--bbox", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-output", help="Optional untouched source-frame output path")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame_id - 1))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {args.frame_id}: {args.video}")
    if args.source_output:
        source_output = Path(args.source_output)
        source_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(source_output), frame):
            raise RuntimeError(f"Cannot write source frame: {source_output}")

    x1, y1, x2, y2 = args.bbox
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError(f"Empty crop: {args.bbox}")
    prepared = OcrRecognizer({"input_width": 480, "input_height": 480})._prepare_frame(crop)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), prepared):
        raise RuntimeError(f"Cannot write: {output}")
    print(f"source={frame.shape} crop={crop.shape} ocr_input={prepared.shape} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
