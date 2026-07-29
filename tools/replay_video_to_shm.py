"""Replay a fixed video into the shm_ar_video protocol at a controlled rate."""

from __future__ import annotations

import argparse
from multiprocessing import resource_tracker, shared_memory
import struct
import time

import cv2
import numpy as np


HEADER = struct.Struct("@QII")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--name", default="shm_ar_video")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--start-frame-id", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    ok, frame_bgr = capture.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"video has no readable frames: {args.video}")
    height, width = frame_bgr.shape[:2]
    frame_bytes = width * height * 3
    required_size = HEADER.size + frame_bytes
    owns_shm = False
    try:
        try:
            shm = shared_memory.SharedMemory(name=args.name, create=False)
            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except FileNotFoundError:
            shm = shared_memory.SharedMemory(
                name=args.name,
                create=True,
                size=required_size,
            )
            owns_shm = True
        if shm.size < required_size:
            raise RuntimeError(
                f"shared memory {args.name} is {shm.size} bytes; "
                f"{required_size} bytes required"
            )
        frame_id = max(1, int(args.start_frame_id))
        started = time.perf_counter()
        next_deadline = started
        frame_period = 1.0 / args.fps
        while args.duration_sec <= 0 or time.perf_counter() - started < args.duration_sec:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            contiguous = np.ascontiguousarray(frame_rgb)
            # Invalidate the header while pixels are copied, then publish the
            # monotonically increasing frame id as the final write.
            shm.buf[: HEADER.size] = HEADER.pack(0, width, height)
            shm.buf[HEADER.size : required_size] = contiguous.reshape(-1)
            shm.buf[: HEADER.size] = HEADER.pack(frame_id, width, height)
            frame_id += 1

            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError("failed to loop benchmark video")
            next_deadline += frame_period
            delay = next_deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -frame_period:
                next_deadline = time.perf_counter()
        return 0
    finally:
        capture.release()
        if "shm" in locals():
            shm.close()
            if owns_shm:
                shm.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
