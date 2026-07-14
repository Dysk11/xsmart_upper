"""Measure the AR shared-memory producer and CameraReader throughput."""

from __future__ import annotations

import argparse
import struct
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.camera import CameraReader


HEADER = struct.Struct("@QII")


def measure_producer(name: str, duration: float) -> dict[str, float | int]:
    shm = shared_memory.SharedMemory(name=name, create=False)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except (AttributeError, KeyError):
        pass
    start = time.perf_counter()
    deadline = start + duration
    first_id: int | None = None
    last_id: int | None = None
    unique_frames = 0
    while time.perf_counter() < deadline:
        frame_id, _width, _height = HEADER.unpack(bytes(shm.buf[: HEADER.size]))
        if frame_id and frame_id != last_id:
            first_id = frame_id if first_id is None else first_id
            last_id = frame_id
            unique_frames += 1
        time.sleep(0.0005)
    elapsed = time.perf_counter() - start
    shm.close()
    return {
        "elapsed_sec": elapsed,
        "unique_frames": unique_frames,
        "observed_fps": unique_frames / elapsed,
        "frame_id_rate": (
            (last_id - first_id) / elapsed
            if first_id is not None and last_id is not None
            else 0.0
        ),
        "first_frame_id": first_id or 0,
        "last_frame_id": last_id or 0,
    }


def measure_reader(name: str, duration: float) -> dict[str, float | int]:
    reader = CameraReader(
        {
            "mode": "shared_memory",
            "shared_memory_name": name,
            "max_reconnect_attempts": 1,
        }
    )
    reader.open()
    start = time.perf_counter()
    deadline = start + duration
    frames = 0
    first_id = 0
    last_id = 0
    while time.perf_counter() < deadline:
        ok, _frame = reader.read()
        if not ok:
            continue
        frames += 1
        last_id = reader._shared_memory_last_frame_id
        first_id = first_id or last_id
    elapsed = time.perf_counter() - start
    reader.release()
    return {
        "elapsed_sec": elapsed,
        "frames": frames,
        "reader_fps": frames / elapsed,
        "frame_id_rate": (last_id - first_id) / elapsed if first_id else 0.0,
        "first_frame_id": first_id,
        "last_frame_id": last_id,
    }


def print_result(label: str, result: dict[str, float | int]) -> None:
    values = " ".join(
        f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in result.items()
    )
    print(f"{label} {values}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="shm_ar_video")
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    print_result("PRODUCER", measure_producer(args.name, args.duration))
    print_result("READER", measure_reader(args.name, args.duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
