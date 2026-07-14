"""Run the live application for a fixed duration and report main-loop FPS."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_module
from core.bridge import MockBridge
from utils.fps import FPSCounter as BaseFPSCounter


class ReportingFPSCounter(BaseFPSCounter):
    """Keep normal smoothing while printing exact interval throughput."""

    instances: list["ReportingFPSCounter"] = []

    def __init__(self, smooth_alpha: float = 0.2) -> None:
        super().__init__(smooth_alpha)
        self.started_at = time.perf_counter()
        self.interval_started_at = self.started_at
        self.frames = 0
        self.interval_frames = 0
        self.last_report_at = self.started_at
        self.instances.append(self)

    def update(self) -> float:
        value = super().update()
        now = time.perf_counter()
        self.frames += 1
        self.interval_frames += 1
        if now - self.last_report_at >= 1.0:
            elapsed = now - self.interval_started_at
            print(
                f"LIVE_FPS interval={self.interval_frames / elapsed:.3f} "
                f"smoothed={value:.3f} frames={self.frames}",
                flush=True,
            )
            self.interval_started_at = now
            self.interval_frames = 0
            self.last_report_at = now
        return value

    def print_summary(self) -> None:
        elapsed = time.perf_counter() - self.started_at
        print(
            f"LIVE_SUMMARY elapsed_sec={elapsed:.3f} frames={self.frames} "
            f"fps={self.frames / max(elapsed, 1e-9):.3f}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--lane-max-age", type=int)
    parser.add_argument("--object-stride", type=int)
    parser.add_argument("--ui-stride", type=int)
    parser.add_argument("--verbose-mock", action="store_true")
    args = parser.parse_args()

    main_module.FPSCounter = ReportingFPSCounter
    original_load_config = main_module.load_config

    def load_benchmark_config(path: Path) -> dict:
        config = original_load_config(path)
        if args.lane_max_age is not None:
            config["rknn_lane_segmenter"]["max_result_age_frames"] = args.lane_max_age
        if args.object_stride is not None:
            config["rknn_object_detector"]["inference_stride"] = args.object_stride
        if args.ui_stride is not None:
            config["visualizer"]["frame_stride"] = args.ui_stride
        return config

    main_module.load_config = load_benchmark_config
    if not args.verbose_mock:
        MockBridge.send = lambda self, payload: None
    sys.argv = [str(PROJECT_ROOT / "main.py"), "--config", args.config]
    if not args.gui:
        sys.argv.append("--no-gui")

    timer = threading.Timer(args.duration, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.daemon = True
    timer.start()
    try:
        return main_module.main()
    finally:
        timer.cancel()
        for counter in ReportingFPSCounter.instances:
            counter.print_summary()


if __name__ == "__main__":
    raise SystemExit(main())
