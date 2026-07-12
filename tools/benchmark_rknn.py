"""Benchmark RKNN core allocation and pipelining on an RK3588 board."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

PROJECT_ROOT = Path(os.environ.get("XSMART_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from core.rknn_lane_segmenter import RknnLaneSegmenter
from core.rknn_object_detector import RknnObjectDetector


NPU_SYSFS = Path("/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu")
DEFAULT_STRATEGIES = [
    {"name": "baseline_lane02_obj1", "lane_masks": ["NPU_CORE_0", "NPU_CORE_2"], "object_mask": "NPU_CORE_1", "object_stride": 1},
    {"name": "lane0_obj1", "lane_masks": ["NPU_CORE_0"], "object_mask": "NPU_CORE_1", "object_stride": 1},
    {"name": "lane0_obj2", "lane_masks": ["NPU_CORE_0"], "object_mask": "NPU_CORE_2", "object_stride": 1},
    {"name": "lane1_obj0", "lane_masks": ["NPU_CORE_1"], "object_mask": "NPU_CORE_0", "object_stride": 1},
    {"name": "lane2_obj0", "lane_masks": ["NPU_CORE_2"], "object_mask": "NPU_CORE_0", "object_stride": 1},
    {"name": "lane01_obj2", "lane_masks": ["NPU_CORE_0_1"], "object_mask": "NPU_CORE_2", "object_stride": 1},
    {"name": "lane2_obj01", "lane_masks": ["NPU_CORE_2"], "object_mask": "NPU_CORE_0_1", "object_stride": 1},
    {"name": "lane012_obj2_overlap", "lane_masks": ["NPU_CORE_0_1_2"], "object_mask": "NPU_CORE_2", "object_stride": 1},
    {"name": "lane012_obj1_stride2", "lane_masks": ["NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2"], "object_mask": "NPU_CORE_1", "object_stride": 2},
    {"name": "lane02_obj1_stride2", "lane_masks": ["NPU_CORE_0", "NPU_CORE_2"], "object_mask": "NPU_CORE_1", "object_stride": 2},
    {"name": "lane01_obj2_stride2", "lane_masks": ["NPU_CORE_0", "NPU_CORE_1"], "object_mask": "NPU_CORE_2", "object_stride": 2},
    {"name": "lane_auto_obj1_stride2", "lane_masks": ["NPU_CORE_AUTO"], "object_mask": "NPU_CORE_1", "object_stride": 2},
]
SINGLE_MODEL_STRATEGIES = [
    {"name": "lane_only_core0", "lane_masks": ["NPU_CORE_0"], "object_mask": None, "object_stride": 1},
    {"name": "lane_only_core01", "lane_masks": ["NPU_CORE_0_1"], "object_mask": None, "object_stride": 1},
    {"name": "lane_only_core012", "lane_masks": ["NPU_CORE_0_1_2"], "object_mask": None, "object_stride": 1},
    {"name": "object_only_core0", "lane_masks": [], "object_mask": "NPU_CORE_0", "object_stride": 1},
    {"name": "object_only_core01", "lane_masks": [], "object_mask": "NPU_CORE_0_1", "object_stride": 1},
    {"name": "object_only_core012", "lane_masks": [], "object_mask": "NPU_CORE_0_1_2", "object_stride": 1},
]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


class NpuSampler:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.loads: list[float] = []
        self.freqs: list[float] = []
        self.temps: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _number(path: Path) -> float | None:
        try:
            text = path.read_text(encoding="utf-8").strip()
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
            return float(match.group(0)) if match else None
        except (OSError, ValueError, IndexError):
            return None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="npu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            "npu_load_avg": float(np.mean(self.loads)) if self.loads else 0.0,
            "npu_load_p95": percentile(self.loads, 95),
            "npu_freq_avg_hz": float(np.mean(self.freqs)) if self.freqs else 0.0,
            "npu_freq_min_hz": min(self.freqs, default=0.0),
            "temperature_avg_c": float(np.mean(self.temps)) if self.temps else 0.0,
            "temperature_max_c": max(self.temps, default=0.0),
        }

    def _run(self) -> None:
        thermal_paths = list(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
        while not self._stop.is_set():
            load = self._number(NPU_SYSFS / "load")
            freq = self._number(NPU_SYSFS / "cur_freq")
            temps = [value for path in thermal_paths if (value := self._number(path)) is not None]
            if load is not None:
                self.loads.append(load)
            if freq is not None:
                self.freqs.append(freq)
            if temps:
                self.temps.append(max(temps) / 1000.0)
            self._stop.wait(self.interval)


def lane_config(model: Path, core_mask: str) -> dict[str, Any]:
    return {
        "enable": True,
        "model_path": str(model),
        "input_size": [640, 480],
        "score_threshold": 0.25,
        "nms_threshold": 0.45,
        "mask_threshold": 0.5,
        "max_instances": 1,
        "runtime_backend": "lite2",
        "core_mask": core_mask,
    }


def object_config(model: Path, core_mask: str) -> dict[str, Any]:
    return {
        "enable": True,
        "model_path": str(model),
        "input_size": [640, 480],
        "input_layout": "nhwc",
        "input_color": "rgb",
        "input_dtype": "uint8",
        "score_threshold": 0.20,
        "nms_threshold": 0.45,
        "max_detections": 30,
        "class_names": ["car", "coin", "Go", "human", "road_sign", "speed_limit", "Stop"],
        "runtime_backend": "lite2",
        "core_mask": core_mask,
    }


def read_next(capture: cv2.VideoCapture, video: Path) -> tuple[cv2.VideoCapture, np.ndarray]:
    ok, frame = capture.read()
    if ok and frame is not None:
        return capture, frame
    capture.release()
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read benchmark video: {video}")
    return capture, frame


def summarize_timings(prefix: str, samples: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
        values = [float(sample.get(key, 0.0)) for sample in samples]
        result[f"{prefix}_{key}_avg"] = float(np.mean(values)) if values else 0.0
        result[f"{prefix}_{key}_p95"] = percentile(values, 95)
    return result


def run_strategy(
    strategy: dict[str, Any],
    repeat: int,
    video: Path,
    lane_model: Path,
    object_model: Path,
    warmup_frames: int,
    duration_seconds: float,
    max_lane_age: int,
) -> dict[str, Any]:
    lane_masks = list(strategy["lane_masks"])
    object_stride = max(1, int(strategy.get("object_stride", 1)))
    lane_workers = [RknnLaneSegmenter(lane_config(lane_model, mask)) for mask in lane_masks]
    object_mask = strategy.get("object_mask")
    object_worker = (
        RknnObjectDetector(object_config(object_model, str(object_mask)))
        if object_mask is not None else None
    )
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(lane_workers) + int(object_worker is not None)))
    lane_futures: list[concurrent.futures.Future[Any] | None] = [None] * len(lane_workers)
    object_future: concurrent.futures.Future[Any] | None = None
    capture = cv2.VideoCapture(str(video))
    frame_id = 0
    measured_frames = 0
    dropped_lane_submissions = 0
    dropped_object_submissions = 0
    stale_lane_results = 0
    last_lane_frame = -1
    latest_lane_mask_pixels = 0
    lane_timings: list[dict[str, float]] = []
    object_timings: list[dict[str, float]] = []
    frame_intervals: list[float] = []
    lane_latencies: list[float] = []
    lane_ages: list[float] = []
    object_latencies: list[float] = []
    detection_counts: list[float] = []
    sampler = NpuSampler()
    measurement_started = 0.0
    previous_frame_time: float | None = None

    def lane_call(worker: RknnLaneSegmenter, fid: int, captured: float, image: np.ndarray) -> tuple[Any, ...]:
        result = worker.segment(image)
        return fid, captured, time.perf_counter(), dict(getattr(worker, "last_timing", {})), result

    def object_call(fid: int, captured: float, image: np.ndarray) -> tuple[Any, ...]:
        assert object_worker is not None
        result = object_worker.detect(image)
        return fid, captured, time.perf_counter(), dict(getattr(object_worker, "last_timing", {})), result

    try:
        while True:
            capture, frame = read_next(capture, video)
            frame_id += 1
            now = time.perf_counter()

            for index, future in enumerate(lane_futures):
                if future is None or not future.done():
                    continue
                fid, captured, completed, timing, result = future.result()
                lane_futures[index] = None
                if fid > last_lane_frame:
                    last_lane_frame = int(fid)
                    latest_lane_mask_pixels = int(np.count_nonzero(result.mask))
                    if measurement_started:
                        lane_timings.append(timing)
                        lane_latencies.append((completed - captured) * 1000.0)
                else:
                    stale_lane_results += 1

            if object_future is not None and object_future.done():
                _fid, captured, completed, timing, detections = object_future.result()
                object_future = None
                if measurement_started:
                    object_timings.append(timing)
                    object_latencies.append((completed - captured) * 1000.0)
                    detection_counts.append(float(len(detections)))

            free_lane = next((i for i, future in enumerate(lane_futures) if future is None), None)
            if free_lane is not None:
                lane_futures[free_lane] = executor.submit(
                    lane_call, lane_workers[free_lane], frame_id, now, frame.copy()
                )
            elif lane_workers:
                dropped_lane_submissions += 1

            if object_worker is not None and frame_id % object_stride == 0:
                if object_future is None:
                    object_future = executor.submit(object_call, frame_id, now, frame.copy())
                else:
                    dropped_object_submissions += 1

            if lane_workers and (last_lane_frame < 0 or frame_id - last_lane_frame > max_lane_age):
                pending = [future for future in lane_futures if future is not None]
                if pending:
                    concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)

            if frame_id == warmup_frames:
                measurement_started = time.perf_counter()
                previous_frame_time = measurement_started
                sampler.start()
            if measurement_started:
                measured_frames += 1
                current = time.perf_counter()
                if previous_frame_time is not None:
                    frame_intervals.append((current - previous_frame_time) * 1000.0)
                previous_frame_time = current
                if last_lane_frame >= 0:
                    lane_ages.append(float(frame_id - last_lane_frame))
                if current - measurement_started >= duration_seconds:
                    break
    finally:
        npu_metrics = sampler.stop() if measurement_started else {}
        capture.release()
        executor.shutdown(wait=True, cancel_futures=False)
        for worker in lane_workers:
            worker.close()
        if object_worker is not None:
            object_worker.close()

    elapsed = max(time.perf_counter() - measurement_started, 1e-9)
    result: dict[str, Any] = {
        "strategy": str(strategy["name"]),
        "repeat": repeat,
        "lane_masks": lane_masks,
        "object_mask": object_mask,
        "object_stride": object_stride,
        "frames": measured_frames,
        "elapsed_seconds": elapsed,
        "fps": measured_frames / elapsed,
        "frame_interval_p50_ms": percentile(frame_intervals, 50),
        "frame_interval_p95_ms": percentile(frame_intervals, 95),
        "frame_interval_p99_ms": percentile(frame_intervals, 99),
        "lane_latency_p50_ms": percentile(lane_latencies, 50),
        "lane_latency_p95_ms": percentile(lane_latencies, 95),
        "lane_latency_p99_ms": percentile(lane_latencies, 99),
        "lane_age_p95_frames": percentile(lane_ages, 95),
        "object_latency_p95_ms": percentile(object_latencies, 95),
        "dropped_lane_submissions": dropped_lane_submissions,
        "dropped_object_submissions": dropped_object_submissions,
        "stale_lane_results": stale_lane_results,
        "last_lane_mask_pixels": latest_lane_mask_pixels,
        "detection_count_avg": float(np.mean(detection_counts)) if detection_counts else 0.0,
        "lane_completed_fps": len(lane_timings) / elapsed,
        "object_completed_fps": len(object_timings) / elapsed,
    }
    result.update(summarize_timings("lane", lane_timings))
    result.update(summarize_timings("object", object_timings))
    result.update(npu_metrics)
    return result


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for strategy in sorted({str(row["strategy"]) for row in rows}):
        selected = [row for row in rows if row["strategy"] == strategy]
        summary = dict(selected[0])
        summary["repeat"] = "median"
        for key, value in list(summary.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                summary[key] = float(np.median([float(row[key]) for row in selected]))
        summaries.append(summary)
    return sorted(summaries, key=lambda row: float(row["fps"]), reverse=True)


def write_results(output: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": rows, "summary": summaries}, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fieldnames = sorted({key for row in rows + summaries for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows + summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--lane-model", type=Path, default=PROJECT_ROOT / "models/yolov5n_seg_track_480x640_int8_rk3588.rknn")
    parser.add_argument("--object-model", type=Path, default=PROJECT_ROOT / "models/rknn_7classes.rknn")
    parser.add_argument("--strategies", type=Path, help="JSON list overriding the built-in strategy matrix")
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--max-lane-age", type=int, default=2)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/benchmark/rknn_benchmark.json")
    parser.add_argument("--only", action="append", help="Run only the named strategy; may be repeated")
    parser.add_argument("--scout", action="store_true", help="Use 20 warmup frames, 8 seconds and one repeat")
    parser.add_argument("--single-models", action="store_true", help="Benchmark isolated lane/object core scaling")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategies = SINGLE_MODEL_STRATEGIES if args.single_models else DEFAULT_STRATEGIES
    if args.strategies:
        strategies = json.loads(args.strategies.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only)
        strategies = [strategy for strategy in strategies if strategy["name"] in wanted]
    if not strategies:
        raise SystemExit("No benchmark strategies selected")
    warmup = 20 if args.scout else max(0, args.warmup_frames)
    duration = 8.0 if args.scout else max(0.1, args.duration_seconds)
    repeats = 1 if args.scout else max(1, args.repeat_count)
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        for repeat in range(1, repeats + 1):
            print(f"BENCHMARK_START strategy={strategy['name']} repeat={repeat}", flush=True)
            row = run_strategy(
                strategy, repeat, args.video, args.lane_model, args.object_model,
                warmup, duration, max(0, args.max_lane_age),
            )
            rows.append(row)
            summaries = aggregate(rows)
            write_results(args.output, rows, summaries)
            print(
                f"BENCHMARK_DONE strategy={strategy['name']} fps={row['fps']:.3f} "
                f"lane_p95_ms={row['lane_latency_p95_ms']:.3f} npu={row.get('npu_load_avg', 0):.1f}",
                flush=True,
            )
    summaries = aggregate(rows)
    baseline = next((row for row in summaries if row["strategy"] == "baseline_lane02_obj1"), None)
    if baseline:
        for row in summaries:
            row["fps_gain_percent_vs_baseline"] = (
                (float(row["fps"]) / max(float(baseline["fps"]), 1e-9) - 1.0) * 100.0
            )
        write_results(args.output, rows, summaries)
    print(json.dumps(summaries[:5], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
