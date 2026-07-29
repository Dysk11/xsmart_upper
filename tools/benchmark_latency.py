"""Run a fixed-duration camera-to-command latency benchmark."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime.app import (  # noqa: E402
    UpperMachineApp,
    load_config,
    prepare_runtime_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure source/NPU/lane/bridge latency on the active runtime."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Base YAML configuration.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("shared_memory", "video"),
        help="Input source under test.",
    )
    parser.add_argument(
        "--bridge",
        default="mock",
        choices=("mock", "serial"),
        help="Command bridge under test.",
    )
    parser.add_argument("--video", help="Video path for --mode video.")
    parser.add_argument(
        "--lane-backend",
        choices=("c_api", "lite2"),
        help="Override the lane inference backend for an A/B run.",
    )
    parser.add_argument(
        "--preprocess",
        choices=("auto", "direct", "rga"),
        help="Override native preprocessing for a C API run.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("float", "native"),
        help="Select preallocated float outputs or native zero-copy outputs.",
    )
    parser.add_argument(
        "--fresh-result-wait-ms",
        type=float,
        help="Override the realtime C API eventfd wait budget.",
    )
    parser.add_argument("--warmup-sec", type=float, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--system-sample-interval-sec", type=float, default=None)
    parser.add_argument("--output-dir", help="Benchmark report directory.")
    parser.add_argument(
        "--serial-safety-confirmed",
        action="store_true",
        help="Required for serial runs after the vehicle is physically made safe.",
    )
    return parser


def build_benchmark_config(args: argparse.Namespace) -> dict:
    config_path = Path(args.config).expanduser().resolve()
    config = copy.deepcopy(load_config(config_path))
    camera_config = config.setdefault("camera", {})
    camera_config["mode"] = args.mode
    if args.video:
        camera_config["video_path"] = args.video
    if args.mode == "video" and not camera_config.get("video_path"):
        raise ValueError("--mode video requires --video or camera.video_path")
    if args.mode == "video":
        # Keep an unthrottled finite file running for the full sampling window.
        camera_config["loop_video"] = True

    if args.bridge == "serial" and not args.serial_safety_confirmed:
        raise RuntimeError(
            "Serial benchmark blocked: physically disable vehicle motion, then pass "
            "--serial-safety-confirmed."
        )
    config.setdefault("bridge", {})["type"] = args.bridge
    lane_config = config.setdefault("rknn_lane_segmenter", {})
    if getattr(args, "lane_backend", None):
        lane_config["runtime_backend"] = args.lane_backend
    if getattr(args, "preprocess", None):
        lane_config["preprocess_backend"] = args.preprocess
    if getattr(args, "output_mode", None):
        lane_config["output_mode"] = args.output_mode
    if getattr(args, "fresh_result_wait_ms", None) is not None:
        lane_config["fresh_result_wait_ms"] = max(
            0.0,
            float(args.fresh_result_wait_ms),
        )

    visualizer = config.setdefault("visualizer", {})
    visualizer["show_window"] = False
    visualizer["show_debug_window"] = False
    visualizer["save_video"] = False
    visualizer["save_screenshot"] = False
    config.setdefault("logger", {})["enable"] = False

    benchmark = config.setdefault("app", {}).setdefault("latency_benchmark", {})
    benchmark["enable"] = True
    benchmark["auto_stop"] = True
    if args.warmup_sec is not None:
        benchmark["warmup_sec"] = args.warmup_sec
    if args.duration_sec is not None:
        benchmark["duration_sec"] = args.duration_sec
    if args.system_sample_interval_sec is not None:
        benchmark["system_sample_interval_sec"] = args.system_sample_interval_sec
    if args.output_dir:
        benchmark["output_dir"] = args.output_dir

    return prepare_runtime_config(config, PROJECT_ROOT)


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = build_benchmark_config(args)
        app = UpperMachineApp(config, PROJECT_ROOT)
    except Exception as error:
        print(f"Benchmark setup failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    return_code = 0
    try:
        app.run()
    except KeyboardInterrupt:
        print("Benchmark interrupted.")
        return_code = 130
    except Exception as error:
        app.latency_benchmark.add_error(f"{type(error).__name__}: {error}")
        print(f"Benchmark failed: {type(error).__name__}: {error}", file=sys.stderr)
        return_code = 1
    finally:
        app.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
