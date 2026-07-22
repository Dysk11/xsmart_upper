"""Benchmark the RK3588 native perception path without UI or vehicle output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.native.runtime import NativePerceptionBackend


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * ratio) - 1))]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(values) if values else 0.0,
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values, default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["camera"].update(
        mode="video",
        video_path=str(Path(args.video).expanduser().resolve()),
        loop_video=False,
    )
    backend = NativePerceptionBackend(config, project_root, want_bgr=False)
    series = {name: [] for name in ("total", "lane", "object")}
    lane_statuses: dict[str, int] = {}
    backend.open()
    try:
        for index in range(max(0, args.frames + args.warmup)):
            ok, _frame = backend.read()
            if not ok:
                break
            if index < args.warmup:
                continue
            timing = backend.last_timing
            series["total"].append(float(timing["total_ms"]))
            series["lane"].append(float(timing["lane"]["total_ms"]))
            series["object"].append(float(timing["object"]["total_ms"]))
            status = backend.last_segmentation.status
            lane_statuses[status] = lane_statuses.get(status, 0) + 1
    finally:
        backend.close()

    report = {
        "frames": len(series["total"]),
        "total": summarize(series["total"]),
        "lane": summarize(series["lane"]),
        "object": summarize(series["object"]),
        "effective_fps_from_mean_total": (
            1000.0 / statistics.mean(series["total"]) if series["total"] else 0.0
        ),
        "lane_statuses": lane_statuses,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if series["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
