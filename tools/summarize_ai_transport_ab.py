"""Summarize and enforce the three-run AI shared-frame transport A/B gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def load_runs(root: Path, transport: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for trial in range(1, 4):
        matches = sorted((root / f"{transport}_run{trial}").glob("latency_*.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {transport} run {trial} report, found {len(matches)}"
            )
        report = json.loads(matches[0].read_text(encoding="utf-8"))
        report["_path"] = str(matches[0])
        runs.append(report)
    return runs


def metric(
    report: dict[str, Any],
    name: str,
    statistic: str = "p95",
) -> float:
    return float(report["metrics"][name][statistic])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()

    copied = load_runs(args.root, "rgb_bgr_copy")
    leased = load_runs(args.root, "rgb_lease")
    copy_p95 = [metric(run, "ai_capture_to_result_ms") for run in copied]
    lease_p95 = [metric(run, "ai_capture_to_result_ms") for run in leased]
    p95_reductions = [
        (baseline - optimized) * 100.0 / baseline
        for baseline, optimized in zip(copy_p95, lease_p95)
    ]
    copy_p99 = [
        metric(run, "ai_capture_to_result_ms", "p99") for run in copied
    ]
    lease_p99 = [
        metric(run, "ai_capture_to_result_ms", "p99") for run in leased
    ]
    p99_changes = [
        (optimized - baseline) * 100.0 / baseline
        for baseline, optimized in zip(copy_p99, lease_p99)
    ]
    copy_fps = [float(run["command_fps"]) for run in copied]
    lease_fps = [float(run["command_fps"]) for run in leased]
    fps_change = (
        statistics.mean(lease_fps) - statistics.mean(copy_fps)
    ) * 100.0 / statistics.mean(copy_fps)
    copy_ai_fps = [float(run["ai_result_fps"]) for run in copied]
    lease_ai_fps = [float(run["ai_result_fps"]) for run in leased]
    ai_fps_change = (
        statistics.mean(lease_ai_fps) - statistics.mean(copy_ai_fps)
    ) * 100.0 / statistics.mean(copy_ai_fps)
    copy_age = [metric(run, "ai_result_age_frames") for run in copied]
    lease_age = [metric(run, "ai_result_age_frames") for run in leased]
    errors = [
        {"path": run["_path"], "errors": run.get("errors", [])}
        for run in [*copied, *leased]
        if run.get("errors")
    ]
    lease_bgr_writes = [
        metric(run, "ai_pool_bgr_write_ms", "max") for run in leased
    ]
    contexts = [
        {
            "mean": metric(run, "ai_context_index", "mean"),
            "max": metric(run, "ai_context_index", "max"),
        }
        for run in leased
    ]
    failure_markers = (
        "resource_tracker:",
        "leaked shared_memory",
        "buffererror:",
        "no space left on device",
        "object detector worker exited unexpectedly",
        "slot exhausted",
    )
    log_failures: list[dict[str, str]] = []
    for transport in ("rgb_bgr_copy", "rgb_lease"):
        for trial in range(1, 4):
            log_path = args.root / f"{transport}_run{trial}" / "benchmark.log"
            log_text = log_path.read_text(encoding="utf-8", errors="replace").lower()
            matched = [marker for marker in failure_markers if marker in log_text]
            if matched:
                log_failures.append(
                    {
                        "path": str(log_path),
                        "markers": ", ".join(matched),
                    }
                )
    gates = {
        "three_runs_each": len(copied) == 3 and len(leased) == 3,
        "capture_p95_reduction_at_least_3_percent": (
            statistics.mean(p95_reductions) >= 3.0
        ),
        "capture_p99_not_worse_over_2_percent": (
            statistics.mean(p99_changes) <= 2.0
        ),
        "result_age_p95_not_increased": all(
            optimized <= baseline
            for baseline, optimized in zip(copy_age, lease_age)
        ),
        "control_fps_drop_not_over_2_percent": fps_change >= -2.0,
        "ai_result_fps_drop_not_over_2_percent": ai_fps_change >= -2.0,
        "no_reported_errors": not errors,
        "no_shared_memory_or_slot_failures": not log_failures,
        "lease_has_no_bgr_pool_writes": all(value == 0.0 for value in lease_bgr_writes),
        "both_contexts_observed": all(
            item["max"] == 1.0 and 0.0 < item["mean"] < 1.0
            for item in contexts
        ),
    }
    summary = {
        "copy_capture_to_result_p95_ms": copy_p95,
        "lease_capture_to_result_p95_ms": lease_p95,
        "paired_p95_reduction_percent": p95_reductions,
        "mean_p95_reduction_percent": statistics.mean(p95_reductions),
        "copy_capture_to_result_p99_ms": copy_p99,
        "lease_capture_to_result_p99_ms": lease_p99,
        "paired_p99_change_percent": p99_changes,
        "mean_p99_change_percent": statistics.mean(p99_changes),
        "copy_result_age_p95_frames": copy_age,
        "lease_result_age_p95_frames": lease_age,
        "copy_command_fps": copy_fps,
        "lease_command_fps": lease_fps,
        "mean_command_fps_change_percent": fps_change,
        "copy_ai_result_fps": copy_ai_fps,
        "lease_ai_result_fps": lease_ai_fps,
        "mean_ai_result_fps_change_percent": ai_fps_change,
        "lease_bgr_write_max_ms": lease_bgr_writes,
        "lease_context_stats": contexts,
        "errors": errors,
        "log_failures": log_failures,
        "gates": gates,
        "accepted": all(gates.values()),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
