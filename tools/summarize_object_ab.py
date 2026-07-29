"""Summarize and enforce the three-run object detector A/B acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def load_runs(root: Path, backend: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for trial in range(1, 4):
        matches = sorted((root / f"{backend}_run{trial}").glob("latency_*.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {backend} run {trial} report, found {len(matches)}"
            )
        report = json.loads(matches[0].read_text(encoding="utf-8"))
        report["_path"] = str(matches[0])
        report["_trial"] = trial
        runs.append(report)
    return runs


def metric(report: dict[str, Any], name: str, statistic: str = "p95") -> float:
    return float(report["metrics"][name][statistic])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()

    lite2 = load_runs(args.root, "lite2")
    capi = load_runs(args.root, "c_api")
    lite_p95 = [metric(run, "ai_capture_to_result_ms") for run in lite2]
    capi_p95 = [metric(run, "ai_capture_to_result_ms") for run in capi]
    reductions = [
        (baseline - native) * 100.0 / baseline
        for baseline, native in zip(lite_p95, capi_p95)
    ]
    lite_age = [metric(run, "ai_result_age_frames") for run in lite2]
    capi_age = [metric(run, "ai_result_age_frames") for run in capi]
    lite_fps = [float(run["command_fps"]) for run in lite2]
    capi_fps = [float(run["command_fps"]) for run in capi]
    fps_change = (
        statistics.mean(capi_fps) - statistics.mean(lite_fps)
    ) * 100.0 / statistics.mean(lite_fps)
    errors = [
        {"path": run["_path"], "errors": run.get("errors", [])}
        for run in [*lite2, *capi]
        if run.get("errors")
    ]
    core_masks = [
        metric(run, "ai_core_mask", "mean")
        for run in capi
    ]
    contexts = [
        {
            "mean": metric(run, "ai_context_index", "mean"),
            "max": metric(run, "ai_context_index", "max"),
        }
        for run in capi
    ]
    gates = {
        "three_runs_each": len(lite2) == 3 and len(capi) == 3,
        "capture_p95_reduction_at_least_20_percent": min(reductions) >= 20.0,
        "result_age_p95_not_increased": all(
            native <= baseline for baseline, native in zip(lite_age, capi_age)
        ),
        "control_fps_drop_not_over_2_percent": fps_change >= -2.0,
        "no_reported_errors": not errors,
        "all_capi_results_report_npu2": all(mask == 4.0 for mask in core_masks),
        "both_contexts_observed": all(
            item["max"] == 1.0 and 0.0 < item["mean"] < 1.0
            for item in contexts
        ),
    }
    summary = {
        "lite2_capture_to_result_p95_ms": lite_p95,
        "c_api_capture_to_result_p95_ms": capi_p95,
        "paired_reduction_percent": reductions,
        "mean_reduction_percent": statistics.mean(reductions),
        "lite2_result_age_p95_frames": lite_age,
        "c_api_result_age_p95_frames": capi_age,
        "lite2_command_fps": lite_fps,
        "c_api_command_fps": capi_fps,
        "mean_command_fps_change_percent": fps_change,
        "c_api_core_mask_means": core_masks,
        "c_api_context_stats": contexts,
        "errors": errors,
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
