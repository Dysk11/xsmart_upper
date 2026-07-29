"""Reusable latency benchmark collection and reporting."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import threading
import time
from typing import Any, Mapping


SUMMARY_PERCENTILES = (50, 95, 99)


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    """Return stable descriptive statistics for finite numeric samples."""

    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    def percentile(percent: int) -> float:
        if len(finite) == 1:
            return finite[0]
        position = (len(finite) - 1) * percent / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return finite[lower]
        weight = position - lower
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        **{f"p{percent}": percentile(percent) for percent in SUMMARY_PERCENTILES},
        "max": finite[-1],
    }


class LinuxSystemSampler:
    """Sample process-tree CPU/RSS and readable RK3588 telemetry once per interval."""

    def __init__(self, interval_sec: float = 1.0) -> None:
        self.interval_sec = max(0.1, float(interval_sec))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._last_cpu_total: int | None = None
        self._last_process_ticks: int | None = None

    def start(self, started_at: float) -> None:
        if self._thread is not None or platform.system() != "Linux":
            return
        self._started_at = float(started_at)
        self._thread = threading.Thread(
            target=self._run,
            name="xsmart-latency-system-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            sampled_at = time.perf_counter()
            self.samples.append(self._sample(sampled_at))
            remaining = self.interval_sec - (time.perf_counter() - sampled_at)
            self._stop.wait(max(0.0, remaining))

    def _sample(self, sampled_at: float) -> dict[str, Any]:
        process_ticks, rss_bytes, process_count = self._read_process_tree()
        cpu_total = self._read_cpu_total()
        cpu_percent: float | None = None
        if (
            process_ticks is not None
            and cpu_total is not None
            and self._last_process_ticks is not None
            and self._last_cpu_total is not None
            and cpu_total > self._last_cpu_total
        ):
            cpu_count = max(1, os.cpu_count() or 1)
            cpu_percent = (
                (process_ticks - self._last_process_ticks)
                / (cpu_total - self._last_cpu_total)
                * cpu_count
                * 100.0
            )
        self._last_process_ticks = process_ticks
        self._last_cpu_total = cpu_total
        return {
            "elapsed_sec": sampled_at - self._started_at,
            "process_tree_cpu_percent": cpu_percent,
            "process_tree_rss_mb": (
                rss_bytes / (1024.0 * 1024.0) if rss_bytes is not None else None
            ),
            "process_count": process_count,
            "temperature_c": self._read_temperature(),
            "npu_frequency_hz": self._read_first_int(
                (
                    "/sys/class/devfreq/fdab0000.npu/cur_freq",
                    "/sys/class/devfreq/fdab0000.npu/device/devfreq/fdab0000.npu/cur_freq",
                )
            ),
            "npu_load": self._read_first_text(
                (
                    "/sys/kernel/debug/rknpu/load",
                    "/sys/class/devfreq/fdab0000.npu/load",
                    "/sys/class/devfreq/fdab0000.npu/device/load",
                )
            ),
        }

    @staticmethod
    def _read_cpu_total() -> int | None:
        try:
            fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
            return sum(int(value) for value in fields[1:])
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def _read_process_tree() -> tuple[int | None, int | None, int]:
        root_pid = os.getpid()
        process_rows: dict[int, tuple[int, int, int]] = {}
        try:
            for stat_path in Path("/proc").glob("[0-9]*/stat"):
                try:
                    text = stat_path.read_text(encoding="utf-8")
                    close_paren = text.rfind(")")
                    fields = text[close_paren + 2 :].split()
                    pid = int(stat_path.parent.name)
                    ppid = int(fields[1])
                    ticks = int(fields[11]) + int(fields[12])
                    rss_pages = int(fields[21])
                    process_rows[pid] = (ppid, ticks, rss_pages)
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            return None, None, 0

        descendants = {root_pid}
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _ticks, _rss) in process_rows.items():
                if ppid in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        selected = [process_rows[pid] for pid in descendants if pid in process_rows]
        if not selected:
            return None, None, 0
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return (
            sum(row[1] for row in selected),
            sum(row[2] for row in selected) * page_size,
            len(selected),
        )

    @staticmethod
    def _read_temperature() -> float | None:
        values: list[float] = []
        for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                raw = float(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            values.append(raw / 1000.0 if raw > 1000.0 else raw)
        return max(values) if values else None

    @staticmethod
    def _read_first_int(paths: tuple[str, ...]) -> int | None:
        text = LinuxSystemSampler._read_first_text(paths)
        if text is None:
            return None
        try:
            return int(text.strip().split()[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _read_first_text(paths: tuple[str, ...]) -> str | None:
        for raw_path in paths:
            try:
                return Path(raw_path).read_text(encoding="utf-8").strip()
            except OSError:
                continue
        return None


class LatencyBenchmark:
    """Collect command-aligned latency samples after a fixed warmup."""

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        config = config or {}
        self.enabled = bool(config.get("enable", False))
        self.warmup_sec = max(0.0, float(config.get("warmup_sec", 10.0)))
        self.duration_sec = max(0.1, float(config.get("duration_sec", 60.0)))
        self.output_dir = Path(
            str(config.get("output_dir", "outputs/benchmarks"))
        ).expanduser()
        self.auto_stop = bool(config.get("auto_stop", True))
        self.system_sampler = LinuxSystemSampler(
            float(config.get("system_sample_interval_sec", 1.0))
        )
        self.metadata = dict(metadata or {})
        self.started_at: float | None = None
        self.sample_started_at: float | None = None
        self.completed_at: float | None = None
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.completed = False
        self.output_paths: dict[str, Path] = {}

    def observe(self, row: Mapping[str, Any], *, observed_at: float) -> bool:
        """Record one command row and return True when the sample window is complete."""

        if not self.enabled or self.completed:
            return self.completed
        now = float(observed_at)
        if self.started_at is None:
            self.started_at = now
            self.sample_started_at = now + self.warmup_sec
            self.system_sampler.start(now)
        assert self.sample_started_at is not None
        if now < self.sample_started_at:
            return False
        if now >= self.sample_started_at + self.duration_sec:
            self.completed = True
            self.completed_at = now
            return True
        output = dict(row)
        output["sample_elapsed_sec"] = now - self.sample_started_at
        self.rows.append(output)
        return False

    def add_error(self, message: str) -> None:
        if self.enabled:
            self.errors.append(str(message))

    def finalize(self) -> dict[str, Any] | None:
        if not self.enabled or self.started_at is None:
            return None
        self.system_sampler.stop()
        if self.completed_at is None:
            self.completed_at = time.perf_counter()
        summary = self._build_summary()
        self._write_outputs(summary)
        return summary

    def _build_summary(self) -> dict[str, Any]:
        numeric_keys = sorted(
            {
                key
                for row in self.rows
                for key, value in row.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        metrics = {
            key: summarize_values(
                [float(row[key]) for row in self.rows if row.get(key) is not None]
            )
            for key in numeric_keys
            if key
            not in {
                "current_frame_id",
                "source_frame_id",
                "lane_frame_id",
                "ai_frame_id",
                "sample_elapsed_sec",
            }
        }
        sample_elapsed = (
            max((float(row["sample_elapsed_sec"]) for row in self.rows), default=0.0)
        )
        unique_lane_frames = len(
            {int(row["lane_frame_id"]) for row in self.rows if int(row.get("lane_frame_id", -1)) >= 0}
        )
        unique_ai_frames = len(
            {int(row["ai_frame_id"]) for row in self.rows if int(row.get("ai_frame_id", -1)) >= 0}
        )
        native_completed_values = [
            int(row["lane_native_completed_count"])
            for row in self.rows
            if row.get("lane_native_completed_count") is not None
        ]
        native_completed_fps = (
            (
                native_completed_values[-1] - native_completed_values[0]
            )
            / sample_elapsed
            if sample_elapsed > 0 and len(native_completed_values) >= 2
            else None
        )
        return {
            "metadata": self.metadata,
            "warmup_sec": self.warmup_sec,
            "requested_duration_sec": self.duration_sec,
            "observed_duration_sec": sample_elapsed,
            "command_samples": len(self.rows),
            "command_fps": len(self.rows) / sample_elapsed if sample_elapsed > 0 else None,
            "lane_result_fps": unique_lane_frames / sample_elapsed if sample_elapsed > 0 else None,
            "lane_native_completed_fps": native_completed_fps,
            "ai_result_fps": unique_ai_frames / sample_elapsed if sample_elapsed > 0 else None,
            "errors": list(self.errors),
            "metrics": metrics,
            "system_samples": self.system_sampler.samples,
            "telemetry_availability": {
                "cpu": any(
                    sample.get("process_tree_cpu_percent") is not None
                    for sample in self.system_sampler.samples
                ),
                "temperature": any(
                    sample.get("temperature_c") is not None
                    for sample in self.system_sampler.samples
                ),
                "npu_frequency": any(
                    sample.get("npu_frequency_hz") is not None
                    for sample in self.system_sampler.samples
                ),
                "npu_load": any(
                    sample.get("npu_load") is not None
                    for sample in self.system_sampler.samples
                ),
            },
        }

    def _write_outputs(self, summary: Mapping[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source = str(self.metadata.get("source_mode", "unknown"))
        bridge = str(self.metadata.get("bridge_type", "unknown"))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = self.output_dir / f"latency_{stamp}_{source}_{bridge}"
        csv_path = base.with_suffix(".csv")
        json_path = base.with_suffix(".json")
        markdown_path = base.with_suffix(".md")
        fieldnames = sorted({key for row in self.rows for key in row})
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_path.write_text(
            self._render_markdown(summary),
            encoding="utf-8",
        )
        self.output_paths = {
            "csv": csv_path,
            "json": json_path,
            "markdown": markdown_path,
        }

    @staticmethod
    def _render_markdown(summary: Mapping[str, Any]) -> str:
        metadata = summary.get("metadata", {})
        lines = [
            "# X-SmartCar latency benchmark",
            "",
            f"- Source: `{metadata.get('source_mode', 'unknown')}`",
            f"- Bridge: `{metadata.get('bridge_type', 'unknown')}`",
            f"- Command samples: `{summary.get('command_samples', 0)}`",
            f"- Observed duration: `{summary.get('observed_duration_sec', 0):.3f} s`",
            f"- Command FPS: `{_format_number(summary.get('command_fps'))}`",
            f"- Lane-result FPS: `{_format_number(summary.get('lane_result_fps'))}`",
            f"- Native-completed FPS: `{_format_number(summary.get('lane_native_completed_fps'))}`",
            f"- AI-result FPS: `{_format_number(summary.get('ai_result_fps'))}`",
            "",
            "## Latency metrics (ms)",
            "",
            "| Metric | Mean | P50 | P95 | P99 | Max | Count |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, stats in summary.get("metrics", {}).items():
            if not name.endswith(("_ms", "_frames")):
                continue
            lines.append(
                "| {name} | {mean} | {p50} | {p95} | {p99} | {max_value} | {count} |".format(
                    name=name,
                    mean=_format_number(stats.get("mean")),
                    p50=_format_number(stats.get("p50")),
                    p95=_format_number(stats.get("p95")),
                    p99=_format_number(stats.get("p99")),
                    max_value=_format_number(stats.get("max")),
                    count=stats.get("count", 0),
                )
            )
        lines.extend(
            [
                "",
                "## Telemetry availability",
                "",
            ]
        )
        for name, available in summary.get("telemetry_availability", {}).items():
            lines.append(f"- {name}: {'available' if available else 'unavailable'}")
        errors = summary.get("errors", [])
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
        if not errors:
            lines.append("- None")
        lines.append("")
        return "\n".join(lines)


def _format_number(value: Any) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.3f}"
