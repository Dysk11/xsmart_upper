from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.runtime.app import UpperMachineApp
from core.runtime.latency_benchmark import LatencyBenchmark, summarize_values
from tools.benchmark_latency import build_benchmark_config


def test_summarize_values_reports_interpolated_percentiles() -> None:
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["p99"] == pytest.approx(3.97)
    assert summary["max"] == pytest.approx(4.0)


def test_benchmark_excludes_warmup_and_stops_at_duration(tmp_path: Path) -> None:
    benchmark = LatencyBenchmark(
        {
            "enable": True,
            "warmup_sec": 2.0,
            "duration_sec": 3.0,
            "output_dir": str(tmp_path),
        },
        metadata={"source_mode": "video", "bridge_type": "mock"},
    )

    assert not benchmark.observe({"bridge_ms": 1.0}, observed_at=10.0)
    assert not benchmark.observe({"bridge_ms": 2.0}, observed_at=11.9)
    assert not benchmark.observe({"bridge_ms": 3.0}, observed_at=12.0)
    assert not benchmark.observe({"bridge_ms": 4.0}, observed_at=14.9)
    assert benchmark.observe({"bridge_ms": 5.0}, observed_at=15.0)
    summary = benchmark.finalize()

    assert summary is not None
    assert summary["command_samples"] == 2
    assert summary["metrics"]["bridge_ms"]["mean"] == pytest.approx(3.5)
    assert benchmark.output_paths["csv"].is_file()
    assert benchmark.output_paths["json"].is_file()
    assert benchmark.output_paths["markdown"].is_file()


def test_latency_sample_matches_command_to_lane_and_ai_frames() -> None:
    app = UpperMachineApp.__new__(UpperMachineApp)
    app.last_segmentation_timing = {
        "preprocess_ms": 0.5,
        "inference_ms": 18.0,
        "postprocess_queue_ms": 1.0,
        "postprocess_ms": 6.0,
        "total_ms": 25.5,
    }
    app.last_segmentation_frame_id = 8
    app.last_segmentation_worker_index = 1
    app.last_segmentation_captured_at = 100.0
    app.last_segmentation_ipc_finished_at = 100.002
    app.last_segmentation_completed_at = 100.026
    app.last_ai_frame_id = 7
    app.last_ai_captured_at = 99.990
    app.last_ai_ipc_finished_at = 99.993
    app.last_ai_completed_at = 100.015
    app.last_ai_timing = {
        "preprocess_ms": 1.0,
        "inference_ms": 10.0,
        "postprocess_ms": 3.0,
    }

    row = app._build_latency_sample(
        current_frame_id=10,
        source_frame_id=42,
        captured_at=100.020,
        bridge_finished=100.040,
        camera_wait_ms=2.0,
        color_conversion_ms=0.4,
        segmentation_wait_ms=0.2,
        geometry_ms=3.0,
        track_ms=0.1,
        planning_ms=0.5,
        protocol_ms=0.02,
        bridge_ms=0.08,
    )

    assert row["lane_result_age_frames"] == 2
    assert row["ai_result_age_frames"] == 3
    assert row["current_frame_to_send_ms"] == pytest.approx(20.0)
    assert row["lane_source_to_send_ms"] == pytest.approx(40.0)
    assert row["lane_source_to_worker_ipc_ms"] == pytest.approx(2.0)
    assert row["lane_npu_inference_ms"] == pytest.approx(18.0)
    assert row["ai_npu_inference_ms"] == pytest.approx(10.0)


def test_serial_benchmark_requires_explicit_physical_safety_confirmation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "camera": {"mode": "video", "video_path": "demo.mp4"},
                "bridge": {"type": "mock"},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config=str(config_path),
        mode="video",
        bridge="serial",
        video=None,
        warmup_sec=None,
        duration_sec=None,
        system_sample_interval_sec=None,
        output_dir=None,
        serial_safety_confirmed=False,
    )

    with pytest.raises(RuntimeError, match="physically disable vehicle motion"):
        build_benchmark_config(args)


def test_benchmark_cli_disables_off_path_ui_and_logging(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "camera": {"mode": "shared_memory"},
                "bridge": {"type": "mock"},
                "visualizer": {"show_window": True, "save_video": True},
                "logger": {"enable": True},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        config=str(config_path),
        mode="shared_memory",
        bridge="mock",
        video=None,
        shared_memory_name="xsmart_isolated_benchmark",
        ai_frame_transport="rgb_lease",
        warmup_sec=10.0,
        duration_sec=60.0,
        system_sample_interval_sec=1.0,
        output_dir=str(tmp_path / "reports"),
        serial_safety_confirmed=False,
    )

    config = build_benchmark_config(args)

    assert config["app"]["latency_benchmark"]["enable"] is True
    assert config["app"]["latency_benchmark"]["duration_sec"] == 60.0
    assert config["visualizer"]["show_window"] is False
    assert config["visualizer"]["save_video"] is False
    assert config["logger"]["enable"] is False
    assert (
        config["camera"]["shared_memory_name"]
        == "xsmart_isolated_benchmark"
    )
    assert (
        config["rknn_object_detector"]["ai_frame_transport"]
        == "rgb_lease"
    )
