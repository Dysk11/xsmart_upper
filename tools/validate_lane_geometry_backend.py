"""Validate native row-run extraction against the NumPy reference on target."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lane.detector import LaneDetector  # noqa: E402
from core.runtime.app import load_config, prepare_runtime_config  # noqa: E402


def materialize(row_runs) -> list[list[tuple[int, int]]]:
    return [list(row_runs[y]) for y in range(len(row_runs))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--cases", type=int, default=200)
    args = parser.parse_args()

    config = prepare_runtime_config(load_config(Path(args.config)), PROJECT_ROOT)
    geometry = copy.deepcopy(config.get("lane_geometry", {}))
    reference_config = copy.deepcopy(geometry)
    reference_config["row_runs_backend"] = "numpy"
    native_config = copy.deepcopy(geometry)
    native_config["row_runs_backend"] = "native"
    reference = LaneDetector(reference_config)
    native = LaneDetector(native_config)

    generator = np.random.default_rng(20260730)
    height = 199
    width = 576
    for index in range(max(1, args.cases)):
        mask = (generator.random((height, width)) > 0.94).astype(np.uint8) * 255
        if index % 4 == 0:
            mask[:, :12] = 255
        if index % 5 == 0:
            mask[:, -14:] = 255
        expected = materialize(reference._build_row_runs(mask))
        actual = materialize(native._build_row_runs(mask))
        if actual != expected:
            print(f"row-run mismatch at case {index}", file=sys.stderr)
            return 1
        for direction in (None, "left", "right"):
            reference_runs = reference._build_row_runs(mask)
            native_runs = native._build_row_runs(mask)
            expected_boundaries = reference._extract_row_boundaries(
                mask,
                route_direction=direction,
                row_runs=reference_runs,
                bottom_center_x=0.5 * width,
            )
            actual_boundaries = native._extract_row_boundaries(
                mask,
                route_direction=direction,
                row_runs=native_runs,
                bottom_center_x=0.5 * width,
            )
            for field_index, (expected_field, actual_field) in enumerate(
                zip(expected_boundaries, actual_boundaries)
            ):
                if isinstance(expected_field, np.ndarray):
                    equal = np.array_equal(expected_field, actual_field)
                else:
                    equal = expected_field == actual_field
                if not equal:
                    print(
                        "boundary mismatch at "
                        f"case {index}, direction={direction}, field={field_index}",
                        file=sys.stderr,
                    )
                    return 1
    print(f"native row-run validation passed: {max(1, args.cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
