from __future__ import annotations

import numpy as np

from core.lane.detector import LaneDetector


def make_detector(min_run_width: int = 3) -> LaneDetector:
    return LaneDetector(
        {
            "row_runs_backend": "numpy",
            "boundary": {"min_run_width_px": min_run_width},
        }
    )


def reference_row_runs(
    mask: np.ndarray,
    min_run_width: int,
) -> list[list[tuple[int, int]]]:
    rows: list[list[tuple[int, int]]] = []
    for row in mask:
        runs: list[tuple[int, int]] = []
        x = 0
        while x < row.size:
            while x < row.size and row[x] == 0:
                x += 1
            if x == row.size:
                break
            start = x
            while x < row.size and row[x] != 0:
                x += 1
            if x - start >= min_run_width:
                runs.append((start, x - 1))
        rows.append(runs)
    return rows


def materialize(row_runs) -> list[list[tuple[int, int]]]:
    return [list(row_runs[y]) for y in range(len(row_runs))]


def test_csr_row_runs_match_scalar_reference_on_random_masks() -> None:
    generator = np.random.default_rng(20260730)
    detector = make_detector(min_run_width=4)
    for _ in range(50):
        mask = (generator.random((37, 83)) > 0.82).astype(np.uint8) * 255
        expected = reference_row_runs(mask, 4)
        assert materialize(detector._build_row_runs(mask)) == expected


def test_csr_row_runs_preserve_roi_edge_contacts_and_filter_narrow_runs() -> None:
    mask = np.zeros((4, 16), dtype=np.uint8)
    mask[0, :5] = 255
    mask[1, 11:] = 255
    mask[2, 3:6] = 255
    mask[3, 2:7] = 255
    mask[3, 10:16] = 255

    detector = make_detector(min_run_width=4)
    assert materialize(detector._build_row_runs(mask)) == [
        [(0, 4)],
        [(11, 15)],
        [],
        [(2, 6), (10, 15)],
    ]


def test_binary_mask_fast_path_preserves_geometry_output() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    for y in range(mask.shape[0]):
        left = 24 + y // 12
        right = 88 + y // 15
        mask[y, left : right + 1] = 255
    detector = make_detector(min_run_width=4)

    contiguous = detector.detect_from_mask(mask)
    non_contiguous = make_detector(min_run_width=4).detect_from_mask(
        np.asfortranarray(mask)
    )

    assert contiguous.centerline_points == non_contiguous.centerline_points
    assert contiguous.left_boundary_points == non_contiguous.left_boundary_points
    assert contiguous.right_boundary_points == non_contiguous.right_boundary_points
    np.testing.assert_array_equal(contiguous.filtered_mask, non_contiguous.filtered_mask)
