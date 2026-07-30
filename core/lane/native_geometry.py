"""ctypes bridge for the stateless native lane row-run extractor."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


class _BoundaryRow(ctypes.Structure):
    _fields_ = [
        ("y", ctypes.c_int32),
        ("left", ctypes.c_int32),
        ("right", ctypes.c_int32),
        ("left_lost", ctypes.c_uint8),
        ("right_lost", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
    ]


class _BranchRow(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class NativeRowRunExtractor:
    """Extract foreground row runs without moving stateful lane logic to C++."""

    def __init__(self, library_path: str | Path) -> None:
        self.library_path = Path(library_path).expanduser()
        if not self.library_path.is_file():
            raise RuntimeError(
                f"native row-run library not found: {self.library_path}"
            )
        self._library = ctypes.CDLL(str(self.library_path))
        self._extract = self._library.xsmart_extract_row_runs
        self._extract.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._extract.restype = ctypes.c_int
        self._trace = self._library.xsmart_trace_row_boundaries
        self._trace.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint32,
            ctypes.POINTER(_BoundaryRow),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(_BranchRow),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_BranchRow),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._trace.restype = ctypes.c_int

    def extract(
        self,
        mask: np.ndarray,
        min_run_width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = np.ascontiguousarray(mask, dtype=np.uint8)
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError("native row-run input must be a non-empty 2-D mask")
        height, width = frame.shape
        row_offsets = np.empty(height + 1, dtype=np.uint32)
        run_count = ctypes.c_uint32(0)
        mask_pointer = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        offsets_pointer = row_offsets.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint32)
        )
        status = self._extract(
            mask_pointer,
            width,
            height,
            frame.strides[0],
            max(1, int(min_run_width)),
            0,
            offsets_pointer,
            None,
            None,
            ctypes.byref(run_count),
        )
        if status not in (0, 1):
            raise RuntimeError(f"native row-run sizing failed, code={status}")

        starts = np.empty(run_count.value, dtype=np.int32)
        ends = np.empty(run_count.value, dtype=np.int32)
        status = self._extract(
            mask_pointer,
            width,
            height,
            frame.strides[0],
            max(1, int(min_run_width)),
            run_count.value,
            offsets_pointer,
            starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(run_count),
        )
        if status != 0:
            raise RuntimeError(f"native row-run extraction failed, code={status}")
        return row_offsets, starts, ends

    def trace_boundaries(
        self,
        offsets: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        *,
        width: int,
        max_single_side_gap_rows: int,
        route_direction: str | None,
        bottom_center_x: float | None,
        historical_center_x: float,
    ) -> tuple[
        list[tuple[int, int, int, bool, bool]],
        list[tuple[int, int]],
        list[tuple[int, int]],
    ]:
        height = int(offsets.size) - 1
        rows = (_BoundaryRow * height)()
        branch_capacity = max(1, int(starts.size))
        left_branches = (_BranchRow * branch_capacity)()
        right_branches = (_BranchRow * branch_capacity)()
        row_count = ctypes.c_uint32(0)
        left_count = ctypes.c_uint32(0)
        right_count = ctypes.c_uint32(0)
        route_mode = 1 if route_direction == "left" else 2 if route_direction == "right" else 0
        status = self._trace(
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            height,
            max(1, int(width)),
            max(0, int(max_single_side_gap_rows)),
            route_mode,
            int(bottom_center_x is not None),
            0.0 if bottom_center_x is None else float(bottom_center_x),
            float(historical_center_x),
            height,
            rows,
            ctypes.byref(row_count),
            branch_capacity,
            left_branches,
            ctypes.byref(left_count),
            right_branches,
            ctypes.byref(right_count),
        )
        if status != 0:
            raise RuntimeError(f"native boundary trace failed, code={status}")
        materialized_rows = [
            (
                int(rows[index].y),
                int(rows[index].left),
                int(rows[index].right),
                bool(rows[index].left_lost),
                bool(rows[index].right_lost),
            )
            for index in range(row_count.value)
        ]
        materialized_left = [
            (int(left_branches[index].x), int(left_branches[index].y))
            for index in range(left_count.value)
        ]
        materialized_right = [
            (int(right_branches[index].x), int(right_branches[index].y))
            for index in range(right_count.value)
        ]
        return materialized_rows, materialized_left, materialized_right
