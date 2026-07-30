#pragma once

#include <stdint.h>

#if defined(_WIN32)
#define XSMART_GEOMETRY_API __declspec(dllexport)
#else
#define XSMART_GEOMETRY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xsmart_boundary_row {
  int32_t y;
  int32_t left;
  int32_t right;
  uint8_t left_lost;
  uint8_t right_lost;
  uint16_t reserved;
} xsmart_boundary_row;

typedef struct xsmart_branch_row {
  int32_t x;
  int32_t y;
} xsmart_branch_row;

XSMART_GEOMETRY_API int xsmart_extract_row_runs(
    const uint8_t* mask,
    uint32_t width,
    uint32_t height,
    uint32_t row_stride,
    uint32_t min_run_width,
    uint32_t capacity,
    uint32_t* row_offsets,
    int32_t* starts,
    int32_t* ends,
    uint32_t* run_count);

XSMART_GEOMETRY_API int xsmart_trace_row_boundaries(
    const uint32_t* row_offsets,
    const int32_t* starts,
    const int32_t* ends,
    uint32_t height,
    uint32_t width,
    uint32_t max_single_side_gap_rows,
    int32_t route_mode,
    int32_t has_bottom_center,
    double bottom_center_x,
    double historical_center_x,
    uint32_t row_capacity,
    xsmart_boundary_row* rows,
    uint32_t* row_count,
    uint32_t branch_capacity,
    xsmart_branch_row* left_branches,
    uint32_t* left_branch_count,
    xsmart_branch_row* right_branches,
    uint32_t* right_branch_count);

#ifdef __cplusplus
}
#endif
