#include "lane_geometry_backend.h"

#include <cmath>
#include <cstddef>
#include <limits>

extern "C" int xsmart_extract_row_runs(
    const uint8_t* mask,
    const uint32_t width,
    const uint32_t height,
    const uint32_t row_stride,
    const uint32_t min_run_width,
    const uint32_t capacity,
    uint32_t* row_offsets,
    int32_t* starts,
    int32_t* ends,
    uint32_t* run_count) {
  if (mask == nullptr || row_offsets == nullptr || run_count == nullptr ||
      width == 0U || height == 0U || row_stride < width ||
      min_run_width == 0U) {
    return -1;
  }

  uint32_t count = 0U;
  bool overflow = false;
  for (uint32_t y = 0; y < height; ++y) {
    row_offsets[y] = count;
    const uint8_t* row = mask + static_cast<std::size_t>(y) * row_stride;
    uint32_t x = 0U;
    while (x < width) {
      while (x < width && row[x] == 0U) {
        ++x;
      }
      if (x == width) {
        break;
      }
      const uint32_t start = x;
      while (x < width && row[x] != 0U) {
        ++x;
      }
      const uint32_t end = x - 1U;
      if (end - start + 1U < min_run_width) {
        continue;
      }
      if (count < capacity && starts != nullptr && ends != nullptr) {
        starts[count] = static_cast<int32_t>(start);
        ends[count] = static_cast<int32_t>(end);
      } else {
        overflow = true;
      }
      ++count;
    }
  }
  row_offsets[height] = count;
  *run_count = count;
  return overflow ? 1 : 0;
}

extern "C" int xsmart_trace_row_boundaries(
    const uint32_t* row_offsets,
    const int32_t* starts,
    const int32_t* ends,
    const uint32_t height,
    const uint32_t width,
    const uint32_t max_single_side_gap_rows,
    const int32_t route_mode,
    const int32_t has_bottom_center,
    const double bottom_center_x,
    const double historical_center_x,
    const uint32_t row_capacity,
    xsmart_boundary_row* rows,
    uint32_t* row_count,
    const uint32_t branch_capacity,
    xsmart_branch_row* left_branches,
    uint32_t* left_branch_count,
    xsmart_branch_row* right_branches,
    uint32_t* right_branch_count) {
  if (row_offsets == nullptr || starts == nullptr || ends == nullptr ||
      rows == nullptr || row_count == nullptr || left_branches == nullptr ||
      left_branch_count == nullptr || right_branches == nullptr ||
      right_branch_count == nullptr || height == 0U || width == 0U ||
      row_capacity < height || (route_mode < 0 || route_mode > 2)) {
    return -1;
  }

  uint32_t output_count = 0U;
  uint32_t left_count = 0U;
  uint32_t right_count = 0U;
  uint32_t single_side_gap = 0U;
  bool first_valid_row = true;
  double prior_center = historical_center_x;
  for (int32_t y = static_cast<int32_t>(height) - 1; y >= 0; --y) {
    const uint32_t begin = row_offsets[y];
    const uint32_t end = row_offsets[y + 1];
    if (begin == end) {
      ++single_side_gap;
      if (single_side_gap <= max_single_side_gap_rows && output_count > 0U) {
        const xsmart_boundary_row& previous = rows[output_count - 1U];
        rows[output_count++] = {
            y, previous.left, previous.right, 1U, 1U, 0U};
      }
      continue;
    }

    single_side_gap = 0U;
    const bool select_from_bottom =
        first_valid_row && end - begin > 1U && has_bottom_center != 0 &&
        route_mode == 0;
    uint32_t chosen = begin;
    double chosen_distance = std::numeric_limits<double>::infinity();
    if (route_mode == 1 && end - begin > 1U) {
      chosen = begin;
    } else if (route_mode == 2 && end - begin > 1U) {
      chosen = end - 1U;
    } else {
      for (uint32_t index = begin; index < end; ++index) {
        const double center =
            0.5 * (static_cast<double>(starts[index]) + ends[index]);
        const double primary = std::abs(
            center - (select_from_bottom ? bottom_center_x : prior_center));
        if (primary < chosen_distance) {
          chosen = index;
          chosen_distance = primary;
          continue;
        }
        if (primary == chosen_distance && select_from_bottom) {
          const double candidate_history =
              std::abs(center - historical_center_x);
          const double chosen_center =
              0.5 * (static_cast<double>(starts[chosen]) + ends[chosen]);
          const double chosen_history =
              std::abs(chosen_center - historical_center_x);
          if (candidate_history < chosen_history) {
            chosen = index;
          }
        }
      }
    }

    const int32_t left = starts[chosen];
    const int32_t right = ends[chosen];
    for (uint32_t index = begin; index < end; ++index) {
      if (index == chosen) {
        continue;
      }
      const double center =
          0.5 * (static_cast<double>(starts[index]) + ends[index]);
      const xsmart_branch_row branch = {
          static_cast<int32_t>(center), y};
      if (center < prior_center) {
        if (left_count >= branch_capacity) {
          return 1;
        }
        left_branches[left_count++] = branch;
      } else {
        if (right_count >= branch_capacity) {
          return 1;
        }
        right_branches[right_count++] = branch;
      }
    }
    rows[output_count++] = {y, left, right, 0U, 0U, 0U};
    first_valid_row = false;
    const double chosen_center =
        0.5 * (static_cast<double>(left) + right);
    prior_center = select_from_bottom
                       ? chosen_center
                       : 0.65 * prior_center + 0.35 * chosen_center;
  }
  *row_count = output_count;
  *left_branch_count = left_count;
  *right_branch_count = right_count;
  return 0;
}
