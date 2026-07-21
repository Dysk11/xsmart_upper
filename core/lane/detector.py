"""蓝色航道检测模块。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.math_utils import (
    clamp,
    compute_curvature,
    evaluate_poly,
    mean_abs_residual,
    polyfit_with_fallback,
    safe_divide,
)


@dataclass
class ForkLaneResult:
    """Debug information for fork branch selection."""

    fork_detected: bool
    requested_direction: str | None
    selected_direction: str | None
    left_points: List[Tuple[int, int]]
    right_points: List[Tuple[int, int]]
    reason: str
    left_detected: bool = False
    right_detected: bool = False
    left_corner: Tuple[int, int] | None = None
    right_corner: Tuple[int, int] | None = None
    confirm_frames: int = 0
    state: str = "MAIN"
    junction_kind: str | None = None
    a_point: Tuple[int, int] | None = None
    p_point: Tuple[int, int] | None = None
    break_point: Tuple[int, int] | None = None
    patch_line: Tuple[Tuple[int, int], Tuple[int, int]] | None = None
    path_overridden: bool = False


@dataclass(frozen=True)
class RightForkObservation:
    """Measured right-fork features before temporal state handling."""

    a_point: Tuple[int, int] | None = None
    p_point: Tuple[int, int] | None = None
    junction_kind: str | None = None
    break_point: Tuple[int, int] | None = None
    right_branch_points: Tuple[Tuple[int, int], ...] = ()
    boundaries_valid: bool = False


@dataclass(frozen=True)
class RightForkDecision:
    """State-machine output consumed by the row-boundary path builder."""

    result: ForkLaneResult
    patch_kind: str | None = None
    base_route_direction: str | None = None


@dataclass
class LaneDetectionResult:
    """保存单帧巡线检测结果。"""

    centerline_points: List[Tuple[int, int]]
    lateral_error_px: float
    heading_error_deg: float
    curvature: float
    confidence: float
    is_lane_lost: bool
    mask: np.ndarray
    filtered_mask: np.ndarray
    fit_coeffs: Optional[np.ndarray]
    lane_width_px: float
    valid_row_count: int
    fit_point_count: int
    fork_result: ForkLaneResult
    left_boundary_points: List[Tuple[int, int]] = field(default_factory=list)
    right_boundary_points: List[Tuple[int, int]] = field(default_factory=list)
    left_lost_rows: int = 0
    right_lost_rows: int = 0
    segmentation_confidence: float = 0.0
    segmentation_status: str = "legacy"


@dataclass(frozen=True)
class LaneAnchorSample:
    """保存单个连通域内部可用于连接中心线的候选锚点。"""

    x: float
    y: int
    width: int


@dataclass
class LaneComponent:
    """保存单个蓝色候选连通域的几何信息。"""

    label: int
    x: int
    y: int
    width: int
    height: int
    area: int
    centroid_x: float
    centroid_y: float
    bottom_y: int
    touches_side: bool
    anchor_samples: Tuple[LaneAnchorSample, ...]


@dataclass
class RouteComponentSelection:
    """Selected component chain and fork debug result."""

    selected_components: List[LaneComponent]
    fork_result: ForkLaneResult


ARTICLE_ROW_WEIGHTS = np.asarray(
    [0] * 46
    + [2] * 5 + [4] * 5 + [5] * 7 + [6] * 6 + [8] + [9] * 3
    + [10] * 8 + [9] * 4 + [8] * 2 + [7] * 3
    + [6, 7, 7, 7, 6, 6, 6, 6, 6, 5]
    + [5, 5, 5, 5, 4, 4, 3, 3, 3, 3] + [0] * 10,
    dtype=np.float32,
)


class LaneDetector:
    """使用传统视觉方法检测蓝色航道中心线。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """读取阈值与拟合参数并初始化检测器。

        输入:
            config: lane_detector 对应配置字典。

        输出:
            无返回值，内部缓存检测参数与少量历史状态。
        """

        self.config = config
        self.color_space = str(config.get("color_space", "hsv")).lower()
        self.hsv_config = config.get("hsv", {})
        self.lab_config = config.get("lab", {})
        self.morphology_config = config.get("morphology", {})
        self.component_config = config.get("connected_components", {})
        self.centerline_config = config.get("centerline", {})
        self.confidence_config = config.get("confidence", {})

        self.scan_step = int(self.centerline_config.get("scan_step", 6))
        self.min_valid_points = int(self.centerline_config.get("min_valid_points", 8))
        self.min_lane_width_px = float(self.centerline_config.get("min_lane_width_px", 18.0))
        self.default_lane_width_px = float(self.centerline_config.get("default_lane_width_px", 60.0))
        self.single_side_infer_ratio = float(self.centerline_config.get("single_side_infer_ratio", 0.55))
        self.enable_single_side_inference = bool(
            self.centerline_config.get("enable_single_side_inference", False)
        )
        self.single_side_edge_margin_px = int(
            self.centerline_config.get("single_side_edge_margin_px", 28)
        )
        self.prediction_blend = float(self.centerline_config.get("prediction_blend", 0.2))
        self.distance_weight = float(self.centerline_config.get("distance_weight", 0.12))
        self.center_bias = float(self.centerline_config.get("center_bias", 0.02))
        self.lookahead_ratio = float(self.centerline_config.get("lookahead_ratio", 0.45))
        self.max_center_jump_px = float(self.centerline_config.get("max_center_jump_px", 120.0))
        self.fit_top_extension_rows = int(self.centerline_config.get("fit_top_extension_rows", 1))
        # 箭头赛道不是连续实线，连通域内部会有“宽头 + 窄颈”结构。
        # 这里优先在组件内部找更窄、更靠上的锚点，避免中心线钻进底部尖角。
        self.anchor_sample_ratios = (0.35, 0.45, 0.55)
        self.anchor_width_weight = 0.12
        self.max_component_dx_px = 210.0
        self.max_component_gap_px = 240.0
        self.start_bottom_ratio = 0.55
        # 下面几条是“不要把环境蓝色接进主航道链”的几何约束。
        self.min_component_chain_score = float(
            self.component_config.get("min_component_chain_score", 12.0)
        )
        self.short_gap_y_threshold = float(
            self.component_config.get("short_gap_y_threshold", 24.0)
        )
        self.short_gap_max_dx_px = float(
            self.component_config.get("short_gap_max_dx_px", 120.0)
        )
        self.max_far_width_growth_ratio = float(
            self.component_config.get("max_far_width_growth_ratio", 1.45)
        )
        self.width_growth_guard_top_ratio = float(
            self.component_config.get("width_growth_guard_top_ratio", 0.48)
        )
        # 这一组参数只在主航道链连接阶段生效，用来忽略墙面广告、
        # 赛道外蓝块等环境蓝色，不会直接删除远处真实航道。
        self.environment_guard_top_ratio = float(
            self.component_config.get("environment_guard_top_ratio", 0.34)
        )
        self.environment_guard_min_chain_length = int(
            self.component_config.get("environment_guard_min_chain_length", 2)
        )
        self.environment_compact_aspect_ratio_max = float(
            self.component_config.get("environment_compact_aspect_ratio_max", 2.2)
        )
        self.environment_small_area_max = float(
            self.component_config.get("environment_small_area_max", 4200.0)
        )
        self.environment_small_width_ratio = float(
            self.component_config.get("environment_small_width_ratio", 0.72)
        )
        self.environment_side_penalty = float(
            self.component_config.get("environment_side_penalty", 28.0)
        )
        self.lost_threshold = float(self.confidence_config.get("lost_threshold", 0.28))
        self.expected_area_ratio = float(self.confidence_config.get("expected_area_ratio", 0.08))
        self.residual_tolerance_px = float(self.confidence_config.get("residual_tolerance_px", 18.0))

        self.last_fit_coeffs: Optional[np.ndarray] = None
        self.last_lane_width_px = self.default_lane_width_px
        self.last_centerline_points: List[Tuple[int, int]] = []
        boundary_config = config.get("boundary", {})
        fork_config = config.get("fork", {})
        temporal_config = config.get("temporal_filter", {})
        self.gradient_jump_ratio = float(boundary_config.get("gradient_jump_ratio", 0.05))
        self.gradient_step_ratio = float(boundary_config.get("gradient_step_ratio", 0.025))
        self.max_single_side_gap_rows = int(boundary_config.get("max_single_side_gap_rows", 12))
        self.min_run_width_px = int(boundary_config.get("min_run_width_px", 6))
        self.fork_corner_span_rows = int(fork_config.get("corner_span_rows", 10))
        self.fork_outward_jump_ratio = float(fork_config.get("outward_jump_ratio", 0.08))
        self.fork_min_lost_rows = int(fork_config.get("min_lost_rows", 3))
        self.fork_corner_min_y_ratio = float(fork_config.get("corner_min_y_ratio", 0.05))
        self.fork_corner_max_y_ratio = float(fork_config.get("corner_max_y_ratio", 0.72))
        self.fork_corner_side_margin_ratio = float(
            fork_config.get("corner_side_margin_ratio", 0.02)
        )
        self.fork_confirm_frames = max(1, int(fork_config.get("confirm_frames", 2)))
        self.fork_release_frames = max(1, int(fork_config.get("release_frames", 3)))
        self.fork_feature_span_rows = max(
            2, int(fork_config.get("feature_span_rows", self.fork_corner_span_rows))
        )
        self.fork_feature_prominence_ratio = float(
            fork_config.get("feature_prominence_ratio", 0.025)
        )
        self.fork_feature_min_y_ratio = float(
            fork_config.get("feature_min_y_ratio", self.fork_corner_min_y_ratio)
        )
        self.fork_feature_max_y_ratio = float(
            fork_config.get("feature_max_y_ratio", 0.85)
        )
        self.fork_feature_side_margin_ratio = float(
            fork_config.get("feature_side_margin_ratio", self.fork_corner_side_margin_ratio)
        )
        self.fork_orientation_min_rows = max(
            1, int(fork_config.get("orientation_min_rows", 3))
        )
        self.fork_orientation_vote_ratio = float(
            fork_config.get("orientation_vote_ratio", 0.65)
        )
        self.fork_ap_min_distance_ratio = float(
            fork_config.get("ap_min_distance_ratio", 0.04)
        )
        self.fork_ap_max_distance_ratio = float(
            fork_config.get("ap_max_distance_ratio", 0.75)
        )
        self.fork_break_jump_ratio = float(
            fork_config.get("break_jump_ratio", 0.05)
        )
        self.fork_recovery_confirm_frames = max(
            1, int(fork_config.get("recovery_confirm_frames", 5))
        )
        self.fork_return_release_frames = max(
            1, int(fork_config.get("return_release_frames", 3))
        )
        self.temporal_weights = tuple(float(v) for v in temporal_config.get("weights", [0.20, 0.50, 0.30]))
        if len(self.temporal_weights) != 3 or sum(self.temporal_weights) <= 0:
            self.temporal_weights = (0.20, 0.50, 0.30)
        self._weighted_center_history: List[float] = []
        self._left_fork_hits = self._right_fork_hits = 0
        self._left_fork_misses = self._right_fork_misses = 0
        self._left_fork_active = self._right_fork_active = False
        self._fork_state = "MAIN"
        self._fork_junction_kind: str | None = None
        self._fork_junction_hits = 0
        self._fork_junction_misses = 0
        self._fork_right_latched = False
        self._fork_last_a_point: Tuple[int, int] | None = None
        self._fork_last_p_point: Tuple[int, int] | None = None
        self._fork_break_hits = 0
        self._fork_recovery_hits = 0
        self._fork_return_clear_hits = 0
        self._fork_last_break_point: Tuple[int, int] | None = None

    def detect(self, roi_frame: np.ndarray, route_direction: str | None = None) -> LaneDetectionResult:
        """对单帧 ROI 图像执行蓝色航道检测。

        输入:
            roi_frame: 经过预处理后的 ROI BGR 图像。

        输出:
            返回 LaneDetectionResult，包含中心线、误差、曲率、置信度和调试掩膜。
        """

        if roi_frame.size == 0:
            return self._empty_result((1, 1))

        # 先做蓝色阈值分割，得到“哪里像蓝色航道”的初始掩膜。
        mask = self._segment_lane(roi_frame)
        return self.detect_from_mask(mask, route_direction=route_direction)

    def detect_from_mask(
        self,
        roi_mask: np.ndarray,
        route_direction: str | None = None,
        segmentation_confidence: float = 0.0,
        segmentation_status: str = "ok",
    ) -> LaneDetectionResult:
        """Extract lane geometry from an externally produced binary ROI mask."""

        if roi_mask.size == 0:
            result = self._empty_result((1, 1))
            result.segmentation_status = segmentation_status
            return result
        if roi_mask.ndim == 3:
            roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)
        mask = np.where(roi_mask > 0, 255, 0).astype(np.uint8)
        return self._detect_from_boundaries(
            mask,
            route_direction=route_direction,
            segmentation_confidence=segmentation_confidence,
            segmentation_status=segmentation_status,
        )

        # Legacy connected-component implementation is intentionally unreachable.
        # 先提取所有可用蓝色连通域，再从中选出“沿地面连续前进”的主链条。
        labels, candidate_components = self._extract_candidate_components(mask)
        branch_result = self._select_components_for_route(
            components=candidate_components,
            shape=mask.shape,
            route_direction=route_direction,
        )
        selected_components = branch_result.selected_components
        selected_mask = self._build_mask_from_components(labels, selected_components)
        # 当主链条太短时，不要让调试窗口里整片蓝箭头忽隐忽现，先显示全部候选区域。
        filtered_mask = selected_mask if len(selected_components) >= 2 else mask

        # 优先使用“箭头块内部锚点链 + 折线插值”生成中心线。
        # 这样能显著减少二次曲线在尖角处钻出、再突然跳到下一块箭头的情况。
        support_points = self._build_support_points_from_components(selected_components, mask.shape)
        centerline_points = self._build_polyline_centerline(support_points, filtered_mask.shape)
        lane_width_px = self._estimate_lane_width_from_components(selected_components)

        # 如果组件链不够稳定，再退回到逐行扫描作为兜底方案。
        if len(centerline_points) < self.min_valid_points:
            fallback_mask = selected_mask if cv2.countNonZero(selected_mask) > 0 else mask
            raw_points, lane_width_px = self._extract_centerline(fallback_mask)
            centerline_points = self._filter_centerline_points(raw_points)
            support_points = list(centerline_points)

        # 曲线拟合只用于提取曲率和做少量历史预测，显示与控制主链路改用折线，避免拟合翻折。
        fit_coeffs = self._fit_centerline(centerline_points)

        # 根据中心线计算车辆当前最关心的三个量：横向误差、航向误差、曲率。
        lateral_error_px, heading_error_deg, curvature = self._compute_lane_metrics(
            centerline_points=centerline_points,
            fit_coeffs=fit_coeffs,
            shape=filtered_mask.shape,
            support_points=support_points,
        )
        # 置信度用于告诉后面的跟踪器和规划器：这帧结果到底靠不靠谱。
        confidence = self._estimate_confidence(
            filtered_mask=filtered_mask,
            raw_points=centerline_points,
            fit_coeffs=fit_coeffs,
            lane_width_px=lane_width_px,
        )
        is_lane_lost = len(centerline_points) < max(2, self.min_valid_points // 2) or confidence < self.lost_threshold

        if not is_lane_lost and centerline_points:
            self.last_fit_coeffs = fit_coeffs
            self.last_lane_width_px = lane_width_px
            self.last_centerline_points = centerline_points

        return LaneDetectionResult(
            centerline_points=centerline_points,
            lateral_error_px=lateral_error_px,
            heading_error_deg=heading_error_deg,
            curvature=curvature,
            confidence=confidence,
            is_lane_lost=is_lane_lost,
            mask=mask,
            filtered_mask=filtered_mask,
            fit_coeffs=fit_coeffs,
            lane_width_px=lane_width_px,
            valid_row_count=len(selected_components),
            fit_point_count=len(centerline_points),
            fork_result=branch_result.fork_result,
            segmentation_confidence=segmentation_confidence,
            segmentation_status=segmentation_status,
        )

    def _detect_from_boundaries(
        self,
        mask: np.ndarray,
        route_direction: str | None,
        segmentation_confidence: float,
        segmentation_status: str,
    ) -> LaneDetectionResult:
        """Build the driving centerline directly from row-wise track boundaries."""

        row_runs = self._build_row_runs(mask)
        (
            left_points,
            right_points,
            raw_centers,
            left_lost,
            right_lost,
            selected_mask,
            left_branch_rows,
            right_branch_rows,
        ) = self._extract_row_boundaries(mask, row_runs=row_runs)
        observation = self._observe_right_fork(
            mask,
            row_runs,
            left_points,
            right_points,
            left_lost,
            right_lost,
            right_branch_rows,
        )
        fork_decision = self._update_right_fork_state(
            observation,
            route_direction=route_direction,
            shape=mask.shape,
        )
        fork_result = fork_decision.result

        if fork_decision.base_route_direction is not None:
            (
                left_points,
                right_points,
                raw_centers,
                left_lost,
                right_lost,
                selected_mask,
                _selected_left_branches,
                _selected_right_branches,
            ) = self._extract_row_boundaries(
                mask,
                route_direction=fork_decision.base_route_direction,
                row_runs=row_runs,
            )

        # Keep the segmentation evidence separate from the virtual corridor.
        # Boundary repair is allowed to guide control, but must not manufacture
        # confidence by painting pixels into the raw model output.
        evidence_mask = selected_mask.copy()
        raw_left_lost_rows = sum(left_lost)
        raw_right_lost_rows = sum(right_lost)
        patch_side: str | None = None
        patch_start: Tuple[int, int] | None = None
        patch_end: Tuple[int, int] | None = None
        if fork_decision.patch_kind == "main_ap":
            patch_side = "right"
            patch_start = fork_result.a_point
            patch_end = fork_result.p_point
        elif fork_decision.patch_kind == "enter_branch":
            bottom_left = self._lowest_valid_boundary_point(left_points, left_lost)
            if bottom_left is not None and fork_result.p_point is not None:
                patch_side = "left"
                patch_start = bottom_left
                patch_end = fork_result.p_point
        elif fork_decision.patch_kind == "return_to_main":
            if fork_result.break_point is not None:
                patch_side = "left"
                patch_start = fork_result.break_point
                patch_end = (mask.shape[1] - 1, 0)

        if patch_side is not None and patch_start is not None and patch_end is not None:
            (
                left_points,
                right_points,
                raw_centers,
                selected_mask,
            ) = self._apply_virtual_boundary_patch(
                left_points=left_points,
                right_points=right_points,
                side=patch_side,
                start=patch_start,
                end=patch_end,
                shape=mask.shape,
            )
            fork_result.patch_line = (patch_start, patch_end)
            fork_result.path_overridden = True

        centerline_points = self._smooth_article_centerline(raw_centers, mask.shape[1])
        fit_coeffs = self._fit_centerline(centerline_points)
        widths = [right[0] - left[0] for left, right in zip(left_points, right_points) if right[0] > left[0]]
        lane_width_px = float(np.median(widths)) if widths else self.last_lane_width_px
        lateral_error_px, heading_error_deg, curvature = self._article_metrics(
            centerline_points, fit_coeffs, mask.shape
        )
        confidence = self._estimate_confidence(
            evidence_mask,
            centerline_points,
            fit_coeffs,
            lane_width_px,
        )
        is_lane_lost = len(centerline_points) < max(2, self.min_valid_points // 2) or confidence < self.lost_threshold
        if not is_lane_lost and centerline_points:
            self.last_fit_coeffs = fit_coeffs
            self.last_lane_width_px = lane_width_px
            self.last_centerline_points = centerline_points

        return LaneDetectionResult(
            centerline_points=centerline_points,
            lateral_error_px=lateral_error_px,
            heading_error_deg=heading_error_deg,
            curvature=curvature,
            confidence=confidence,
            is_lane_lost=is_lane_lost,
            mask=mask,
            filtered_mask=selected_mask,
            fit_coeffs=fit_coeffs,
            lane_width_px=lane_width_px,
            valid_row_count=len(raw_centers),
            fit_point_count=len(centerline_points),
            fork_result=fork_result,
            left_boundary_points=left_points,
            right_boundary_points=right_points,
            left_lost_rows=raw_left_lost_rows,
            right_lost_rows=raw_right_lost_rows,
            segmentation_confidence=segmentation_confidence,
            segmentation_status=segmentation_status,
        )

    def _build_row_runs(self, mask: np.ndarray) -> list[list[tuple[int, int]]]:
        """Precompute valid foreground runs for every row in one NumPy pass."""

        height, width = mask.shape[:2]
        padded = np.zeros((height, width + 2), dtype=np.uint8)
        padded[:, 1 : width + 1] = mask > 0
        transitions = np.diff(padded.astype(np.int8, copy=False), axis=1)
        start_rows, start_xs = np.nonzero(transitions == 1)
        end_rows, end_exclusive_xs = np.nonzero(transitions == -1)

        row_runs: list[list[tuple[int, int]]] = [[] for _ in range(height)]
        for start_row, start_x, end_row, end_exclusive_x in zip(
            start_rows,
            start_xs,
            end_rows,
            end_exclusive_xs,
        ):
            if start_row != end_row:
                continue
            if int(end_exclusive_x) - int(start_x) < self.min_run_width_px:
                continue
            row_runs[int(start_row)].append((int(start_x), int(end_exclusive_x) - 1))
        return row_runs

    def _extract_row_boundaries(
        self,
        mask: np.ndarray,
        route_direction: str | None = None,
        row_runs: Sequence[Sequence[tuple[int, int]]] | None = None,
    ):
        """Follow the run nearest the previous center from the vehicle upward."""

        height, width = mask.shape[:2]
        if row_runs is None:
            row_runs = self._build_row_runs(mask)
        selected_mask = np.zeros_like(mask)
        prior_center = float(self.last_centerline_points[0][0]) if self.last_centerline_points else width * 0.5
        prior_width = max(float(self.last_lane_width_px), float(self.min_run_width_px))
        rows: list[tuple[int, int, int, bool, bool]] = []
        left_branch_rows: list[tuple[int, int]] = []
        right_branch_rows: list[tuple[int, int]] = []
        single_side_gap = 0

        for y in range(height - 1, -1, -1):
            runs = row_runs[y]
            if not runs:
                single_side_gap += 1
                if single_side_gap <= self.max_single_side_gap_rows and rows:
                    last_left, last_right = rows[-1][1], rows[-1][2]
                    rows.append((y, last_left, last_right, True, True))
                continue

            single_side_gap = 0
            centers = [0.5 * (left + right) for left, right in runs]
            if route_direction == "left" and len(runs) > 1:
                chosen_index = min(range(len(runs)), key=lambda index: centers[index])
            elif route_direction == "right" and len(runs) > 1:
                chosen_index = max(range(len(runs)), key=lambda index: centers[index])
            else:
                chosen_index = min(range(len(runs)), key=lambda index: abs(centers[index] - prior_center))
            left, right = runs[chosen_index]
            for index, (candidate_left, candidate_right) in enumerate(runs):
                if index == chosen_index:
                    continue
                candidate_center = 0.5 * (candidate_left + candidate_right)
                if candidate_center < prior_center:
                    left_branch_rows.append((int(candidate_center), y))
                else:
                    right_branch_rows.append((int(candidate_center), y))

            left_missing = left <= 1
            right_missing = right >= width - 2
            if left_missing and not right_missing:
                left = max(0, int(round(right - prior_width)))
            elif right_missing and not left_missing:
                right = min(width - 1, int(round(left + prior_width)))
            rows.append((y, left, right, left_missing, right_missing))
            selected_mask[y, left : right + 1] = 255
            prior_center = 0.65 * prior_center + 0.35 * (0.5 * (left + right))
            prior_width = 0.8 * prior_width + 0.2 * max(1, right - left)

        left_points = [(left, y) for y, left, _right, _ll, _rl in rows]
        right_points = [(right, y) for y, _left, right, _ll, _rl in rows]
        centers = [(int(round((left + right) * 0.5)), y) for y, left, right, _ll, _rl in rows]
        left_lost = [left_lost for _y, _left, _right, left_lost, _right_lost in rows]
        right_lost = [right_lost for _y, _left, _right, _left_lost, right_lost in rows]
        return left_points, right_points, centers, left_lost, right_lost, selected_mask, left_branch_rows, right_branch_rows

    def _observe_right_fork(
        self,
        mask: np.ndarray,
        row_runs: Sequence[Sequence[tuple[int, int]]],
        left_points: Sequence[Tuple[int, int]],
        right_points: Sequence[Tuple[int, int]],
        left_lost: Sequence[bool],
        right_lost: Sequence[bool],
        right_branch_rows: Sequence[Tuple[int, int]],
    ) -> RightForkObservation:
        """Measure A/P points and the branch-return break without changing state."""

        _ = row_runs
        a_point, p_point = self._find_right_fork_ap_pair(
            mask,
            right_points=right_points,
            right_lost=right_lost,
        )
        junction_kind = self._classify_right_junction(a_point, right_branch_rows)
        if junction_kind is None:
            a_point = None
            p_point = None

        break_point = self._find_left_boundary_break(
            left_points,
            left_lost,
            mask.shape,
        )
        boundaries_valid = bool(left_points and right_points) and (
            sum(left_lost) < self.fork_min_lost_rows
            and sum(right_lost) < self.fork_min_lost_rows
        )
        return RightForkObservation(
            a_point=a_point,
            p_point=p_point,
            junction_kind=junction_kind,
            break_point=break_point,
            right_branch_points=tuple(right_branch_rows),
            boundaries_valid=boundaries_valid,
        )

    def _find_right_fork_ap_pair(
        self,
        mask: np.ndarray,
        right_points: Sequence[Tuple[int, int]],
        right_lost: Sequence[bool],
    ) -> tuple[Tuple[int, int] | None, Tuple[int, int] | None]:
        """Find a concave A point and an outward P point on the right boundary."""

        height, width = mask.shape[:2]
        prominence = max(6.0, width * self.fork_feature_prominence_ratio)
        min_y = int(round(height * self.fork_feature_min_y_ratio))
        max_y = int(round(height * self.fork_feature_max_y_ratio))
        side_margin = max(2, int(round(width * self.fork_feature_side_margin_ratio)))
        diagonal = math.hypot(width, height)
        min_distance = diagonal * self.fork_ap_min_distance_ratio
        max_distance = diagonal * self.fork_ap_max_distance_ratio

        p_candidates: list[tuple[Tuple[int, int], float]] = []
        span = min(
            self.fork_feature_span_rows,
            max(2, (len(right_points) - 1) // 2),
        )
        if len(right_points) >= span * 2 + 1:
            for index in range(span, len(right_points) - span):
                if (
                    right_lost[index]
                    or right_lost[index - span]
                    or right_lost[index + span]
                ):
                    continue
                x, y = right_points[index]
                if y < min_y or y > max_y or x >= width - 1 - side_margin:
                    continue
                neighbor_x = max(
                    right_points[index - span][0],
                    right_points[index + span][0],
                )
                outward = float(x - neighbor_x)
                if outward >= prominence:
                    p_candidates.append(((int(x), int(y)), outward))
        if not p_candidates:
            return None, None

        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            return None, None
        bottom_limit = int(round(height * 0.55))
        bottom_contours = [
            contour
            for contour in contours
            if cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] >= bottom_limit
        ]
        contour = max(bottom_contours or contours, key=cv2.contourArea)
        if len(contour) < 4:
            return None, None
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 4:
            return None, None
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return None, None

        best: tuple[float, Tuple[int, int], Tuple[int, int]] | None = None
        right_side_min_x = int(round(width * 0.35))
        for defect in defects[:, 0, :]:
            _start_index, _end_index, far_index, raw_depth = [int(value) for value in defect]
            depth = raw_depth / 256.0
            if depth < prominence:
                continue
            a_x, a_y = [int(value) for value in contour[far_index][0]]
            if (
                a_x < right_side_min_x
                or a_x <= side_margin
                or a_x >= width - 1 - side_margin
                or a_y < min_y
                or a_y > max_y
            ):
                continue
            for p_point, p_prominence in p_candidates:
                p_x, p_y = p_point
                if p_x <= a_x:
                    continue
                distance = math.hypot(p_x - a_x, p_y - a_y)
                if distance < min_distance or distance > max_distance:
                    continue
                score = depth + p_prominence - 0.05 * distance
                if best is None or score > best[0]:
                    best = (score, (a_x, a_y), p_point)
        if best is None:
            return None, None
        return best[1], best[2]

    def _classify_right_junction(
        self,
        a_point: Tuple[int, int] | None,
        right_branch_rows: Sequence[Tuple[int, int]],
    ) -> str | None:
        """Classify split/merge from the branch rows above or below A."""

        if a_point is None:
            return None
        a_y = int(a_point[1])
        above = sum(1 for _x, y in right_branch_rows if y < a_y - 1)
        below = sum(1 for _x, y in right_branch_rows if y > a_y + 1)
        total = above + below
        if total < self.fork_orientation_min_rows:
            return None
        if above >= self.fork_orientation_min_rows and above / total >= self.fork_orientation_vote_ratio:
            return "split"
        if below >= self.fork_orientation_min_rows and below / total >= self.fork_orientation_vote_ratio:
            return "merge"
        return None

    def _find_left_boundary_break(
        self,
        left_points: Sequence[Tuple[int, int]],
        left_lost: Sequence[bool],
        shape: Tuple[int, int],
    ) -> Tuple[int, int] | None:
        """Return the strongest measured internal jump on the left boundary."""

        height, width = shape[:2]
        threshold = max(8.0, width * self.fork_break_jump_ratio)
        min_y = int(round(height * self.fork_feature_min_y_ratio))
        max_y = int(round(height * self.fork_feature_max_y_ratio))
        side_margin = max(2, int(round(width * self.fork_feature_side_margin_ratio)))
        best: tuple[float, Tuple[int, int]] | None = None
        for index in range(len(left_points) - 1):
            if left_lost[index] or left_lost[index + 1]:
                continue
            near_x, near_y = left_points[index]
            far_x, far_y = left_points[index + 1]
            if abs(near_y - far_y) > 2:
                continue
            if near_y < min_y or near_y > max_y:
                continue
            if near_x <= side_margin or near_x >= width - 1 - side_margin:
                continue
            jump = abs(float(far_x - near_x))
            if jump >= threshold and (best is None or jump > best[0]):
                best = (jump, (int(near_x), int(near_y)))
        return None if best is None else best[1]

    def _update_right_fork_state(
        self,
        observation: RightForkObservation,
        route_direction: str | None,
        shape: Tuple[int, int],
    ) -> RightForkDecision:
        """Advance MAIN/JUNCTION/BRANCH/RETURNING in strict order."""

        _ = shape
        requested = self._normalize_route_direction(route_direction)
        has_junction = bool(
            observation.a_point is not None
            and observation.p_point is not None
            and observation.junction_kind in {"merge", "split"}
        )

        if self._fork_state == "MAIN":
            if has_junction:
                if self._fork_junction_kind != observation.junction_kind:
                    self._fork_junction_hits = 0
                self._fork_junction_kind = observation.junction_kind
                self._fork_junction_hits += 1
                self._fork_last_a_point = observation.a_point
                self._fork_last_p_point = observation.p_point
                self._fork_recovery_hits = 0
                if self._fork_junction_hits >= self.fork_confirm_frames:
                    self._fork_state = "JUNCTION"
                    self._fork_junction_misses = 0
                    self._fork_right_latched = False
            else:
                self._fork_junction_hits = 0
                self._fork_junction_kind = None
                if observation.break_point is not None:
                    self._fork_recovery_hits += 1
                    self._fork_last_break_point = observation.break_point
                    if self._fork_recovery_hits >= self.fork_recovery_confirm_frames:
                        self._fork_state = "RETURNING"
                        self._fork_return_clear_hits = 0
                else:
                    self._fork_recovery_hits = 0

        if self._fork_state == "JUNCTION":
            if has_junction and observation.junction_kind == self._fork_junction_kind:
                self._fork_junction_misses = 0
                self._fork_last_a_point = observation.a_point
                self._fork_last_p_point = observation.p_point
            else:
                self._fork_junction_misses += 1
            if self._fork_junction_kind == "split" and requested == "right":
                self._fork_right_latched = True
            if self._fork_junction_misses >= self.fork_release_frames:
                enter_branch = (
                    self._fork_junction_kind == "split" and self._fork_right_latched
                )
                self._fork_state = "BRANCH" if enter_branch else "MAIN"
                self._clear_junction_memory()

        elif self._fork_state == "BRANCH":
            if observation.break_point is not None:
                self._fork_break_hits += 1
                self._fork_last_break_point = observation.break_point
                if self._fork_break_hits >= self.fork_confirm_frames:
                    self._fork_state = "RETURNING"
                    self._fork_return_clear_hits = 0
            else:
                self._fork_break_hits = 0

        elif self._fork_state == "RETURNING":
            if observation.break_point is not None:
                self._fork_last_break_point = observation.break_point
                self._fork_return_clear_hits = 0
            elif observation.boundaries_valid:
                self._fork_return_clear_hits += 1
                if self._fork_return_clear_hits >= self.fork_return_release_frames:
                    self._fork_state = "MAIN"
                    self._fork_last_break_point = None
                    self._fork_break_hits = 0
                    self._fork_recovery_hits = 0
                    self._fork_return_clear_hits = 0
            else:
                self._fork_return_clear_hits = 0

        state = self._fork_state
        junction_kind = self._fork_junction_kind if state == "JUNCTION" else None
        selected_direction: str | None = None
        patch_kind: str | None = None
        base_route_direction: str | None = None
        reason = f"state={state.lower()}"
        if state == "JUNCTION":
            if junction_kind == "merge":
                selected_direction = "left"
                patch_kind = "main_ap"
                reason = "right branch merges; keep main with A-P"
            elif junction_kind == "split" and self._fork_right_latched:
                selected_direction = "right"
                patch_kind = "enter_branch"
                base_route_direction = "right"
                reason = "right split selected; connect bottom-left to P"
            elif junction_kind == "split":
                selected_direction = "left"
                patch_kind = "main_ap"
                reason = "right split visible; keep main with A-P"
        elif state == "BRANCH":
            reason = "right branch locked; waiting for left-boundary break"
        elif state == "RETURNING":
            patch_kind = "return_to_main"
            reason = "return break; connect B to ROI top-right"

        a_point = self._fork_last_a_point if state == "JUNCTION" else None
        p_point = self._fork_last_p_point if state == "JUNCTION" else None
        break_point = self._fork_last_break_point if state == "RETURNING" else None
        result = ForkLaneResult(
            fork_detected=state == "JUNCTION" and junction_kind == "split",
            requested_direction=requested,
            selected_direction=selected_direction,
            left_points=[],
            right_points=list(observation.right_branch_points),
            reason=reason,
            left_detected=False,
            right_detected=state == "JUNCTION",
            left_corner=None,
            right_corner=p_point,
            confirm_frames=self._fork_junction_hits,
            state=state,
            junction_kind=junction_kind,
            a_point=a_point,
            p_point=p_point,
            break_point=break_point,
        )
        return RightForkDecision(
            result=result,
            patch_kind=patch_kind,
            base_route_direction=base_route_direction,
        )

    def _clear_junction_memory(self) -> None:
        self._fork_junction_kind = None
        self._fork_junction_hits = 0
        self._fork_junction_misses = 0
        self._fork_right_latched = False
        self._fork_last_a_point = None
        self._fork_last_p_point = None

    @staticmethod
    def _lowest_valid_boundary_point(
        points: Sequence[Tuple[int, int]],
        lost_flags: Sequence[bool],
    ) -> Tuple[int, int] | None:
        valid = [
            (int(x), int(y))
            for (x, y), lost in zip(points, lost_flags)
            if not lost
        ]
        return max(valid, key=lambda point: point[1]) if valid else None

    def _apply_virtual_boundary_patch(
        self,
        left_points: Sequence[Tuple[int, int]],
        right_points: Sequence[Tuple[int, int]],
        side: str,
        start: Tuple[int, int],
        end: Tuple[int, int],
        shape: Tuple[int, int],
    ) -> tuple[
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        np.ndarray,
    ]:
        """Replace one boundary with a two-point virtual line, row by row."""

        height, width = shape[:2]
        left_by_y = {int(y): int(x) for x, y in left_points}
        right_by_y = {int(y): int(x) for x, y in right_points}
        y_start, y_end = int(start[1]), int(end[1])
        y_min, y_max = sorted((y_start, y_end))
        denominator = float(y_end - y_start)
        if abs(denominator) < 1.0:
            denominator = 1.0
        for y in sorted(set(left_by_y).intersection(right_by_y)):
            if y < y_min or y > y_max:
                continue
            ratio = (float(y) - float(y_start)) / denominator
            x = int(round(float(start[0]) + ratio * (float(end[0]) - float(start[0]))))
            if side == "left":
                left_by_y[y] = int(clamp(x, 0, max(0, right_by_y[y] - self.min_run_width_px)))
            elif side == "right":
                right_by_y[y] = int(clamp(x, left_by_y[y] + self.min_run_width_px, width - 1))
            else:
                raise ValueError(f"unsupported virtual boundary side: {side}")

        rows: list[tuple[int, int, int]] = []
        selected_mask = np.zeros((height, width), dtype=np.uint8)
        for y in sorted(set(left_by_y).intersection(right_by_y), reverse=True):
            left = int(clamp(left_by_y[y], 0, width - 1))
            right = int(clamp(right_by_y[y], left + 1, width - 1))
            if right <= left:
                continue
            rows.append((y, left, right))
            selected_mask[y, left : right + 1] = 255
        patched_left = [(left, y) for y, left, _right in rows]
        patched_right = [(right, y) for y, _left, right in rows]
        centers = [(int(round((left + right) * 0.5)), y) for y, left, right in rows]
        return patched_left, patched_right, centers, selected_mask

    def _smooth_article_centerline(
        self, raw_points: Sequence[Tuple[int, int]], width: int
    ) -> List[Tuple[int, int]]:
        if not raw_points:
            return []
        jump = max(1.0, width * self.gradient_jump_ratio)
        step = max(1.0, width * self.gradient_step_ratio)
        limited: list[tuple[float, int]] = []
        previous_x = float(raw_points[0][0])
        for x, y in raw_points:
            current_x = float(x)
            delta = current_x - previous_x
            if abs(delta) > jump:
                current_x = previous_x + math.copysign(step, delta)
            limited.append((current_x, y))
            previous_x = current_x
        smoothed: list[tuple[int, int]] = []
        for index, (_x, y) in enumerate(limited):
            start = max(0, index - 2)
            end = min(len(limited), index + 3)
            mean_x = sum(item[0] for item in limited[start:end]) / float(end - start)
            if index % max(1, self.scan_step) == 0:
                smoothed.append((int(round(mean_x)), y))
        return smoothed

    def _article_metrics(self, centerline_points, fit_coeffs, shape):
        height, width = shape[:2]
        if not centerline_points:
            return 0.0, 0.0, 0.0
        weighted_sum = 0.0
        weight_sum = 0.0
        for x, y in centerline_points:
            index = int(round((y / max(1, height - 1)) * (len(ARTICLE_ROW_WEIGHTS) - 1)))
            weight = float(ARTICLE_ROW_WEIGHTS[index])
            weighted_sum += float(x) * weight
            weight_sum += weight
        weighted_center = weighted_sum / weight_sum if weight_sum > 0 else float(centerline_points[0][0])
        self._weighted_center_history.append(weighted_center)
        self._weighted_center_history = self._weighted_center_history[-3:]
        if len(self._weighted_center_history) == 3:
            weights = self.temporal_weights
            total = sum(weights)
            weighted_center = sum(value * weight for value, weight in zip(self._weighted_center_history, weights)) / total
        ordered = sorted(centerline_points, key=lambda point: point[1], reverse=True)
        bottom_x, bottom_y = ordered[0]
        lookahead_index = min(len(ordered) - 1, max(1, int(len(ordered) * self.lookahead_ratio)))
        lookahead_x, lookahead_y = ordered[lookahead_index]
        heading = math.degrees(math.atan2(lookahead_x - bottom_x, max(1, bottom_y - lookahead_y)))
        return float(weighted_center - width * 0.5), float(heading), float(compute_curvature(fit_coeffs, bottom_y))

    def _geometric_fork_result(
        self, left_points, right_points, left_lost, right_lost,
        left_branch_rows, right_branch_rows, shape,
    ) -> ForkLaneResult:
        height, width = shape[:2]
        span = max(2, self.fork_corner_span_rows)
        threshold = max(6.0, width * self.fork_outward_jump_ratio)

        min_corner_y = int(height * self.fork_corner_min_y_ratio)
        max_corner_y = int(height * self.fork_corner_max_y_ratio)
        side_margin = max(2, int(round(width * self.fork_corner_side_margin_ratio)))

        def outward_corner(points, lost_flags, side):
            for index in range(span, len(points) - span):
                x, y = points[index]
                # A boundary reconstructed after touching the ROI edge is not a
                # measured corner.  Using it here creates LF/RF markers at the
                # lower image edges when the lane leaves the camera view.
                if lost_flags[index] or lost_flags[index - span] or lost_flags[index + span]:
                    continue
                if y < min_corner_y or y > max_corner_y:
                    continue
                if x <= side_margin or x >= width - 1 - side_margin:
                    continue
                near_x = points[index - span][0]
                far_x = points[index + span][0]
                outward = (x < near_x - threshold and x <= far_x + threshold * 0.5) if side == "left" else (x > near_x + threshold and x >= far_x - threshold * 0.5)
                if outward:
                    return (x, y)
            return None

        left_corner = outward_corner(left_points, left_lost, "left")
        right_corner = outward_corner(right_points, right_lost, "right")
        min_branch_rows = max(3, self.fork_min_lost_rows)
        left_raw = len(left_branch_rows) >= min_branch_rows or (
            left_corner is not None and sum(left_lost) >= self.fork_min_lost_rows
        )
        right_raw = len(right_branch_rows) >= min_branch_rows or (
            right_corner is not None and sum(right_lost) >= self.fork_min_lost_rows
        )
        self._left_fork_active, self._left_fork_hits, self._left_fork_misses = self._debounce_fork(
            left_raw, self._left_fork_active, self._left_fork_hits, self._left_fork_misses
        )
        self._right_fork_active, self._right_fork_hits, self._right_fork_misses = self._debounce_fork(
            right_raw, self._right_fork_active, self._right_fork_hits, self._right_fork_misses
        )
        reason = f"geometry left={self._left_fork_active} right={self._right_fork_active}"
        return ForkLaneResult(
            fork_detected=self._left_fork_active or self._right_fork_active,
            requested_direction=None,
            selected_direction=None,
            left_points=list(left_branch_rows),
            right_points=list(right_branch_rows),
            reason=reason,
            left_detected=self._left_fork_active,
            right_detected=self._right_fork_active,
            left_corner=left_corner,
            right_corner=right_corner,
            confirm_frames=max(self._left_fork_hits, self._right_fork_hits),
        )

    def _debounce_fork(self, raw, active, hits, misses):
        if raw:
            hits += 1
            misses = 0
            if hits >= self.fork_confirm_frames:
                active = True
        else:
            hits = 0
            misses += 1
            if misses >= self.fork_release_frames:
                active = False
        return active, hits, misses

    def _segment_lane(self, roi_frame: np.ndarray) -> np.ndarray:
        """根据颜色空间阈值提取蓝色航道候选区域。

        输入:
            roi_frame: 预处理后的 ROI BGR 图像。

        输出:
            返回单通道二值掩膜，非零像素表示蓝色候选区域。
        """

        if self.color_space == "lab":
            converted = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
            lower = np.asarray(self.lab_config.get("lower", [20, 120, 80]), dtype=np.uint8)
            upper = np.asarray(self.lab_config.get("upper", [255, 170, 135]), dtype=np.uint8)
        else:
            converted = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
            lower = np.asarray(self.hsv_config.get("lower", [85, 70, 40]), dtype=np.uint8)
            upper = np.asarray(self.hsv_config.get("upper", [140, 255, 255]), dtype=np.uint8)

        mask = cv2.inRange(converted, lower, upper)
        return self._apply_morphology(mask)

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """对二值掩膜执行开闭运算去噪与补洞。

        输入:
            mask: 初始二值掩膜。

        输出:
            返回形态学处理后的掩膜。
        """

        open_kernel_size = max(1, int(self.morphology_config.get("open_kernel", 3)))
        close_kernel_size = max(1, int(self.morphology_config.get("close_kernel", 7)))
        erode_iterations = int(self.morphology_config.get("erode_iterations", 0))
        dilate_iterations = int(self.morphology_config.get("dilate_iterations", 1))

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_size, open_kernel_size))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))

        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        if erode_iterations > 0:
            cleaned = cv2.erode(cleaned, open_kernel, iterations=erode_iterations)
        if dilate_iterations > 0:
            cleaned = cv2.dilate(cleaned, close_kernel, iterations=dilate_iterations)
        return cleaned

    def _suppress_environment_blue(self, mask: np.ndarray) -> np.ndarray:
        """抑制明显不属于地面航道的蓝色环境干扰。

        输入:
            mask: 颜色阈值分割并完成形态学处理后的二值掩膜。

        输出:
            返回去除了高位紧凑蓝色干扰后的二值掩膜。
        """

        if mask.size == 0:
            return mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered_mask = mask.copy()

        for label in range(1, num_labels):
            x, y, component_width, component_height, area = stats[label]
            if self._is_high_compact_environment_component(
                x=int(x),
                y=int(y),
                width=int(component_width),
                height=int(component_height),
                area=int(area),
                shape=mask.shape,
            ):
                filtered_mask[labels == label] = 0

        return filtered_mask

    def _is_high_compact_environment_component(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        area: int,
        shape: Tuple[int, int],
    ) -> bool:
        """判断一个蓝色连通域是否更像墙面广告牌等环境干扰。

        输入:
            x: 连通域包围框左上角 x 坐标。
            y: 连通域包围框左上角 y 坐标。
            width: 连通域包围框宽度。
            height: 连通域包围框高度。
            area: 连通域面积。
            shape: 当前 ROI 尺寸，格式为 (高, 宽)。

        输出:
            如果该连通域明显属于高位紧凑干扰，则返回 True，否则返回 False。
        """

        image_height, image_width = shape[:2]
        if width <= 0 or height <= 0 or area <= 0:
            return False

        aspect_ratio = safe_divide(float(width), float(height), default=999.0)
        near_top = y <= int(image_height * self.high_compact_top_ratio)
        near_side = (
            x <= self.high_compact_side_margin_px
            or (x + width) >= (image_width - self.high_compact_side_margin_px)
        )
        compact_shape = aspect_ratio <= self.compact_aspect_ratio_threshold
        small_to_mid_area = area <= self.high_compact_max_area

        return near_top and near_side and compact_shape and small_to_mid_area

    def _extract_candidate_components(
        self,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, List[LaneComponent]]:
        """提取所有满足基础面积条件的蓝色候选连通域。

        输入:
            mask: 形态学处理后的二值掩膜。

        输出:
            返回二元组 (labels, components)，其中 labels 为连通域标签图，
            components 为所有候选连通域的几何信息列表。
        """

        height, width = mask.shape[:2]
        min_area = int(self.component_config.get("min_area", 250))
        min_height = int(self.component_config.get("min_height", 12))
        side_margin = int(self.centerline_config.get("single_side_edge_margin_px", 28))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components: List[LaneComponent] = []
        for label in range(1, num_labels):
            x, y, component_width, component_height, area = stats[label]
            if area < min_area or component_height < min_height:
                continue
            component_mask = labels == label
            ys, xs = np.where(component_mask)
            if xs.size == 0 or ys.size == 0:
                continue
            centroid_x = float(np.mean(xs))
            centroid_y = float(np.mean(ys))
            bottom_y = int(np.max(ys))
            touches_side = x <= side_margin or (x + component_width) >= (width - side_margin)
            anchor_samples = self._build_component_anchor_samples(
                component_mask=component_mask,
                top_y=int(y),
                height=int(component_height),
                fallback_x=centroid_x,
            )
            if not anchor_samples:
                continue
            components.append(
                LaneComponent(
                    label=label,
                    x=int(x),
                    y=int(y),
                    width=int(component_width),
                    height=int(component_height),
                    area=int(area),
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    bottom_y=bottom_y,
                    touches_side=touches_side,
                    anchor_samples=anchor_samples,
                )
            )

        return labels, components

    def _build_component_anchor_samples(
        self,
        component_mask: np.ndarray,
        top_y: int,
        height: int,
        fallback_x: float,
    ) -> Tuple[LaneAnchorSample, ...]:
        """在单个连通域内部采样多个候选锚点。

        输入:
            component_mask: 当前连通域的布尔掩膜。
            top_y: 连通域包围框顶部 y 坐标。
            height: 连通域高度。
            fallback_x: 当某一采样行为空时使用的回退横坐标。

        输出:
            返回候选锚点元组，每个锚点包含横坐标、纵坐标和该行宽度。
        """

        samples: List[LaneAnchorSample] = []
        bottom_y = component_mask.shape[0] - 1

        for ratio in self.anchor_sample_ratios:
            sample_y = top_y + int(height * ratio)
            sample_y = int(clamp(float(sample_y), float(top_y), float(bottom_y)))
            row_indices = np.where(component_mask[sample_y])[0]
            if row_indices.size > 0:
                sample_x = float(row_indices[0] + row_indices[-1]) * 0.5
                sample_width = int(row_indices[-1] - row_indices[0] + 1)
            else:
                sample_x = fallback_x
                sample_width = max(1, int(round(self.last_lane_width_px)))

            samples.append(
                LaneAnchorSample(
                    x=float(sample_x),
                    y=int(sample_y),
                    width=int(sample_width),
                )
            )

        return tuple(samples)

    def _choose_component_anchor(
        self,
        component: LaneComponent,
        predicted_x: float,
    ) -> LaneAnchorSample:
        """从组件候选锚点中挑出最贴合当前航向预测的一点。

        输入:
            component: 待评估的连通域组件。
            predicted_x: 当前帧根据历史或链条趋势预测的横向位置。

        输出:
            返回一个最适合作为中心线支撑点的锚点。
        """

        best_sample = component.anchor_samples[0]
        best_score = float("inf")

        for sample in component.anchor_samples:
            score = abs(sample.x - predicted_x)
            score += sample.width * self.anchor_width_weight
            if score < best_score:
                best_score = score
                best_sample = sample

        return best_sample

    def _compute_environment_penalty(
        self,
        component: LaneComponent,
        anchor: LaneAnchorSample,
        current_anchor: LaneAnchorSample,
        chain_length: int,
        shape: Tuple[int, int],
    ) -> float:
        """????????????????????????
        ??:
            component: ????????????
            anchor: ?????????????????
            current_anchor: ????????????
            chain_length: ????????????
            shape: ?? ROI ?????? (?, ?)?
        ??:
            ??????????????????????????
        """

        if chain_length < self.environment_guard_min_chain_length:
            return 0.0

        top_limit = int(shape[0] * self.environment_guard_top_ratio)
        if anchor.y > top_limit:
            return 0.0

        aspect_ratio = safe_divide(float(component.width), float(component.height), default=999.0)
        width_ratio = safe_divide(float(anchor.width), float(max(current_anchor.width, 1)), default=1.0)

        penalty = 0.0
        # ??????/??????????????????????????
        if aspect_ratio <= self.environment_compact_aspect_ratio_max:
            penalty += 42.0
        # ????????????????????????????????
        if component.area <= self.environment_small_area_max and width_ratio <= self.environment_small_width_ratio:
            penalty += 38.0
        if component.touches_side:
            penalty += self.environment_side_penalty
        return penalty

    def _select_main_lane_components(
        self,
        components: Sequence[LaneComponent],
        shape: Tuple[int, int],
    ) -> List[LaneComponent]:
        """从所有蓝色候选块中挑出沿地面主方向连续排列的一串主航道箭头。

        输入:
            components: 候选连通域列表。
            shape: 当前 ROI 尺寸，格式为 (高, 宽)。

        输出:
            返回按从下往上排序的主航道组件列表。
        """

        if not components:
            return []

        height, width = shape[:2]
        image_center = width * 0.5
        remaining = sorted(components, key=lambda item: item.bottom_y, reverse=True)

        start_candidates = [
            component
            for component in remaining
            if component.bottom_y >= int(height * self.start_bottom_ratio)
        ]
        if not start_candidates:
            start_candidates = remaining

        best_start_score = -1e9
        best_start = start_candidates[0]
        best_start_anchor = self._choose_component_anchor(best_start, image_center)
        for component in start_candidates:
            target_y = component.anchor_samples[len(component.anchor_samples) // 2].y
            predicted_x = self._predict_chain_x([], target_y, image_center)
            anchor = self._choose_component_anchor(component, predicted_x)

            score = component.bottom_y * 1.35
            score += min(component.area, 5000) * 0.02
            score -= abs(anchor.x - predicted_x) * 0.95
            score -= anchor.width * 0.05
            if component.touches_side:
                score -= 80.0
            if score > best_start_score:
                best_start_score = score
                best_start = component
                best_start_anchor = anchor

        chain: List[LaneComponent] = [best_start]
        chain_points: List[Tuple[float, float]] = [(best_start_anchor.x, float(best_start_anchor.y))]
        chain_anchors: List[LaneAnchorSample] = [best_start_anchor]
        used_labels = {best_start.label}

        while True:
            current_x, current_y = chain_points[-1]
            current_anchor = chain_anchors[-1]
            best_next: Optional[LaneComponent] = None
            best_next_anchor: Optional[LaneAnchorSample] = None
            best_next_score = -1e9

            for component in remaining:
                if component.label in used_labels:
                    continue

                target_y = component.anchor_samples[len(component.anchor_samples) // 2].y
                predicted_x = self._predict_chain_x(chain_points, float(target_y), image_center)
                anchor = self._choose_component_anchor(component, predicted_x)
                delta_y = current_y - anchor.y
                if delta_y <= self.scan_step:
                    continue
                if delta_y > self.max_component_gap_px:
                    continue

                delta_x = abs(anchor.x - predicted_x)
                max_allowed_dx = self.max_component_dx_px + max(0.0, delta_y - 80.0) * 0.45
                if delta_x > max_allowed_dx:
                    continue
                # 如果两个组件在纵向上几乎贴在一起，但横向却猛地跳到一边，
                # 往往是连到了墙面广告牌或赛道外蓝色物体，而不是继续沿地面航道前进。
                if delta_y < self.short_gap_y_threshold and delta_x > self.short_gap_max_dx_px:
                    continue
                # 远处组件通常不会突然比前一个组件“宽出很多”，
                # 一旦高处宽度暴增，常见原因就是接到了墙面蓝色标牌。
                if (
                    anchor.y <= int(shape[0] * self.width_growth_guard_top_ratio)
                    and anchor.width > current_anchor.width * self.max_far_width_growth_ratio
                ):
                    continue
                environment_penalty = self._compute_environment_penalty(
                    component=component,
                    anchor=anchor,
                    current_anchor=current_anchor,
                    chain_length=len(chain),
                    shape=shape,
                )
                # 明显更像环境蓝块时，直接停止把它接进主航道链。
                if environment_penalty >= 75.0:
                    continue

                score = 220.0
                score -= delta_x * 1.08
                score -= abs(delta_y - 72.0) * 0.18
                score += min(component.area, 4000) * 0.01
                score -= anchor.width * 0.035
                score -= environment_penalty
                if component.touches_side:
                    score -= 55.0

                if score > best_next_score:
                    best_next_score = score
                    best_next = component
                    best_next_anchor = anchor

            if best_next is None or best_next_anchor is None:
                break
            if best_next_score < self.min_component_chain_score:
                break

            chain.append(best_next)
            chain_points.append((best_next_anchor.x, float(best_next_anchor.y)))
            chain_anchors.append(best_next_anchor)
            used_labels.add(best_next.label)
            if len(chain) >= 12:
                break

        return chain

    def _select_components_for_route(
        self,
        components: Sequence[LaneComponent],
        shape: Tuple[int, int],
        route_direction: str | None,
    ) -> RouteComponentSelection:
        base_chain = self._select_main_lane_components(components, shape)
        left_chain: List[LaneComponent] = list(base_chain)
        right_chain: List[LaneComponent] = []
        if base_chain:
            left_chain, right_chain = self._build_fork_branch_chains(
                components=components,
                base_chain=base_chain,
                shape=shape,
            )

        left_points = (
            self._build_support_points_from_components(left_chain, shape)
            if right_chain
            else []
        )
        right_points = self._build_support_points_from_components(right_chain, shape)
        fork_detected = bool(right_chain)
        requested = self._normalize_route_direction(route_direction)
        selected_direction: str | None = None
        selected_chain = base_chain
        reason = "no fork candidate"

        if fork_detected:
            reason = "right fork candidate ready"
            if requested == "left" and left_chain:
                selected_chain = left_chain
                selected_direction = "left"
                reason = "selected current branch"
            elif requested == "right" and right_chain:
                selected_chain = right_chain
                selected_direction = "right"
                reason = "selected right branch"
            elif requested is not None:
                reason = f"requested {requested} but branch missing"
        elif requested == "right":
            reason = "waiting right branch"
        elif requested == "left" and left_chain:
            selected_direction = "left"
            reason = "selected current branch"

        return RouteComponentSelection(
            selected_components=selected_chain,
            fork_result=ForkLaneResult(
                fork_detected=fork_detected,
                requested_direction=requested,
                selected_direction=selected_direction,
                left_points=left_points,
                right_points=right_points,
                reason=reason,
            ),
        )

    def _build_fork_branch_chains(
        self,
        components: Sequence[LaneComponent],
        base_chain: Sequence[LaneComponent],
        shape: Tuple[int, int],
    ) -> tuple[List[LaneComponent], List[LaneComponent]]:
        if not base_chain:
            return [], []

        _, width = shape[:2]
        image_center = width * 0.5
        base_labels = {component.label for component in base_chain}
        base_points = self._component_chain_points(base_chain, image_center)
        trunk = self._fork_trunk_components(base_chain)
        trunk_points = self._component_chain_points(trunk, image_center)
        current_x, current_y = trunk_points[-1]
        base_tip = base_chain[-1]
        base_tip_anchor = self._choose_component_anchor(base_tip, current_x)
        base_tip_side: str | None = None
        if base_tip.label not in {component.label for component in trunk}:
            base_delta = base_tip_anchor.x - current_x
            if abs(base_delta) >= max(18.0, self.last_lane_width_px * 0.35):
                base_tip_side = "left" if base_delta < 0 else "right"

        candidates: list[tuple[str, int, LaneComponent, LaneAnchorSample, float]] = []
        for component in components:
            if component.label in base_labels:
                continue
            best_component_candidate = self._best_component_branch_candidate(
                component=component,
                base_chain=base_chain,
                base_points=base_points,
                image_center=image_center,
                shape=shape,
                base_tip_anchor=base_tip_anchor,
            )
            if best_component_candidate is not None:
                candidates.append(best_component_candidate)

        right = self._best_branch_candidate(candidates, "right")
        right_chain: List[LaneComponent] = []
        if base_tip_side == "right":
            right_chain = list(base_chain)
        elif right is not None:
            split_index, right_component = right
            right_chain = list(base_chain[: split_index + 1]) + [right_component]

        left_chain = list(base_chain)
        return left_chain, right_chain

    def _best_component_branch_candidate(
        self,
        component: LaneComponent,
        base_chain: Sequence[LaneComponent],
        base_points: Sequence[Tuple[float, float]],
        image_center: float,
        shape: Tuple[int, int],
        base_tip_anchor: LaneAnchorSample,
    ) -> tuple[str, int, LaneComponent, LaneAnchorSample, float] | None:
        best: tuple[str, int, LaneComponent, LaneAnchorSample, float] | None = None
        target_y = component.anchor_samples[len(component.anchor_samples) // 2].y

        for split_index, (current_x, current_y) in enumerate(base_points):
            trunk_points = base_points[: split_index + 1]
            predicted_x = self._predict_chain_x(trunk_points, float(target_y), image_center)
            anchor = self._choose_component_anchor(component, predicted_x)
            delta_y = current_y - anchor.y
            if delta_y < -self.short_gap_y_threshold or delta_y > self.max_component_gap_px:
                continue

            delta_from_current = anchor.x - current_x
            min_branch_dx = max(14.0, self.last_lane_width_px * 0.25)
            if abs(delta_from_current) < min_branch_dx:
                continue

            side = "left" if delta_from_current < 0 else "right"
            delta_from_prediction = abs(anchor.x - predicted_x)
            max_allowed_dx = self.max_component_dx_px + max(0.0, delta_y - 80.0) * 0.45
            if side == "right":
                max_allowed_dx *= 1.45
            if delta_from_prediction > max_allowed_dx:
                continue
            if (
                side != "right"
                and delta_y < self.short_gap_y_threshold
                and delta_from_prediction > self.short_gap_max_dx_px
            ):
                continue
            if (
                anchor.y <= int(shape[0] * self.width_growth_guard_top_ratio)
                and anchor.width > base_tip_anchor.width * self.max_far_width_growth_ratio
            ):
                continue

            score = 220.0
            score -= delta_from_prediction * (0.42 if side == "right" else 0.65)
            score -= abs(delta_y - 72.0) * 0.10
            score += min(component.area, 4000) * 0.01
            score -= anchor.width * 0.02
            score += split_index * 4.0
            if side == "right":
                score += max(0.0, delta_from_current) * 0.06

            candidate = (side, split_index, component, anchor, score)
            if best is None or score > best[4]:
                best = candidate

        return best

    def _fork_trunk_components(self, base_chain: Sequence[LaneComponent]) -> List[LaneComponent]:
        if len(base_chain) <= 1:
            return list(base_chain)
        return list(base_chain[:-1])

    def _component_chain_points(
        self,
        components: Sequence[LaneComponent],
        image_center: float,
    ) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for component in components:
            target_y = component.anchor_samples[len(component.anchor_samples) // 2].y
            predicted_x = self._predict_chain_x(points, float(target_y), image_center)
            anchor = self._choose_component_anchor(component, predicted_x)
            points.append((anchor.x, float(anchor.y)))
        return points or [(image_center, 0.0)]

    def _best_branch_candidate(
        self,
        candidates: Sequence[tuple[str, int, LaneComponent, LaneAnchorSample, float]],
        side: str,
    ) -> tuple[int, LaneComponent] | None:
        best: tuple[float, int, LaneComponent] | None = None
        for candidate_side, split_index, component, _anchor, score in candidates:
            if candidate_side != side:
                continue
            if best is None or score > best[0]:
                best = (score, split_index, component)
        return (best[1], best[2]) if best is not None else None

    def _normalize_route_direction(self, route_direction: str | None) -> str | None:
        if route_direction is None:
            return None
        direction = str(route_direction).casefold()
        if direction in ("left", "right"):
            return direction
        return None

    def _predict_chain_x(
        self,
        chain_points: Sequence[Tuple[float, float]],
        target_y: float,
        image_center: Optional[float] = None,
    ) -> float:
        """根据历史链条走势预测目标高度处的横向位置。

        输入:
            chain_points: 已经选中的主航道锚点链。
            target_y: 目标纵坐标。
            image_center: 当前 ROI 的中心横坐标，作为没有历史时的回退值。

        输出:
            返回预测的横向位置。
        """

        history_prediction: Optional[float] = None
        if self.last_fit_coeffs is not None:
            history_prediction = float(evaluate_poly(self.last_fit_coeffs, [target_y])[0])

        if len(chain_points) >= 2:
            x1, y1 = chain_points[-2]
            x2, y2 = chain_points[-1]
            if abs(y2 - y1) > 1e-3:
                slope = (x2 - x1) / (y2 - y1)
                chain_prediction = float(x2 + slope * (target_y - y2))
            else:
                chain_prediction = float(x2)
        elif chain_points:
            chain_prediction = float(chain_points[-1][0])
        else:
            chain_prediction = float(image_center if image_center is not None else 0.0)

        if history_prediction is None:
            return chain_prediction
        if not chain_points:
            return history_prediction

        return float(
            chain_prediction * (1.0 - self.prediction_blend)
            + history_prediction * self.prediction_blend
        )

    def _build_mask_from_components(
        self,
        labels: np.ndarray,
        components: Sequence[LaneComponent],
    ) -> np.ndarray:
        """根据选中的组件列表重新构造主航道掩膜。

        输入:
            labels: 连通域标签图。
            components: 需要保留的组件列表。

        输出:
            返回只包含主航道组件的二值掩膜。
        """

        mask = np.zeros_like(labels, dtype=np.uint8)
        for component in components:
            mask[labels == component.label] = 255
        return mask

    def _build_support_points_from_components(
        self,
        components: Sequence[LaneComponent],
        shape: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """从主航道组件链中提取用于拟合的支撑点。

        输入:
            components: 按从下往上排序的主航道组件链。
            shape: 当前 ROI 尺寸，格式为 (高, 宽)。

        输出:
            返回用于拟合中心线的支撑点列表，顺序为从下往上。
        """

        if not components:
            return []

        _, width = shape[:2]
        image_center = width * 0.5
        support_points: List[Tuple[int, int]] = []

        for component in components:
            target_y = component.anchor_samples[len(component.anchor_samples) // 2].y
            predicted_x = self._predict_chain_x(
                [(float(x), float(y)) for x, y in support_points],
                float(target_y),
                image_center,
            )
            anchor = self._choose_component_anchor(component, predicted_x)
            support_points.append((int(round(anchor.x)), int(anchor.y)))

        return self._filter_centerline_points(support_points)

    def _build_polyline_centerline(
        self,
        support_points: Sequence[Tuple[int, int]],
        shape: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """根据支撑点链做折线插值，生成稳定的中心线点集。

        输入:
            support_points: 按从下往上排序的支撑点列表。
            shape: 当前 ROI 尺寸，格式为 (高, 宽)。

        输出:
            返回经过折线插值和轻度平滑后的中心线点集。
        """

        if not support_points:
            return []
        if len(support_points) == 1:
            return list(support_points)

        _, width = shape[:2]
        ordered_points = sorted(support_points, key=lambda item: item[1], reverse=True)
        polyline_points: List[Tuple[int, int]] = []

        for index in range(len(ordered_points) - 1):
            x1, y1 = ordered_points[index]
            x2, y2 = ordered_points[index + 1]
            if y1 == y2:
                continue

            for y in range(y1, y2, -self.scan_step):
                ratio = safe_divide(float(y - y1), float(y2 - y1), default=0.0)
                x = x1 + (x2 - x1) * ratio
                polyline_points.append((int(round(clamp(x, 0.0, width - 1.0))), int(y)))

        polyline_points.append(ordered_points[-1])
        return self._smooth_polyline_points(polyline_points, width)

    def _smooth_polyline_points(
        self,
        points: Sequence[Tuple[int, int]],
        image_width: int,
    ) -> List[Tuple[int, int]]:
        """对折线中心线做轻度一维平滑，减少局部小抖动。

        输入:
            points: 折线插值后的中心线点集。
            image_width: 当前 ROI 宽度。

        输出:
            返回轻度平滑后的中心线点集。
        """

        if len(points) <= 2:
            return list(points)

        x_values = np.asarray([point[0] for point in points], dtype=np.float32)
        smoothed_x = x_values.copy()
        for index in range(1, len(points) - 1):
            smoothed_x[index] = (
                x_values[index - 1] * 0.25
                + x_values[index] * 0.50
                + x_values[index + 1] * 0.25
            )

        return [
            (int(round(clamp(float(smoothed_x[index]), 0.0, image_width - 1.0))), int(point[1]))
            for index, point in enumerate(points)
        ]

    def _estimate_lane_width_from_components(
        self,
        components: Sequence[LaneComponent],
    ) -> float:
        """根据主航道组件链估计航道宽度，用于置信度和备用逻辑。

        输入:
            components: 按从下往上排序的主航道组件链。

        输出:
            返回估计的航道宽度像素值。
        """

        if not components:
            return self.last_lane_width_px

        widths = [
            min(sample.width for sample in component.anchor_samples)
            for component in components
        ]
        lane_width_px = float(np.median(np.asarray(widths, dtype=np.float32)))
        return max(lane_width_px, self.min_lane_width_px)

    def _extract_centerline(self, mask: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
        """按行扫描主航道掩膜，提取中心线原始点集。

        输入:
            mask: 连通域筛选后的主航道掩膜。

        输出:
            返回二元组 (raw_points, lane_width_px)，分别是原始中心线点集与估计航道宽度。
        """

        height, width = mask.shape[:2]
        raw_points: List[Tuple[int, int]] = []
        widths: List[float] = []
        previous_center: Optional[float] = None

        for y in range(height - 1, -1, -self.scan_step):
            row = mask[y]
            segments = self._find_row_segments(row)
            if not segments:
                continue

            # 优先参考上一帧拟合结果，帮助当前行在遮挡、断裂时少跑偏。
            predicted_center: Optional[float] = None
            if self.last_fit_coeffs is not None:
                predicted_center = float(evaluate_poly(self.last_fit_coeffs, [y])[0])
            elif previous_center is not None:
                predicted_center = previous_center

            start, end = self._select_best_segment(segments, predicted_center, width)
            segment_width = float(end - start + 1)
            segment_center = (start + end) * 0.5
            estimated_center = segment_center
            width_for_stat = segment_width

            # 单边推中心只适合“真正只露出一侧边界”的情况。
            # 对现在这种块状蓝白航道，误用它会把中心线强行推到航道外侧，所以默认关闭。
            infer_threshold = max(self.min_lane_width_px, self.last_lane_width_px * self.single_side_infer_ratio)
            touches_border = (
                start <= self.single_side_edge_margin_px
                or end >= width - 1 - self.single_side_edge_margin_px
            )
            if (
                self.enable_single_side_inference
                and predicted_center is not None
                and segment_width < infer_threshold
                and touches_border
            ):
                if segment_center < predicted_center - self.last_lane_width_px * 0.15:
                    estimated_center = end + self.last_lane_width_px * 0.5
                    width_for_stat = self.last_lane_width_px
                elif segment_center > predicted_center + self.last_lane_width_px * 0.15:
                    estimated_center = start - self.last_lane_width_px * 0.5
                    width_for_stat = self.last_lane_width_px

            # 当前观测值和历史预测值做一点融合，减少中心线跳动。
            if predicted_center is not None:
                estimated_center = (
                    estimated_center * (1.0 - self.prediction_blend)
                    + predicted_center * self.prediction_blend
                )

            estimated_center = clamp(estimated_center, 0.0, width - 1.0)
            raw_points.append((int(round(estimated_center)), int(y)))
            widths.append(float(width_for_stat))
            previous_center = estimated_center

        if not widths:
            return raw_points, self.last_lane_width_px

        lane_width_px = float(np.median(np.asarray(widths, dtype=np.float32)))
        return raw_points, max(lane_width_px, self.min_lane_width_px)

    def _filter_centerline_points(
        self,
        raw_points: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """根据行间连续性筛掉容易导致拟合翻折的离群点。

        输入:
            raw_points: 原始中心线点集，顺序为从下往上。

        输出:
            返回经过连续性筛选后的稳定点集。
        """

        if len(raw_points) <= 2:
            return list(raw_points)

        filtered_points: List[Tuple[int, int]] = []
        previous_x: Optional[float] = None
        max_jump = max(self.max_center_jump_px, self.last_lane_width_px * 0.9)

        for x, y in raw_points:
            current_x = float(x)
            if previous_x is None:
                filtered_points.append((x, y))
                previous_x = current_x
                continue

            if abs(current_x - previous_x) <= max_jump:
                filtered_points.append((x, y))
                previous_x = current_x
                continue

            if self.last_fit_coeffs is not None:
                predicted_x = float(evaluate_poly(self.last_fit_coeffs, [y])[0])
                if abs(current_x - predicted_x) <= max_jump:
                    filtered_points.append((x, y))
                    previous_x = current_x

        if len(filtered_points) < max(2, self.min_valid_points // 2):
            return list(raw_points)
        return filtered_points

    def _find_row_segments(self, row: np.ndarray) -> List[Tuple[int, int]]:
        """在单行掩膜中寻找连续非零像素段。

        输入:
            row: 单行二值掩膜数组。

        输出:
            返回连续区间列表，每项格式为 (start_x, end_x)。
        """

        indices = np.where(row > 0)[0]
        if indices.size == 0:
            return []

        segments: List[Tuple[int, int]] = []
        start = int(indices[0])
        end = int(indices[0])

        for index in indices[1:]:
            current = int(index)
            if current == end + 1:
                end = current
            else:
                segments.append((start, end))
                start = current
                end = current

        segments.append((start, end))
        return segments

    def _select_best_segment(
        self,
        segments: Sequence[Tuple[int, int]],
        predicted_center: Optional[float],
        image_width: int,
    ) -> Tuple[int, int]:
        """从多个候选蓝色段中选出最可能属于主航道的段。

        输入:
            segments: 当前扫描行的连续蓝色区间列表。
            predicted_center: 根据上一帧或上一行估计得到的中心位置。
            image_width: 当前 ROI 的宽度。

        输出:
            返回最优区间，格式为 (start_x, end_x)。
        """

        if len(segments) == 1:
            return segments[0]

        image_center = image_width * 0.5
        best_score = -1e9
        best_segment = segments[0]

        for start, end in segments:
            width = float(end - start + 1)
            center = (start + end) * 0.5
            score = width
            if predicted_center is not None:
                score -= abs(center - predicted_center) * self.distance_weight
            else:
                score -= abs(center - image_center) * self.center_bias

            if score > best_score:
                best_score = score
                best_segment = (start, end)

        return best_segment

    def _fit_centerline(self, raw_points: Sequence[Tuple[int, int]]) -> Optional[np.ndarray]:
        """对原始中心线点集进行二次曲线拟合。

        输入:
            raw_points: 原始中心线点集，格式为 [(x, y), ...]。

        输出:
            返回二次曲线系数数组 [a, b, c]；若点数不足则可能返回低阶退化结果或 None。
        """

        if not raw_points:
            return None

        x_values = [point[0] for point in raw_points]
        y_values = [point[1] for point in raw_points]
        return polyfit_with_fallback(y_values=y_values, x_values=x_values, degree=2)

    def _generate_centerline_points(
        self,
        fit_coeffs: Optional[np.ndarray],
        shape: Tuple[int, int],
        support_points: Sequence[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """根据拟合曲线重新采样得到更平滑的中心线点集。

        输入:
            fit_coeffs: 拟合得到的曲线系数。
            shape: 当前 ROI 掩膜尺寸，格式为 (高, 宽)。
            support_points: 真正参与拟合的中心线点集。

        输出:
            返回平滑后的中心线点集。
        """

        if fit_coeffs is None or not support_points:
            return []

        _, width = shape[:2]
        support_y_values = [point[1] for point in support_points]
        bottom_y = max(support_y_values)
        top_y = max(0, min(support_y_values) - self.fit_top_extension_rows * self.scan_step)

        points: List[Tuple[int, int]] = []
        for y in range(bottom_y, top_y - 1, -self.scan_step):
            x = float(evaluate_poly(fit_coeffs, [y])[0])
            x = clamp(x, 0.0, width - 1.0)
            points.append((int(round(x)), int(y)))
        return points

    def _compute_lane_metrics(
        self,
        centerline_points: Sequence[Tuple[int, int]],
        fit_coeffs: Optional[np.ndarray],
        shape: Tuple[int, int],
        support_points: Sequence[Tuple[int, int]],
    ) -> Tuple[float, float, float]:
        """根据中心线计算横向误差、航向误差与曲率。

        输入:
            centerline_points: 平滑后的中心线点集。
            fit_coeffs: 中心线拟合系数。
            shape: 当前 ROI 尺寸，格式为 (高, 宽)。
            support_points: 真正参与拟合的中心线点集。

        输出:
            返回三元组 (lateral_error_px, heading_error_deg, curvature)。
        """

        _, width = shape[:2]
        if not centerline_points or not support_points:
            return 0.0, 0.0, 0.0

        support_y_values = [point[1] for point in support_points]
        bottom_y = max(support_y_values)
        top_y = min(support_y_values)
        support_span = max(self.scan_step * 2, bottom_y - top_y)
        lookahead_distance = max(self.scan_step * 2, int(support_span * self.lookahead_ratio))
        lookahead_y = max(top_y, bottom_y - lookahead_distance)
        image_center_x = width * 0.5

        bottom_x = float(centerline_points[0][0])
        lookahead_x = self._sample_centerline_x(centerline_points, lookahead_y)

        lateral_error_px = bottom_x - image_center_x
        heading_error_deg = math.degrees(
            math.atan2(lookahead_x - bottom_x, max(1.0, bottom_y - lookahead_y))
        )
        curvature = compute_curvature(fit_coeffs, bottom_y)
        return float(lateral_error_px), float(heading_error_deg), float(curvature)

    def _sample_centerline_x(
        self,
        centerline_points: Sequence[Tuple[int, int]],
        target_y: int,
    ) -> float:
        """按目标纵坐标在中心线上插值采样横坐标。

        输入:
            centerline_points: 按从下往上排序的中心线点集。
            target_y: 需要采样的目标纵坐标。

        输出:
            返回插值得到的横坐标。
        """

        if not centerline_points:
            return 0.0
        if len(centerline_points) == 1:
            return float(centerline_points[0][0])

        ordered_points = sorted(centerline_points, key=lambda item: item[1], reverse=True)
        if target_y >= ordered_points[0][1]:
            return float(ordered_points[0][0])
        if target_y <= ordered_points[-1][1]:
            return float(ordered_points[-1][0])

        for index in range(len(ordered_points) - 1):
            x1, y1 = ordered_points[index]
            x2, y2 = ordered_points[index + 1]
            if y1 >= target_y >= y2 and y1 != y2:
                ratio = safe_divide(float(target_y - y1), float(y2 - y1), default=0.0)
                return float(x1 + (x2 - x1) * ratio)

        return float(ordered_points[-1][0])

    def _estimate_confidence(
        self,
        filtered_mask: np.ndarray,
        raw_points: Sequence[Tuple[int, int]],
        fit_coeffs: Optional[np.ndarray],
        lane_width_px: float,
    ) -> float:
        """综合覆盖率、拟合残差与面积特征估计当前检测置信度。

        输入:
            filtered_mask: 筛选后的主航道掩膜。
            raw_points: 原始中心线点集。
            fit_coeffs: 中心线拟合系数。
            lane_width_px: 当前帧估计的航道宽度。

        输出:
            返回 0 到 1 之间的置信度分数。
        """

        height, width = filtered_mask.shape[:2]
        expected_rows = max(1, int(math.ceil(height / float(self.scan_step))))
        if len(raw_points) <= 12:
            expected_points = max(4.0, min(8.0, float(expected_rows)))
        else:
            expected_points = float(expected_rows)
        row_score = clamp(len(raw_points) / expected_points, 0.0, 1.0)

        non_zero_pixels = float(cv2.countNonZero(filtered_mask))
        expected_area = max(1.0, height * width * self.expected_area_ratio)
        area_score = clamp(non_zero_pixels / expected_area, 0.0, 1.0)
        if non_zero_pixels > expected_area * 3.0:
            overflow_ratio = safe_divide(non_zero_pixels - expected_area * 3.0, height * width, default=0.0)
            area_score *= clamp(1.0 - overflow_ratio * 4.0, 0.2, 1.0)

        if fit_coeffs is not None and raw_points:
            x_values = [point[0] for point in raw_points]
            y_values = [point[1] for point in raw_points]
            residual = mean_abs_residual(fit_coeffs, y_values=y_values, x_values=x_values)
            fit_score = clamp(1.0 - residual / max(self.residual_tolerance_px, 1.0), 0.0, 1.0)
        else:
            fit_score = 0.0

        width_score = clamp(lane_width_px / max(self.default_lane_width_px, 1.0), 0.0, 1.0)
        if lane_width_px > self.default_lane_width_px * 1.8:
            width_score *= 0.7

        confidence = (
            0.45 * row_score
            + 0.25 * fit_score
            + 0.20 * area_score
            + 0.10 * width_score
        )
        if len(raw_points) < self.min_valid_points:
            confidence *= 0.6

        return clamp(confidence, 0.0, 1.0)

    def _empty_result(self, shape: Tuple[int, int]) -> LaneDetectionResult:
        """在输入异常时构造一份空检测结果。

        输入:
            shape: 需要构造的掩膜尺寸，格式为 (高, 宽)。

        输出:
            返回表示丢线状态的 LaneDetectionResult。
        """

        mask = np.zeros(shape, dtype=np.uint8)
        return LaneDetectionResult(
            centerline_points=[],
            lateral_error_px=0.0,
            heading_error_deg=0.0,
            curvature=0.0,
            confidence=0.0,
            is_lane_lost=True,
            mask=mask,
            filtered_mask=mask.copy(),
            fit_coeffs=None,
            lane_width_px=self.last_lane_width_px,
            valid_row_count=0,
            fit_point_count=0,
            fork_result=ForkLaneResult(
                fork_detected=False,
                requested_direction=None,
                selected_direction=None,
                left_points=[],
                right_points=[],
                reason="empty input",
            ),
        )
