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
    left_centerline_points: List[Tuple[int, int]]
    right_centerline_points: List[Tuple[int, int]]
    reason: str
    left_detected: bool = False
    right_detected: bool = False
    left_corner: Tuple[int, int] | None = None
    right_corner: Tuple[int, int] | None = None
    confirm_frames: int = 0
    left_smoothness_residual_px: float | None = None
    right_smoothness_residual_px: float | None = None
    rejected_direction: str | None = None


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
CURRENT_ROUTE_DISTANCE_MARGIN_PX = 1.0


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
        self.perspective_width_top_px = float(
            self.centerline_config.get("perspective_width_top_px", 30.0)
        )
        self.perspective_width_bottom_px = float(
            self.centerline_config.get("perspective_width_bottom_px", 60.0)
        )
        if self.perspective_width_top_px <= 0.0:
            raise ValueError("centerline.perspective_width_top_px must be greater than zero")
        if self.perspective_width_bottom_px <= 0.0:
            raise ValueError("centerline.perspective_width_bottom_px must be greater than zero")
        if self.perspective_width_bottom_px < self.perspective_width_top_px:
            raise ValueError(
                "centerline.perspective_width_bottom_px must be greater than or equal to "
                "centerline.perspective_width_top_px"
            )
        self.single_side_infer_ratio = float(self.centerline_config.get("single_side_infer_ratio", 0.55))
        self.enable_single_side_inference = bool(
            self.centerline_config.get("enable_single_side_inference", False)
        )
        self.single_side_edge_margin_px = int(
            self.centerline_config.get("single_side_edge_margin_px", 28)
        )
        self.enable_boundary_smoothness_fallback = bool(
            self.centerline_config.get("enable_boundary_smoothness_fallback", False)
        )
        self.boundary_smoothness_residual_threshold_px = float(
            self.centerline_config.get("boundary_smoothness_residual_threshold_px", 3.0)
        )
        self.boundary_smoothness_tie_margin_px = float(
            self.centerline_config.get("boundary_smoothness_tie_margin_px", 0.1)
        )
        self.boundary_smoothness_confirm_frames = int(
            self.centerline_config.get("boundary_smoothness_confirm_frames", 2)
        )
        if (
            not math.isfinite(self.boundary_smoothness_residual_threshold_px)
            or self.boundary_smoothness_residual_threshold_px <= 0.0
        ):
            raise ValueError(
                "centerline.boundary_smoothness_residual_threshold_px must be finite "
                "and greater than zero"
            )
        if (
            not math.isfinite(self.boundary_smoothness_tie_margin_px)
            or self.boundary_smoothness_tie_margin_px < 0.0
        ):
            raise ValueError(
                "centerline.boundary_smoothness_tie_margin_px must be finite and not negative"
            )
        if self.boundary_smoothness_confirm_frames < 1:
            raise ValueError(
                "centerline.boundary_smoothness_confirm_frames must be at least one"
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
        self.fork_split_enter_ratio = float(fork_config.get("split_enter_ratio", 0.60))
        self.fork_split_exit_ratio = float(fork_config.get("split_exit_ratio", 0.35))
        self.fork_split_min_rows = max(1, int(fork_config.get("split_min_rows", 5)))
        self.fork_smoothness_residual_threshold_px = float(
            fork_config.get("smoothness_residual_threshold_px", 3.0)
        )
        self.fork_smoothness_tie_margin_px = float(
            fork_config.get("smoothness_tie_margin_px", 0.1)
        )
        if self.fork_split_enter_ratio <= 0.0:
            raise ValueError("fork.split_enter_ratio must be greater than zero")
        if self.fork_split_exit_ratio < 0.0:
            raise ValueError("fork.split_exit_ratio must not be negative")
        if self.fork_split_exit_ratio >= self.fork_split_enter_ratio:
            raise ValueError("fork.split_exit_ratio must be less than fork.split_enter_ratio")
        if (
            not math.isfinite(self.fork_smoothness_residual_threshold_px)
            or self.fork_smoothness_residual_threshold_px <= 0.0
        ):
            raise ValueError(
                "fork.smoothness_residual_threshold_px must be finite and greater than zero"
            )
        if (
            not math.isfinite(self.fork_smoothness_tie_margin_px)
            or self.fork_smoothness_tie_margin_px < 0.0
        ):
            raise ValueError(
                "fork.smoothness_tie_margin_px must be finite and not negative"
            )
        self.temporal_weights = tuple(float(v) for v in temporal_config.get("weights", [0.20, 0.50, 0.30]))
        if len(self.temporal_weights) != 3 or sum(self.temporal_weights) <= 0:
            self.temporal_weights = (0.20, 0.50, 0.30)
        self._weighted_center_history: List[float] = []
        self._left_fork_hits = self._right_fork_hits = 0
        self._left_fork_misses = self._right_fork_misses = 0
        self._left_fork_active = self._right_fork_active = False
        self._held_fork_direction: str | None = None
        self._normal_centerline_mode = "normal"
        self._pending_normal_centerline_mode: str | None = None
        self._pending_normal_centerline_hits = 0

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
        vehicle_center_x: float | None = None,
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
            vehicle_center_x=vehicle_center_x,
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
        vehicle_center_x: float | None,
        segmentation_confidence: float,
        segmentation_status: str,
    ) -> LaneDetectionResult:
        """Build the driving centerline directly from row-wise track boundaries."""

        row_runs = self._build_row_runs(mask)
        roi_center_x = (
            0.5 * float(mask.shape[1])
            if vehicle_center_x is None
            else clamp(float(vehicle_center_x), 0.0, float(max(0, mask.shape[1] - 1)))
        )
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
        fork_result = self._geometric_fork_result(
            left_points,
            right_points,
            left_lost,
            right_lost,
            left_branch_rows,
            right_branch_rows,
            mask.shape,
        )
        requested_direction = self._normalize_route_direction(route_direction)
        fork_result.requested_direction = requested_direction
        display_left_points = left_points
        display_right_points = right_points
        if not fork_result.fork_detected:
            self._held_fork_direction = None
            raw_centers = self._apply_boundary_smoothness_fallback(
                left_points=left_points,
                right_points=right_points,
                raw_centers=raw_centers,
                left_lost=left_lost,
                right_lost=right_lost,
                shape=mask.shape,
            )
            if requested_direction is not None:
                fork_result.reason = f"waiting for {requested_direction} fork"
        else:
            self._reset_normal_centerline_mode()
            (
                left_candidate_points,
                right_candidate_points,
                shared_centerline_points,
                outer_left_points,
                outer_right_points,
            ) = self._build_perspective_fork_centerlines(row_runs, mask.shape)
            fork_result.left_centerline_points = self._smooth_article_centerline(
                left_candidate_points, mask.shape[1]
            )
            fork_result.right_centerline_points = self._smooth_article_centerline(
                right_candidate_points, mask.shape[1]
            )
            left_smoothness_residual = self._fork_smoothness_residual(
                left_candidate_points
            )
            right_smoothness_residual = self._fork_smoothness_residual(
                right_candidate_points
            )
            fork_result.left_smoothness_residual_px = left_smoothness_residual
            fork_result.right_smoothness_residual_px = right_smoothness_residual
            if outer_left_points and outer_right_points:
                display_left_points = outer_left_points
                display_right_points = outer_right_points

            center_distance_scores: tuple[float, float] | None = None
            smoothness_selected_direction: str | None = None
            if requested_direction is not None:
                self._held_fork_direction = requested_direction
            elif (
                self._held_fork_direction is None
                and left_candidate_points
                and right_candidate_points
            ):
                (
                    smoothness_selected_direction,
                    rejected_direction,
                ) = self._choose_fork_direction_by_smoothness(
                    left_smoothness_residual,
                    right_smoothness_residual,
                )
                fork_result.rejected_direction = rejected_direction
                if rejected_direction == "left":
                    fork_result.left_centerline_points = []
                elif rejected_direction == "right":
                    fork_result.right_centerline_points = []

                if smoothness_selected_direction is not None:
                    self._held_fork_direction = smoothness_selected_direction
                else:
                    (
                        current_direction,
                        left_center_distance,
                        right_center_distance,
                    ) = self._choose_current_fork_direction(
                        roi_center_x,
                        left_candidate_points,
                        right_candidate_points,
                    )
                    center_distance_scores = (left_center_distance, right_center_distance)
                    if current_direction is not None:
                        self._held_fork_direction = current_direction

            def format_residual(value: float | None) -> str:
                return "n/a" if value is None else f"{value:.2f}px"

            smoothness_debug = (
                f"smoothness left={format_residual(left_smoothness_residual)} "
                f"right={format_residual(right_smoothness_residual)} "
                f"threshold={self.fork_smoothness_residual_threshold_px:.2f}px "
                f"rejected={fork_result.rejected_direction or 'none'}"
            )

            selected_direction = self._held_fork_direction
            fork_result.selected_direction = selected_direction
            selected_candidates = (
                left_candidate_points
                if selected_direction == "left"
                else right_candidate_points
                if selected_direction == "right"
                else []
            )
            if selected_candidates:
                raw_centers = self._merge_fork_centerline(
                    raw_centers,
                    shared_centerline_points,
                    selected_candidates,
                )
                selected_mask = self._build_fork_selected_mask(
                    selected_mask,
                    row_runs,
                    selected_candidates,
                )
                if requested_direction is not None:
                    fork_result.reason = (
                        f"selected {selected_direction} branch (requested); {smoothness_debug}"
                    )
                elif smoothness_selected_direction is not None:
                    fork_result.reason = (
                        f"selected {selected_direction} branch by smoothness; {smoothness_debug}"
                    )
                elif center_distance_scores is not None:
                    fork_result.reason = (
                        f"selected {selected_direction} branch by frame center "
                        f"left={center_distance_scores[0]:.2f}px "
                        f"right={center_distance_scores[1]:.2f}px; {smoothness_debug}"
                    )
                else:
                    fork_result.reason = (
                        f"selected {selected_direction} branch (held); {smoothness_debug}"
                    )
            elif selected_direction is not None:
                raw_centers = self._merge_fork_centerline(
                    raw_centers,
                    shared_centerline_points,
                    [],
                )
                fork_result.reason = f"holding {selected_direction} through shared fork region"
            else:
                raw_centers = self._merge_fork_centerline(
                    raw_centers,
                    shared_centerline_points,
                    [],
                )
                if center_distance_scores is not None:
                    fork_result.reason = (
                        "frame-center distances too close; following normal centerline "
                        f"left={center_distance_scores[0]:.2f}px "
                        f"right={center_distance_scores[1]:.2f}px "
                        f"margin={CURRENT_ROUTE_DISTANCE_MARGIN_PX:.2f}px; "
                        f"{smoothness_debug}"
                    )
                else:
                    fork_result.reason = (
                        f"fork shared region; following normal centerline; {smoothness_debug}"
                    )

        centerline_points = self._smooth_article_centerline(raw_centers, mask.shape[1])
        fit_coeffs = self._fit_centerline(centerline_points)
        widths = [right[0] - left[0] for left, right in zip(left_points, right_points) if right[0] > left[0]]
        lane_width_px = float(np.median(widths)) if widths else self.last_lane_width_px
        lateral_error_px, heading_error_deg, curvature = self._article_metrics(
            centerline_points, fit_coeffs, mask.shape
        )
        confidence = self._estimate_confidence(
            selected_mask,
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
            left_boundary_points=display_left_points,
            right_boundary_points=display_right_points,
            left_lost_rows=sum(left_lost),
            right_lost_rows=sum(right_lost),
            segmentation_confidence=segmentation_confidence,
            segmentation_status=segmentation_status,
        )

    def _perspective_lane_width(self, y: int, height: int) -> float:
        """Interpolate the configured lane width from ROI top to bottom."""

        ratio = 0.0 if height <= 1 else clamp(float(y) / float(height - 1), 0.0, 1.0)
        return self.perspective_width_top_px + (
            self.perspective_width_bottom_px - self.perspective_width_top_px
        ) * ratio

    def _build_perspective_fork_centerlines(
        self,
        row_runs: Sequence[Sequence[tuple[int, int]]],
        shape: tuple[int, int],
    ) -> tuple[
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
        list[tuple[int, int]],
    ]:
        """Infer left/right branch centerlines from the outermost measured edges."""

        height, _width = shape[:2]
        samples: list[tuple[int, int, int, int, int, float]] = []
        for y in range(height - 1, -1, -1):
            runs = row_runs[y]
            if not runs:
                continue
            outer_left = min(run[0] for run in runs)
            outer_right = max(run[1] for run in runs)
            expected_width = self._perspective_lane_width(y, height)
            half_width = 0.5 * expected_width
            left_center = int(round(min(float(outer_right), float(outer_left) + half_width)))
            right_center = int(round(max(float(outer_left), float(outer_right) - half_width)))
            gap_ratio = max(0.0, float(right_center - left_center)) / max(expected_width, 1.0)
            samples.append(
                (y, outer_left, outer_right, left_center, right_center, gap_ratio)
            )

        split_flags = [False] * len(samples)
        split_active = False
        enter_start: int | None = None
        exit_start: int | None = None
        for index, sample in enumerate(samples):
            gap_ratio = sample[5]
            if not split_active:
                if gap_ratio >= self.fork_split_enter_ratio:
                    enter_start = index if enter_start is None else enter_start
                    if index - enter_start + 1 >= self.fork_split_min_rows:
                        split_active = True
                        for buffered_index in range(enter_start, index + 1):
                            split_flags[buffered_index] = True
                        enter_start = None
                else:
                    enter_start = None
                continue

            split_flags[index] = True
            if gap_ratio <= self.fork_split_exit_ratio:
                exit_start = index if exit_start is None else exit_start
                if index - exit_start + 1 >= self.fork_split_min_rows:
                    for buffered_index in range(exit_start, index + 1):
                        split_flags[buffered_index] = False
                    split_active = False
                    exit_start = None
            else:
                exit_start = None

        left_candidates = [
            (sample[3], sample[0])
            for sample, is_split in zip(samples, split_flags)
            if is_split
        ]
        right_candidates = [
            (sample[4], sample[0])
            for sample, is_split in zip(samples, split_flags)
            if is_split
        ]
        shared_centerline_points = [
            (int(round(0.5 * (sample[1] + sample[2]))), sample[0])
            for sample, is_split in zip(samples, split_flags)
            if not is_split
        ]
        outer_left_points = [(sample[1], sample[0]) for sample in samples]
        outer_right_points = [(sample[2], sample[0]) for sample in samples]
        return (
            left_candidates,
            right_candidates,
            shared_centerline_points,
            outer_left_points,
            outer_right_points,
        )

    def _choose_current_fork_direction(
        self,
        vehicle_center_x: float,
        left_candidates: Sequence[Tuple[int, int]],
        right_candidates: Sequence[Tuple[int, int]],
    ) -> tuple[str | None, float, float]:
        """Choose the branch whose average horizontal distance to frame center is shorter."""

        def mean_distance(points: Sequence[Tuple[int, int]]) -> float:
            if not points:
                return float("inf")
            return float(
                np.mean([
                    abs(float(x) - float(vehicle_center_x))
                    for x, _y in points
                ])
            )

        left_score = mean_distance(left_candidates)
        right_score = mean_distance(right_candidates)
        if abs(left_score - right_score) <= CURRENT_ROUTE_DISTANCE_MARGIN_PX:
            return None, left_score, right_score
        direction = "left" if left_score < right_score else "right"
        return direction, left_score, right_score

    def _fork_smoothness_residual(
        self,
        candidate_points: Sequence[Tuple[int, int]],
    ) -> float | None:
        """Measure raw candidate roughness as its mean residual from a quadratic fit."""

        return self._polyline_smoothness_residual(candidate_points, min_points=3)

    def _polyline_smoothness_residual(
        self,
        points: Sequence[Tuple[int, int]],
        min_points: int,
    ) -> float | None:
        """Return quadratic mean absolute residual for a sufficiently long polyline."""

        if len(points) < max(3, int(min_points)):
            return None
        x_values = [float(x) for x, _y in points]
        y_values = [float(y) for _x, y in points]
        fit_coeffs = polyfit_with_fallback(y_values, x_values, degree=2)
        if fit_coeffs is None:
            return None
        residual = mean_abs_residual(fit_coeffs, y_values, x_values)
        return float(residual) if math.isfinite(residual) else None

    def _boundary_smoothness_residual(
        self,
        points: Sequence[Tuple[int, int]],
        lost_flags: Sequence[bool],
    ) -> float | None:
        """Measure only real boundary samples, excluding inferred gap rows."""

        measured = [
            (int(x), int(y))
            for (x, y), lost in zip(points, lost_flags)
            if not lost
        ]
        return self._polyline_smoothness_residual(
            measured,
            min_points=self.min_valid_points,
        )

    def _choose_normal_centerline_mode(
        self,
        left_residual: float | None,
        right_residual: float | None,
    ) -> str:
        """Choose a reliable boundary for normal-lane centerline reconstruction."""

        if left_residual is None or right_residual is None:
            return "normal"

        threshold = self.boundary_smoothness_residual_threshold_px
        left_rough = left_residual > threshold
        right_rough = right_residual > threshold
        if left_rough and not right_rough:
            return "right"
        if right_rough and not left_rough:
            return "left"
        if left_rough and right_rough:
            if abs(left_residual - right_residual) <= self.boundary_smoothness_tie_margin_px:
                return "normal"
            return "left" if left_residual < right_residual else "right"
        return "normal"

    def _update_normal_centerline_mode(
        self,
        desired_mode: str,
        left_residual: float | None,
        right_residual: float | None,
    ) -> str:
        """Debounce normal/left/right switching and return the safe mode for this frame."""

        if desired_mode == self._normal_centerline_mode:
            self._pending_normal_centerline_mode = None
            self._pending_normal_centerline_hits = 0
            return self._normal_centerline_mode

        if desired_mode != self._pending_normal_centerline_mode:
            self._pending_normal_centerline_mode = desired_mode
            self._pending_normal_centerline_hits = 1
        else:
            self._pending_normal_centerline_hits += 1

        if self._pending_normal_centerline_hits >= self.boundary_smoothness_confirm_frames:
            self._normal_centerline_mode = desired_mode
            self._pending_normal_centerline_mode = None
            self._pending_normal_centerline_hits = 0
            return self._normal_centerline_mode

        active_residual = (
            left_residual
            if self._normal_centerline_mode == "left"
            else right_residual
            if self._normal_centerline_mode == "right"
            else None
        )
        if (
            active_residual is not None
            and active_residual <= self.boundary_smoothness_residual_threshold_px
        ):
            return self._normal_centerline_mode
        return "normal"

    def _apply_boundary_smoothness_fallback(
        self,
        left_points: Sequence[Tuple[int, int]],
        right_points: Sequence[Tuple[int, int]],
        raw_centers: Sequence[Tuple[int, int]],
        left_lost: Sequence[bool],
        right_lost: Sequence[bool],
        shape: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Rebuild normal-lane centers from the smoother measured boundary."""

        if not self.enable_boundary_smoothness_fallback:
            self._reset_normal_centerline_mode()
            return [(int(x), int(y)) for x, y in raw_centers]

        left_residual = self._boundary_smoothness_residual(left_points, left_lost)
        right_residual = self._boundary_smoothness_residual(right_points, right_lost)
        desired_mode = self._choose_normal_centerline_mode(left_residual, right_residual)
        active_mode = self._update_normal_centerline_mode(
            desired_mode,
            left_residual,
            right_residual,
        )
        if active_mode == "normal":
            return [(int(x), int(y)) for x, y in raw_centers]

        _height, width = shape[:2]
        rebuilt: list[tuple[int, int]] = []
        for index, ((left_x, y), (right_x, _right_y), (center_x, _center_y)) in enumerate(
            zip(left_points, right_points, raw_centers)
        ):
            use_left = active_mode == "left" and not left_lost[index]
            use_right = active_mode == "right" and not right_lost[index]
            if use_left:
                candidate_x = float(left_x) + 0.5 * self._perspective_lane_width(y, _height)
            elif use_right:
                candidate_x = float(right_x) - 0.5 * self._perspective_lane_width(y, _height)
            else:
                candidate_x = float(center_x)
            rebuilt.append(
                (
                    int(round(clamp(candidate_x, 0.0, float(max(0, width - 1))))),
                    int(y),
                )
            )
        return rebuilt

    def _reset_normal_centerline_mode(self) -> None:
        self._normal_centerline_mode = "normal"
        self._pending_normal_centerline_mode = None
        self._pending_normal_centerline_hits = 0

    def _choose_fork_direction_by_smoothness(
        self,
        left_residual: float | None,
        right_residual: float | None,
    ) -> tuple[str | None, str | None]:
        """Reject a clearly rougher branch before automatic frame-center selection."""

        if left_residual is None or right_residual is None:
            return None, None

        threshold = self.fork_smoothness_residual_threshold_px
        left_rough = left_residual > threshold
        right_rough = right_residual > threshold
        if left_rough and not right_rough:
            return "right", "left"
        if right_rough and not left_rough:
            return "left", "right"
        if left_rough and right_rough:
            if abs(left_residual - right_residual) <= self.fork_smoothness_tie_margin_px:
                return None, None
            if left_residual > right_residual:
                return "right", "left"
            return "left", "right"
        return None, None

    @staticmethod
    def _merge_fork_centerline(
        ordinary_points: Sequence[Tuple[int, int]],
        shared_centerline_points: Sequence[Tuple[int, int]],
        selected_candidates: Sequence[Tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Use the outer-edge midpoint on shared rows and the selected split branch."""

        candidate_by_y = {int(y): int(x) for x, y in shared_centerline_points}
        candidate_by_y.update({int(y): int(x) for x, y in selected_candidates})
        return [
            (candidate_by_y.get(int(y), int(x)), int(y))
            for x, y in ordinary_points
        ]

    @staticmethod
    def _build_fork_selected_mask(
        base_mask: np.ndarray,
        row_runs: Sequence[Sequence[tuple[int, int]]],
        selected_candidates: Sequence[Tuple[int, int]],
    ) -> np.ndarray:
        """Keep the foreground run nearest the selected branch on separated rows."""

        selected_mask = base_mask.copy()
        for candidate_x, y in selected_candidates:
            runs = row_runs[int(y)]
            if not runs:
                continue
            left, right = min(
                runs,
                key=lambda run: abs(0.5 * (run[0] + run[1]) - float(candidate_x)),
            )
            selected_mask[int(y), :] = 0
            selected_mask[int(y), int(left) : int(right) + 1] = 255
        return selected_mask

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

            # A foreground run that reaches an ROI edge still has a valid
            # boundary for this row.  Keep the measured endpoint instead of
            # rebuilding it inward from the historical lane width.
            rows.append((y, left, right, False, False))
            selected_mask[y, left : right + 1] = 255
            prior_center = 0.65 * prior_center + 0.35 * (0.5 * (left + right))

        left_points = [(left, y) for y, left, _right, _ll, _rl in rows]
        right_points = [(right, y) for y, _left, right, _ll, _rl in rows]
        centers = [(int(round((left + right) * 0.5)), y) for y, left, right, _ll, _rl in rows]
        left_lost = [left_lost for _y, _left, _right, left_lost, _right_lost in rows]
        right_lost = [right_lost for _y, _left, _right, _left_lost, right_lost in rows]
        return left_points, right_points, centers, left_lost, right_lost, selected_mask, left_branch_rows, right_branch_rows

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
                # A boundary copied across a missing-mask gap is not a measured
                # corner and must not produce an LF/RF marker.
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
            left_centerline_points=[],
            right_centerline_points=[],
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
                left_centerline_points=left_points,
                right_centerline_points=right_points,
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
                left_centerline_points=[],
                right_centerline_points=[],
                reason="empty input",
            ),
        )
