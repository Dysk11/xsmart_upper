"""X-SmartCar 人工智能模型组上位机主程序。"""

from __future__ import annotations

import os

# RK3588 上小矩阵 mask 后处理让 BLAS 自动开满 8 核反而更慢。必须在导入
# NumPy/OpenCV 前设置，spawn 出来的推理进程也会继承这些限制。
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import argparse
import copy
import multiprocessing as mp
from multiprocessing import resource_tracker
from multiprocessing import shared_memory
import queue
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bridge import BaseVehicleBridge, build_vehicle_bridge
from core.camera import CameraReader
from core.avoidance_target_planner import AvoidanceTargetPlanner, AvoidanceTargetResult
from core.blocking_analyzer import BlockingAnalyzer, BlockingAnalysisResult, DetectedObject, attach_roi_bboxes
from core.fork_route_planner import ForkRoutePlanner, ForkRouteResult
from core.gold_target_planner import GoldTargetPlanner, GoldTargetResult
from core.lane_detector import LaneDetectionResult, LaneDetector
from core.lane_tracker import LaneTracker, TrackedLaneState
from core.logger import CsvLogger
from core.planner import ControlCommand, HighLevelPlanner, ModuleHints
from core.preprocess import ImagePreprocessor, PreprocessResult
from core.rknn_object_detector import RknnObjectDetector
from core.rknn_lane_segmenter import RknnLaneSegmenter, SegmentationResult
from core.target_selector import TargetPointResult, TargetSelector
from core.visualizer import Visualizer
from utils.fps import FPSCounter


SHARED_ARRAY_MARKER = "__xsmart_shared_array__"
SHARED_PREPROCESS_MARKER = "__xsmart_shared_preprocess__"


def _unregister_shared_memory(shm: shared_memory.SharedMemory) -> None:
    """Prevent a temporary opener from owning shared-memory cleanup."""

    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


class SharedArrayPool:
    """A small ring of shared-memory ndarray slots owned by the main process."""

    def __init__(self, pool_id: str, ack_queue: Any, slot_count: int = 3) -> None:
        self.pool_id = pool_id
        self.ack_queue = ack_queue
        self.slot_count = max(2, int(slot_count))
        self.slots: list[shared_memory.SharedMemory] = []
        self.shape: tuple[int, ...] | None = None
        self.dtype: np.dtype | None = None
        self.pending_slots: set[int] = set()

    def write(self, array: np.ndarray) -> Dict[str, Any] | None:
        """Copy an ndarray into a free slot and return its queue-safe descriptor."""

        contiguous = np.ascontiguousarray(array)
        dtype = contiguous.dtype
        shape = tuple(contiguous.shape)
        self._drain_acks()

        if not self.slots:
            self._allocate_slots(shape, dtype, contiguous.nbytes)
        elif shape != self.shape or dtype != self.dtype:
            if self.pending_slots:
                return None
            self.close()
            self._allocate_slots(shape, dtype, contiguous.nbytes)

        slot_index = self._next_free_slot()
        if slot_index is None:
            return None

        shm = self.slots[slot_index]
        shared_view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        shared_view[...] = contiguous
        self.pending_slots.add(slot_index)
        return {
            SHARED_ARRAY_MARKER: True,
            "pool_id": self.pool_id,
            "slot": slot_index,
            "name": shm.name,
            "shape": shape,
            "dtype": dtype.str,
        }

    def release_descriptor(self, descriptor: Dict[str, Any]) -> None:
        """Mark a queued-but-dropped slot as available again."""

        if descriptor.get("pool_id") != self.pool_id:
            return
        self.pending_slots.discard(int(descriptor["slot"]))

    def close(self) -> None:
        """Release all shared-memory slots owned by this pool."""

        for shm in self.slots:
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        self.slots = []
        self.shape = None
        self.dtype = None
        self.pending_slots.clear()

    def _allocate_slots(self, shape: tuple[int, ...], dtype: np.dtype, nbytes: int) -> None:
        self.shape = shape
        self.dtype = dtype
        self.slots = [
            shared_memory.SharedMemory(create=True, size=nbytes)
            for _ in range(self.slot_count)
        ]

    def _drain_acks(self) -> None:
        while True:
            try:
                slot_index = self.ack_queue.get_nowait()
            except queue.Empty:
                break
            self.pending_slots.discard(int(slot_index))

    def _next_free_slot(self) -> int | None:
        for slot_index in range(len(self.slots)):
            if slot_index not in self.pending_slots:
                return slot_index
        return None


def _is_shared_array_descriptor(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get(SHARED_ARRAY_MARKER))


def _ack_shared_descriptor(descriptor: Dict[str, Any], ack_queues: Dict[str, Any]) -> None:
    try:
        ack_queues[str(descriptor["pool_id"])].put_nowait(int(descriptor["slot"]))
    except Exception:
        pass


def _ack_shared_payload(value: Any, ack_queues: Dict[str, Any]) -> None:
    if _is_shared_array_descriptor(value):
        _ack_shared_descriptor(value, ack_queues)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _ack_shared_payload(nested, ack_queues)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _ack_shared_payload(nested, ack_queues)


def _take_shared_ndarray(descriptor: Dict[str, Any], ack_queues: Dict[str, Any]) -> np.ndarray:
    """Copy a pooled shared-memory ndarray locally and acknowledge its slot."""

    try:
        shm = shared_memory.SharedMemory(name=str(descriptor["name"]))
    except FileNotFoundError:
        _ack_shared_descriptor(descriptor, ack_queues)
        raise
    _unregister_shared_memory(shm)
    try:
        array = np.ndarray(
            tuple(descriptor["shape"]),
            dtype=np.dtype(str(descriptor["dtype"])),
            buffer=shm.buf,
        ).copy()
    finally:
        shm.close()
        _ack_shared_descriptor(descriptor, ack_queues)
    return array


def _release_shared_payload(value: Any, pools: Dict[str, SharedArrayPool] | None = None) -> None:
    """Recursively release shared-memory slots held by a dropped queue item."""

    if _is_shared_array_descriptor(value):
        if pools is not None:
            pool = pools.get(str(value.get("pool_id")))
            if pool is not None:
                pool.release_descriptor(value)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _release_shared_payload(nested, pools)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _release_shared_payload(nested, pools)


def _share_preprocess_result(
    preprocess_result: PreprocessResult,
    resized_pool: SharedArrayPool,
    roi_pool: SharedArrayPool,
) -> Dict[str, Any] | None:
    """Move PreprocessResult images into shared memory while preserving metadata."""

    descriptor: Dict[str, Any] = {
        SHARED_PREPROCESS_MARKER: True,
        "roi_rect": preprocess_result.roi_rect,
    }
    try:
        descriptor["resized_frame"] = resized_pool.write(preprocess_result.resized_frame)
        descriptor["roi_frame"] = roi_pool.write(preprocess_result.roi_frame)
        if descriptor["resized_frame"] is None or descriptor["roi_frame"] is None:
            _release_shared_payload(
                descriptor,
                {
                    resized_pool.pool_id: resized_pool,
                    roi_pool.pool_id: roi_pool,
                },
            )
            return None
        return descriptor
    except Exception:
        _release_shared_payload(
            descriptor,
            {
                resized_pool.pool_id: resized_pool,
                roi_pool.pool_id: roi_pool,
            },
        )
        raise


def _take_shared_preprocess_result(
    descriptor: Dict[str, Any],
    ack_queues: Dict[str, Any],
) -> PreprocessResult:
    """Rebuild a local PreprocessResult from shared-memory image descriptors."""

    remaining = dict(descriptor)
    try:
        resized_frame = _take_shared_ndarray(remaining.pop("resized_frame"), ack_queues)
        roi_frame = _take_shared_ndarray(remaining.pop("roi_frame"), ack_queues)
        return PreprocessResult(
            original_frame=resized_frame,
            resized_frame=resized_frame,
            roi_raw_frame=roi_frame,
            roi_frame=roi_frame,
            roi_rect=tuple(descriptor["roi_rect"]),
        )
    except Exception:
        _ack_shared_payload(remaining, ack_queues)
        raise


def _put_latest(work_queue: Any, item: Any, release_func: Any = _release_shared_payload) -> None:
    """Put an item into a size-limited multiprocessing queue, dropping stale data."""

    try:
        work_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        dropped_item = work_queue.get_nowait()
    except queue.Empty:
        dropped_item = None
    release_func(dropped_item)

    try:
        work_queue.put_nowait(item)
    except queue.Full:
        release_func(item)


def _drain_latest(work_queue: Any, release_func: Any = _release_shared_payload) -> Any | None:
    """Return the newest available queue item without blocking."""

    latest = None
    while True:
        try:
            item = work_queue.get_nowait()
        except queue.Empty:
            return latest
        if latest is not None:
            release_func(latest)
        latest = item


def _ai_inference_worker(
    detector_config: Dict[str, Any],
    input_queue: Any,
    output_queue: Any,
    ack_queue: Any,
    stop_event: Any,
) -> None:
    """Run RKNN object detection in an independent process."""

    detector = RknnObjectDetector(detector_config)
    ack_queues = {"ai_frame": ack_queue}
    try:
        while not stop_event.is_set():
            try:
                item = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break

            frame_id, frame_payload = item
            try:
                frame = (
                    _take_shared_ndarray(frame_payload, ack_queues)
                    if _is_shared_array_descriptor(frame_payload)
                    else frame_payload
                )
                detections = detector.detect(frame)
                _put_latest(output_queue, (frame_id, detections))
            except Exception:
                traceback.print_exc()
                _put_latest(output_queue, (frame_id, []))
    finally:
        detector.close()


def _lane_inference_worker(
    worker_index: int,
    segmenter_config: Dict[str, Any],
    input_queue: Any,
    output_queue: Any,
    ack_queue: Any,
    pool_id: str,
    stop_event: Any,
) -> None:
    """Run one lane-segmentation runtime on its configured NPU core."""

    segmenter = RknnLaneSegmenter(segmenter_config)
    ack_queues = {pool_id: ack_queue}
    try:
        while not stop_event.is_set():
            try:
                item = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break

            frame_id, frame_payload = item
            try:
                frame = _take_shared_ndarray(frame_payload, ack_queues)
                result = segmenter.segment(frame)
            except Exception:
                traceback.print_exc()
                shape = tuple(frame.shape[:2]) if "frame" in locals() else (1, 1)
                result = SegmentationResult(np.zeros(shape, dtype=np.uint8), [], 0.0, "worker_error")
            packed_mask = np.packbits(result.mask.reshape(-1), bitorder="little")
            output_queue.put(
                (
                    frame_id,
                    worker_index,
                    tuple(result.mask.shape),
                    packed_mask,
                    result.instances,
                    result.confidence,
                    result.status,
                )
            )
    finally:
        segmenter.close()


def _ui_worker(
    visualizer_config: Dict[str, Any],
    input_queue: Any,
    resized_ack_queue: Any,
    roi_ack_queue: Any,
    stop_event: Any,
) -> None:
    """Render OpenCV UI and video output in an independent process."""

    visualizer = Visualizer(visualizer_config)
    ack_queues = {
        "ui_resized": resized_ack_queue,
        "ui_roi": roi_ack_queue,
    }
    try:
        while not stop_event.is_set():
            try:
                packet = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if packet is None:
                break

            try:
                packet = dict(packet)
                preprocess_payload = packet.get("preprocess_result")
                if isinstance(preprocess_payload, dict) and preprocess_payload.get(SHARED_PREPROCESS_MARKER):
                    packet["preprocess_result"] = _take_shared_preprocess_result(preprocess_payload, ack_queues)
                should_continue = visualizer.render(**packet)
            except Exception:
                traceback.print_exc()
                should_continue = False
            if not should_continue:
                stop_event.set()
                break
    finally:
        visualizer.close()


def load_config(config_path: Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。

    输入:
        config_path: 配置文件路径。

    输出:
        返回配置字典；若文件为空则返回空字典。
    """

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def resolve_project_path(project_root: Path, raw_path: str) -> Path:
    """将相对路径解析为基于项目根目录的绝对路径。

    输入:
        project_root: 项目根目录。
        raw_path: 配置中的原始路径字符串。

    输出:
        返回解析后的绝对路径。
    """

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def prepare_runtime_config(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """复制配置并将路径字段转换为绝对路径。

    输入:
        config: 原始配置字典。
        project_root: 项目根目录。

    输出:
        返回可直接用于运行的配置副本。
    """

    runtime_config = copy.deepcopy(config)

    camera_config = runtime_config.setdefault("camera", {})
    if camera_config.get("video_path"):
        camera_config["video_path"] = str(resolve_project_path(project_root, str(camera_config["video_path"])))

    logger_config = runtime_config.setdefault("logger", {})
    logger_config["output_dir"] = str(resolve_project_path(project_root, str(logger_config.get("output_dir", "outputs/logs"))))

    visualizer_config = runtime_config.setdefault("visualizer", {})
    visualizer_config["save_dir"] = str(resolve_project_path(project_root, str(visualizer_config.get("save_dir", "outputs/visual"))))

    rknn_detector_config = runtime_config.setdefault("rknn_object_detector", {})
    if rknn_detector_config.get("model_path"):
        rknn_detector_config["model_path"] = str(
            resolve_project_path(project_root, str(rknn_detector_config["model_path"]))
        )

    rknn_segmenter_config = runtime_config.setdefault("rknn_lane_segmenter", {})
    if rknn_segmenter_config.get("model_path"):
        rknn_segmenter_config["model_path"] = str(
            resolve_project_path(project_root, str(rknn_segmenter_config["model_path"]))
        )

    return runtime_config


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """根据命令行参数覆写配置项。

    输入:
        config: 原始配置字典。
        args: 解析后的命令行参数。

    输出:
        返回覆写后的配置字典。
    """

    runtime_config = copy.deepcopy(config)

    if args.mode:
        runtime_config.setdefault("camera", {})["mode"] = args.mode
    if args.video:
        runtime_config.setdefault("camera", {})["video_path"] = args.video
    if args.bridge:
        runtime_config.setdefault("bridge", {})["type"] = args.bridge
    if args.no_gui:
        runtime_config.setdefault("visualizer", {})["show_window"] = False
    if args.save_video:
        runtime_config.setdefault("visualizer", {})["save_video"] = True

    return runtime_config


class UpperMachineApp:
    """串联摄像头、视觉、控制、通信与可视化的上位机主应用。"""

    def __init__(self, config: Dict[str, Any], project_root: Path) -> None:
        """根据配置初始化整条巡线主链路。

        输入:
            config: 已经完成路径解析的运行时配置。
            project_root: 项目根目录。

        输出:
            无返回值，内部创建所有模块实例。
        """

        self.config = config
        self.project_root = project_root

        self.camera = CameraReader(config.get("camera", {}))
        self.preprocessor = ImagePreprocessor(config.get("preprocess", {}))
        self.detector = LaneDetector(config.get("detector", {}))
        lane_config = config.get("rknn_lane_segmenter", {})
        worker_core_masks = list(lane_config.get("worker_core_masks", []))
        self.lane_parallel_enabled = bool(lane_config.get("parallel", False) and worker_core_masks)
        self.lane_segmenter = None if self.lane_parallel_enabled else RknnLaneSegmenter(lane_config)
        self.tracker = LaneTracker(config.get("tracker", {}))
        self.target_selector = TargetSelector(config.get("target_selector", {}))
        self.fork_route_planner = ForkRoutePlanner(config.get("fork_route", {}))
        self.gold_target_planner = GoldTargetPlanner(config.get("gold_target", {}))
        self.blocking_config = config.get("blocking_analyzer", {})
        self.blocking_class_names = {
            str(name).casefold()
            for name in self.blocking_config.get("allowed_class_names", ["car", "human"])
        }
        self.blocking_analyzer = BlockingAnalyzer(self.blocking_config)
        self.avoidance_planner = AvoidanceTargetPlanner(
            config.get("avoidance_target_planner", {}),
            target_selector=self.target_selector,
        )
        self.planner = HighLevelPlanner(config.get("planner", {}))
        self.bridge: BaseVehicleBridge = build_vehicle_bridge(config.get("bridge", {}))
        self.csv_logger = CsvLogger(config.get("logger", {}))
        self.fps_counter = FPSCounter(config.get("app", {}).get("fps_smoothing_alpha", 0.2))
        timing_config = config.get("app", {}).get("lane_timing", {})
        self.lane_timing_enabled = bool(timing_config.get("enable", True))
        self.lane_timing_interval = max(1, int(timing_config.get("print_interval_frames", 30)))
        self.lane_timing_count = 0
        self.lane_timing_total_ms = 0.0
        self.lane_timing_preprocess_ms = 0.0
        self.lane_timing_detect_ms = 0.0
        self.lane_timing_track_ms = 0.0
        self.lane_timing_max_ms = 0.0
        self.last_target_result: TargetPointResult | None = None
        self.last_blocking_result: BlockingAnalysisResult | None = None
        self.last_avoidance_result: AvoidanceTargetResult | None = None
        self.last_detected_objects: list[DetectedObject] = []
        self.last_gold_result: GoldTargetResult | None = None
        self.last_fork_result: ForkRouteResult | None = None
        self.frame_id = 0

        self.mp_context = mp.get_context("spawn")
        self.stop_event = self.mp_context.Event()
        self.ai_input_queue = self.mp_context.Queue(maxsize=1)
        self.ai_output_queue = self.mp_context.Queue(maxsize=1)
        self.ui_queue = self.mp_context.Queue(maxsize=1)
        self.ai_ack_queue = self.mp_context.Queue()
        self.lane_output_queue = self.mp_context.Queue(maxsize=max(2, len(worker_core_masks) * 2))
        self.lane_input_queues: list[Any] = []
        self.lane_ack_queues: list[Any] = []
        self.lane_frame_pools: list[SharedArrayPool] = []
        self.lane_processes: list[Any] = []
        self.lane_worker_busy: list[bool] = []
        self.last_segmentation_result: SegmentationResult | None = None
        self.last_segmentation_frame_id = -1
        self.ui_resized_ack_queue = self.mp_context.Queue()
        self.ui_roi_ack_queue = self.mp_context.Queue()
        self.ai_frame_pool = SharedArrayPool("ai_frame", self.ai_ack_queue)
        self.ui_resized_pool = SharedArrayPool("ui_resized", self.ui_resized_ack_queue)
        self.ui_roi_pool = SharedArrayPool("ui_roi", self.ui_roi_ack_queue)
        self.shared_pools = {
            self.ai_frame_pool.pool_id: self.ai_frame_pool,
            self.ui_resized_pool.pool_id: self.ui_resized_pool,
            self.ui_roi_pool.pool_id: self.ui_roi_pool,
        }
        if self.lane_parallel_enabled:
            for worker_index, core_mask in enumerate(worker_core_masks):
                input_queue = self.mp_context.Queue(maxsize=1)
                ack_queue = self.mp_context.Queue()
                pool_id = f"lane_frame_{worker_index}"
                frame_pool = SharedArrayPool(pool_id, ack_queue, slot_count=2)
                worker_config = copy.deepcopy(lane_config)
                worker_config["core_mask"] = str(core_mask)
                process = self.mp_context.Process(
                    target=_lane_inference_worker,
                    args=(worker_index, worker_config, input_queue, self.lane_output_queue, ack_queue, pool_id, self.stop_event),
                    name=f"xsmart-lane-inference-{worker_index}",
                )
                self.lane_input_queues.append(input_queue)
                self.lane_ack_queues.append(ack_queue)
                self.lane_frame_pools.append(frame_pool)
                self.lane_processes.append(process)
                self.lane_worker_busy.append(False)
                self.shared_pools[pool_id] = frame_pool
        self.ai_process = self.mp_context.Process(
            target=_ai_inference_worker,
            args=(
                config.get("rknn_object_detector", {}),
                self.ai_input_queue,
                self.ai_output_queue,
                self.ai_ack_queue,
                self.stop_event,
            ),
            name="xsmart-ai-inference",
        )
        self.ui_process = self.mp_context.Process(
            target=_ui_worker,
            args=(
                config.get("visualizer", {}),
                self.ui_queue,
                self.ui_resized_ack_queue,
                self.ui_roi_ack_queue,
                self.stop_event,
            ),
            name="xsmart-ui",
        )
        visualizer_config = config.get("visualizer", {})
        self.ui_active = bool(
            visualizer_config.get("show_window", True)
            or visualizer_config.get("save_video", False)
            or visualizer_config.get("save_screenshot", False)
        )
        self.ui_frame_stride = max(1, int(visualizer_config.get("frame_stride", 3)))

    def run(self) -> None:
        """执行上位机主循环。

        输入:
            无。

        输出:
            无返回值，函数内部持续运行直到视频结束或用户退出。
        """

        self.camera.open()
        self.bridge.connect()
        self.csv_logger.open()
        self.ai_process.start()
        for process in self.lane_processes:
            process.start()
        if self.ui_active:
            self.ui_process.start()

        while not self.stop_event.is_set():
            success, frame = self.camera.read()
            if not success or frame is None:
                if self.camera.mode == "video" and not self.camera.loop_video:
                    print("视频读取结束，主循环退出。")
                    break
                print("图像读取失败，等待下一次重试。")
                continue

            # 第 1 步：先预处理，再用 YOLO 读取 coin、障碍物和岔路标志。
            lane_start_time = time.perf_counter()
            preprocess_result = self.preprocessor.process(frame)
            lane_preprocess_time = time.perf_counter()
            self.frame_id += 1
            ai_frame_payload = self.ai_frame_pool.write(preprocess_result.resized_frame)
            if ai_frame_payload is not None:
                _put_latest(
                    self.ai_input_queue,
                    (self.frame_id, ai_frame_payload),
                    release_func=self._release_owned_shared_payload,
                )
            latest_ai_result = _drain_latest(self.ai_output_queue)
            if latest_ai_result is not None:
                _frame_id, detected_objects = latest_ai_result
                self.last_detected_objects = detected_objects
            self.last_fork_result = self.fork_route_planner.update(self.last_detected_objects)

            # 第 2 步：在 ROI 中找蓝色航道，并根据已锁定的岔路方向选左/右分支。
            segmentation_result = self._segment_lane(preprocess_result.resized_frame)
            roi_x1, roi_y1, roi_x2, roi_y2 = preprocess_result.roi_rect
            roi_mask = segmentation_result.mask[roi_y1:roi_y2, roi_x1:roi_x2]
            detection_result = self.detector.detect_from_mask(
                roi_mask,
                route_direction=self.last_fork_result.requested_direction,
                segmentation_confidence=segmentation_result.confidence,
                segmentation_status=segmentation_result.status,
            )
            lane_detect_time = time.perf_counter()
            self.last_fork_result = self.fork_route_planner.update(
                self.last_detected_objects,
                fork_detected=detection_result.fork_result.fork_detected,
            )

            # 第 3 步：把当前帧结果和历史结果融合，岔路选中帧优先相信当前分支。
            tracked_state = self.tracker.update(
                detection_result,
                prefer_current=detection_result.fork_result.selected_direction is not None,
            )
            lane_track_time = time.perf_counter()
            self._record_lane_timing(
                total_ms=(lane_track_time - lane_start_time) * 1000.0,
                preprocess_ms=(lane_preprocess_time - lane_start_time) * 1000.0,
                detect_ms=(lane_detect_time - lane_preprocess_time) * 1000.0,
                track_ms=(lane_track_time - lane_detect_time) * 1000.0,
            )
            planning_state = self._build_planning_state(
                preprocess_result=preprocess_result,
                detection_result=detection_result,
                tracked_state=tracked_state,
            )
            # 第 4 步：为后续 OCR、红绿灯、金币规划等模块预留融合入口。
            module_hints = self._collect_future_module_hints(
                preprocess_result=preprocess_result,
                detection_result=detection_result,
                tracked_state=planning_state,
            )
            # 第 5 步：把视觉结果变成“高层目标速度、目标转向”。
            control_command = self.planner.plan(planning_state, module_hints=module_hints)

            # 第 6 步：通过桥接层发给下位机，至于串口协议细节由 bridge/protocol 负责。
            payload = self._build_payload(control_command, planning_state)
            self.bridge.send(payload)
            fps_value = self.fps_counter.update()

            # 第 7 步：把关键数据落盘，方便赛后分析和调参。
            self.csv_logger.log(
                {
                    "timestamp_ms": control_command.ts_ms,
                    "lateral_error_px": planning_state.lateral_error_px,
                    "heading_error_deg": planning_state.heading_error_deg,
                    "curvature": planning_state.curvature,
                    "confidence": planning_state.confidence,
                    "target_speed": control_command.target_speed,
                    "steer_deg": control_command.steer_deg,
                    "lane_lost_count": tracked_state.lane_lost_count,
                }
            )

            # 第 8 步：显示调试画面，看掩膜、中心线、误差和控制量是否正常。
            ui_preprocess_payload = None
            should_render_ui = self.ui_active and self.frame_id % self.ui_frame_stride == 0
            if should_render_ui:
                ui_preprocess_payload = _share_preprocess_result(
                    preprocess_result,
                    resized_pool=self.ui_resized_pool,
                    roi_pool=self.ui_roi_pool,
                )
            if should_render_ui and ui_preprocess_payload is not None:
                _put_latest(
                    self.ui_queue,
                    {
                        "preprocess_result": ui_preprocess_payload,
                        "detection_result": detection_result,
                        "tracked_state": planning_state,
                        "control_command": control_command,
                        "fps_value": fps_value,
                        "target_result": self.last_target_result,
                        "blocking_result": self.last_blocking_result,
                        "avoidance_result": self.last_avoidance_result,
                        "detected_objects": self.last_detected_objects,
                        "gold_result": self.last_gold_result,
                        "fork_route_result": self.last_fork_result,
                    },
                    release_func=self._release_owned_shared_payload,
                )

    def _segment_lane(self, frame: np.ndarray) -> SegmentationResult:
        """Segment synchronously or through the two-core pipelined worker pool."""

        if not self.lane_parallel_enabled:
            assert self.lane_segmenter is not None
            return self.lane_segmenter.segment(frame)

        free_worker = next((i for i, busy in enumerate(self.lane_worker_busy) if not busy), None)
        if free_worker is None:
            self._receive_lane_result(block=True)
            free_worker = next(i for i, busy in enumerate(self.lane_worker_busy) if not busy)

        descriptor = self.lane_frame_pools[free_worker].write(frame)
        if descriptor is not None:
            self.lane_input_queues[free_worker].put((self.frame_id, descriptor))
            self.lane_worker_busy[free_worker] = True

        while self._receive_lane_result(block=False):
            pass
        if self.last_segmentation_result is None:
            self._receive_lane_result(block=True)
        assert self.last_segmentation_result is not None
        return self.last_segmentation_result

    def _receive_lane_result(self, block: bool) -> bool:
        try:
            item = self.lane_output_queue.get(timeout=2.0) if block else self.lane_output_queue.get_nowait()
        except queue.Empty:
            return False
        frame_id, worker_index, mask_shape, packed_mask, instances, confidence, status = item
        self.lane_worker_busy[int(worker_index)] = False
        if int(frame_id) > self.last_segmentation_frame_id:
            mask_size = int(mask_shape[0]) * int(mask_shape[1])
            mask = np.unpackbits(packed_mask, count=mask_size, bitorder="little").reshape(mask_shape)
            mask = mask.astype(np.uint8, copy=False) * np.uint8(255)
            result = SegmentationResult(mask, instances, confidence, status)
            self.last_segmentation_frame_id = int(frame_id)
            self.last_segmentation_result = result
        return True

    def _release_owned_shared_payload(self, value: Any) -> None:
        """Release queued shared-memory slots owned by this app instance."""

        _release_shared_payload(value, self.shared_pools)

    def _record_lane_timing(
        self,
        total_ms: float,
        preprocess_ms: float,
        detect_ms: float,
        track_ms: float,
    ) -> None:
        """Collect lane timing stats and print a low-frequency summary."""

        if not self.lane_timing_enabled:
            return

        self.lane_timing_count += 1
        self.lane_timing_total_ms += total_ms
        self.lane_timing_preprocess_ms += preprocess_ms
        self.lane_timing_detect_ms += detect_ms
        self.lane_timing_track_ms += track_ms
        self.lane_timing_max_ms = max(self.lane_timing_max_ms, total_ms)

        if self.lane_timing_count < self.lane_timing_interval:
            return

        count = float(self.lane_timing_count)
        print(
            "巡线耗时: "
            f"avg={self.lane_timing_total_ms / count:.2f} ms, "
            f"max={self.lane_timing_max_ms:.2f} ms, "
            f"pre={self.lane_timing_preprocess_ms / count:.2f} ms, "
            f"detect={self.lane_timing_detect_ms / count:.2f} ms, "
            f"track={self.lane_timing_track_ms / count:.2f} ms"
        )

        self.lane_timing_count = 0
        self.lane_timing_total_ms = 0.0
        self.lane_timing_preprocess_ms = 0.0
        self.lane_timing_detect_ms = 0.0
        self.lane_timing_track_ms = 0.0
        self.lane_timing_max_ms = 0.0

    def close(self) -> None:
        """释放主程序中创建的所有资源。

        输入:
            无。

        输出:
            无返回值。
        """

        self.camera.release()
        if self.lane_segmenter is not None:
            self.lane_segmenter.close()
        self.bridge.close()
        self.stop_event.set()
        _put_latest(self.ai_input_queue, None, release_func=self._release_owned_shared_payload)
        if self.ui_active:
            _put_latest(self.ui_queue, None, release_func=self._release_owned_shared_payload)
        for lane_queue in self.lane_input_queues:
            _put_latest(lane_queue, None, release_func=self._release_owned_shared_payload)
        processes = [self.ai_process, *self.lane_processes]
        if self.ui_active:
            processes.append(self.ui_process)
        for process in processes:
            if process.pid is None:
                continue
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        for pool in self.shared_pools.values():
            pool.close()
        self.csv_logger.close()

    def _build_payload(
        self,
        control_command: ControlCommand,
        tracked_state: TrackedLaneState,
    ) -> Dict[str, Any]:
        """整理发送给下位机的高层指令负载。

        输入:
            control_command: 当前帧高层控制量。
            tracked_state: 当前帧平滑后的巡线状态。

        输出:
            返回协议打包前的字段字典。
        """

        return {
            "ts_ms": control_command.ts_ms,
            "mode": control_command.mode,
            "target_speed": control_command.target_speed,
            "steer_deg": control_command.steer_deg,
            "lateral_error_px": tracked_state.lateral_error_px,
            "heading_error_deg": tracked_state.heading_error_deg,
            "curvature": tracked_state.curvature,
            "confidence": tracked_state.confidence,
            "is_lane_lost": tracked_state.is_lane_lost,
        }

    def _collect_future_module_hints(
        self,
        preprocess_result: Any,
        detection_result: Any,
        tracked_state: TrackedLaneState,
    ) -> ModuleHints:
        """为后续目标检测、OCR、红绿灯和金币规划模块预留提示接口。

        输入:
            preprocess_result: 当前帧预处理结果，便于后续模块直接复用 ROI。
            detection_result: 当前帧巡线检测结果，便于其他模块参考中心线或置信度。
            tracked_state: 当前帧平滑后的巡线状态。

        输出:
            返回 ModuleHints。
            当前版本默认返回空提示，不改变主链路行为；
            后续可在此接入红绿灯限速、OCR 区域规则、金币规划限速等高层策略。
        """

        _ = preprocess_result
        _ = detection_result
        _ = tracked_state
        if (
            self.last_gold_result is not None
            and self.last_gold_result.active
            and self.last_avoidance_result is not None
            and self.last_avoidance_result.mode == "gold_target"
        ):
            return ModuleHints(
                speed_limit=self.last_gold_result.speed_limit,
                force_mode="GOLD",
                note=self.last_gold_result.reason,
            )
        return ModuleHints()

    def _detect_objects_for_roi(
        self,
        preprocess_result: PreprocessResult,
        detection_result: LaneDetectionResult,
        tracked_state: TrackedLaneState,
    ) -> list[DetectedObject]:
        """Return vehicle/person boxes with ROI coordinates attached.

        This hook intentionally defaults to no detections. A real ObjectDetector should
        provide DetectedObject(class_name, confidence, bbox_frame) in resized_frame
        coordinates; this method clips each bbox into bbox_roi for BlockingAnalyzer.
        """

        _ = detection_result
        _ = tracked_state
        objects_from_detector = [
            obj for obj in self.last_detected_objects
            if obj.class_name.casefold() in self.blocking_class_names
        ]
        roi_height, roi_width = preprocess_result.roi_frame.shape[:2]
        return attach_roi_bboxes(
            objects=objects_from_detector,
            roi_rect=preprocess_result.roi_rect,
            roi_width=roi_width,
            roi_height=roi_height,
        )

    def _build_planning_state(
        self,
        preprocess_result: PreprocessResult,
        detection_result: LaneDetectionResult,
        tracked_state: TrackedLaneState,
    ) -> TrackedLaneState:
        """Replace lane metrics with target-point metrics before high-level planning."""

        roi_height, roi_width = preprocess_result.roi_frame.shape[:2]
        centerline_points = tracked_state.centerline_points or detection_result.centerline_points
        normal_target = self.target_selector.select(
            centerline_points=centerline_points,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=tracked_state.confidence,
            curvature=tracked_state.curvature,
        )
        if self.last_fork_result is not None and self.last_fork_result.active:
            gold_result = GoldTargetResult(
                active=False,
                target_object=None,
                target_point_roi=(float(roi_width) * 0.5, float(max(0, roi_height - 1))),
                final_lateral_error_px=0.0,
                final_heading_error_deg=0.0,
                confidence=0.0,
                speed_limit=None,
                reason=f"fork route active: {self.last_fork_result.requested_direction}",
            )
        else:
            gold_result = self.gold_target_planner.plan(
                objects=self.last_detected_objects,
                roi_rect=preprocess_result.roi_rect,
                roi_width=roi_width,
                roi_height=roi_height,
            )
        self.last_gold_result = gold_result
        objects = self._detect_objects_for_roi(
            preprocess_result=preprocess_result,
            detection_result=detection_result,
            tracked_state=tracked_state,
        )
        blocking_result = self.blocking_analyzer.analyze(
            objects=objects,
            centerline_points=centerline_points,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        avoidance_result = self.avoidance_planner.plan(
            centerline_points=centerline_points,
            normal_target=normal_target,
            blocking_result=blocking_result,
            roi_width=roi_width,
            roi_height=roi_height,
            lane_confidence=tracked_state.confidence,
            curvature=tracked_state.curvature,
        )

        if blocking_result.need_avoid or blocking_result.too_close:
            self.last_target_result = normal_target
            self.last_blocking_result = blocking_result
            self.last_avoidance_result = avoidance_result
            return replace(
                tracked_state,
                centerline_points=[
                    (int(round(x)), int(round(y)))
                    for x, y in avoidance_result.shifted_centerline_points
                ],
                lateral_error_px=avoidance_result.final_lateral_error_px,
                heading_error_deg=avoidance_result.final_heading_error_deg,
                confidence=min(tracked_state.confidence, avoidance_result.confidence),
            )

        if gold_result.active:
            self.last_target_result = normal_target
            self.last_blocking_result = blocking_result
            self.last_avoidance_result = AvoidanceTargetResult(
                mode="gold_target",
                shifted_centerline_points=[
                    (float(x), float(y))
                    for x, y in centerline_points
                ],
                target_point_roi=gold_result.target_point_roi,
                avoid_bias_px=0.0,
                final_lateral_error_px=gold_result.final_lateral_error_px,
                final_heading_error_deg=gold_result.final_heading_error_deg,
                confidence=gold_result.confidence,
                reason=gold_result.reason,
            )
            return replace(
                tracked_state,
                lateral_error_px=gold_result.final_lateral_error_px,
                heading_error_deg=gold_result.final_heading_error_deg,
                confidence=max(tracked_state.confidence, min(1.0, gold_result.confidence)),
                is_lane_lost=False,
            )

        self.last_target_result = normal_target
        self.last_blocking_result = blocking_result
        self.last_avoidance_result = avoidance_result

        return replace(
            tracked_state,
            centerline_points=[
                (int(round(x)), int(round(y)))
                for x, y in avoidance_result.shifted_centerline_points
            ],
            lateral_error_px=avoidance_result.final_lateral_error_px,
            heading_error_deg=avoidance_result.final_heading_error_deg,
            confidence=min(tracked_state.confidence, avoidance_result.confidence),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """构建主程序命令行参数解析器。

    输入:
        无。

    输出:
        返回配置完成的 ArgumentParser 对象。
    """

    parser = argparse.ArgumentParser(description="X-SmartCar 上位机视觉巡线主程序")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--mode", choices=["camera", "video"], help="图像源模式")
    parser.add_argument("--video", help="视频文件路径，仅视频模式下有效")
    parser.add_argument("--bridge", choices=["mock", "serial"], help="桥接层类型")
    parser.add_argument("--no-gui", action="store_true", help="关闭图像显示窗口")
    parser.add_argument("--save-video", action="store_true", help="保存调试视频")
    return parser


def main() -> int:
    """主程序入口，负责加载配置、创建应用并处理退出逻辑。

    输入:
        无。

    输出:
        返回进程退出码，0 表示成功退出。
    """

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    config = load_config(config_path)
    config = apply_cli_overrides(config, args)
    config = prepare_runtime_config(config, PROJECT_ROOT)

    app = UpperMachineApp(config, PROJECT_ROOT)
    try:
        app.run()
    except KeyboardInterrupt:
        print("收到键盘中断，程序准备退出。")
    except Exception as error:
        print(f"主程序异常退出: {error}")
        traceback.print_exc()
        return_code = 1
    else:
        return_code = 0
    finally:
        app.close()

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
