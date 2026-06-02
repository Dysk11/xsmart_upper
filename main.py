"""X-SmartCar 人工智能模型组上位机主程序。"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import queue
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

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
from core.target_selector import TargetPointResult, TargetSelector
from core.visualizer import Visualizer
from utils.fps import FPSCounter


def _put_latest(work_queue: Any, item: Any) -> None:
    """Put an item into a size-limited multiprocessing queue, dropping stale data."""

    try:
        work_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        work_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        work_queue.put_nowait(item)
    except queue.Full:
        pass


def _drain_latest(work_queue: Any) -> Any | None:
    """Return the newest available queue item without blocking."""

    latest = None
    while True:
        try:
            latest = work_queue.get_nowait()
        except queue.Empty:
            return latest


def _ai_inference_worker(
    detector_config: Dict[str, Any],
    input_queue: Any,
    output_queue: Any,
    stop_event: Any,
) -> None:
    """Run RKNN object detection in an independent process."""

    detector = RknnObjectDetector(detector_config)
    try:
        while not stop_event.is_set():
            try:
                item = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break

            frame_id, frame = item
            try:
                detections = detector.detect(frame)
                _put_latest(output_queue, (frame_id, detections))
            except Exception:
                traceback.print_exc()
                _put_latest(output_queue, (frame_id, []))
    finally:
        detector.close()


def _ui_worker(
    visualizer_config: Dict[str, Any],
    input_queue: Any,
    stop_event: Any,
) -> None:
    """Render OpenCV UI and video output in an independent process."""

    visualizer = Visualizer(visualizer_config)
    try:
        while not stop_event.is_set():
            try:
                packet = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if packet is None:
                break

            try:
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
        self.tracker = LaneTracker(config.get("tracker", {}))
        self.target_selector = TargetSelector(config.get("target_selector", {}))
        self.fork_route_planner = ForkRoutePlanner(config.get("fork_route", {}))
        self.gold_target_planner = GoldTargetPlanner(config.get("gold_target", {}))
        self.blocking_config = config.get("blocking_analyzer", {})
        self.blocking_class_names = {
            str(name).casefold()
            for name in self.blocking_config.get("allowed_class_names", ["Car", "Human"])
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
        self.ai_process = self.mp_context.Process(
            target=_ai_inference_worker,
            args=(
                config.get("rknn_object_detector", {}),
                self.ai_input_queue,
                self.ai_output_queue,
                self.stop_event,
            ),
            name="xsmart-ai-inference",
        )
        self.ui_process = self.mp_context.Process(
            target=_ui_worker,
            args=(
                config.get("visualizer", {}),
                self.ui_queue,
                self.stop_event,
            ),
            name="xsmart-ui",
        )

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
        self.ui_process.start()

        while not self.stop_event.is_set():
            success, frame = self.camera.read()
            if not success or frame is None:
                if self.camera.mode == "video" and not self.camera.loop_video:
                    print("视频读取结束，主循环退出。")
                    break
                print("图像读取失败，等待下一次重试。")
                continue

            # 第 1 步：先预处理，再用 YOLO 读取 Gold、障碍物和岔路标志。
            lane_start_time = time.perf_counter()
            preprocess_result = self.preprocessor.process(frame)
            self.frame_id += 1
            _put_latest(self.ai_input_queue, (self.frame_id, preprocess_result.resized_frame))
            latest_ai_result = _drain_latest(self.ai_output_queue)
            if latest_ai_result is not None:
                _frame_id, detected_objects = latest_ai_result
                self.last_detected_objects = detected_objects
            self.last_fork_result = self.fork_route_planner.update(self.last_detected_objects)

            # 第 2 步：在 ROI 中找蓝色航道，并根据已锁定的岔路方向选左/右分支。
            detection_result = self.detector.detect(
                preprocess_result.roi_frame,
                route_direction=self.last_fork_result.requested_direction,
            )
            self.last_fork_result = self.fork_route_planner.update(
                self.last_detected_objects,
                fork_detected=detection_result.fork_result.fork_detected,
            )

            # 第 3 步：把当前帧结果和历史结果融合，岔路选中帧优先相信当前分支。
            tracked_state = self.tracker.update(
                detection_result,
                prefer_current=detection_result.fork_result.selected_direction is not None,
            )
            lane_elapsed_ms = (time.perf_counter() - lane_start_time) * 1000.0
            print(f"巡线耗时: {lane_elapsed_ms:.2f} ms")
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
            _put_latest(
                self.ui_queue,
                {
                    "preprocess_result": preprocess_result,
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
            )

    def close(self) -> None:
        """释放主程序中创建的所有资源。

        输入:
            无。

        输出:
            无返回值。
        """

        self.camera.release()
        self.bridge.close()
        self.stop_event.set()
        _put_latest(self.ai_input_queue, None)
        _put_latest(self.ui_queue, None)
        for process in (self.ai_process, self.ui_process):
            if process.pid is None:
                continue
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
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
