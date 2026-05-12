"""X-SmartCar 人工智能模型组上位机主程序。"""

from __future__ import annotations

import argparse
import copy
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bridge import BaseVehicleBridge, build_vehicle_bridge
from core.camera import CameraReader
from core.lane_detector import LaneDetectionResult, LaneDetector
from core.lane_tracker import LaneTracker, TrackedLaneState
from core.logger import CsvLogger
from core.planner import ControlCommand, HighLevelPlanner, ModuleHints
from core.preprocess import ImagePreprocessor, PreprocessResult
from core.visualizer import Visualizer
from utils.fps import FPSCounter


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
        self.planner = HighLevelPlanner(config.get("planner", {}))
        self.bridge: BaseVehicleBridge = build_vehicle_bridge(config.get("bridge", {}))
        self.visualizer = Visualizer(config.get("visualizer", {}))
        self.csv_logger = CsvLogger(config.get("logger", {}))
        self.fps_counter = FPSCounter(config.get("app", {}).get("fps_smoothing_alpha", 0.2))

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

        while True:
            success, frame = self.camera.read()
            if not success or frame is None:
                if self.camera.mode == "video" and not self.camera.loop_video:
                    print("视频读取结束，主循环退出。")
                    break
                print("图像读取失败，等待下一次重试。")
                continue

            # 第 1 步：把原始图像裁成适合巡线的 ROI，并做基础增强。
            preprocess_result = self.preprocessor.process(frame)
            # 第 2 步：在 ROI 中找蓝色航道，输出中心线和误差等信息。
            detection_result = self.detector.detect(preprocess_result.roi_frame)
            # 第 3 步：把当前帧结果和历史结果融合，减少抖动。
            tracked_state = self.tracker.update(detection_result)
            # 第 4 步：为后续 OCR、红绿灯、金币规划等模块预留融合入口。
            module_hints = self._collect_future_module_hints(
                preprocess_result=preprocess_result,
                detection_result=detection_result,
                tracked_state=tracked_state,
            )
            # 第 5 步：把视觉结果变成“高层目标速度、目标转向”。
            control_command = self.planner.plan(tracked_state, module_hints=module_hints)

            # 第 6 步：通过桥接层发给下位机，至于串口协议细节由 bridge/protocol 负责。
            payload = self._build_payload(control_command, tracked_state)
            self.bridge.send(payload)
            fps_value = self.fps_counter.update()

            # 第 7 步：把关键数据落盘，方便赛后分析和调参。
            self.csv_logger.log(
                {
                    "timestamp_ms": control_command.ts_ms,
                    "lateral_error_px": tracked_state.lateral_error_px,
                    "heading_error_deg": tracked_state.heading_error_deg,
                    "curvature": tracked_state.curvature,
                    "confidence": tracked_state.confidence,
                    "target_speed": control_command.target_speed,
                    "steer_deg": control_command.steer_deg,
                    "lane_lost_count": tracked_state.lane_lost_count,
                }
            )

            # 第 8 步：显示调试画面，看掩膜、中心线、误差和控制量是否正常。
            should_continue = self.visualizer.render(
                preprocess_result=preprocess_result,
                detection_result=detection_result,
                tracked_state=tracked_state,
                control_command=control_command,
                fps_value=fps_value,
            )
            if not should_continue:
                break

    def close(self) -> None:
        """释放主程序中创建的所有资源。

        输入:
            无。

        输出:
            无返回值。
        """

        self.camera.release()
        self.bridge.close()
        self.visualizer.close()
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
        return ModuleHints()


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
