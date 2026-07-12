"""HSV 实时滑块调参工具。"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.camera import CameraReader
from utils.image_utils import draw_text_lines, ensure_bgr, overlay_mask, stack_images


# OpenCV 原生滑块标题在 Windows 下对中文支持不稳定，
# 所以这里统一使用 ASCII 名称，中文说明放在预览画面里显示。
CONTROL_WINDOW_NAME = "HSV Tuner Controls"
PREVIEW_WINDOW_NAME = "HSV Tuner Preview"


@dataclass
class TrackbarState:
    """保存当前滑块对应的调参状态。"""

    roi_top_ratio: float
    h_low: int
    h_high: int
    s_low: int
    v_low: int
    close_kernel: int
    min_area: int
    s_high: int
    v_high: int
    open_kernel: int
    dilate_iterations: int
    min_height: int


def load_config(config_path: Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。

    输入:
        config_path: 配置文件路径。

    输出:
        返回配置字典；若配置文件为空则返回空字典。
    """

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_project_path(project_root: Path, raw_path: str) -> Path:
    """将相对路径解析成相对于项目根目录的绝对路径。

    输入:
        project_root: 项目根目录。
        raw_path: 原始路径字符串。

    输出:
        返回解析后的绝对路径。
    """

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def prepare_runtime_config(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """把配置中的路径字段转换为可直接运行的绝对路径。

    输入:
        config: 原始配置字典。
        project_root: 项目根目录。

    输出:
        返回路径已处理好的配置副本。
    """

    runtime_config = copy.deepcopy(config)
    camera_config = runtime_config.setdefault("camera", {})
    if camera_config.get("video_path"):
        camera_config["video_path"] = str(resolve_project_path(project_root, str(camera_config["video_path"])))

    visualizer_config = runtime_config.setdefault("visualizer", {})
    if visualizer_config.get("font_path"):
        visualizer_config["font_path"] = str(resolve_project_path(project_root, str(visualizer_config["font_path"])))

    return runtime_config


def noop(_: int) -> None:
    """作为 OpenCV 滑块的空回调函数。

    输入:
        _: OpenCV 传入的滑块数值。

    输出:
        无返回值。
    """


def ensure_odd(value: int) -> int:
    """把卷积核尺寸修正为大于等于 1 的奇数。

    输入:
        value: 原始核尺寸。

    输出:
        返回修正后的奇数核尺寸。
    """

    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


class HsvTunerApp:
    """通过滑块实时查看蓝色阈值分割效果的调参应用。"""

    def __init__(self, config: Dict[str, Any], config_path: Path) -> None:
        """根据现有项目配置初始化调参工具。

        输入:
            config: 运行时配置字典。
            config_path: 当前使用的配置文件路径。

        输出:
            无返回值。
        """

        self.config = copy.deepcopy(config)
        self.config_path = config_path
        self.camera = CameraReader(self.config.get("camera", {}))
        self.base_roi_config = copy.deepcopy(self.config.get("lane_geometry", {}).get("roi", {}))
        self.visualizer_config = copy.deepcopy(self.config.get("visualizer", {}))
        self.font_path = str(self.visualizer_config.get("font_path", ""))
        self.font_size = int(self.visualizer_config.get("font_size", 22))
        self.snapshot_output_path = PROJECT_ROOT / "outputs" / "hsv_tuner_last_snippet.yaml"
        self.snapshot_output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """启动摄像头和滑块窗口，进入实时调参循环。

        输入:
            无。

        输出:
            无返回值；用户退出窗口后自动释放资源。
        """

        self.camera.open()
        self._create_windows()

        while True:
            success, frame = self.camera.read()
            if not success or frame is None:
                if self.camera.mode == "video" and not self.camera.loop_video:
                    print("视频读取结束，调参工具退出。")
                    break
                print("图像读取失败，等待重试。")
                continue

            state = self._read_trackbar_state()
            roi_frame, roi_rect = self._crop_roi(frame, state)
            raw_mask, filtered_mask = self._build_masks(roi_frame, state)
            canvas = self._build_canvas(frame, roi_rect, raw_mask, filtered_mask, state)

            cv2.imshow(PREVIEW_WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("p"), ord("P")):
                print("\n当前推荐配置片段：")
                print(self._build_yaml_snippet(state))
            if key in (ord("s"), ord("S")):
                self._save_snapshot(state)

        self.close()

    def close(self) -> None:
        """释放相机资源并关闭所有窗口。

        输入:
            无。

        输出:
            无返回值。
        """

        self.camera.release()
        cv2.destroyAllWindows()

    def _create_windows(self) -> None:
        """创建预览窗口与滑块控制窗口。

        输入:
            无。

        输出:
            无返回值。
        """

        cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

        detector_config = self.config.get("detector", {})
        hsv_config = detector_config.get("hsv", {})
        morphology_config = detector_config.get("morphology", {})
        component_config = detector_config.get("connected_components", {})
        roi_config = self.base_roi_config

        lower = hsv_config.get("lower", [85, 70, 40])
        upper = hsv_config.get("upper", [140, 255, 255])

        cv2.createTrackbar(
            "1_ROI_top_pct",
            CONTROL_WINDOW_NAME,
            int(float(roi_config.get("top_ratio", 0.35)) * 100.0),
            95,
            noop,
        )
        cv2.createTrackbar(
            "2_H_low",
            CONTROL_WINDOW_NAME,
            int(lower[0]),
            179,
            noop,
        )
        cv2.createTrackbar(
            "3_H_high",
            CONTROL_WINDOW_NAME,
            int(upper[0]),
            179,
            noop,
        )
        cv2.createTrackbar(
            "4_S_low",
            CONTROL_WINDOW_NAME,
            int(lower[1]),
            255,
            noop,
        )
        cv2.createTrackbar(
            "5_V_low",
            CONTROL_WINDOW_NAME,
            int(lower[2]),
            255,
            noop,
        )
        cv2.createTrackbar(
            "6_CloseLevel",
            CONTROL_WINDOW_NAME,
            max(0, (int(morphology_config.get("close_kernel", 7)) - 1) // 2),
            10,
            noop,
        )
        cv2.createTrackbar(
            "7_MinArea",
            CONTROL_WINDOW_NAME,
            int(component_config.get("min_area", 250)),
            5000,
            noop,
        )

    def _read_trackbar_state(self) -> TrackbarState:
        """读取当前所有滑块数值，并做合法性修正。

        输入:
            无。

        输出:
            返回整理后的 TrackbarState。
        """

        roi_top_ratio = cv2.getTrackbarPos("1_ROI_top_pct", CONTROL_WINDOW_NAME) / 100.0
        h_low = cv2.getTrackbarPos("2_H_low", CONTROL_WINDOW_NAME)
        h_high = cv2.getTrackbarPos("3_H_high", CONTROL_WINDOW_NAME)
        s_low = cv2.getTrackbarPos("4_S_low", CONTROL_WINDOW_NAME)
        v_low = cv2.getTrackbarPos("5_V_low", CONTROL_WINDOW_NAME)
        close_level = cv2.getTrackbarPos("6_CloseLevel", CONTROL_WINDOW_NAME)
        min_area = cv2.getTrackbarPos("7_MinArea", CONTROL_WINDOW_NAME)

        close_kernel = ensure_odd(1 + close_level * 2)
        s_high = 255
        v_high = 255

        h_high = max(h_high, h_low)
        cv2.setTrackbarPos("3_H_high", CONTROL_WINDOW_NAME, h_high)

        morphology_config = self.config.get("detector", {}).get("morphology", {})
        component_config = self.config.get("detector", {}).get("connected_components", {})
        open_kernel = ensure_odd(int(morphology_config.get("open_kernel", 1)))
        dilate_iterations = int(morphology_config.get("dilate_iterations", 1))
        min_height = int(component_config.get("min_height", 12))

        return TrackbarState(
            roi_top_ratio=roi_top_ratio,
            h_low=h_low,
            h_high=h_high,
            s_low=s_low,
            v_low=v_low,
            close_kernel=close_kernel,
            min_area=min_area,
            s_high=s_high,
            v_high=v_high,
            open_kernel=open_kernel,
            dilate_iterations=dilate_iterations,
            min_height=min_height,
        )

    def _crop_roi(self, frame: np.ndarray, state: TrackbarState):
        """基于当前滑块状态动态生成预处理器。

        输入:
            state: 当前滑块参数状态。

        输出:
            返回当前帧 ROI 和其矩形。
        """

        height, width = frame.shape[:2]
        config = self.base_roi_config
        x1 = max(0, min(width - 1, int(width * float(config.get("left_ratio", 0.05)))))
        x2 = max(x1 + 1, min(width, int(width * float(config.get("right_ratio", 0.95)))))
        y1 = max(0, min(height - 1, int(height * state.roi_top_ratio)))
        y2 = max(y1 + 1, min(height, int(height * float(config.get("bottom_ratio", 1.0)))))
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _build_masks(
        self,
        roi_frame: np.ndarray,
        state: TrackbarState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据当前滑块参数构建原始掩膜和筛选后的主航道掩膜。

        输入:
            roi_frame: 预处理后的 ROI 图像。
            state: 当前滑块参数状态。

        输出:
            返回二元组 (raw_mask, filtered_mask)。
        """

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        lower = np.asarray([state.h_low, state.s_low, state.v_low], dtype=np.uint8)
        upper = np.asarray([state.h_high, state.s_high, state.v_high], dtype=np.uint8)
        raw_mask = cv2.inRange(hsv, lower, upper)

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (state.open_kernel, state.open_kernel))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (state.close_kernel, state.close_kernel))
        filtered_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel)
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, close_kernel)
        if state.dilate_iterations > 0:
            filtered_mask = cv2.dilate(filtered_mask, close_kernel, iterations=state.dilate_iterations)

        filtered_mask = self._filter_connected_components(
            filtered_mask,
            min_area=state.min_area,
            min_height=state.min_height,
        )
        return raw_mask, filtered_mask

    def _filter_connected_components(
        self,
        mask: np.ndarray,
        min_area: int,
        min_height: int,
    ) -> np.ndarray:
        """按面积和高度筛选主航道候选连通域。

        输入:
            mask: 形态学处理后的二值掩膜。
            min_area: 最小连通域面积阈值。
            min_height: 最小连通域高度阈值。

        输出:
            返回筛选后的二值掩膜。
        """

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for label in range(1, num_labels):
            _, _, _, height, area = stats[label]
            if area >= min_area and height >= min_height:
                filtered[labels == label] = 255
        return filtered

    def _build_canvas(
        self,
        frame: np.ndarray,
        roi_rect: tuple[int, int, int, int],
        raw_mask: np.ndarray,
        filtered_mask: np.ndarray,
        state: TrackbarState,
    ) -> np.ndarray:
        """构建调参工具的双画面预览。

        输入:
            frame: 当前原始帧。
            raw_mask: 当前 HSV 阈值得到的原始掩膜。
            filtered_mask: 筛选后的主航道掩膜。
            state: 当前滑块参数状态。

        输出:
            返回拼接好的预览画面。
        """

        x1, y1, x2, y2 = roi_rect
        original_panel = frame.copy()
        cv2.rectangle(original_panel, (x1, y1), (x2, y2), (0, 255, 255), 2)
        original_panel = draw_text_lines(
            original_panel,
            [
                "窗口1：原始画面与ROI范围",
                "第1步先调：ROI上边界，让黄色框只看地面和航道",
                "ROI调大 = 看得更近更稳；ROI调小 = 看得更远但更容易误检",
            ],
            font_path=self.font_path,
            font_size=self.font_size,
        )

        filtered_panel = ensure_bgr(filtered_mask)
        filtered_panel = draw_text_lines(
            filtered_panel,
            [
                "窗口3：筛选后的主航道掩膜",
                "第6到7步调：补洞强度、最小面积",
                "理想状态是白色区域连续、干净，别有太多零碎小白块",
            ],
            font_path=self.font_path,
            font_size=self.font_size,
        )

        cell_width = max(360, frame.shape[1] // 2)
        cell_height = max(220, frame.shape[0] // 2)
        return stack_images(
            [original_panel, filtered_panel],
            cols=2,
            cell_size=(cell_width, cell_height),
        )

    def _build_yaml_snippet(self, state: TrackbarState) -> str:
        """把当前滑块状态转换成推荐粘贴回配置文件的 YAML 片段。

        输入:
            state: 当前滑块参数状态。

        输出:
            返回 YAML 文本字符串。
        """

        snippet = {
            "lane_geometry": {
                "roi": {
                    "top_ratio": round(state.roi_top_ratio, 3),
                }
            },
            "detector": {
                "color_space": "hsv",
                "hsv": {
                    "lower": [state.h_low, state.s_low, state.v_low],
                    "upper": [state.h_high, state.s_high, state.v_high],
                },
                "morphology": {
                    "open_kernel": state.open_kernel,
                    "close_kernel": state.close_kernel,
                    "dilate_iterations": state.dilate_iterations,
                },
                "connected_components": {
                    "min_area": state.min_area,
                    "min_height": state.min_height,
                },
            },
        }
        return yaml.safe_dump(snippet, allow_unicode=True, sort_keys=False)

    def _save_snapshot(self, state: TrackbarState) -> None:
        """把当前调参结果保存为 YAML 快照，方便后续复制回主配置。

        输入:
            state: 当前滑块参数状态。

        输出:
            无返回值；函数会在终端提示快照路径。
        """

        snapshot_text = self._build_yaml_snippet(state)
        self.snapshot_output_path.write_text(snapshot_text, encoding="utf-8")
        print(f"\n已保存当前参数快照到: {self.snapshot_output_path}")
        print(snapshot_text)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建调参工具的命令行参数解析器。

    输入:
        无。

    输出:
        返回 ArgumentParser 对象。
    """

    parser = argparse.ArgumentParser(description="X-SmartCar HSV 实时滑块调参工具")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--mode", choices=["camera", "video", "shared_memory"], help="图像源模式")
    parser.add_argument("--video", help="视频路径，仅视频模式下有效")
    parser.add_argument("--device-id", type=int, help="摄像头设备号")
    parser.add_argument("--loop-video", action="store_true", help="视频模式下循环播放，方便反复调参")
    return parser


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """根据命令行参数覆盖图像源配置。

    输入:
        config: 原始配置字典。
        args: 解析后的命令行参数。

    输出:
        返回覆盖后的配置字典。
    """

    runtime_config = copy.deepcopy(config)
    camera_config = runtime_config.setdefault("camera", {})
    if args.mode:
        camera_config["mode"] = args.mode
    if args.video:
        camera_config["video_path"] = args.video
        camera_config["mode"] = "video"
    if args.device_id is not None:
        camera_config["device_id"] = args.device_id
    if args.loop_video:
        camera_config["loop_video"] = True
    return runtime_config


def main() -> int:
    """HSV 调参工具入口。

    输入:
        无。

    输出:
        返回进程退出码，0 表示正常结束。
    """

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    config = load_config(config_path)
    config = apply_cli_overrides(config, args)
    config = prepare_runtime_config(config, PROJECT_ROOT)

    app = HsvTunerApp(config=config, config_path=config_path)
    try:
        app.run()
    except KeyboardInterrupt:
        print("收到键盘中断，调参工具退出。")
    finally:
        app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
