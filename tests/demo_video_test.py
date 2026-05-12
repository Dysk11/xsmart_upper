"""本地视频回放测试脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import UpperMachineApp, apply_cli_overrides, load_config, prepare_runtime_config


def build_arg_parser() -> argparse.ArgumentParser:
    """构建视频回放测试脚本的命令行参数解析器。

    输入:
        无。

    输出:
        返回配置完成的 ArgumentParser 对象。
    """

    parser = argparse.ArgumentParser(description="X-SmartCar 视频回放测试脚本")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--video", required=True, help="待回放的视频文件路径")
    parser.add_argument("--no-gui", action="store_true", help="关闭图像显示窗口")
    parser.add_argument("--save-video", action="store_true", help="保存调试视频")
    return parser


def main() -> int:
    """读取配置并以视频模式运行主程序，方便离线调试巡线算法。

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
    args.mode = "video"
    args.bridge = "mock"
    config = apply_cli_overrides(config, args)
    config = prepare_runtime_config(config, PROJECT_ROOT)

    app = UpperMachineApp(config, PROJECT_ROOT)
    try:
        app.run()
    finally:
        app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
