from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.camera import CameraReader


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def build_camera_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    camera_config = dict(config.get("camera", {}))

    if args.mode is not None:
        camera_config["mode"] = args.mode
    if args.device_id is not None:
        camera_config["device_id"] = args.device_id
    if args.video is not None:
        camera_config["mode"] = "video"
        camera_config["video_path"] = str(resolve_project_path(args.video))
    elif camera_config.get("video_path"):
        camera_config["video_path"] = str(resolve_project_path(str(camera_config["video_path"])))
    if args.stream_url is not None:
        camera_config["mode"] = "stream"
        camera_config["stream_url"] = args.stream_url
    if args.width is not None:
        camera_config["width"] = args.width
    if args.height is not None:
        camera_config["height"] = args.height
    if args.fps is not None:
        camera_config["fps"] = args.fps
    if args.mirror:
        camera_config["mirror"] = True

    return camera_config


def save_frame(frame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"screenshot_{timestamp}_{int(time.time() * 1000) % 1000:03d}.png"
    if not cv2.imwrite(str(file_path), frame):
        raise RuntimeError(f"Failed to save screenshot: {file_path}")
    return file_path


def run(args: argparse.Namespace) -> int:
    camera_config = build_camera_config(args)
    output_dir = resolve_project_path(args.output_dir)
    camera = CameraReader(camera_config)

    print("Opening image source...")
    print(f"Mode: {camera_config.get('mode', 'camera')}")
    print(f"Press 's' to save screenshots to: {output_dir}")
    print("Press 'q' or Esc to exit.")

    try:
        camera.open()
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("No frame was read from the image source.")
                return 1

            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return 0
            if key in (ord("s"), ord("S")):
                file_path = save_frame(frame, output_dir)
                print(f"Saved screenshot: {file_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a preview window and save screenshots with the s key.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"), help="Config file path")
    parser.add_argument("--mode", choices=["camera", "video", "stream", "http"], help="Image source mode")
    parser.add_argument("--device-id", type=int, help="Camera device id")
    parser.add_argument("--video", help="Video file path; sets mode to video")
    parser.add_argument("--stream-url", help="HTTP/RTSP stream URL; sets mode to stream")
    parser.add_argument("--width", type=int, help="Capture width")
    parser.add_argument("--height", type=int, help="Capture height")
    parser.add_argument("--fps", type=int, help="Capture FPS")
    parser.add_argument("--mirror", action="store_true", help="Mirror the preview horizontally")
    parser.add_argument("--output-dir", default="output/visual", help="Screenshot output directory")
    parser.add_argument("--window-name", default="X-SmartCar Screenshot", help="OpenCV preview window name")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
