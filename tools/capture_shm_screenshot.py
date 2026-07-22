#!/usr/bin/env python3
"""Preview the AR shared-memory video and save screenshots on key press."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.io.camera import CameraReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="预览 shm_ar_video 共享内存画面；按 s 保存截图，按 q 或 Esc 退出。"
    )
    parser.add_argument(
        "--shared-memory-name",
        default="shm_ar_video",
        help="POSIX 共享内存名称（默认：shm_ar_video）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "screenshots",
        help="截图保存目录（默认：outputs/screenshots）",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="水平镜像预览和保存的画面",
    )
    parser.add_argument(
        "--window-name",
        default="Shared Memory Capture",
        help="预览窗口名称",
    )
    return parser.parse_args()


def save_screenshot(image, output_dir: Path, frame_id: int) -> Path:
    """Save one BGR frame and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_path = output_dir / f"shm_{timestamp}_frame_{frame_id}.png"
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"截图写入失败: {output_path}")
    return output_path


def main() -> int:
    args = parse_args()
    reader = CameraReader(
        {
            "mode": "shared_memory",
            "shared_memory_name": args.shared_memory_name,
            "mirror": args.mirror,
            "reconnect_interval_sec": 0.5,
            "max_reconnect_attempts": 5,
        }
    )

    try:
        reader.open()
    except RuntimeError as exc:
        print(f"无法打开共享内存画面: {exc}", file=sys.stderr)
        print("请先启动 AR/SetupUI，使其创建共享内存。", file=sys.stderr)
        return 1

    print(f"共享内存: {args.shared_memory_name}")
    print(f"截图目录: {args.output_dir.resolve()}")
    print("按 s 保存截图；按 q 或 Esc 退出。")

    last_failure_notice = 0.0
    try:
        while True:
            success, captured = reader.read()
            if not success or captured is None:
                now = time.monotonic()
                if now - last_failure_notice >= 2.0:
                    print("等待共享内存恢复或产生新帧……", file=sys.stderr)
                    last_failure_notice = now
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            cv2.imshow(args.window_name, captured.bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                try:
                    path = save_screenshot(
                        captured.bgr,
                        args.output_dir,
                        captured.source_frame_id,
                    )
                    print(f"已保存: {path.resolve()}")
                except RuntimeError as exc:
                    print(exc, file=sys.stderr)
            elif key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
