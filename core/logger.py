"""CSV 日志记录模块。"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional


class CsvLogger:
    """将关键巡线与控制量记录到 CSV 文件。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """读取日志配置并延迟打开文件。

        输入:
            config: logger 对应配置字典。

        输出:
            无返回值。
        """

        self.enabled = bool(config.get("enable", True))
        self.output_dir = Path(str(config.get("output_dir", "outputs/logs")))
        self.fieldnames = [
            "timestamp_ms",
            "lateral_error_px",
            "heading_error_deg",
            "curvature",
            "confidence",
            "target_speed",
            "steer_deg",
            "lane_lost_count",
        ]
        self.file_handle: Optional[Any] = None
        self.writer: Optional[csv.DictWriter] = None
        self.file_path: Optional[Path] = None

    def open(self) -> None:
        """创建日志目录并打开新的 CSV 文件。

        输入:
            无。

        输出:
            无返回值；若日志关闭则直接跳过。
        """

        if not self.enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_name = time.strftime("run_%Y%m%d_%H%M%S.csv")
        self.file_path = self.output_dir / file_name
        self.file_handle = self.file_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """写入一行关键数据到 CSV 文件。

        输入:
            row: 单帧日志字典，字段名需与配置的 fieldnames 对齐。

        输出:
            无返回值。
        """

        if not self.enabled or self.writer is None or self.file_handle is None:
            return

        normalized_row = {field: row.get(field, "") for field in self.fieldnames}
        self.writer.writerow(normalized_row)
        self.file_handle.flush()

    def close(self) -> None:
        """关闭日志文件句柄。

        输入:
            无。

        输出:
            无返回值。
        """

        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            self.writer = None
