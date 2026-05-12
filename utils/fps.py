"""FPS 统计工具。"""

from __future__ import annotations

import time

from utils.math_utils import ema


class FPSCounter:
    """用于统计主循环实时帧率。"""

    def __init__(self, smooth_alpha: float = 0.2) -> None:
        """初始化 FPS 统计器。

        输入:
            smooth_alpha: FPS 指数滑动平均权重。

        输出:
            无返回值，内部记录初始时间戳。
        """

        self.smooth_alpha = smooth_alpha
        self.last_timestamp = time.perf_counter()
        self.current_fps = 0.0

    def update(self) -> float:
        """根据当前时间更新一次 FPS 数值。

        输入:
            无。

        输出:
            返回平滑后的 FPS 数值。
        """

        now = time.perf_counter()
        delta = max(now - self.last_timestamp, 1e-6)
        instant_fps = 1.0 / delta

        if self.current_fps <= 1e-6:
            self.current_fps = instant_fps
        else:
            self.current_fps = ema(self.current_fps, instant_fps, self.smooth_alpha)

        self.last_timestamp = now
        return self.current_fps

    def get(self) -> float:
        """获取当前缓存的 FPS 数值。

        输入:
            无。

        输出:
            返回最近一次更新后的 FPS 数值。
        """

        return self.current_fps
