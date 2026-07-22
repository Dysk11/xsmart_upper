"""图像绘制与拼接工具函数。"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """确保输入图像为三通道 BGR 格式。

    输入:
        image: 单通道或三通道图像。

    输出:
        返回三通道 BGR 图像副本。
    """

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def overlay_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """将二值掩膜以半透明形式覆盖到图像上。

    输入:
        image: 原始 BGR 图像。
        mask: 单通道二值掩膜，非零区域将被着色。
        color: 掩膜显示颜色，默认使用蓝色。
        alpha: 掩膜混合透明度。

    输出:
        返回叠加后的 BGR 图像。
    """

    output = ensure_bgr(image)
    if mask is None or mask.size == 0:
        return output

    overlay = np.zeros_like(output)
    overlay[mask > 0] = color
    return cv2.addWeighted(output, 1.0, overlay, alpha, 0.0)


def draw_centerline(
    image: np.ndarray,
    points: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 3,
    thickness: int = 2,
    offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """在图像上绘制中心线点集与折线。

    输入:
        image: 需要绘制的 BGR 图像。
        points: 中心线点集，格式为 [(x, y), ...]。
        color: 绘制颜色。
        radius: 点半径。
        thickness: 折线宽度。
        offset: 绘制前统一叠加的坐标偏移。

    输出:
        返回绘制后的 BGR 图像。
    """

    output = ensure_bgr(image)
    if not points:
        return output

    shifted_points = [
        (int(point[0] + offset[0]), int(point[1] + offset[1]))
        for point in points
    ]

    for point in shifted_points:
        cv2.circle(output, point, radius, color, -1)

    if len(shifted_points) >= 2:
        polyline = np.asarray(shifted_points, dtype=np.int32)
        cv2.polylines(output, [polyline], False, color, thickness, cv2.LINE_AA)

    return output


def draw_text_lines(
    image: np.ndarray,
    lines: Iterable[str],
    origin: Tuple[int, int] = (12, 28),
    color: Tuple[int, int, int] = (255, 255, 255),
    line_height: int = 24,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    background_alpha: float = 0.45,
    font_path: str = "",
    font_size: int = 22,
) -> np.ndarray:
    """在图像上绘制多行文字，并添加半透明底板提升可读性。

    输入:
        image: 需要绘制的 BGR 图像。
        lines: 文字内容序列，每个元素代表一行。
        origin: 第一行文字左下角坐标。
        color: 文字颜色。
        line_height: 行高。
        background_color: 文字底板颜色。
        background_alpha: 文字底板透明度。
        font_path: 中文字体文件路径，留空时自动搜索常见字体。
        font_size: 使用 Pillow 绘制中文时的字体大小。

    输出:
        返回绘制后的 BGR 图像。
    """

    output = ensure_bgr(image)
    text_lines = list(lines)
    if not text_lines:
        return output

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, float(font_size) / 36.0)
    thickness = 2

    pil_font = _load_font(font_path, font_size)
    if pil_font is not None:
        widths = [_measure_text_width(line, pil_font) for line in text_lines]
        max_width = max(widths) if widths else 0
    else:
        max_width = 0
        for line in text_lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_size[0])

    x, y = origin
    top = max(0, y - 22)
    bottom = min(output.shape[0] - 1, y + line_height * (len(text_lines) - 1) + 10)
    left = max(0, x - 10)
    right = min(output.shape[1] - 1, x + max_width + 20)

    overlay = output.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), background_color, -1)
    output = cv2.addWeighted(overlay, background_alpha, output, 1.0 - background_alpha, 0.0)

    if pil_font is not None and Image is not None and ImageDraw is not None:
        rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        pil_color = (int(color[2]), int(color[1]), int(color[0]))
        for index, line in enumerate(text_lines):
            position = (x, y - font_size + index * line_height)
            draw.text(position, line, font=pil_font, fill=pil_color)
        output = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    else:
        for index, line in enumerate(text_lines):
            position = (x, y + index * line_height)
            cv2.putText(output, line, position, font, font_scale, color, thickness, cv2.LINE_AA)

    return output


def measure_text_width(text: str, font_path: str = "", font_size: int = 22) -> int:
    """Return the rendered pixel width used by :func:`draw_text_lines`."""

    pil_font = _load_font(font_path, font_size)
    if pil_font is not None:
        return _measure_text_width(text, pil_font)
    font_scale = max(0.4, float(font_size) / 36.0)
    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        2,
    )
    return int(text_size[0])


def wrap_text_lines(
    lines: Iterable[str],
    max_width: int,
    font_path: str = "",
    font_size: int = 22,
) -> List[str]:
    """Wrap mixed Chinese/ASCII text to a measured pixel width.

    Wrapping prefers the last whitespace that fits.  A single token wider than
    the available space is split by character so diagnostic identifiers and
    unbroken error messages can never escape the panel.
    """

    width_limit = max(1, int(max_width))
    wrapped: List[str] = []
    for source_line in lines:
        remaining = str(source_line)
        if not remaining:
            wrapped.append("")
            continue
        while remaining:
            if measure_text_width(remaining, font_path, font_size) <= width_limit:
                wrapped.append(remaining)
                break

            low, high = 1, len(remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if measure_text_width(remaining[:middle], font_path, font_size) <= width_limit:
                    low = middle
                else:
                    high = middle - 1
            fit_count = max(1, low)
            prefix = remaining[:fit_count]
            whitespace_index = max(prefix.rfind(" "), prefix.rfind("\t"))
            if whitespace_index > 0:
                wrapped.append(prefix[:whitespace_index].rstrip())
                remaining = remaining[whitespace_index + 1 :].lstrip()
            else:
                wrapped.append(prefix)
                remaining = remaining[fit_count:].lstrip()
    return wrapped


def stack_images(
    images: Sequence[np.ndarray],
    cols: int = 2,
    cell_size: Optional[Tuple[int, int]] = None,
    background_color: Tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    """将多张图像按网格形式拼接为一张大图。

    输入:
        images: 输入图像序列。
        cols: 每行显示的图像数量。
        cell_size: 单个网格的目标尺寸，格式为 (宽, 高)；若为空则自动取最大尺寸。
        background_color: 画布背景颜色。

    输出:
        返回拼接后的 BGR 图像。
    """

    prepared_images = [ensure_bgr(image) for image in images]
    if not prepared_images:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if cell_size is None:
        cell_width = max(image.shape[1] for image in prepared_images)
        cell_height = max(image.shape[0] for image in prepared_images)
    else:
        cell_width, cell_height = cell_size

    rows = int(math.ceil(len(prepared_images) / float(cols)))
    canvas = np.full(
        (rows * cell_height, cols * cell_width, 3),
        background_color,
        dtype=np.uint8,
    )

    for index, image in enumerate(prepared_images):
        row = index // cols
        col = index % cols
        resized = cv2.resize(image, (cell_width, cell_height))
        y1 = row * cell_height
        y2 = y1 + cell_height
        x1 = col * cell_width
        x2 = x1 + cell_width
        canvas[y1:y2, x1:x2] = resized

    return canvas


def _contains_non_ascii(lines: Sequence[str]) -> bool:
    """判断文字列表中是否包含非 ASCII 字符。

    输入:
        lines: 文本行列表。

    输出:
        若包含中文等非 ASCII 字符则返回 True，否则返回 False。
    """

    return any(any(ord(char) > 127 for char in line) for line in lines)


@lru_cache(maxsize=16)
def _load_font(font_path: str, font_size: int) -> Optional["ImageFont.FreeTypeFont"]:
    """加载可用于绘制中文的字体对象。

    输入:
        font_path: 用户配置的字体文件路径。
        font_size: 字体大小。

    输出:
        若找到可用字体则返回字体对象，否则返回 None。
    """

    if ImageFont is None:
        return None

    candidates: List[str] = []
    if font_path:
        candidates.append(font_path)

    candidates.extend(
        [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), font_size)
        except Exception:
            continue

    return None


def _measure_text_width(text: str, font: "ImageFont.FreeTypeFont") -> int:
    """测量使用 Pillow 字体绘制文本时的像素宽度。

    输入:
        text: 待测量的文本。
        font: Pillow 字体对象。

    输出:
        返回文本像素宽度。
    """

    if Image is None or ImageDraw is None:
        return len(text) * 12

    image = Image.new("RGB", (8, 8), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return int(right - left)
