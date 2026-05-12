"""数学工具函数。"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def clamp(value: float, min_value: float, max_value: float) -> float:
    """将数值限制在指定区间内。

    输入:
        value: 需要限制的数值。
        min_value: 区间下界。
        max_value: 区间上界。

    输出:
        返回限制后的数值。
    """

    return max(min_value, min(max_value, value))


def ema(previous_value: float, current_value: float, alpha: float) -> float:
    """计算指数滑动平均值。

    输入:
        previous_value: 上一次的平滑结果。
        current_value: 当前测量值。
        alpha: 当前值权重，范围建议为 0 到 1。

    输出:
        返回新的平滑结果。
    """

    alpha = clamp(alpha, 0.0, 1.0)
    return previous_value * (1.0 - alpha) + current_value * alpha


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """执行安全除法，避免除以零。

    输入:
        numerator: 分子。
        denominator: 分母。
        default: 当分母接近零时返回的默认值。

    输出:
        返回除法结果或默认值。
    """

    if abs(denominator) < 1e-6:
        return default
    return numerator / denominator


def polyfit_with_fallback(
    y_values: Sequence[float],
    x_values: Sequence[float],
    degree: int = 2,
) -> Optional[np.ndarray]:
    """对中心线点执行多项式拟合，并在点数不足时自动降阶。

    输入:
        y_values: 点集的纵坐标序列。
        x_values: 点集的横坐标序列。
        degree: 期望的多项式阶数，默认使用二次多项式。

    输出:
        返回长度为 3 的多项式系数数组，格式为 [a, b, c]；
        如果输入点为空，则返回 None。
    """

    if len(y_values) == 0 or len(x_values) == 0:
        return None

    y_array = np.asarray(y_values, dtype=np.float32)
    x_array = np.asarray(x_values, dtype=np.float32)
    point_count = min(len(y_array), len(x_array))

    if point_count >= max(3, degree + 1):
        coeffs = np.polyfit(y_array[:point_count], x_array[:point_count], degree)
        return np.asarray(coeffs, dtype=np.float32)

    if point_count >= 2:
        slope, intercept = np.polyfit(y_array[:point_count], x_array[:point_count], 1)
        return np.asarray([0.0, slope, intercept], dtype=np.float32)

    return np.asarray([0.0, 0.0, float(x_array[0])], dtype=np.float32)


def evaluate_poly(coeffs: Optional[np.ndarray], y_values: Sequence[float] | float) -> np.ndarray:
    """根据多项式系数计算指定纵坐标对应的横坐标。

    输入:
        coeffs: 多项式系数数组，格式为 [a, b, c]。
        y_values: 单个纵坐标或纵坐标序列。

    输出:
        返回计算得到的横坐标数组；若系数为空则返回零数组。
    """

    y_array = np.asarray(y_values, dtype=np.float32)
    if coeffs is None:
        return np.zeros_like(y_array, dtype=np.float32)
    return np.polyval(coeffs, y_array).astype(np.float32)


def compute_curvature(coeffs: Optional[np.ndarray], y_value: float) -> float:
    """根据二次曲线系数计算指定位置的曲率。

    输入:
        coeffs: 多项式系数数组，格式为 [a, b, c]。
        y_value: 需要计算曲率的纵坐标。

    输出:
        返回该位置的曲率绝对值；若系数为空则返回 0.0。
    """

    if coeffs is None:
        return 0.0

    a, b, _ = coeffs
    first_derivative = 2.0 * a * y_value + b
    second_derivative = 2.0 * a
    denominator = np.power(1.0 + first_derivative * first_derivative, 1.5)
    return float(abs(second_derivative) / max(denominator, 1e-6))


def mean_abs_residual(
    coeffs: Optional[np.ndarray],
    y_values: Sequence[float],
    x_values: Sequence[float],
) -> float:
    """计算点集相对拟合曲线的平均绝对残差。

    输入:
        coeffs: 多项式系数数组，格式为 [a, b, c]。
        y_values: 点集纵坐标序列。
        x_values: 点集横坐标序列。

    输出:
        返回平均绝对残差；若系数为空或输入为空则返回 0.0。
    """

    if coeffs is None or len(y_values) == 0 or len(x_values) == 0:
        return 0.0

    y_array = np.asarray(y_values, dtype=np.float32)
    x_array = np.asarray(x_values, dtype=np.float32)
    fitted_x = evaluate_poly(coeffs, y_array)
    return float(np.mean(np.abs(fitted_x - x_array)))
