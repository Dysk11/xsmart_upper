"""Small numeric helpers shared by lane tracking and planning."""

from __future__ import annotations


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a numeric value to the inclusive interval."""

    return max(min_value, min(max_value, value))


def ema(previous_value: float, current_value: float, alpha: float) -> float:
    """Return an exponential moving average update."""

    alpha = clamp(alpha, 0.0, 1.0)
    return previous_value * (1.0 - alpha) + current_value * alpha


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide while returning a fallback for near-zero denominators."""

    if abs(denominator) < 1e-6:
        return default
    return numerator / denominator
