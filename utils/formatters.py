"""Small formatting helpers for numbers and text shown in the UI/reports."""

from __future__ import annotations

import math
from typing import Any


def human_number(value: Any, decimals: int = 2) -> str:
    """Format a number compactly (1_234_567 -> '1.23M'). Non-numbers pass through."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num != num or math.isinf(num):  # NaN / inf guard
        return "n/a"
    sign = "-" if num < 0 else ""
    num = abs(num)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if num >= threshold:
            return f"{sign}{num / threshold:.{decimals}f}{suffix}"
    if num == int(num):
        return f"{sign}{int(num)}"
    return f"{sign}{num:.{decimals}f}"


def pct(value: Any, decimals: int = 1) -> str:
    """Format a 0-1 fraction as a percentage string."""
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "n/a"


def strength_label(correlation: float) -> str:
    """Map an absolute correlation coefficient to a plain-English strength."""
    r = abs(correlation)
    if r >= 0.7:
        return "strong"
    if r >= 0.4:
        return "moderate"
    if r >= 0.2:
        return "weak"
    return "negligible"


def titleize(name: str) -> str:
    """Turn a snake_case column name into a human-friendly Title Case label."""
    return name.replace("_", " ").strip().title()
