"""Entry predicates. NaN is unevaluable, never filled.

All comparisons are strict. Thresholds come from the caller (strategy
config); this module contains no period or bound literals.
"""

from __future__ import annotations

import pandas as pd


def _unevaluable(*values: object) -> bool:
    return any(value is None or pd.isna(value) for value in values)


def ema_trend(ema_fast: object, ema_slow: object) -> bool | None:
    """True iff the fast EMA is strictly above the slow EMA."""
    if _unevaluable(ema_fast, ema_slow):
        return None
    return bool(float(ema_fast) > float(ema_slow))


def volume_confirmation(
    volume: object, volume_ma: object, multiplier: object
) -> bool | None:
    """True iff volume is strictly greater than volume_ma * multiplier."""
    if _unevaluable(volume, volume_ma, multiplier):
        return None
    return bool(float(volume) > float(volume_ma) * float(multiplier))


def rsi_filter(rsi: object, rsi_min: object, rsi_max: object) -> bool | None:
    """True iff rsi_min < rsi < rsi_max (both sides strict)."""
    if _unevaluable(rsi, rsi_min, rsi_max):
        return None
    value = float(rsi)
    return bool(float(rsi_min) < value < float(rsi_max))
