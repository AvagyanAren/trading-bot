"""Previous-N-high breakout level. The current candle's HIGH is never an input.

For candle N the caller must pass highs of candles N-period through N-1.
The current close is compared by the caller after this function returns.

``breakout_period`` is a strategy parameter. It is intentionally independent
of indicator periods such as ``volume_ma_period``, even when both happen to
be 20. This module does not read volume SMA length and must not be passed
that value in place of the breakout lookback.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def breakout_level(previous_highs: Any) -> float:
    """Return max(previous_highs), or NaN if the window is empty or contains NaN.

    NaNs are not skipped. Using nanmax would silently shorten the lookback.
    """
    values = np.asarray(previous_highs, dtype="float64")
    if values.size == 0:
        return float("nan")
    if np.isnan(values).any():
        return float("nan")
    return float(np.max(values))


def close_breaks_out(close: float, level: float) -> bool | None:
    """True iff close is strictly greater than the previous-N high level."""
    if pd.isna(close) or pd.isna(level):
        return None
    return bool(close > level)
