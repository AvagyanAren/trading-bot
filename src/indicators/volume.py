"""Simple moving average of volume. No exponential weighting, no look-ahead.

Calculation inputs
------------------
Only the values of ``series`` (volume) in row order. This function never reads
a timestamp, a calendar month, or a future row.

Warm-up
-------
``volume_ma`` at index t is the mean of ``series[t - period + 1 : t + 1]``
(the current row included). The first ``period - 1`` values are NaN. They are
not filled.

A window that contains a NaN is NaN. No forward-fill, no back-fill.
"""

from __future__ import annotations

import pandas as pd

from .ema import _require_period


def calculate_volume_sma(series: pd.Series, period: int) -> pd.Series:
    """Return the trailing SMA of ``series`` with the given ``period``."""
    period = _require_period(period)
    numeric = pd.to_numeric(series, errors="raise").astype("float64")
    result = numeric.rolling(window=period, min_periods=period).mean()
    return pd.Series(result.to_numpy(), index=series.index, dtype="float64", name=series.name)
