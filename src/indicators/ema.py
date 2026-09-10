"""Standard exponential moving average, SMA-seeded, no look-ahead.

Calculation inputs
------------------
Only the values of ``series`` in row order. This function never reads a
timestamp, a calendar month, or a future row. A NaN is a hole in the series,
not a value to invent.

Warm-up
-------
The first valid EMA is the SMA of the first ``period`` consecutive valid
observations. The ``period - 1`` rows before that are NaN. They are not filled
with zero or with the first price.

Recursion (after the seed):

    alpha = 2 / (period + 1)
    EMA_t = alpha * price_t + (1 - alpha) * EMA_(t-1)

NaN policy
----------
A NaN input produces a NaN output and **resets** state. After a hole, a new
SMA of ``period`` consecutive valid values is required before recursion
resumes. Leading NaNs stay NaN. No forward-fill, no back-fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IndicatorError(ValueError):
    """Invalid period or other caller error in an indicator calculator."""


def _require_period(period: int) -> int:
    if not isinstance(period, (int, np.integer)) or isinstance(period, bool):
        raise IndicatorError(f"period must be a positive integer, got {period!r}")
    period = int(period)
    if period < 1:
        raise IndicatorError(f"period must be >= 1, got {period}")
    return period


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Return SMA-seeded EMA of ``series`` with the given ``period``.

    ``period == 1`` is allowed: the output equals the input (alpha = 1).
    """
    period = _require_period(period)
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype="float64", copy=True)
    out = np.full(values.shape[0], np.nan, dtype="float64")
    alpha = 2.0 / (period + 1)
    one_minus = 1.0 - alpha

    window: list[float] = []
    ema: float | None = None

    for index, price in enumerate(values):
        if np.isnan(price):
            out[index] = np.nan
            window = []
            ema = None
            continue

        if ema is not None:
            ema = alpha * price + one_minus * ema
            out[index] = ema
            continue

        window.append(float(price))
        if len(window) == period:
            ema = float(np.mean(window))
            out[index] = ema

    return pd.Series(out, index=series.index, dtype="float64", name=series.name)
