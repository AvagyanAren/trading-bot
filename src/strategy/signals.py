"""Signal schema for the v0.3 Strategy Engine.

A signal records that candle N has closed and that the hypothesis did or
did not fire. ``timestamp`` is that candle's open time (the dataset key).
``earliest_execution_time`` is the next interval boundary — a timing label,
not a fill.

There is no execution price, fill price, or entry price on this object.
``reference_close`` is the closed candle's close, for audit only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SignalType(str, Enum):
    """Allowed strategy outputs. SHORT is intentionally absent."""

    NONE = "NONE"
    LONG_ENTRY = "LONG_ENTRY"


@dataclass(frozen=True)
class ConditionBreakdown:
    """Per-condition result at candle close.

    ``True`` / ``False`` means the predicate was evaluated. ``None`` means
    it was unevaluable (NaN input or insufficient history).
    """

    ema_trend: bool | None
    breakout: bool | None
    volume_confirmation: bool | None
    rsi_filter: bool | None


@dataclass(frozen=True)
class Signal:
    """Look-ahead-free evaluation of one already-closed candle.

    ``earliest_execution_time`` may fall after the last row of a historical
    frame. In v0.3 that is only a label: no order is placed and no fill is
    assumed.
    """

    strategy_name: str
    strategy_version: str
    symbol: str
    interval: str
    timestamp: pd.Timestamp
    candle_close_time: pd.Timestamp | None
    earliest_execution_time: pd.Timestamp
    signal: SignalType
    reference_close: float | None
    breakout_level: float | None
    warmup: bool
    conditions: ConditionBreakdown
