"""Synthetic candles and injected signals for backtest tests. Offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.config import BacktestConfig
from src.strategy.signals import SignalType


def make_backtest_config(
    tmp_path: Path,
    *,
    starting_capital: float = 20.0,
    risk_per_trade: float = 0.01,
    stop_loss: float = 0.01,
    take_profit: float = 0.02,
    slippage: float = 0.0005,
    entry_rate: float = 0.001,
    exit_rate: float = 0.001,
    entry_model: str = "next_open",
    same_candle_priority: str = "stop_loss",
) -> BacktestConfig:
    return BacktestConfig(
        starting_capital=starting_capital,
        risk_per_trade=risk_per_trade,
        stop_loss=stop_loss,
        take_profit=take_profit,
        entry_model=entry_model,
        slippage=slippage,
        same_candle_priority=same_candle_priority,
        entry_rate=entry_rate,
        exit_rate=exit_rate,
        project_root=tmp_path,
    )


def ohlcv_frame(
    n: int,
    *,
    start: str = "2024-01-01 00:00:00",
    freq: str = "5min",
    open_price: float | np.ndarray | list[float] = 100.0,
    high: float | np.ndarray | list[float] | None = None,
    low: float | np.ndarray | list[float] | None = None,
    close: float | np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    open_arr = np.full(n, open_price, dtype="float64") if np.isscalar(open_price) else np.asarray(open_price, dtype="float64")
    if close is None:
        close_arr = open_arr.copy()
    elif np.isscalar(close):
        close_arr = np.full(n, close, dtype="float64")
    else:
        close_arr = np.asarray(close, dtype="float64")
    if high is None:
        high_arr = np.maximum(open_arr, close_arr) + 0.5
    elif np.isscalar(high):
        high_arr = np.full(n, high, dtype="float64")
    else:
        high_arr = np.asarray(high, dtype="float64")
    if low is None:
        low_arr = np.minimum(open_arr, close_arr) - 0.5
    elif np.isscalar(low):
        low_arr = np.full(n, low, dtype="float64")
    else:
        low_arr = np.asarray(low, dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
        }
    )


def signal_frame(candles: pd.DataFrame, long_at: set[int] | frozenset[int]) -> pd.DataFrame:
    signal = np.full(len(candles), SignalType.NONE.value, dtype=object)
    for index in long_at:
        signal[index] = SignalType.LONG_ENTRY.value
    return pd.DataFrame(
        {
            "timestamp": candles["timestamp"].copy(),
            "signal": signal,
        }
    )
