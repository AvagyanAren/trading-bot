"""Development/test windows, lookback for breakout continuity, leakage guards."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .config import ResearchError

CANONICAL_DEVELOPMENT_START = date(2024, 1, 1)
CANONICAL_DEVELOPMENT_END = date(2024, 12, 31)
CANONICAL_TEST_START = date(2025, 1, 1)
CANONICAL_TEST_END = date(2025, 12, 31)
TEST_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")


class ResearchPeriodError(ResearchError):
    """The requested experiment window is invalid or leaked."""


def window_end_exclusive(end: date) -> pd.Timestamp:
    """First instant after the inclusive end date (UTC)."""
    return pd.Timestamp(end + timedelta(days=1), tz="UTC")


def window_start(start: date) -> pd.Timestamp:
    return pd.Timestamp(start, tz="UTC")


def window_mask(timestamps: pd.Series, start: date, end: date) -> pd.Series:
    start_ts = window_start(start)
    end_ts = window_end_exclusive(end)
    values = pd.to_datetime(timestamps, utc=True)
    return (values >= start_ts) & (values < end_ts)


def prepare_experiment_frames(
    candles: pd.DataFrame,
    start: date,
    end: date,
    lookback_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(signal_candles, trade_candles)``.

    ``trade_candles`` is the inclusive ``start``..``end`` window.
    ``signal_candles`` prepends up to ``lookback_bars`` prior rows so
    breakout history is available; those extra rows are not traded.
    """
    if lookback_bars < 0:
        raise ResearchPeriodError(f"lookback_bars must be >= 0, got {lookback_bars}")
    if "timestamp" not in candles.columns:
        raise ResearchPeriodError("candles must contain a timestamp column")
    if candles.empty:
        raise ResearchPeriodError("candles are empty")

    frame = candles.sort_values("timestamp").reset_index(drop=True)
    mask = window_mask(frame["timestamp"], start, end)
    trade = frame.loc[mask].reset_index(drop=True)
    if trade.empty:
        raise ResearchPeriodError(
            f"No candles in experiment window {start.isoformat()} .. {end.isoformat()}"
        )
    first_pos = int(frame.index[mask][0])
    last_pos = int(frame.index[mask][-1])
    lookback_start = max(0, first_pos - int(lookback_bars))
    signal = frame.iloc[lookback_start : last_pos + 1].reset_index(drop=True)
    return signal, trade


def assert_development_window(candles: pd.DataFrame) -> None:
    if candles.empty:
        raise ResearchPeriodError("development candles are empty")
    latest = pd.Timestamp(candles["timestamp"].max())
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    else:
        latest = latest.tz_convert("UTC")
    if latest >= TEST_CUTOFF:
        raise ResearchPeriodError(
            "Development candles include 2025 timestamps; "
            f"max timestamp is {latest}"
        )


def assert_trades_in_window(
    trades,
    start: date,
    end: date,
) -> None:
    start_ts = window_start(start)
    end_ts = window_end_exclusive(end)
    for trade in trades:
        entry = pd.Timestamp(trade.entry_timestamp)
        if entry.tzinfo is None:
            entry = entry.tz_localize("UTC")
        else:
            entry = entry.tz_convert("UTC")
        if entry < start_ts or entry >= end_ts:
            raise ResearchPeriodError(
                f"Trade {trade.trade_id} entry {entry} is outside the window"
            )
        if trade.exit_timestamp is None:
            continue
        exit_ts = pd.Timestamp(trade.exit_timestamp)
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize("UTC")
        else:
            exit_ts = exit_ts.tz_convert("UTC")
        if exit_ts < start_ts or exit_ts >= end_ts:
            raise ResearchPeriodError(
                f"Trade {trade.trade_id} exit {exit_ts} is outside the window"
            )


def align_signals_to_candles(
    signals: pd.DataFrame,
    trade_candles: pd.DataFrame,
) -> pd.DataFrame:
    """Keep signal rows whose timestamp matches the trade window, in candle order."""
    if "timestamp" not in signals.columns:
        raise ResearchPeriodError("signals must contain a timestamp column")
    ordered = trade_candles[["timestamp"]].merge(signals, on="timestamp", how="left")
    if ordered["signal"].isna().any():
        raise ResearchPeriodError(
            "Signal frame is missing rows for the trade-window candles"
        )
    if len(ordered) != len(trade_candles):
        raise ResearchPeriodError("Aligned signals length does not match candles")
    return ordered.reset_index(drop=True)
