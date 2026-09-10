"""Reconstruct candle-close MTM equity from the v0.4 ledger. No engine change."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.backtest.ledger import STATUS_CLOSED, STATUS_OPEN, Trade

from .config import ResearchError

EQUITY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "cash",
    "position_quantity",
    "position_mark_value",
    "realized_equity",
    "mtm_equity",
    "unrealized_pnl",
    "drawdown",
    "drawdown_pct",
)


class ResearchAnalyticsError(ResearchError):
    """Equity or metric reconstruction failed."""


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def reconstruct_equity(
    candles: pd.DataFrame,
    trades: Sequence[Trade],
    starting_capital: float,
) -> pd.DataFrame:
    """One row per trade-window candle close.

    Entry debit and exit credit use ledger amounts. Mark-to-market uses this
    candle's close only. OPEN/END_OF_DATA never invents net_pnl.
    """
    if "timestamp" not in candles.columns or "close" not in candles.columns:
        raise ResearchAnalyticsError("candles must include timestamp and close")

    by_entry: dict[pd.Timestamp, Trade] = {}
    for trade in trades:
        key = _utc(trade.entry_timestamp)
        if key in by_entry:
            raise ResearchAnalyticsError(
                f"Duplicate entry_timestamp {key} in the trade ledger"
            )
        by_entry[key] = trade

    cash = float(starting_capital)
    realized = float(starting_capital)
    quantity = 0.0
    current: Trade | None = None
    peak: float | None = None
    rows: list[dict[str, object]] = []

    for index in range(len(candles)):
        timestamp = _utc(candles["timestamp"].iloc[index])
        close = float(candles["close"].iloc[index])

        if current is None:
            incoming = by_entry.get(timestamp)
            if incoming is not None:
                current = incoming
                cash -= float(incoming.position_value) + float(incoming.entry_fee)
                quantity = float(incoming.quantity)

        if (
            current is not None
            and current.status == STATUS_CLOSED
            and current.exit_timestamp is not None
            and _utc(current.exit_timestamp) == timestamp
        ):
            if current.exit_price is None or current.exit_fee is None:
                raise ResearchAnalyticsError(
                    f"CLOSED trade {current.trade_id} is missing exit_price or exit_fee"
                )
            if current.net_pnl is None:
                raise ResearchAnalyticsError(
                    f"CLOSED trade {current.trade_id} is missing net_pnl"
                )
            cash += float(current.quantity) * float(current.exit_price) - float(
                current.exit_fee
            )
            realized += float(current.net_pnl)
            quantity = 0.0
            current = None

        if current is not None and current.status not in {STATUS_CLOSED, STATUS_OPEN}:
            raise ResearchAnalyticsError(
                f"Unexpected trade status {current.status!r}"
            )

        mark = quantity * close if quantity else 0.0
        mtm = cash + mark
        if peak is None or mtm > peak:
            peak = mtm
        drawdown = peak - mtm
        drawdown_pct = (drawdown / peak) if peak > 0 else float("nan")
        rows.append(
            {
                "timestamp": timestamp,
                "cash": cash,
                "position_quantity": quantity,
                "position_mark_value": mark,
                "realized_equity": realized,
                "mtm_equity": mtm,
                "unrealized_pnl": mtm - realized,
                "drawdown": drawdown,
                "drawdown_pct": drawdown_pct,
            }
        )

    if not rows:
        return pd.DataFrame({column: [] for column in EQUITY_COLUMNS})
    return pd.DataFrame(rows, columns=list(EQUITY_COLUMNS))


def global_max_drawdown(equity: pd.DataFrame) -> tuple[float, float | None, object, object]:
    """Max peak-to-trough on the complete-period MTM curve."""
    if equity.empty:
        return 0.0, None, None, None
    series = equity["drawdown"].to_numpy(dtype="float64")
    trough_pos = int(np.argmax(series))
    max_dd = float(series[trough_pos])
    pct_raw = equity["drawdown_pct"].iloc[trough_pos]
    max_pct = None if pd.isna(pct_raw) else float(pct_raw)
    trough_ts = equity["timestamp"].iloc[trough_pos]
    peak_pos = int(equity["mtm_equity"].iloc[: trough_pos + 1].to_numpy().argmax())
    peak_ts = equity["timestamp"].iloc[peak_pos]
    return max_dd, max_pct, peak_ts, trough_ts


def intra_window_max_drawdown(mtm: pd.Series) -> tuple[float, float | None]:
    """Peak-to-trough with the peak reset at the first value (monthly/yearly)."""
    peak: float | None = None
    max_dd = 0.0
    max_pct: float | None = 0.0
    for value in mtm.to_numpy(dtype="float64"):
        number = float(value)
        if peak is None or number > peak:
            peak = number
        drawdown = peak - number
        if drawdown > max_dd:
            max_dd = drawdown
            max_pct = (drawdown / peak) if peak > 0 else None
    if peak is None:
        return 0.0, None
    return max_dd, max_pct
