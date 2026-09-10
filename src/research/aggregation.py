"""Monthly and yearly tables from the trade ledger and MTM equity curve."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.backtest.ledger import EXIT_SL, EXIT_TP, STATUS_CLOSED, Trade

from .equity import intra_window_max_drawdown

MONTHLY_COLUMNS: tuple[str, ...] = (
    "period",
    "trades_closed",
    "tp",
    "sl",
    "win_rate",
    "gross_pnl",
    "fees",
    "slippage",
    "net_pnl",
    "cumulative_net_pnl",
    "ending_cash",
    "ending_realized_equity",
    "ending_mtm_equity",
    "max_drawdown",
    "max_drawdown_pct",
)

YEARLY_COLUMNS = MONTHLY_COLUMNS


def _closed(trades: Sequence[Trade]) -> list[Trade]:
    return [trade for trade in trades if trade.status == STATUS_CLOSED]


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _period_key(timestamp: pd.Timestamp, kind: str) -> str:
    stamp = _utc(timestamp)
    if kind == "month":
        return f"{stamp.year:04d}-{stamp.month:02d}"
    return f"{stamp.year:04d}"


def _empty(kind: str) -> pd.DataFrame:
    columns = MONTHLY_COLUMNS if kind == "month" else YEARLY_COLUMNS
    return pd.DataFrame({column: [] for column in columns})


def aggregate_periods(
    trades: Sequence[Trade],
    equity: pd.DataFrame,
    *,
    kind: str,
) -> pd.DataFrame:
    """Attribute realized P&L to the UTC calendar month/year of ``exit_timestamp``.

    ``max_drawdown`` here is intra-window (peak reset at the window start).
    It is an additional diagnostic and not the experiment global max drawdown.
    """
    if kind not in {"month", "year"}:
        raise ValueError(f"kind must be 'month' or 'year', got {kind!r}")
    if equity.empty:
        return _empty(kind)

    equity_frame = equity.copy()
    equity_frame["timestamp"] = pd.to_datetime(equity_frame["timestamp"], utc=True)
    equity_frame["period"] = equity_frame["timestamp"].map(
        lambda value: _period_key(value, kind)
    )

    closed = _closed(trades)
    buckets: dict[str, list[Trade]] = {}
    for trade in closed:
        if trade.exit_timestamp is None:
            continue
        key = _period_key(_utc(trade.exit_timestamp), kind)
        buckets.setdefault(key, []).append(trade)

    periods = sorted(set(equity_frame["period"]).union(buckets))
    rows: list[dict[str, object]] = []
    cumulative = 0.0
    for period in periods:
        group = buckets.get(period, [])
        wins = [
            trade
            for trade in group
            if trade.net_pnl is not None and trade.net_pnl > 0
        ]
        n = len(group)
        net = float(sum(trade.net_pnl or 0.0 for trade in group))
        cumulative += net
        slice_eq = equity_frame.loc[equity_frame["period"] == period]
        last = slice_eq.iloc[-1] if not slice_eq.empty else None
        intra_dd, intra_pct = (
            intra_window_max_drawdown(slice_eq["mtm_equity"])
            if not slice_eq.empty
            else (0.0, None)
        )
        rows.append(
            {
                "period": period,
                "trades_closed": n,
                "tp": sum(1 for trade in group if trade.exit_reason == EXIT_TP),
                "sl": sum(1 for trade in group if trade.exit_reason == EXIT_SL),
                "win_rate": (len(wins) / n) if n else None,
                "gross_pnl": float(sum(trade.gross_pnl or 0.0 for trade in group)),
                "fees": float(
                    sum(
                        (trade.entry_fee or 0.0) + (trade.exit_fee or 0.0)
                        for trade in group
                    )
                ),
                "slippage": float(sum(trade.slippage or 0.0 for trade in group)),
                "net_pnl": net,
                "cumulative_net_pnl": cumulative,
                "ending_cash": None if last is None else float(last["cash"]),
                "ending_realized_equity": (
                    None if last is None else float(last["realized_equity"])
                ),
                "ending_mtm_equity": None if last is None else float(last["mtm_equity"]),
                "max_drawdown": intra_dd,
                "max_drawdown_pct": intra_pct,
            }
        )
    return pd.DataFrame(rows, columns=list(MONTHLY_COLUMNS))


def monthly_table(trades: Sequence[Trade], equity: pd.DataFrame) -> pd.DataFrame:
    return aggregate_periods(trades, equity, kind="month")


def yearly_table(trades: Sequence[Trade], equity: pd.DataFrame) -> pd.DataFrame:
    return aggregate_periods(trades, equity, kind="year")
