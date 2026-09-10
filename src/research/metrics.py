"""Closed-trade and account metrics from the v0.4 ledger.

Does not redefine risk_amount, net_pnl, or R. OPEN/END_OF_DATA rows are
excluded from realized totals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.ledger import (
    EXIT_END_OF_DATA,
    EXIT_SL,
    EXIT_TP,
    IGNORE_INSUFFICIENT_CASH,
    IGNORE_NO_NEXT_CANDLE,
    IGNORE_POSITION_OPEN,
    STATUS_CLOSED,
    STATUS_OPEN,
    Trade,
)

from .equity import global_max_drawdown, reconstruct_equity

CANDLE_SECONDS = 300.0
RECONCILE_EPS = 1e-8


def _is_close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=RECONCILE_EPS, equal_nan=False))


def closed_trades(trades: Sequence[Trade]) -> list[Trade]:
    return [trade for trade in trades if trade.status == STATUS_CLOSED]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def _mean_timedelta(values: list[pd.Timedelta]) -> pd.Timedelta | None:
    if not values:
        return None
    seconds = [float(item.total_seconds()) for item in values]
    return pd.to_timedelta(float(np.mean(seconds)), unit="s")


def _median_timedelta(values: list[pd.Timedelta]) -> pd.Timedelta | None:
    if not values:
        return None
    seconds = [float(item.total_seconds()) for item in values]
    return pd.to_timedelta(float(np.median(seconds)), unit="s")


def _candles(duration: pd.Timedelta | None) -> float | None:
    if duration is None:
        return None
    return float(duration.total_seconds()) / CANDLE_SECONDS


def max_consecutive(trades: Sequence[Trade]) -> tuple[int, int]:
    max_wins = 0
    max_losses = 0
    cur_wins = 0
    cur_losses = 0
    for trade in trades:
        pnl = trade.net_pnl
        if pnl is None:
            continue
        if pnl > 0:
            cur_wins += 1
            cur_losses = 0
        elif pnl < 0:
            cur_losses += 1
            cur_wins = 0
        else:
            cur_wins = 0
            cur_losses = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


@dataclass(frozen=True)
class ExperimentMetrics:
    starting_capital: float
    ending_cash: float
    ending_realized_equity: float
    ending_mtm_equity: float
    total_net_pnl: float
    total_gross_pnl: float
    total_fees: float
    total_slippage: float
    signals: int
    filled_trades: int
    closed_trades: int
    open_end_of_data: int
    ignored_signals: int
    ignored_position_open: int
    ignored_no_next_candle: int
    ignored_insufficient_cash: int
    tp_count: int
    sl_count: int
    win_count: int
    loss_count: int
    scratch_count: int
    win_rate: float | None
    loss_rate: float | None
    average_winning_trade: float | None
    average_losing_trade: float | None
    median_winning_trade: float | None
    median_losing_trade: float | None
    profit_factor: float | None
    average_net_pnl_per_closed_trade: float | None
    average_R_per_closed_trade: float | None
    expectancy_per_trade: float | None
    expectancy_R: float | None
    median_R: float | None
    total_R: float | None
    average_holding_time: str | None
    median_holding_time: str | None
    max_holding_time: str | None
    average_holding_candles: float | None
    median_holding_candles: float | None
    max_holding_candles: float | None
    max_consecutive_wins: int
    max_consecutive_losses: int
    max_drawdown: float
    max_drawdown_pct: float | None
    max_drawdown_peak_timestamp: str | None
    max_drawdown_trough_timestamp: str | None
    largest_win_net_pnl: float | None
    sum_of_winning_net_pnl: float | None
    months_with_closed_trades: int
    max_month_share_of_net: float | None
    open_entry_fees: float
    unresolved_reserved: float

    def as_pairs(self) -> list[tuple[str, object]]:
        return [(item.name, getattr(self, item.name)) for item in fields(self)]


def compute_metrics(
    result: BacktestResult,
    equity: pd.DataFrame,
    *,
    monthly: pd.DataFrame | None = None,
) -> ExperimentMetrics:
    closed = closed_trades(result.trades)
    open_rows = [trade for trade in result.trades if trade.status == STATUS_OPEN]
    wins = [trade for trade in closed if trade.net_pnl is not None and trade.net_pnl > 0]
    losses = [trade for trade in closed if trade.net_pnl is not None and trade.net_pnl < 0]
    scratches = [
        trade for trade in closed if trade.net_pnl is not None and trade.net_pnl == 0
    ]
    win_pnls = [float(trade.net_pnl) for trade in wins]
    loss_pnls = [float(trade.net_pnl) for trade in losses]
    all_pnls = [float(trade.net_pnl) for trade in closed if trade.net_pnl is not None]
    all_r = [float(trade.R) for trade in closed if trade.R is not None]
    holdings = [
        trade.holding_time
        for trade in closed
        if trade.holding_time is not None and not pd.isna(trade.holding_time)
    ]

    gross = float(sum(trade.gross_pnl or 0.0 for trade in closed))
    fees = float(
        sum((trade.entry_fee or 0.0) + (trade.exit_fee or 0.0) for trade in closed)
    )
    slippage = float(sum(trade.slippage or 0.0 for trade in closed))
    net = float(sum(trade.net_pnl or 0.0 for trade in closed))
    if closed and not _is_close(net, result.realized_net_pnl):
        raise ValueError("realized_net_pnl does not match the sum of CLOSED net_pnl")

    win_sum = float(sum(win_pnls)) if win_pnls else None
    loss_sum = float(sum(loss_pnls)) if loss_pnls else None
    if loss_sum is None:
        profit_factor = None
    elif loss_sum == 0:
        profit_factor = None
    else:
        profit_factor = (win_sum or 0.0) / abs(loss_sum)

    avg_net = _mean(all_pnls)
    avg_r = _mean(all_r)
    n_closed = len(closed)
    win_rate = (len(wins) / n_closed) if n_closed else None
    loss_rate = (len(losses) / n_closed) if n_closed else None

    avg_hold = _mean_timedelta(holdings)
    med_hold = _median_timedelta(holdings)
    max_hold = max(holdings) if holdings else None

    max_dd, max_dd_pct, peak_ts, trough_ts = global_max_drawdown(equity)
    last = equity.iloc[-1] if not equity.empty else None
    ending_cash = float(result.cash)
    ending_realized = float(result.starting_capital) + float(result.realized_net_pnl)
    if last is None:
        ending_mtm = ending_cash
    else:
        ending_mtm = float(last["mtm_equity"])
        if not _is_close(float(last["cash"]), ending_cash):
            raise ValueError("equity cash path does not match BacktestResult.cash")
        if not _is_close(float(last["realized_equity"]), ending_realized):
            raise ValueError("equity realized path does not match realized_net_pnl")

    reasons = [item.reason for item in result.ignored]
    reserved = 0.0
    open_entry_fees = 0.0
    for trade in open_rows:
        reserved += float(trade.position_value) + float(trade.entry_fee)
        open_entry_fees += float(trade.entry_fee)
        if trade.exit_reason != EXIT_END_OF_DATA:
            raise ValueError("OPEN trade is missing END_OF_DATA")
        if trade.net_pnl is not None or trade.R is not None:
            raise ValueError("OPEN / END_OF_DATA realized fields must stay NULL")

    months_with = 0
    max_month_share: float | None = None
    if monthly is not None and not monthly.empty:
        months_with = int((monthly["trades_closed"] > 0).sum())
        if net > 0:
            max_month_share = float(monthly["net_pnl"].max() / net)

    return ExperimentMetrics(
        starting_capital=float(result.starting_capital),
        ending_cash=ending_cash,
        ending_realized_equity=ending_realized,
        ending_mtm_equity=ending_mtm,
        total_net_pnl=net,
        total_gross_pnl=gross,
        total_fees=fees,
        total_slippage=slippage,
        signals=int(result.signals_received),
        filled_trades=int(result.entries_executed),
        closed_trades=n_closed,
        open_end_of_data=len(open_rows),
        ignored_signals=len(result.ignored),
        ignored_position_open=reasons.count(IGNORE_POSITION_OPEN),
        ignored_no_next_candle=reasons.count(IGNORE_NO_NEXT_CANDLE),
        ignored_insufficient_cash=reasons.count(IGNORE_INSUFFICIENT_CASH),
        tp_count=sum(1 for trade in result.trades if trade.exit_reason == EXIT_TP),
        sl_count=sum(1 for trade in result.trades if trade.exit_reason == EXIT_SL),
        win_count=len(wins),
        loss_count=len(losses),
        scratch_count=len(scratches),
        win_rate=win_rate,
        loss_rate=loss_rate,
        average_winning_trade=_mean(win_pnls),
        average_losing_trade=_mean(loss_pnls),
        median_winning_trade=_median(win_pnls),
        median_losing_trade=_median(loss_pnls),
        profit_factor=profit_factor,
        average_net_pnl_per_closed_trade=avg_net,
        average_R_per_closed_trade=avg_r,
        expectancy_per_trade=avg_net,
        expectancy_R=avg_r,
        median_R=_median(all_r),
        total_R=float(sum(all_r)) if all_r else None,
        average_holding_time=None if avg_hold is None else str(avg_hold),
        median_holding_time=None if med_hold is None else str(med_hold),
        max_holding_time=None if max_hold is None else str(max_hold),
        average_holding_candles=_candles(avg_hold),
        median_holding_candles=_candles(med_hold),
        max_holding_candles=_candles(max_hold),
        max_consecutive_wins=max_consecutive(closed)[0],
        max_consecutive_losses=max_consecutive(closed)[1],
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_peak_timestamp=None if peak_ts is None else str(peak_ts),
        max_drawdown_trough_timestamp=None if trough_ts is None else str(trough_ts),
        largest_win_net_pnl=max(win_pnls) if win_pnls else None,
        sum_of_winning_net_pnl=win_sum,
        months_with_closed_trades=months_with,
        max_month_share_of_net=max_month_share,
        open_entry_fees=open_entry_fees,
        unresolved_reserved=reserved,
    )


def metrics_from_components(
    result: BacktestResult,
    candles: pd.DataFrame,
    monthly: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ExperimentMetrics]:
    equity = reconstruct_equity(candles, result.trades, result.starting_capital)
    metrics = compute_metrics(result, equity, monthly=monthly)
    return equity, metrics
