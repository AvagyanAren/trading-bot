"""Synthetic trades and configs for v0.5 research tests. Offline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.ledger import (
    EXIT_SL,
    EXIT_TP,
    STATUS_CLOSED,
    STATUS_OPEN,
    SIDE_LONG,
    Trade,
)
from src.research.config import (
    ROLE_DEVELOPMENT,
    ROLE_TEST,
    EligibilityGates,
    PeriodSpec,
    ResearchConfig,
    VariantSpec,
)
from tests.backtest.conftest import make_backtest_config
from tests.strategy.conftest import strategy_config


def make_trade(
    *,
    trade_id: int = 1,
    status: str = STATUS_CLOSED,
    net_pnl: float | None = 0.2,
    gross_pnl: float | None = 0.4,
    R: float | None = 1.0,
    risk_amount: float = 0.2,
    exit_reason: str = EXIT_TP,
    entry_timestamp: str = "2024-01-01 00:05:00",
    exit_timestamp: str | None = "2024-01-01 00:15:00",
    entry_fee: float = 0.05,
    exit_fee: float | None = 0.05,
    slippage: float | None = 0.1,
    holding_minutes: float | None = 10.0,
    quantity: float = 0.001,
    position_value: float = 0.1,
    entry_price: float = 100.0,
    exit_price: float | None = 102.0,
) -> Trade:
    entry_ts = pd.Timestamp(entry_timestamp, tz="UTC")
    exit_ts = None if exit_timestamp is None else pd.Timestamp(exit_timestamp, tz="UTC")
    holding = None
    if holding_minutes is not None and status == STATUS_CLOSED:
        holding = pd.to_timedelta(holding_minutes, unit="min")
    return Trade(
        trade_id=trade_id,
        status=status,
        symbol="BTCUSDT",
        side=SIDE_LONG,
        signal_timestamp=entry_ts - pd.to_timedelta(5, unit="min"),
        entry_timestamp=entry_ts,
        entry_reference_price=100.0,
        entry_price=entry_price,
        exit_timestamp=exit_ts,
        exit_reference_price=exit_price,
        exit_price=exit_price,
        quantity=quantity,
        position_value=position_value,
        risk_amount=risk_amount,
        stop_loss=99.0,
        take_profit=102.0,
        exit_reason=exit_reason,
        gross_pnl=gross_pnl,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        entry_slippage=0.05,
        exit_slippage=None if slippage is None else 0.05,
        slippage=slippage,
        net_pnl=net_pnl,
        R=R,
        holding_time=holding,
    )


def make_result(
    trades: list[Trade],
    *,
    starting_capital: float = 20.0,
    cash: float | None = None,
    signals_received: int = 0,
) -> BacktestResult:
    closed = [trade for trade in trades if trade.status == STATUS_CLOSED]
    realized = float(sum(trade.net_pnl or 0.0 for trade in closed))
    reserved = 0.0
    open_rows = [trade for trade in trades if trade.status == STATUS_OPEN]
    if open_rows:
        reserved = open_rows[0].position_value + open_rows[0].entry_fee
    if cash is None:
        cash = starting_capital + realized - reserved
    return BacktestResult(
        starting_capital=starting_capital,
        cash=cash,
        realized_net_pnl=realized,
        min_cash=min(cash, starting_capital),
        candles_processed=0,
        signals_received=signals_received,
        entries_executed=len(trades),
        trades_closed=len(closed),
        unresolved_count=len(open_rows),
        last_close=100.0,
        unrealized_mark_value=None,
        informational_equity=None,
        pending_cleared=True,
        trades=trades,
        ignored=[],
    )


def flat_equity(
    starting: float = 20.0,
    n: int = 3,
    start: str = "2024-01-01 00:00:00",
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "cash": starting,
            "position_quantity": 0.0,
            "position_mark_value": 0.0,
            "realized_equity": starting,
            "mtm_equity": starting,
            "unrealized_pnl": 0.0,
            "drawdown": 0.0,
            "drawdown_pct": 0.0,
        }
    )


def default_gates() -> EligibilityGates:
    return EligibilityGates(
        min_closed_trades=30,
        max_drawdown_pct_of_starting_capital=0.50,
        max_largest_win_share_of_wins=0.40,
        min_months_with_trades=6,
        max_single_month_net_share=0.70,
    )


def make_research_config(tmp_path: Path, output_dir: Path | None = None) -> ResearchConfig:
    strategy = strategy_config(tmp_path)
    backtest = make_backtest_config(tmp_path)
    variants = {
        "A": VariantSpec("A", "A_baseline", "BASELINE", 20, 1.5, 50.0, 70.0, True, 0.02),
        "B": VariantSpec("B", "B_no_rsi", "NO RSI", 20, 1.5, 50.0, 70.0, False, 0.02),
        "C": VariantSpec("C", "C_higher_tp", "HIGHER TAKE PROFIT", 20, 1.5, 50.0, 70.0, True, 0.03),
        "D": VariantSpec("D", "D_stronger_breakout", "STRONGER BREAKOUT", 50, 1.5, 50.0, 70.0, True, 0.02),
    }
    return ResearchConfig(
        project_root=tmp_path,
        output_dir=output_dir or (tmp_path / "reports" / "research"),
        development=PeriodSpec(date(2024, 1, 1), date(2024, 12, 31), ROLE_DEVELOPMENT),
        test=PeriodSpec(date(2025, 1, 1), date(2025, 12, 31), ROLE_TEST),
        gates=default_gates(),
        variants=variants,
        base_strategy=strategy,
        base_backtest=backtest,
    )
