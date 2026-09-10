"""MTM equity reconstruction and global vs monthly drawdown."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.ledger import EXIT_END_OF_DATA, STATUS_OPEN
from src.research.equity import (
    EQUITY_COLUMNS,
    global_max_drawdown,
    intra_window_max_drawdown,
    reconstruct_equity,
)

from .conftest import make_trade


def test_global_max_drawdown_on_complete_curve():
    equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="MS", tz="UTC"),
            "mtm_equity": [20.0, 22.0, 18.0, 21.0],
            "drawdown": [0.0, 0.0, 4.0, 1.0],
            "drawdown_pct": [0.0, 0.0, 4.0 / 22.0, 1.0 / 22.0],
        }
    )
    max_dd, max_pct, peak_ts, trough_ts = global_max_drawdown(equity)
    assert max_dd == pytest.approx(4.0)
    assert max_pct == pytest.approx(4.0 / 22.0)
    assert trough_ts == equity["timestamp"].iloc[2]
    assert peak_ts == equity["timestamp"].iloc[1]


def test_monthly_drawdown_resets_peak():
    mtm_jan = pd.Series([20.0, 22.0])
    mtm_feb = pd.Series([18.0, 21.0])
    jan_dd, _ = intra_window_max_drawdown(mtm_jan)
    feb_dd, _ = intra_window_max_drawdown(mtm_feb)
    assert jan_dd == pytest.approx(0.0)
    assert feb_dd == pytest.approx(0.0)


def test_reconstruct_equity_schema_and_running_drawdown():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
            "close": [100.0, 100.0, 100.0],
        }
    )
    equity = reconstruct_equity(candles, [], starting_capital=20.0)
    assert list(equity.columns) == list(EQUITY_COLUMNS)
    assert list(equity["drawdown"]) == [0.0, 0.0, 0.0]
    assert list(equity["mtm_equity"]) == [20.0, 20.0, 20.0]


def test_open_position_mtm_does_not_change_realized_equity():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
            "close": [100.0, 110.0, 120.0],
        }
    )
    trade = make_trade(
        status=STATUS_OPEN,
        exit_reason=EXIT_END_OF_DATA,
        net_pnl=None,
        gross_pnl=None,
        R=None,
        exit_timestamp=None,
        exit_price=None,
        exit_fee=None,
        slippage=None,
        holding_minutes=None,
        entry_timestamp="2024-01-01 00:05:00",
        quantity=0.1,
        position_value=10.0,
        entry_fee=0.1,
        entry_price=100.0,
    )
    equity = reconstruct_equity(candles, [trade], starting_capital=20.0)
    assert equity["realized_equity"].iloc[0] == pytest.approx(20.0)
    assert equity["realized_equity"].iloc[-1] == pytest.approx(20.0)
    assert equity["cash"].iloc[-1] == pytest.approx(20.0 - 10.0 - 0.1)
    assert equity["mtm_equity"].iloc[-1] == pytest.approx(
        equity["cash"].iloc[-1] + 0.1 * 120.0
    )
    assert equity["unrealized_pnl"].iloc[-1] != pytest.approx(0.0)
    assert list(equity.columns) == list(EQUITY_COLUMNS)
