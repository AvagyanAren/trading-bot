"""Monthly and yearly aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.ledger import EXIT_SL, EXIT_TP
from src.research.aggregation import monthly_table, yearly_table

from .conftest import make_trade


def test_monthly_attribution_by_exit_timestamp():
    trades = [
        make_trade(
            trade_id=1,
            net_pnl=0.2,
            exit_reason=EXIT_TP,
            exit_timestamp="2024-01-15 00:00:00",
            entry_timestamp="2024-01-15 00:00:00",
        ),
        make_trade(
            trade_id=2,
            net_pnl=-0.1,
            exit_reason=EXIT_SL,
            exit_timestamp="2024-02-02 00:00:00",
            entry_timestamp="2024-02-01 00:00:00",
        ),
    ]
    jan = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    feb = pd.date_range("2024-02-01", periods=3, freq="D", tz="UTC")
    timestamps = jan.append(feb)
    equity = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cash": 20.0,
            "position_quantity": 0.0,
            "position_mark_value": 0.0,
            "realized_equity": 20.0,
            "mtm_equity": [20.0, 21.0, 20.5, 20.0, 19.5, 20.2],
            "unrealized_pnl": 0.0,
            "drawdown": 0.0,
            "drawdown_pct": 0.0,
        }
    )
    monthly = monthly_table(trades, equity)
    assert list(monthly["period"]) == ["2024-01", "2024-02"]
    assert monthly.loc[0, "trades_closed"] == 1
    assert monthly.loc[1, "trades_closed"] == 1
    assert monthly.loc[0, "tp"] == 1
    assert monthly.loc[1, "sl"] == 1
    assert monthly.loc[0, "net_pnl"] == pytest.approx(0.2)
    assert monthly.loc[1, "cumulative_net_pnl"] == pytest.approx(0.1)
    assert "max_drawdown" in monthly.columns
    jan_dd = monthly.loc[0, "max_drawdown"]
    global_like = 21.0 - 20.0
    assert jan_dd == pytest.approx(0.5)


def test_yearly_split_december_versus_january():
    trades = [
        make_trade(
            trade_id=1,
            net_pnl=0.2,
            exit_timestamp="2024-12-31 00:00:00",
            entry_timestamp="2024-12-30 00:00:00",
        ),
        make_trade(
            trade_id=2,
            net_pnl=-0.05,
            exit_reason=EXIT_SL,
            exit_timestamp="2025-01-02 00:00:00",
            entry_timestamp="2025-01-01 00:00:00",
        ),
    ]
    timestamps = pd.to_datetime(
        ["2024-12-31 00:00:00", "2025-01-01 00:00:00", "2025-01-02 00:00:00"],
        utc=True,
    )
    equity = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cash": 20.0,
            "position_quantity": 0.0,
            "position_mark_value": 0.0,
            "realized_equity": 20.0,
            "mtm_equity": 20.0,
            "unrealized_pnl": 0.0,
            "drawdown": 0.0,
            "drawdown_pct": 0.0,
        }
    )
    yearly = yearly_table(trades, equity)
    assert list(yearly["period"]) == ["2024", "2025"]
    assert yearly.loc[0, "trades_closed"] == 1
    assert yearly.loc[1, "trades_closed"] == 1
