"""Report wording and equity.csv schema."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.research.config import ROLE_DEVELOPMENT
from src.research.equity import EQUITY_COLUMNS
from src.research.experiments import run_experiment
from src.research.pipeline import write_experiment_dir
from src.research.reports import render_comparison, render_summary
from tests.backtest.conftest import make_backtest_config
from tests.strategy.conftest import entry_ready_frame, indicator_config, strategy_config

from .conftest import make_research_config


def test_summary_and_comparison_avoid_ending_capital(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    research = make_research_config(tmp_path)
    candles = entry_ready_frame(20, indicator, strategy)
    artifacts = run_experiment(
        candles,
        variant=research.variant("A"),
        strategy_config=strategy,
        indicator_config=indicator,
        backtest_config=make_backtest_config(tmp_path),
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        period_role=ROLE_DEVELOPMENT,
        symbol="BTCUSDT",
        interval="5m",
    )
    dest = tmp_path / "A_baseline"
    write_experiment_dir(
        artifacts,
        dest,
        summary_name="summary.txt",
        symbol="BTCUSDT",
        interval="5m",
    )
    text = (dest / "summary.txt").read_text(encoding="utf-8")
    assert "ending_cash" in text
    assert "ending_realized_equity" in text
    assert "ending_mtm_equity" in text
    assert "ending capital" not in text.lower()
    assert "final capital" not in text.lower()
    assert "DEVELOPMENT" in text
    equity = pd.read_csv(dest / "equity.csv")
    assert list(equity.columns) == list(EQUITY_COLUMNS)
    metrics = pd.read_csv(dest / "metrics.csv")
    names = set(metrics["metric"])
    assert "average_net_pnl_per_closed_trade" in names
    assert "average_R_per_closed_trade" in names

    comparison = render_comparison(
        [
            {
                "experiment_id": experiment_id,
                "label": experiment_id,
                "closed_trades": 1,
                "win_rate": 0.3,
                "average_winning_trade": 0.1,
                "average_losing_trade": -0.1,
                "average_net_pnl_per_closed_trade": 0.01,
                "average_R_per_closed_trade": 0.1,
                "expectancy_per_trade": 0.01,
                "expectancy_R": 0.1,
                "profit_factor": 1.1,
                "total_gross_pnl": 1.0,
                "total_fees": 0.2,
                "total_slippage": 0.1,
                "total_net_pnl": -0.1,
                "max_drawdown": 1.0,
                "max_drawdown_pct": 0.05,
                "total_R": 1.0,
                "max_consecutive_losses": 3,
                "ending_cash": 19.0,
                "ending_realized_equity": 19.0,
                "ending_mtm_equity": 19.0,
            }
            for experiment_id in ("A", "B", "C", "D")
        ],
        generated="2026-01-01 00:00:00 UTC",
    )
    assert "does not declare a winner" in comparison.lower()
    assert "ending capital" not in comparison.lower()
    assert "DEVELOPMENT" in comparison
