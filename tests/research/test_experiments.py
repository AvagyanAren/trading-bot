"""Experiment wiring against generate_signals + run_backtest."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.research.config import ROLE_DEVELOPMENT
from src.research.experiments import (
    ResearchExperimentError,
    configs_from_snapshot,
    run_experiment,
)
from src.strategy.engine import generate_signals
from tests.backtest.conftest import make_backtest_config
from tests.strategy.conftest import (
    entry_ready_frame,
    indicator_config,
    strategy_config,
)

from .conftest import make_research_config


def test_experiment_matches_direct_backtest(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    research = make_research_config(tmp_path)
    variant = research.variant("A")
    strategy_a = strategy
    backtest = make_backtest_config(tmp_path)
    candles = entry_ready_frame(40, indicator, strategy, start="2024-01-01 00:00:00")
    artifacts = run_experiment(
        candles,
        variant=variant,
        strategy_config=strategy_a,
        indicator_config=indicator,
        backtest_config=backtest,
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        period_role=ROLE_DEVELOPMENT,
        symbol="BTCUSDT",
        interval="5m",
    )
    signals = generate_signals(
        candles, strategy_a, indicator, interval="5m", symbol="BTCUSDT"
    )
    direct = run_backtest(
        candles, signals, backtest, symbol="BTCUSDT", interval="5m"
    )
    pd.testing.assert_frame_equal(artifacts.result.trade_frame, direct.trade_frame)
    assert artifacts.result.cash == direct.cash
    assert artifacts.metrics.ending_cash == pytest.approx(direct.cash)
    assert artifacts.metrics.ending_realized_equity == pytest.approx(
        direct.starting_capital + direct.realized_net_pnl
    )
    assert list(artifacts.equity.columns) == [
        "timestamp",
        "cash",
        "position_quantity",
        "position_mark_value",
        "realized_equity",
        "mtm_equity",
        "unrealized_pnl",
        "drawdown",
        "drawdown_pct",
    ]


def test_experiment_is_deterministic(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    research = make_research_config(tmp_path)
    candles = entry_ready_frame(30, indicator, strategy, start="2024-01-01 00:00:00")
    kwargs = dict(
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
    first = run_experiment(candles, **kwargs)
    second = run_experiment(candles, **kwargs)
    assert first.metrics == second.metrics
    pd.testing.assert_frame_equal(first.equity, second.equity)


def test_variant_b_fires_when_rsi_is_80(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    enabled = strategy_config(tmp_path, breakout_period=3, rsi_filter_enabled=True)
    disabled = strategy_config(tmp_path, breakout_period=3, rsi_filter_enabled=False)
    frame = entry_ready_frame(8, indicator, enabled)
    index = enabled.breakout_period
    frame.loc[index, indicator.rsi_column] = 80.0
    with_rsi = generate_signals(frame, enabled, indicator, interval="5m", symbol="BTCUSDT")
    without = generate_signals(frame, disabled, indicator, interval="5m", symbol="BTCUSDT")
    assert with_rsi["signal"].iloc[index] == "NONE"
    assert without["signal"].iloc[index] == "LONG_ENTRY"


def test_configs_from_snapshot_round_trip(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3, rsi_filter_enabled=False)
    research = make_research_config(tmp_path)
    candles = entry_ready_frame(30, indicator, strategy, start="2024-01-01 00:00:00")
    artifacts = run_experiment(
        candles,
        variant=research.variant("B"),
        strategy_config=strategy,
        indicator_config=indicator,
        backtest_config=make_backtest_config(tmp_path, take_profit=0.02),
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        period_role=ROLE_DEVELOPMENT,
        symbol="BTCUSDT",
        interval="5m",
    )
    reconstructed_root = tmp_path / "from-snapshot"
    strategy_out, backtest_out = configs_from_snapshot(
        artifacts.snapshot, project_root=reconstructed_root
    )
    assert strategy_out.breakout_period == strategy.breakout_period
    assert strategy_out.rsi_filter_enabled is False
    assert backtest_out.take_profit == pytest.approx(0.02)
    assert backtest_out.starting_capital == pytest.approx(20.0)
    assert strategy_out.project_root == reconstructed_root
    assert backtest_out.project_root == reconstructed_root


def test_configs_from_snapshot_rejects_wrong_capital(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    research = make_research_config(tmp_path)
    candles = entry_ready_frame(20, indicator, strategy, start="2024-01-01 00:00:00")
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
    tampered = dict(artifacts.snapshot)
    tampered["backtest"] = dict(tampered["backtest"], starting_capital=100.0)
    with pytest.raises(ResearchExperimentError, match="starting_capital"):
        configs_from_snapshot(tampered, project_root=tmp_path)


def test_variant_d_uses_50_prior_highs(tmp_path: Path):
    from src.research.config import load_research_config
    from pathlib import Path as P

    research = load_research_config(P(__file__).resolve().parents[2] / "config" / "research.yaml")
    assert research.strategy_for("D").breakout_period == 50
    assert research.strategy_for("A").breakout_period == 20
