"""Run one named research experiment through v0.3 signals and the v0.4 engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestResult, run_backtest
from src.indicators.engine import IndicatorConfig
from src.strategy.engine import SIGNAL_FRAME_COLUMNS, StrategyConfig, generate_signals

from .aggregation import monthly_table, yearly_table
from .config import ROLE_DEVELOPMENT, ResearchError, VariantSpec
from .distributions import distribution_table
from .equity import reconstruct_equity
from .metrics import ExperimentMetrics, compute_metrics
from .periods import (
    align_signals_to_candles,
    assert_development_window,
    assert_trades_in_window,
    prepare_experiment_frames,
)


class ResearchExperimentError(ResearchError):
    """An experiment could not be completed."""


@dataclass
class ExperimentArtifacts:
    experiment_id: str
    folder: str
    period_role: str
    period_start: date
    period_end: date
    variant: VariantSpec
    strategy_config: StrategyConfig
    backtest_config: BacktestConfig
    result: BacktestResult
    trade_candles: pd.DataFrame
    trade_signals: pd.DataFrame
    equity: pd.DataFrame
    monthly: pd.DataFrame
    yearly: pd.DataFrame
    distributions: pd.DataFrame
    metrics: ExperimentMetrics
    snapshot: dict


def _snapshot(
    variant: VariantSpec,
    strategy: StrategyConfig,
    backtest: BacktestConfig,
    period_role: str,
    period_start: date,
    period_end: date,
) -> dict:
    return {
        "experiment_id": variant.experiment_id,
        "folder": variant.folder,
        "label": variant.label,
        "period_role": period_role,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "strategy": {
            "name": strategy.name,
            "version": strategy.version,
            "breakout_period": strategy.breakout_period,
            "volume_multiplier": strategy.volume_multiplier,
            "rsi_min": strategy.rsi_min,
            "rsi_max": strategy.rsi_max,
            "rsi_filter_enabled": strategy.rsi_filter_enabled,
        },
        "backtest": {
            "starting_capital": backtest.starting_capital,
            "risk_per_trade": backtest.risk_per_trade,
            "stop_loss": backtest.stop_loss,
            "take_profit": backtest.take_profit,
            "entry_model": backtest.entry_model,
            "slippage": backtest.slippage,
            "same_candle_priority": backtest.same_candle_priority,
            "entry_rate": backtest.entry_rate,
            "exit_rate": backtest.exit_rate,
        },
    }


def run_experiment(
    candles: pd.DataFrame,
    *,
    variant: VariantSpec,
    strategy_config: StrategyConfig,
    indicator_config: IndicatorConfig,
    backtest_config: BacktestConfig,
    period_start: date,
    period_end: date,
    period_role: str,
    symbol: str,
    interval: str,
) -> ExperimentArtifacts:
    original = candles.copy(deep=True)
    signal_candles, trade_candles = prepare_experiment_frames(
        candles,
        period_start,
        period_end,
        lookback_bars=strategy_config.breakout_period,
    )
    if period_role == ROLE_DEVELOPMENT:
        assert_development_window(trade_candles)
        assert_development_window(signal_candles)

    signals = generate_signals(
        signal_candles,
        strategy_config,
        indicator_config,
        interval=interval,
        symbol=symbol,
    )
    trade_signals = align_signals_to_candles(signals, trade_candles)
    missing = [column for column in SIGNAL_FRAME_COLUMNS if column not in trade_signals.columns]
    if missing:
        raise ResearchExperimentError(
            f"Aligned signals missing columns: {missing}"
        )

    first = run_backtest(
        trade_candles,
        trade_signals,
        backtest_config,
        symbol=symbol,
        interval=interval,
    )
    second = run_backtest(
        trade_candles,
        trade_signals,
        backtest_config,
        symbol=symbol,
        interval=interval,
    )
    if not (
        first.trade_frame.equals(second.trade_frame)
        and first.ignored_frame.equals(second.ignored_frame)
        and first.cash == second.cash
        and first.realized_net_pnl == second.realized_net_pnl
    ):
        raise ResearchExperimentError(
            f"Experiment {variant.experiment_id} was not deterministic"
        )
    if not original.equals(candles):
        raise ResearchExperimentError("Experiment mutated the input candle frame")

    assert_trades_in_window(first.trades, period_start, period_end)

    equity = reconstruct_equity(trade_candles, first.trades, backtest_config.starting_capital)
    monthly = monthly_table(first.trades, equity)
    yearly = yearly_table(first.trades, equity)
    distributions = distribution_table(first.trades)
    metrics = compute_metrics(first, equity, monthly=monthly)

    return ExperimentArtifacts(
        experiment_id=variant.experiment_id,
        folder=variant.folder,
        period_role=period_role,
        period_start=period_start,
        period_end=period_end,
        variant=variant,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        result=first,
        trade_candles=trade_candles,
        trade_signals=trade_signals,
        equity=equity,
        monthly=monthly,
        yearly=yearly,
        distributions=distributions,
        metrics=metrics,
        snapshot=_snapshot(
            variant,
            strategy_config,
            backtest_config,
            period_role,
            period_start,
            period_end,
        ),
    )
