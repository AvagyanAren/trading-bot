"""Run one named research experiment through v0.3 signals and the v0.4 engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.config import BacktestConfig
from src.backtest.engine import BacktestResult, run_backtest
from src.indicators.engine import IndicatorConfig
from src.strategy.engine import SIGNAL_FRAME_COLUMNS, StrategyConfig, generate_signals

from .aggregation import monthly_table, yearly_table
from .config import REQUIRED_STARTING_CAPITAL, ROLE_DEVELOPMENT, ResearchError, VariantSpec
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


def _snapshot_str(raw: object, key: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ResearchExperimentError(f"{key} must be a non-empty string, got {raw!r}")
    return raw.strip()


def _snapshot_bool(raw: object, key: str) -> bool:
    if not isinstance(raw, bool):
        raise ResearchExperimentError(f"{key} must be a boolean, got {raw!r}")
    return raw


def _snapshot_int(raw: object, key: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ResearchExperimentError(f"{key} must be an integer, got {raw!r}")
    if raw < 1:
        raise ResearchExperimentError(f"{key} must be >= 1, got {raw}")
    return raw


def _snapshot_number(raw: object, key: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ResearchExperimentError(f"{key} must be a number, got {raw!r}")
    return float(raw)


def configs_from_snapshot(
    snapshot: dict,
    *,
    project_root: Path,
) -> tuple[StrategyConfig, BacktestConfig]:
    """Rebuild strategy/backtest configs from a frozen development snapshot.

    ``project_root`` comes from the live research config. Strategy and
    backtest knobs come only from ``snapshot``.
    """
    if not isinstance(snapshot, dict):
        raise ResearchExperimentError("snapshot must be a mapping")
    strategy_raw = snapshot.get("strategy")
    backtest_raw = snapshot.get("backtest")
    if not isinstance(strategy_raw, dict):
        raise ResearchExperimentError("snapshot.strategy must be a mapping")
    if not isinstance(backtest_raw, dict):
        raise ResearchExperimentError("snapshot.backtest must be a mapping")

    starting_capital = _snapshot_number(
        backtest_raw.get("starting_capital"),
        "snapshot.backtest.starting_capital",
    )
    if starting_capital != REQUIRED_STARTING_CAPITAL:
        raise ResearchExperimentError(
            "snapshot.backtest.starting_capital must be "
            f"{REQUIRED_STARTING_CAPITAL}, got {starting_capital}"
        )

    strategy = StrategyConfig(
        name=_snapshot_str(strategy_raw.get("name"), "snapshot.strategy.name"),
        version=_snapshot_str(strategy_raw.get("version"), "snapshot.strategy.version"),
        breakout_period=_snapshot_int(
            strategy_raw.get("breakout_period"),
            "snapshot.strategy.breakout_period",
        ),
        volume_multiplier=_snapshot_number(
            strategy_raw.get("volume_multiplier"),
            "snapshot.strategy.volume_multiplier",
        ),
        rsi_min=_snapshot_number(
            strategy_raw.get("rsi_min"), "snapshot.strategy.rsi_min"
        ),
        rsi_max=_snapshot_number(
            strategy_raw.get("rsi_max"), "snapshot.strategy.rsi_max"
        ),
        project_root=Path(project_root),
        rsi_filter_enabled=_snapshot_bool(
            strategy_raw.get("rsi_filter_enabled"),
            "snapshot.strategy.rsi_filter_enabled",
        ),
    )
    backtest = BacktestConfig(
        starting_capital=starting_capital,
        risk_per_trade=_snapshot_number(
            backtest_raw.get("risk_per_trade"),
            "snapshot.backtest.risk_per_trade",
        ),
        stop_loss=_snapshot_number(
            backtest_raw.get("stop_loss"), "snapshot.backtest.stop_loss"
        ),
        take_profit=_snapshot_number(
            backtest_raw.get("take_profit"), "snapshot.backtest.take_profit"
        ),
        entry_model=_snapshot_str(
            backtest_raw.get("entry_model"), "snapshot.backtest.entry_model"
        ),
        slippage=_snapshot_number(
            backtest_raw.get("slippage"), "snapshot.backtest.slippage"
        ),
        same_candle_priority=_snapshot_str(
            backtest_raw.get("same_candle_priority"),
            "snapshot.backtest.same_candle_priority",
        ),
        entry_rate=_snapshot_number(
            backtest_raw.get("entry_rate"), "snapshot.backtest.entry_rate"
        ),
        exit_rate=_snapshot_number(
            backtest_raw.get("exit_rate"), "snapshot.backtest.exit_rate"
        ),
        project_root=Path(project_root),
    )
    return strategy, backtest


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
