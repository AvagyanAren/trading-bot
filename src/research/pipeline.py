"""Orchestrate v0.5 research stages. Never overwrites v0.4 reports."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from src.backtest.ledger import STATUS_CLOSED
from src.data.pipeline import DataConfig
from src.indicators.engine import IndicatorConfig
from src.indicators.pipeline import enriched_dataset_dir
from src.strategy.pipeline import read_enriched

from .config import (
    ROLE_DEVELOPMENT,
    ROLE_TEST,
    VARIANT_ORDER,
    ResearchConfig,
    ResearchConfigError,
    ResearchError,
)
from .experiments import ExperimentArtifacts, run_experiment
from .integrity import FROZEN_V04_FULL_PERIOD, RECONCILE_EPS
from .metrics import ExperimentMetrics
from .reports import (
    comparison_rows,
    render_checklist,
    render_comparison,
    render_summary,
    write_frame_csv,
    write_metrics_csv,
    write_snapshot,
    write_trades_csv,
)
from .selection import ResearchSelectionError, evaluate_selection

ProgressCallback = Callable[[str], None]


class ResearchPipelineError(ResearchError):
    """The research pipeline could not complete."""


def _log(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _strategy_lines(artifacts: ExperimentArtifacts) -> list[str]:
    strategy = artifacts.strategy_config
    return [
        f"name:                {strategy.name}",
        f"version:             {strategy.version}",
        f"breakout_period:     {strategy.breakout_period}",
        f"volume_multiplier:   {strategy.volume_multiplier}",
        f"rsi_min:             {strategy.rsi_min}",
        f"rsi_max:             {strategy.rsi_max}",
        f"rsi_filter_enabled:  {strategy.rsi_filter_enabled}",
    ]


def _backtest_lines(artifacts: ExperimentArtifacts) -> list[str]:
    config = artifacts.backtest_config
    return [
        f"starting_capital:    {config.starting_capital}",
        f"risk_per_trade:      {config.risk_per_trade}",
        f"stop_loss:           {config.stop_loss}",
        f"take_profit:          {config.take_profit}",
        f"entry_model:         {config.entry_model}",
        f"slippage:            {config.slippage}",
        f"same_candle_priority: {config.same_candle_priority}",
        f"entry_rate:          {config.entry_rate}",
        f"exit_rate:           {config.exit_rate}",
    ]


def write_experiment_dir(
    artifacts: ExperimentArtifacts,
    dest: Path,
    *,
    summary_name: str,
    symbol: str,
    interval: str,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    write_snapshot(artifacts.snapshot, dest / "config_snapshot.yaml")
    summary = render_summary(
        period_role=artifacts.period_role,
        period_start=artifacts.period_start,
        period_end=artifacts.period_end,
        symbol=symbol,
        interval=interval,
        variant=artifacts.variant,
        metrics=artifacts.metrics,
        strategy_lines=_strategy_lines(artifacts),
        backtest_lines=_backtest_lines(artifacts),
    )
    (dest / summary_name).write_text(summary, encoding="utf-8")
    write_metrics_csv(artifacts.metrics, dest / "metrics.csv")
    write_frame_csv(artifacts.monthly, dest / "monthly.csv")
    write_frame_csv(artifacts.yearly, dest / "yearly.csv")
    write_frame_csv(artifacts.distributions, dest / "distributions.csv")
    write_trades_csv(artifacts.result.trade_frame, dest / "trades.csv")
    artifacts.result.ignored_frame.to_csv(dest / "ignored.csv", index=False)
    write_frame_csv(artifacts.equity, dest / "equity.csv")


def _run_variant(
    candles: pd.DataFrame,
    research: ResearchConfig,
    indicator_config: IndicatorConfig,
    experiment_id: str,
    *,
    period_role: str,
    symbol: str,
    interval: str,
) -> ExperimentArtifacts:
    variant = research.variant(experiment_id)
    if period_role == ROLE_DEVELOPMENT:
        start, end = research.development.start, research.development.end
        role = research.development.role
    elif period_role == ROLE_TEST:
        start, end = research.test.start, research.test.end
        role = research.test.role
    else:
        raise ResearchPipelineError(f"Unknown period role {period_role!r}")
    return run_experiment(
        candles,
        variant=variant,
        strategy_config=research.strategy_for(experiment_id),
        indicator_config=indicator_config,
        backtest_config=research.backtest_for(experiment_id),
        period_start=start,
        period_end=end,
        period_role=role,
        symbol=symbol,
        interval=interval,
    )


def run_development(
    data_config: DataConfig,
    indicator_config: IndicatorConfig,
    research: ResearchConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, ExperimentArtifacts]:
    dest_dir = enriched_dataset_dir(data_config, indicator_config)
    _log(progress, "Reading enriched Parquet (read-only)")
    candles = read_enriched(dest_dir)
    artifacts_by_id: dict[str, ExperimentArtifacts] = {}
    root = research.output_dir / "development"
    for experiment_id in VARIANT_ORDER:
        _log(progress, f"Running development experiment {experiment_id}")
        artifacts = _run_variant(
            candles,
            research,
            indicator_config,
            experiment_id,
            period_role=ROLE_DEVELOPMENT,
            symbol=data_config.symbol,
            interval=data_config.interval,
        )
        write_experiment_dir(
            artifacts,
            root / artifacts.folder,
            summary_name="summary.txt",
            symbol=data_config.symbol,
            interval=data_config.interval,
        )
        artifacts_by_id[experiment_id] = artifacts
    return artifacts_by_id


def _load_snapshot(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ResearchPipelineError(f"{path} is not a mapping")
    return payload


def _metrics_from_csv(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    return {
        str(row["metric"]): (None if pd.isna(row["value"]) else row["value"])
        for _, row in frame.iterrows()
    }


def _parse_optional_float(raw: object) -> float | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if text == "" or text == "n/a":
        return None
    return float(text)


def _metrics_from_pairs(pairs: dict[str, object]) -> ExperimentMetrics:
    def i(name: str) -> int:
        return int(float(pairs[name]))

    def f(name: str) -> float:
        return float(pairs[name])

    def opt(name: str) -> float | None:
        return _parse_optional_float(pairs.get(name))

    def s(name: str) -> str | None:
        value = pairs.get(name)
        if value is None or str(value) == "n/a":
            return None
        return str(value)

    return ExperimentMetrics(
        starting_capital=f("starting_capital"),
        ending_cash=f("ending_cash"),
        ending_realized_equity=f("ending_realized_equity"),
        ending_mtm_equity=f("ending_mtm_equity"),
        total_net_pnl=f("total_net_pnl"),
        total_gross_pnl=f("total_gross_pnl"),
        total_fees=f("total_fees"),
        total_slippage=f("total_slippage"),
        signals=i("signals"),
        filled_trades=i("filled_trades"),
        closed_trades=i("closed_trades"),
        open_end_of_data=i("open_end_of_data"),
        ignored_signals=i("ignored_signals"),
        ignored_position_open=i("ignored_position_open"),
        ignored_no_next_candle=i("ignored_no_next_candle"),
        ignored_insufficient_cash=i("ignored_insufficient_cash"),
        tp_count=i("tp_count"),
        sl_count=i("sl_count"),
        win_count=i("win_count"),
        loss_count=i("loss_count"),
        scratch_count=i("scratch_count"),
        win_rate=opt("win_rate"),
        loss_rate=opt("loss_rate"),
        average_winning_trade=opt("average_winning_trade"),
        average_losing_trade=opt("average_losing_trade"),
        median_winning_trade=opt("median_winning_trade"),
        median_losing_trade=opt("median_losing_trade"),
        profit_factor=opt("profit_factor"),
        average_net_pnl_per_closed_trade=opt("average_net_pnl_per_closed_trade"),
        average_R_per_closed_trade=opt("average_R_per_closed_trade"),
        expectancy_per_trade=opt("expectancy_per_trade"),
        expectancy_R=opt("expectancy_R"),
        median_R=opt("median_R"),
        total_R=opt("total_R"),
        average_holding_time=s("average_holding_time"),
        median_holding_time=s("median_holding_time"),
        max_holding_time=s("max_holding_time"),
        average_holding_candles=opt("average_holding_candles"),
        median_holding_candles=opt("median_holding_candles"),
        max_holding_candles=opt("max_holding_candles"),
        max_consecutive_wins=i("max_consecutive_wins"),
        max_consecutive_losses=i("max_consecutive_losses"),
        max_drawdown=f("max_drawdown"),
        max_drawdown_pct=opt("max_drawdown_pct"),
        max_drawdown_peak_timestamp=s("max_drawdown_peak_timestamp"),
        max_drawdown_trough_timestamp=s("max_drawdown_trough_timestamp"),
        largest_win_net_pnl=opt("largest_win_net_pnl"),
        sum_of_winning_net_pnl=opt("sum_of_winning_net_pnl"),
        months_with_closed_trades=i("months_with_closed_trades"),
        max_month_share_of_net=opt("max_month_share_of_net"),
        open_entry_fees=f("open_entry_fees"),
        unresolved_reserved=f("unresolved_reserved"),
    )


def run_compare(
    research: ResearchConfig,
    progress: ProgressCallback | None = None,
) -> Path:
    root = research.output_dir / "development"
    metrics_by_id: dict[str, ExperimentMetrics] = {}
    labels: dict[str, str] = {}
    roles: dict[str, str] = {}
    for experiment_id in VARIANT_ORDER:
        folder = research.variant(experiment_id).folder
        dest = root / folder
        snapshot_path = dest / "config_snapshot.yaml"
        metrics_path = dest / "metrics.csv"
        if not snapshot_path.exists() or not metrics_path.exists():
            raise ResearchPipelineError(
                f"Missing development outputs for {experiment_id} at {dest}"
            )
        snapshot = _load_snapshot(snapshot_path)
        role = str(snapshot.get("period_role", ""))
        if role != ROLE_DEVELOPMENT:
            raise ResearchSelectionError(
                f"compare refused {dest}: period_role={role!r}"
            )
        roles[experiment_id] = role
        labels[experiment_id] = str(snapshot.get("label", experiment_id))
        metrics_by_id[experiment_id] = _metrics_from_pairs(_metrics_from_csv(metrics_path))
        _log(progress, f"Loaded development {experiment_id}")

    rows = comparison_rows(metrics_by_id, labels)
    (root / "comparison.txt").write_text(render_comparison(rows), encoding="utf-8")
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False, na_rep="n/a")
    selection = evaluate_selection(metrics_by_id, research.gates, period_roles=roles)
    (root / "selection_checklist.txt").write_text(
        render_checklist(selection), encoding="utf-8"
    )
    _log(progress, f"Wrote comparison to {root}")
    return root


def run_test(
    data_config: DataConfig,
    indicator_config: IndicatorConfig,
    research: ResearchConfig,
    candidate: str,
    progress: ProgressCallback | None = None,
) -> ExperimentArtifacts:
    experiment_id = str(candidate).strip().upper()
    if experiment_id not in VARIANT_ORDER:
        raise ResearchConfigError(
            f"--candidate must be one of {list(VARIANT_ORDER)}, got {candidate!r}"
        )
    dest_dir = enriched_dataset_dir(data_config, indicator_config)
    _log(progress, "Reading enriched Parquet (read-only)")
    candles = read_enriched(dest_dir)
    _log(progress, f"Running OUT-OF-SAMPLE test for {experiment_id}")
    artifacts = _run_variant(
        candles,
        research,
        indicator_config,
        experiment_id,
        period_role=ROLE_TEST,
        symbol=data_config.symbol,
        interval=data_config.interval,
    )
    write_experiment_dir(
        artifacts,
        research.output_dir / "test" / artifacts.folder,
        summary_name="OUT_OF_SAMPLE.txt",
        symbol=data_config.symbol,
        interval=data_config.interval,
    )
    return artifacts


def run_integrity(
    data_config: DataConfig,
    indicator_config: IndicatorConfig,
    research: ResearchConfig,
    progress: ProgressCallback | None = None,
) -> ExperimentArtifacts:
    dest_dir = enriched_dataset_dir(data_config, indicator_config)
    _log(progress, "Reading enriched Parquet for integrity replay")
    candles = read_enriched(dest_dir)
    artifacts = run_experiment(
        candles,
        variant=research.variant("A"),
        strategy_config=research.strategy_for("A"),
        indicator_config=indicator_config,
        backtest_config=research.backtest_for("A"),
        period_start=data_config.start_date,
        period_end=data_config.end_date,
        period_role="INTEGRITY / FULL PERIOD / NOT USED FOR SELECTION",
        symbol=data_config.symbol,
        interval=data_config.interval,
    )
    frozen = FROZEN_V04_FULL_PERIOD
    closed = [trade for trade in artifacts.result.trades if trade.status == STATUS_CLOSED]
    entry_fees = float(sum(trade.entry_fee for trade in artifacts.result.trades))
    exit_fees = float(sum(trade.exit_fee or 0.0 for trade in closed))
    slippage = float(sum(trade.slippage or 0.0 for trade in closed))
    checks = {
        "signals": artifacts.result.signals_received == frozen.signals,
        "filled_trades": artifacts.result.entries_executed == frozen.filled_trades,
        "closed_trades": artifacts.metrics.closed_trades == frozen.closed_trades,
        "tp_count": artifacts.metrics.tp_count == frozen.tp_count,
        "sl_count": artifacts.metrics.sl_count == frozen.sl_count,
        "total_entry_fees": bool(
            np.isclose(entry_fees, frozen.total_entry_fees, rtol=0.0, atol=RECONCILE_EPS)
        ),
        "total_exit_fees": bool(
            np.isclose(exit_fees, frozen.total_exit_fees, rtol=0.0, atol=RECONCILE_EPS)
        ),
        "total_slippage": bool(
            np.isclose(slippage, frozen.total_slippage, rtol=0.0, atol=RECONCILE_EPS)
        ),
        "total_gross_pnl": bool(
            np.isclose(
                artifacts.metrics.total_gross_pnl,
                frozen.total_gross_pnl,
                rtol=0.0,
                atol=RECONCILE_EPS,
            )
        ),
        "total_net_pnl": bool(
            np.isclose(
                artifacts.metrics.total_net_pnl,
                frozen.total_net_pnl,
                rtol=0.0,
                atol=RECONCILE_EPS,
            )
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ResearchPipelineError(
            "Integrity replay diverged from frozen v0.4 totals: " + ", ".join(failed)
        )
    write_experiment_dir(
        artifacts,
        research.output_dir / "integrity" / "A_full_period",
        summary_name="summary.txt",
        symbol=data_config.symbol,
        interval=data_config.interval,
    )
    _log(progress, "Integrity replay matched frozen v0.4 totals")
    return artifacts
