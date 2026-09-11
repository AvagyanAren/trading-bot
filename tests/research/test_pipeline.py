"""Pipeline file outputs and CLI argument checks."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.indicators.pipeline import (
    enriched_dataset_dir,
    split_by_calendar_month,
    write_monthly_parquet,
)
from src.research.config import (
    ROLE_DEVELOPMENT,
    ROLE_TEST,
    VARIANT_ORDER,
    ResearchConfig,
)
from src.research.equity import EQUITY_COLUMNS
from src.research.experiments import _snapshot, run_experiment
from src.research.pipeline import (
    ResearchPipelineError,
    run_compare,
    run_test,
    write_experiment_dir,
)
from src.research.reports import write_snapshot
from tests.backtest.conftest import make_backtest_config
from tests.strategy.conftest import (
    entry_ready_frame,
    indicator_config,
    make_data_config,
    strategy_config,
)

from .conftest import make_research_config


def test_cli_test_stage_requires_candidate():
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "run_research.py"), "--stage", "test"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2


def test_pipeline_writes_expected_files(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    research = make_research_config(tmp_path)
    candles = entry_ready_frame(24, indicator, strategy, start="2024-01-01 00:00:00")
    extra = entry_ready_frame(8, indicator, strategy, start="2024-02-01 00:00:00")
    candles = pd.concat([candles, extra], ignore_index=True)
    for experiment_id in VARIANT_ORDER:
        variant_strategy = research.strategy_for(experiment_id)
        strategy_for_run = strategy_config(
            tmp_path,
            breakout_period=3,
            rsi_filter_enabled=variant_strategy.rsi_filter_enabled,
        )
        artifacts = run_experiment(
            candles,
            variant=research.variant(experiment_id),
            strategy_config=strategy_for_run,
            indicator_config=indicator,
            backtest_config=research.backtest_for(experiment_id),
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            period_role=ROLE_DEVELOPMENT,
            symbol="BTCUSDT",
            interval="5m",
        )
        write_experiment_dir(
            artifacts,
            research.output_dir / "development" / artifacts.folder,
            summary_name="summary.txt",
            symbol="BTCUSDT",
            interval="5m",
        )
    dest = run_compare(research)
    assert (dest / "comparison.txt").exists()
    assert (dest / "comparison.csv").exists()
    assert (dest / "selection_checklist.txt").exists()
    checklist = (dest / "selection_checklist.txt").read_text(encoding="utf-8")
    assert "NOT statistically validated" in checklist
    equity = pd.read_csv(
        research.output_dir / "development" / "A_baseline" / "equity.csv"
    )
    assert list(equity.columns) == list(EQUITY_COLUMNS)
    assert "ending capital" not in checklist.lower()


def _write_development_snapshot(research, strategy, backtest, experiment_id: str = "B"):
    variant = research.variant(experiment_id)
    payload = _snapshot(
        variant,
        strategy,
        backtest,
        ROLE_DEVELOPMENT,
        research.development.start,
        research.development.end,
    )
    path = research.output_dir / "development" / variant.folder / "config_snapshot.yaml"
    write_snapshot(payload, path)
    return payload, path


def _write_oos_store(tmp_path: Path, data, indicator, strategy, start: str = "2025-01-01 00:00:00"):
    candles = entry_ready_frame(24, indicator, strategy, start=start)
    dest = enriched_dataset_dir(data, indicator)
    write_monthly_parquet(
        split_by_calendar_month(candles),
        dest_dir=dest,
        symbol=data.symbol,
        interval=data.interval,
        staging_dir=data.downloads_dir,
    )
    return candles


def test_run_test_uses_frozen_snapshot_not_live_yaml(tmp_path: Path, monkeypatch):
    indicator = indicator_config(tmp_path)
    frozen_strategy = strategy_config(
        tmp_path, breakout_period=3, rsi_filter_enabled=False
    )
    frozen_backtest = make_backtest_config(tmp_path, take_profit=0.02)
    research = make_research_config(tmp_path)
    freeze, _ = _write_development_snapshot(
        research, frozen_strategy, frozen_backtest, "B"
    )
    data = make_data_config(tmp_path)
    _write_oos_store(tmp_path, data, indicator, frozen_strategy)

    mutated = replace(
        research.variant("B"),
        rsi_filter_enabled=True,
        take_profit=0.99,
        breakout_period=50,
    )
    research = replace(research, variants={**research.variants, "B": mutated})
    assert research.strategy_for("B").rsi_filter_enabled is True
    assert research.strategy_for("B").breakout_period == 50
    assert research.backtest_for("B").take_profit == pytest.approx(0.99)

    def _forbid_strategy_for(self, experiment_id: str):
        raise AssertionError("strategy_for must not be used for OOS")

    def _forbid_backtest_for(self, experiment_id: str):
        raise AssertionError("backtest_for must not be used for OOS")

    monkeypatch.setattr(ResearchConfig, "strategy_for", _forbid_strategy_for)
    monkeypatch.setattr(ResearchConfig, "backtest_for", _forbid_backtest_for)

    artifacts = run_test(data, indicator, research, "B")
    assert artifacts.strategy_config.rsi_filter_enabled is False
    assert artifacts.strategy_config.breakout_period == 3
    assert artifacts.backtest_config.take_profit == pytest.approx(0.02)
    assert artifacts.snapshot["strategy"] == freeze["strategy"]
    assert artifacts.snapshot["backtest"] == freeze["backtest"]
    assert artifacts.snapshot["period_role"] == ROLE_TEST

    written = yaml.safe_load(
        (
            research.output_dir / "test" / "B_no_rsi" / "config_snapshot.yaml"
        ).read_text(encoding="utf-8")
    )
    loaded_freeze = yaml.safe_load(
        (
            research.output_dir
            / "development"
            / "B_no_rsi"
            / "config_snapshot.yaml"
        ).read_text(encoding="utf-8")
    )
    assert written["strategy"] == loaded_freeze["strategy"]
    assert written["backtest"] == loaded_freeze["backtest"]
    assert written["period_role"] == ROLE_TEST


def test_run_test_missing_snapshot_errors(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    research = make_research_config(tmp_path)
    data = make_data_config(tmp_path)
    with pytest.raises(ResearchPipelineError, match="Missing development snapshot"):
        run_test(data, indicator, research, "B")


def test_run_test_snapshot_experiment_id_mismatch_errors(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    research = make_research_config(tmp_path)
    _, path = _write_development_snapshot(
        research, strategy, make_backtest_config(tmp_path), "B"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["experiment_id"] = "A"
    write_snapshot(payload, path)
    data = make_data_config(tmp_path)
    with pytest.raises(ResearchPipelineError, match="experiment_id"):
        run_test(data, indicator, research, "B")


def test_run_test_non_development_snapshot_errors(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, breakout_period=3)
    research = make_research_config(tmp_path)
    _, path = _write_development_snapshot(
        research, strategy, make_backtest_config(tmp_path), "B"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["period_role"] = ROLE_TEST
    write_snapshot(payload, path)
    data = make_data_config(tmp_path)
    with pytest.raises(ResearchPipelineError, match="period_role"):
        run_test(data, indicator, research, "B")
