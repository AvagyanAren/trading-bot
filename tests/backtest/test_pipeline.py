"""Backtest pipeline: concat, report, no Parquet writes, strategy wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.pipeline import run_backtest_pipeline
from src.indicators.pipeline import (
    enriched_dataset_dir,
    split_by_calendar_month,
    write_monthly_parquet,
)
from src.strategy.engine import generate_signals
from src.strategy.signals import SignalType

from tests.strategy.conftest import (
    entry_ready_frame,
    indicator_config,
    make_data_config,
    strategy_config,
)

from .conftest import make_backtest_config


def _write_two_month_store(tmp_path: Path):
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path)
    data = make_data_config(tmp_path)
    january = entry_ready_frame(12, indicator, strategy, start="2024-01-01 00:00:00")
    february = entry_ready_frame(8, indicator, strategy, start="2024-02-01 00:00:00")
    full = pd.concat([january, february], ignore_index=True)
    dest = enriched_dataset_dir(data, indicator)
    parts = split_by_calendar_month(full)
    write_monthly_parquet(
        parts,
        dest_dir=dest,
        symbol=data.symbol,
        interval=data.interval,
        staging_dir=data.downloads_dir,
    )
    return data, indicator, strategy, full, dest


def test_pipeline_uses_v03_signals_and_two_months(tmp_path: Path):
    data, indicator, strategy, full, dest = _write_two_month_store(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in dest.glob("*.parquet")}
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    backtest = make_backtest_config(tmp_path)
    result = run_backtest_pipeline(data, indicator, strategy, backtest)
    expected_signals = generate_signals(
        full, strategy, indicator, interval=data.interval, symbol=data.symbol
    )
    long_count = int((expected_signals["signal"] == SignalType.LONG_ENTRY.value).sum())
    assert result.first.signals_received == long_count
    assert result.report.passed
    text = result.report_path.read_text(encoding="utf-8")
    assert "does NOT mean the strategy is profitable" in text
    assert "STATUS: PASS" in text or "STATUS: FAIL" in text
    assert result.trades_csv_path.exists()
    assert result.ignored_csv_path.exists()
    after = {path: path.stat().st_mtime_ns for path in dest.glob("*.parquet")}
    assert before == after
    assert list(processed.glob("*.parquet")) == []


def test_pipeline_is_deterministic(tmp_path: Path):
    data, indicator, strategy, _full, _dest = _write_two_month_store(tmp_path)
    backtest = make_backtest_config(tmp_path)
    first = run_backtest_pipeline(data, indicator, strategy, backtest)
    second = run_backtest_pipeline(data, indicator, strategy, backtest)
    pd.testing.assert_frame_equal(first.first.trade_frame, second.first.trade_frame)
    assert first.first.cash == second.first.cash
    assert first.report.status == second.report.status
