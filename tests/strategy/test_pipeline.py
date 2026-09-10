"""Strategy pipeline tests: concat, report, CSV, month-file continuity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.indicators.pipeline import (
    enriched_dataset_dir,
    split_by_calendar_month,
    write_monthly_parquet,
)
from src.strategy.engine import generate_signals
from src.strategy.pipeline import (
    StrategyEvaluationError,
    read_enriched,
    run_strategy_evaluation,
)
from src.strategy.signals import SignalType

from .conftest import (
    entry_ready_frame,
    indicator_config,
    make_data_config,
    strategy_config,
)


def _write_two_month_store(
    tmp_path: Path,
    indicator,
    strategy,
    data,
    january_rows: int = 10,
    february_rows: int = 6,
) -> pd.DataFrame:
    january = entry_ready_frame(
        january_rows, indicator, strategy, start="2024-01-01 00:00:00"
    )
    february = entry_ready_frame(
        february_rows, indicator, strategy, start="2024-02-01 00:00:00"
    )
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
    return full


def test_read_enriched_concatenates_monthly_files(tmp_path: Path):
    data = make_data_config(tmp_path)
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path)
    full = _write_two_month_store(tmp_path, indicator, strategy, data)
    loaded = read_enriched(enriched_dataset_dir(data, indicator))
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        full.reset_index(drop=True),
        check_dtype=True,
    )


def test_pipeline_matches_in_memory_concatenated_evaluation(tmp_path: Path):
    data = make_data_config(tmp_path)
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path)
    full = _write_two_month_store(tmp_path, indicator, strategy, data)
    expected = generate_signals(
        full, strategy, indicator, interval=data.interval, symbol=data.symbol
    )
    result = run_strategy_evaluation(data, indicator, strategy)
    pd.testing.assert_frame_equal(result.signals, expected)
    assert result.passed


def test_pipeline_february_matches_full_series_not_month_only_reset(tmp_path: Path):
    data = make_data_config(tmp_path)
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path)
    full = _write_two_month_store(tmp_path, indicator, strategy, data)
    result = run_strategy_evaluation(data, indicator, strategy)
    february = full.loc[full["timestamp"] >= pd.Timestamp("2024-02-01", tz="UTC")]
    month_only = generate_signals(
        february.reset_index(drop=True),
        strategy,
        indicator,
        interval=data.interval,
        symbol=data.symbol,
    )
    jan_len = int((full["timestamp"] < pd.Timestamp("2024-02-01", tz="UTC")).sum())
    assert pd.isna(month_only["breakout_level"].iloc[0])
    assert not pd.isna(result.signals["breakout_level"].iloc[jan_len])
    assert result.report.continuity is not None
    assert result.report.continuity.carried_forward


def test_pipeline_does_not_rewrite_enriched_parquet(tmp_path: Path):
    data = make_data_config(tmp_path)
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path)
    _write_two_month_store(tmp_path, indicator, strategy, data)
    dest = enriched_dataset_dir(data, indicator)
    before = {path.name: path.read_bytes() for path in dest.glob("*.parquet")}
    run_strategy_evaluation(data, indicator, strategy)
    after = {path.name: path.read_bytes() for path in dest.glob("*.parquet")}
    assert before == after


def test_report_and_csv_are_written(tmp_path: Path):
    data = make_data_config(tmp_path)
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path)
    _write_two_month_store(tmp_path, indicator, strategy, data)
    result = run_strategy_evaluation(data, indicator, strategy)
    text = result.report_path.read_text(encoding="utf-8")
    assert "STATUS: PASS" in text
    assert "independent of volume_ma_period" in text
    assert "not a fill" in text
    assert "P&L" not in text
    assert "Sharpe" not in text
    assert "win rate" not in text
    csv = pd.read_csv(result.signals_csv_path)
    assert list(csv.columns) == [
        "timestamp",
        "earliest_execution_time",
        "reference_close",
        "breakout_level",
        "ema_trend",
        "breakout",
        "volume_confirmation",
        "rsi_filter",
    ]
    assert len(csv) == result.report.long_entry_count
    assert result.report.long_entry_count > 0


def test_csv_header_is_written_when_there_are_no_entries(tmp_path: Path):
    data = make_data_config(tmp_path)
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path, rsi_min=90.0, rsi_max=100.0)
    _write_two_month_store(tmp_path, indicator, strategy, data)
    result = run_strategy_evaluation(data, indicator, strategy)
    assert result.report.long_entry_count == 0
    csv = pd.read_csv(result.signals_csv_path)
    assert csv.empty
    assert "timestamp" in csv.columns


def test_missing_enriched_store_raises(tmp_path: Path):
    data = make_data_config(tmp_path)
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path)
    with pytest.raises(StrategyEvaluationError, match="Enriched store is missing"):
        run_strategy_evaluation(data, indicator, strategy)


def test_last_bar_timing_label_is_outside_the_dataset(tmp_path: Path):
    data = make_data_config(tmp_path)
    indicator = indicator_config(tmp_path)
    strategy = strategy_config(tmp_path)
    full = _write_two_month_store(tmp_path, indicator, strategy, data)
    result = run_strategy_evaluation(data, indicator, strategy)
    last_signal = result.signals.iloc[-1]
    last_time = pd.Timestamp(full["timestamp"].iloc[-1])
    expected = last_time + np.timedelta64(5, "m")
    assert last_signal["earliest_execution_time"] == expected
    assert expected not in set(pd.Timestamp(value) for value in full["timestamp"])
    assert result.report.last_bar_execution_outside_dataset
    assert last_signal["signal"] in {
        SignalType.NONE.value,
        SignalType.LONG_ENTRY.value,
    }
    assert "execution_price" not in result.signals.columns
