"""Engine and enrichment-split tests. No trading logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.indicators.engine import (
    IndicatorInputError,
    calculate_indicators,
    load_indicator_config,
)
from src.indicators.pipeline import (
    split_by_calendar_month,
    write_monthly_parquet,
)

from .conftest import indicator_config, ohlcv_frame


def test_required_output_columns_are_present(tiny_config):
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    result = calculate_indicators(frame, tiny_config)

    for column in ("timestamp", "open", "high", "low", "close", "volume"):
        assert column in result.columns
    assert tiny_config.ema_fast_column in result.columns
    assert tiny_config.ema_slow_column in result.columns
    assert tiny_config.rsi_column in result.columns
    assert tiny_config.volume_ma_column in result.columns


def test_input_frame_is_not_mutated(tiny_config):
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    original = frame.copy(deep=True)

    calculate_indicators(frame, tiny_config)

    pd.testing.assert_frame_equal(frame, original)


def test_original_ohlcv_values_are_unchanged(tiny_config):
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    result = calculate_indicators(frame, tiny_config)

    for column in ("timestamp", "open", "high", "low", "close", "volume"):
        pd.testing.assert_series_equal(result[column], frame[column])


def test_row_count_and_order_are_unchanged(tiny_config):
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    result = calculate_indicators(frame, tiny_config)

    assert len(result) == len(frame)
    pd.testing.assert_series_equal(result["timestamp"], frame["timestamp"])


def test_indicator_dtypes_are_float64(tiny_config):
    frame = ohlcv_frame(list(range(20)))
    result = calculate_indicators(frame, tiny_config)

    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"
    for column in tiny_config.indicator_columns:
        assert result[column].dtype == np.float64


def test_pass_through_columns_are_kept(tiny_config):
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0])
    frame["quote_asset_volume"] = frame["volume"] * 2.0
    result = calculate_indicators(frame, tiny_config)

    pd.testing.assert_series_equal(
        result["quote_asset_volume"], frame["quote_asset_volume"]
    )


def test_config_driven_column_names_and_seed(tmp_path: Path):
    config = indicator_config(tmp_path, ema_fast=3, ema_slow=5, rsi_period=2, volume_ma_period=4)
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    result = calculate_indicators(frame, config)

    assert "ema3" in result.columns
    assert "ema5" in result.columns
    assert "rsi2" in result.columns
    assert "volume_ma4" in result.columns
    assert "ema20" not in result.columns
    # SMA seed for period 3: (10+11+12)/3 = 11
    assert result["ema3"].iloc[2] == pytest.approx(11.0)


def test_load_indicator_config_reads_yaml(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "indicators.yaml").write_text(
        "indicators:\n"
        "  ema_fast: 20\n"
        "  ema_slow: 50\n"
        "  rsi_period: 14\n"
        "  volume_ma_period: 20\n"
        "enriched_dir: data/enriched\n",
        encoding="utf-8",
    )
    loaded = load_indicator_config(config_dir / "indicators.yaml")

    assert loaded.ema_fast == 20
    assert loaded.ema_fast_column == "ema20"
    assert loaded.rsi_column == "rsi14"
    assert loaded.enriched_dir == tmp_path / "data" / "enriched"


def test_missing_column_fails_without_repair(tiny_config):
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0]).drop(columns=["volume"])

    with pytest.raises(IndicatorInputError, match="volume"):
        calculate_indicators(frame, tiny_config)


def test_non_numeric_close_fails_without_coercion(tiny_config):
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0])
    frame["close"] = frame["close"].astype(object)
    frame.loc[2, "close"] = "not-a-number"

    with pytest.raises(IndicatorInputError, match="floating-point"):
        calculate_indicators(frame, tiny_config)


def test_integer_close_fails_without_coercion(tiny_config):
    frame = ohlcv_frame([10.0, 11.0, 12.0, 13.0, 14.0])
    frame["close"] = frame["close"].astype("int64")

    with pytest.raises(IndicatorInputError, match="floating-point"):
        calculate_indicators(frame, tiny_config)


def _two_month_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """January and February synthetics with enough rows to leave warmup."""
    jan_close = [100.0 + i for i in range(40)]
    feb_close = [140.0 + i for i in range(20)]
    january = ohlcv_frame(jan_close, start="2024-01-01 00:00:00")
    february = ohlcv_frame(feb_close, start="2024-02-01 00:00:00")
    return january, february


def test_february_full_series_differs_from_february_only_reset(tmp_path: Path):
    config = indicator_config(tmp_path, ema_fast=5, ema_slow=8, rsi_period=4, volume_ma_period=5)
    january, february = _two_month_frames()
    full = pd.concat([january, february], ignore_index=True)

    full_indicators = calculate_indicators(full, config)
    february_only = calculate_indicators(february.reset_index(drop=True), config)

    full_feb = full_indicators.iloc[len(january) :].reset_index(drop=True)
    assert not np.isclose(
        full_feb[config.ema_fast_column].iloc[0],
        february_only[config.ema_fast_column].iloc[0],
    )
    assert not np.isclose(
        full_feb[config.rsi_column].iloc[0],
        february_only[config.rsi_column].iloc[0],
    )
    assert not np.isclose(
        full_feb[config.volume_ma_column].iloc[0],
        february_only[config.volume_ma_column].iloc[0],
    )


def test_enrichment_split_writes_full_series_february_values(tmp_path: Path):
    config = indicator_config(tmp_path, ema_fast=5, ema_slow=8, rsi_period=4, volume_ma_period=5)
    january, february = _two_month_frames()
    full = pd.concat([january, february], ignore_index=True)
    full_indicators = calculate_indicators(full, config)

    parts = split_by_calendar_month(full_indicators)
    assert set(parts) == {"2024-01", "2024-02"}

    dest = tmp_path / "enriched" / "BTCUSDT" / "5m"
    staging = tmp_path / "downloads"
    paths = write_monthly_parquet(
        parts, dest_dir=dest, symbol="BTCUSDT", interval="5m", staging_dir=staging
    )
    by_name = {path.name: path for path in paths}
    written_feb = pd.read_parquet(by_name["BTCUSDT-5m-2024-02.parquet"], engine="pyarrow")

    expected_feb = full_indicators.iloc[len(january) :].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        written_feb.reset_index(drop=True),
        expected_feb.reset_index(drop=True),
        check_dtype=True,
    )

    february_only = calculate_indicators(february.reset_index(drop=True), config)
    assert not np.isclose(
        written_feb[config.ema_fast_column].iloc[0],
        february_only[config.ema_fast_column].iloc[0],
    )


def test_split_does_not_recompute_indicators(tmp_path: Path):
    """Timestamps are a storage key. Splitting must not change already-computed values."""
    config = indicator_config(tmp_path)
    january, february = _two_month_frames()
    full = pd.concat([january, february], ignore_index=True)
    computed = calculate_indicators(full, config)
    parts = split_by_calendar_month(computed)
    rebuilt = pd.concat(parts.values(), ignore_index=True)

    pd.testing.assert_frame_equal(rebuilt, computed.reset_index(drop=True))
