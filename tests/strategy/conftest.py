"""Synthetic enriched frames for strategy tests. Offline, no Binance access."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import DataConfig
from src.indicators.engine import IndicatorConfig
from src.strategy.engine import StrategyConfig


def indicator_config(
    tmp_path: Path,
    ema_fast: int = 3,
    ema_slow: int = 5,
    rsi_period: int = 2,
    volume_ma_period: int = 3,
) -> IndicatorConfig:
    return IndicatorConfig(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_period=rsi_period,
        volume_ma_period=volume_ma_period,
        enriched_dir=tmp_path / "enriched",
        project_root=tmp_path,
    )


def strategy_config(
    tmp_path: Path,
    *,
    name: str = "breakout_volume_ema_rsi",
    version: str = "0.3",
    breakout_period: int = 3,
    volume_multiplier: float = 1.5,
    rsi_min: float = 50.0,
    rsi_max: float = 70.0,
) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        version=version,
        breakout_period=breakout_period,
        volume_multiplier=volume_multiplier,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        project_root=tmp_path,
    )


def make_data_config(
    tmp_path: Path,
    *,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start: date = date(2024, 1, 1),
    end: date = date(2024, 2, 29),
) -> DataConfig:
    return DataConfig(
        project_root=tmp_path,
        symbol=symbol,
        market="spot",
        data_type="klines",
        interval=interval,
        start_date=start,
        end_date=end,
        base_url="https://example.invalid",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        downloads_dir=tmp_path / "downloads",
        reports_dir=tmp_path / "reports",
        max_retries=1,
        backoff_seconds=0.1,
        timeout_seconds=1.0,
        chunk_size=1024,
        verify_existing=False,
    )


def enriched_frame(
    n: int,
    config: IndicatorConfig,
    *,
    start: str = "2024-01-01 00:00:00",
    close: list[float] | np.ndarray | None = None,
    high: list[float] | np.ndarray | None = None,
    volume: list[float] | np.ndarray | None = None,
    ema_fast: list[float] | np.ndarray | None = None,
    ema_slow: list[float] | np.ndarray | None = None,
    rsi: list[float] | np.ndarray | None = None,
    volume_ma: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a v0.2-looking frame. Indicator values are supplied, not calculated."""
    timestamps = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    if close is None:
        close_arr = np.full(n, 100.0, dtype="float64")
    else:
        close_arr = np.asarray(close, dtype="float64")
    if high is None:
        high_arr = close_arr + 1.0
    else:
        high_arr = np.asarray(high, dtype="float64")
    if volume is None:
        volume_arr = np.full(n, 20.0, dtype="float64")
    else:
        volume_arr = np.asarray(volume, dtype="float64")
    if ema_fast is None:
        ema_fast_arr = np.full(n, 2.0, dtype="float64")
    else:
        ema_fast_arr = np.asarray(ema_fast, dtype="float64")
    if ema_slow is None:
        ema_slow_arr = np.full(n, 1.0, dtype="float64")
    else:
        ema_slow_arr = np.asarray(ema_slow, dtype="float64")
    if rsi is None:
        rsi_arr = np.full(n, 60.0, dtype="float64")
    else:
        rsi_arr = np.asarray(rsi, dtype="float64")
    if volume_ma is None:
        volume_ma_arr = np.full(n, 10.0, dtype="float64")
    else:
        volume_ma_arr = np.asarray(volume_ma, dtype="float64")

    close_time = timestamps + np.timedelta64(5, "m") - np.timedelta64(1, "ms")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close_arr,
            "high": high_arr,
            "low": close_arr - 1.0,
            "close": close_arr,
            "volume": volume_arr,
            "close_time": close_time,
            config.ema_fast_column: ema_fast_arr,
            config.ema_slow_column: ema_slow_arr,
            config.rsi_column: rsi_arr,
            config.volume_ma_column: volume_ma_arr,
        }
    )


def entry_ready_frame(
    n: int,
    indicator: IndicatorConfig,
    strategy: StrategyConfig,
    *,
    start: str = "2024-01-01 00:00:00",
) -> pd.DataFrame:
    """Rows after the breakout lookback satisfy all four entry conditions."""
    highs = np.full(n, 100.0, dtype="float64")
    close = np.full(n, 100.0, dtype="float64")
    close[strategy.breakout_period :] = 101.0
    return enriched_frame(
        n,
        indicator,
        start=start,
        close=close,
        high=highs,
        volume=np.full(n, 20.0),
        ema_fast=np.full(n, 2.0),
        ema_slow=np.full(n, 1.0),
        rsi=np.full(n, 60.0),
        volume_ma=np.full(n, 10.0),
    )


@pytest.fixture
def tiny_indicator(tmp_path: Path) -> IndicatorConfig:
    return indicator_config(tmp_path)


@pytest.fixture
def tiny_strategy(tmp_path: Path) -> StrategyConfig:
    return strategy_config(tmp_path)
