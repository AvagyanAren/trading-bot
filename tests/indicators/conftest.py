"""Synthetic OHLCV frames for indicator tests. Offline, no Binance access."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.indicators.engine import IndicatorConfig


def ohlcv_frame(
    close: list[float] | np.ndarray,
    start: str = "2024-01-01 00:00:00",
    volume: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a minimal validated-looking OHLCV frame from a close series."""
    close_arr = np.asarray(close, dtype="float64")
    n = len(close_arr)
    timestamps = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    if volume is None:
        volume_arr = np.arange(1, n + 1, dtype="float64")
    else:
        volume_arr = np.asarray(volume, dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close_arr,
            "high": close_arr + 1.0,
            "low": close_arr - 1.0,
            "close": close_arr,
            "volume": volume_arr,
        }
    )


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


@pytest.fixture
def tiny_config(tmp_path: Path) -> IndicatorConfig:
    return indicator_config(tmp_path)
