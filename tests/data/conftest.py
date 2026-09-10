"""Synthetic dataset builders for the data pipeline tests.

Every helper here is offline. No test touches Binance or the network, so the
suite is deterministic and runnable without credentials or connectivity.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.normalizer import (
    PROCESSED_COLUMNS,
    PROCESSED_DTYPES,
    TIMESTAMP_COLUMN,
    interval_to_timedelta,
)
from src.data.parser import KLINE_COLUMNS, ParsedArchive

# One full UTC day of 5m candles is the smallest dataset that still exercises
# the real expected-index, boundary and gap logic.
DEFAULT_DAY = date(2024, 1, 1)
DEFAULT_INTERVAL = "5m"

_UNIT_FACTORS: dict[str, int] = {"s": 1, "ms": 1_000, "us": 1_000_000}


def make_klines(
    day: date = DEFAULT_DAY, interval: str = DEFAULT_INTERVAL
) -> pd.DataFrame:
    """Build one valid UTC day of candles on the processed schema."""
    step = interval_to_timedelta(interval)
    timestamps = pd.date_range(
        start=pd.Timestamp(day, tz="UTC"),
        end=pd.Timestamp(day, tz="UTC") + timedelta(days=1) - step,
        freq=step,
        tz="UTC",
    )

    opens = [100.0 + index for index in range(len(timestamps))]
    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            "open": opens,
            "high": [value + 2.0 for value in opens],
            "low": [value - 2.0 for value in opens],
            "close": [value + 1.0 for value in opens],
            "volume": [10.0 + index for index in range(len(timestamps))],
            "close_time": timestamps + step - timedelta(milliseconds=1),
            "quote_asset_volume": [1_000.0 + index for index in range(len(timestamps))],
            "number_of_trades": list(range(1, len(timestamps) + 1)),
            "taker_buy_base_asset_volume": [5.0] * len(timestamps),
            "taker_buy_quote_asset_volume": [500.0] * len(timestamps),
        }
    )
    return frame[list(PROCESSED_COLUMNS)].astype(PROCESSED_DTYPES)


def make_raw_rows(
    unit: str,
    count: int = 3,
    day: date = DEFAULT_DAY,
    interval: str = DEFAULT_INTERVAL,
) -> list[list[str]]:
    """Build raw Binance kline rows with epochs in the requested unit."""
    factor = _UNIT_FACTORS[unit]
    step_seconds = int(interval_to_timedelta(interval).total_seconds())
    base_seconds = int(pd.Timestamp(day, tz="UTC").timestamp())

    rows: list[list[str]] = []
    for index in range(count):
        open_epoch = (base_seconds + index * step_seconds) * factor
        close_epoch = open_epoch + step_seconds * factor - 1
        price = 100 + index
        rows.append(
            [
                str(open_epoch),
                f"{price}.00000000",
                f"{price + 2}.00000000",
                f"{price - 2}.00000000",
                f"{price + 1}.00000000",
                "12.50000000",
                str(close_epoch),
                "1250.00000000",
                str(10 + index),
                "6.00000000",
                "600.00000000",
                "0",
            ]
        )
    return rows


def make_parsed_archive(
    rows: list[list[str]],
    name: str = "BTCUSDT-5m-2024-01.zip",
    had_header: bool = False,
) -> ParsedArchive:
    """Wrap raw rows in a ParsedArchive without touching the filesystem."""
    frame = pd.DataFrame(rows, columns=list(KLINE_COLUMNS), dtype=str)
    return ParsedArchive(
        path=Path(name), csv_name=name.replace(".zip", ".csv"),
        had_header=had_header, frame=frame,
    )


@pytest.fixture
def valid_klines() -> pd.DataFrame:
    """One clean UTC day of 5m candles (288 rows)."""
    return make_klines()


@pytest.fixture
def period() -> tuple[date, date]:
    """The configured period matching :func:`make_klines`."""
    return DEFAULT_DAY, DEFAULT_DAY
