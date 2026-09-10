"""Casts parsed Binance klines onto the stable internal schema.

Two things make this more than an ``astype`` call:

1. Binance Spot archives switched epoch units mid-range. Timestamps are
   milliseconds before 2025-01-01 and microseconds from 2025-01-01 onward, so
   the unit is detected per archive rather than assumed. Detection is validated
   against the calendar period the archive's filename claims to cover, so a
   wrong guess fails loudly instead of producing plausible nonsense.
2. Row order is preserved exactly as published. Reordering here would hide a
   genuine upstream ordering fault from the validator.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from .parser import IGNORED_COLUMN, ParsedArchive

# Canonical processed schema. ``timestamp`` is the candle open time, which is the
# only sane index for sequential time-series processing.
TIMESTAMP_COLUMN = "timestamp"

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_COLUMNS: tuple[str, ...] = (
    "volume",
    "quote_asset_volume",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
)
COUNT_COLUMNS: tuple[str, ...] = ("number_of_trades",)
TIME_COLUMNS: tuple[str, ...] = (TIMESTAMP_COLUMN, "close_time")

PROCESSED_COLUMNS: tuple[str, ...] = (
    TIMESTAMP_COLUMN,
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
)

PROCESSED_DTYPES: dict[str, str] = {
    TIMESTAMP_COLUMN: "datetime64[ns, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "datetime64[ns, UTC]",
    "quote_asset_volume": "float64",
    "number_of_trades": "int64",
    "taker_buy_base_asset_volume": "float64",
    "taker_buy_quote_asset_volume": "float64",
}

# Epoch units Binance has used, with their divisor to seconds.
_UNIT_DIVISORS: dict[str, int] = {
    "s": 1,
    "ms": 1_000,
    "us": 1_000_000,
    "ns": 1_000_000_000,
}

# Any real Binance candle falls inside this window. Used to identify the epoch
# unit: only one interpretation of a given magnitude lands in a plausible year.
_PLAUSIBLE_START = datetime(2010, 1, 1, tzinfo=UTC)
_PLAUSIBLE_END = datetime(2100, 1, 1, tzinfo=UTC)

_INTERVAL_PATTERN = re.compile(r"^(\d+)(mo|[smhdw])$")
_INTERVAL_UNITS: dict[str, str] = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}

# Values Binance writes into the documented-unused trailing field.
_EMPTY_IGNORE_VALUES = frozenset({"0", "0.0", "0.00000000", ""})


class NormalizerError(RuntimeError):
    """The archive could not be normalized without guessing."""


def interval_to_timedelta(interval: str) -> timedelta:
    """Convert a Binance interval token such as ``5m`` into a fixed duration."""
    match = _INTERVAL_PATTERN.match(interval.strip())
    if match is None:
        raise ValueError(f"Unrecognized Binance interval: {interval!r}")

    amount, unit = int(match.group(1)), match.group(2)
    if unit == "mo":
        raise ValueError(
            f"Interval {interval!r} is a calendar month and has no fixed "
            "duration; v0.1 supports second- through week-based intervals only."
        )
    if amount <= 0:
        raise ValueError(f"Interval {interval!r} must be positive")

    return timedelta(**{_INTERVAL_UNITS[unit]: amount})


def detect_timestamp_unit(values: pd.Series) -> str:
    """Identify the epoch unit of a timestamp column from its magnitude.

    Rather than assuming milliseconds, every unit Binance has published is
    tested and the one that decodes to a plausible calendar year wins. Adjacent
    units differ by three orders of magnitude, so exactly one ever fits.
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise NormalizerError("Cannot detect timestamp unit: no numeric values")

    probe = int(numeric.median())
    if probe <= 0:
        raise NormalizerError(f"Cannot detect timestamp unit from value {probe}")

    candidates = [
        unit
        for unit, divisor in _UNIT_DIVISORS.items()
        if _decodes_to_plausible_datetime(probe, divisor)
    ]

    if len(candidates) != 1:
        raise NormalizerError(
            f"Ambiguous timestamp unit for epoch value {probe}: "
            f"candidates {candidates or ['none']}. Refusing to guess."
        )
    return candidates[0]


def _decodes_to_plausible_datetime(value: int, divisor: int) -> bool:
    try:
        decoded = datetime.fromtimestamp(value / divisor, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return False
    return _PLAUSIBLE_START <= decoded < _PLAUSIBLE_END


def epoch_to_utc(values: pd.Series, unit: str) -> pd.Series:
    """Convert an integer epoch column to timezone-aware UTC timestamps."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        bad = values[numeric.isna()].head(3).tolist()
        raise NormalizerError(f"Non-numeric timestamp values, e.g. {bad!r}")
    return pd.to_datetime(numeric.astype("int64"), unit=unit, utc=True)


def normalize_klines(
    parsed: ParsedArchive,
    expected_start: date | None = None,
    expected_end: date | None = None,
) -> pd.DataFrame:
    """Cast one parsed archive onto the processed schema.

    ``expected_start`` and ``expected_end`` bound the calendar period the
    archive's filename advertises. Supplying them turns unit detection into a
    checked conversion: if the decoded timestamps fall outside the advertised
    period, the unit was misread and the archive is rejected.
    """
    frame = parsed.frame
    _assert_ignore_column_is_empty(frame, parsed.path.name)

    unit = detect_timestamp_unit(frame["open_time"])
    normalized = pd.DataFrame(index=frame.index)
    normalized[TIMESTAMP_COLUMN] = epoch_to_utc(frame["open_time"], unit)
    normalized["close_time"] = epoch_to_utc(frame["close_time"], unit)

    if expected_start is not None and expected_end is not None:
        _assert_within_period(
            normalized[TIMESTAMP_COLUMN], expected_start, expected_end,
            parsed.path.name, unit,
        )

    for column in PRICE_COLUMNS + VOLUME_COLUMNS:
        normalized[column] = _to_float(frame[column], column, parsed.path.name)
    for column in COUNT_COLUMNS:
        normalized[column] = _to_int(frame[column], column, parsed.path.name)

    # Row order is deliberately untouched; see the module docstring.
    return normalized[list(PROCESSED_COLUMNS)].astype(PROCESSED_DTYPES)


def _assert_ignore_column_is_empty(frame: pd.DataFrame, source: str) -> None:
    """Confirm the documented-unused field carries nothing before dropping it."""
    if IGNORED_COLUMN not in frame.columns:
        return
    unexpected = set(frame[IGNORED_COLUMN].str.strip().unique()) - _EMPTY_IGNORE_VALUES
    if unexpected:
        raise NormalizerError(
            f"{source}: the {IGNORED_COLUMN!r} field carries unexpected values "
            f"{sorted(unexpected)[:5]}. It is dropped from the processed schema, "
            "so refusing to discard it silently."
        )


def _assert_within_period(
    timestamps: pd.Series,
    expected_start: date,
    expected_end: date,
    source: str,
    unit: str,
) -> None:
    lower = pd.Timestamp(expected_start, tz="UTC")
    upper = pd.Timestamp(expected_end, tz="UTC") + timedelta(days=1)
    first, last = timestamps.iloc[0], timestamps.iloc[-1]
    if not (lower <= first < upper and lower <= last < upper):
        raise NormalizerError(
            f"{source}: timestamps decoded with unit {unit!r} span "
            f"{first} to {last}, outside the advertised period "
            f"{expected_start} to {expected_end}. The epoch unit was misread."
        )


def _to_float(values: pd.Series, column: str, source: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    _assert_fully_converted(values, numeric, column, source)
    return numeric.astype("float64")


def _to_int(values: pd.Series, column: str, source: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    _assert_fully_converted(values, numeric, column, source)
    if not (numeric % 1 == 0).all():
        offenders = numeric[numeric % 1 != 0].head(3).tolist()
        raise NormalizerError(
            f"{source}: column {column!r} expects whole numbers, got {offenders!r}"
        )
    return numeric.astype("int64")


def _assert_fully_converted(
    original: pd.Series, numeric: pd.Series, column: str, source: str
) -> None:
    failed = numeric.isna()
    if failed.any():
        offenders = original[failed].head(3).tolist()
        raise NormalizerError(
            f"{source}: column {column!r} has {int(failed.sum())} non-numeric "
            f"value(s), e.g. {offenders!r}"
        )


def expected_timestamp_index(
    start_date: date, end_date: date, interval: str
) -> pd.DatetimeIndex:
    """Build every timestamp the configured period should contain.

    ``end_date`` is inclusive of its final candle, so a 5m range ending
    2025-12-31 expects a last candle at 23:55 UTC. This index is the single
    source of truth for expected count, missing candles and boundary checks.
    """
    step = interval_to_timedelta(interval)
    first = pd.Timestamp(start_date, tz="UTC")
    last = pd.Timestamp(end_date, tz="UTC") + timedelta(days=1) - step
    if last < first:
        raise ValueError(f"Period {start_date}..{end_date} is shorter than {interval}")
    return pd.date_range(start=first, end=last, freq=step, tz="UTC")
