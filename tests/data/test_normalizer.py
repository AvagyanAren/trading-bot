"""Normalizer tests: epoch unit detection, UTC conversion and explicit dtypes."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from src.data.normalizer import (
    PROCESSED_COLUMNS,
    PROCESSED_DTYPES,
    TIMESTAMP_COLUMN,
    NormalizerError,
    detect_timestamp_unit,
    expected_timestamp_index,
    interval_to_timedelta,
    normalize_klines,
)

from .conftest import make_parsed_archive, make_raw_rows

FIRST_CANDLE = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


# --- Interval parsing -------------------------------------------------------


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1s", timedelta(seconds=1)),
        ("5m", timedelta(minutes=5)),
        ("15m", timedelta(minutes=15)),
        ("1h", timedelta(hours=1)),
        ("1d", timedelta(days=1)),
        ("1w", timedelta(weeks=1)),
    ],
)
def test_interval_to_timedelta_parses_binance_tokens(interval, expected):
    assert interval_to_timedelta(interval) == expected


def test_interval_to_timedelta_rejects_calendar_month():
    with pytest.raises(ValueError, match="no fixed"):
        interval_to_timedelta("1mo")


def test_interval_to_timedelta_rejects_garbage():
    with pytest.raises(ValueError, match="Unrecognized"):
        interval_to_timedelta("5minutes")


# --- Epoch unit detection ---------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "epoch"),
    [
        ("s", 1_704_067_200),
        ("ms", 1_704_067_200_000),
        ("us", 1_704_067_200_000_000),
    ],
)
def test_detect_timestamp_unit_identifies_each_binance_unit(unit, epoch):
    assert detect_timestamp_unit(pd.Series([epoch, epoch + 1, epoch + 2])) == unit


def test_detect_timestamp_unit_rejects_implausible_magnitude():
    with pytest.raises(NormalizerError, match="Ambiguous timestamp unit"):
        detect_timestamp_unit(pd.Series([42, 43, 44]))


def test_detect_timestamp_unit_rejects_empty_column():
    with pytest.raises(NormalizerError, match="no numeric values"):
        detect_timestamp_unit(pd.Series(["", "abc"]))


# --- Timestamp normalization ------------------------------------------------


def test_milliseconds_are_normalized_to_utc():
    """Archives before 2025-01-01 publish millisecond epochs."""
    parsed = make_parsed_archive(make_raw_rows("ms", count=3))

    frame = normalize_klines(parsed)

    assert frame[TIMESTAMP_COLUMN].iloc[0] == FIRST_CANDLE
    assert frame[TIMESTAMP_COLUMN].iloc[1] == FIRST_CANDLE + timedelta(minutes=5)
    assert frame[TIMESTAMP_COLUMN].iloc[2] == FIRST_CANDLE + timedelta(minutes=10)


def test_microseconds_are_normalized_to_utc():
    """Archives from 2025-01-01 onward publish microsecond epochs."""
    parsed = make_parsed_archive(make_raw_rows("us", count=3))

    frame = normalize_klines(parsed)

    assert frame[TIMESTAMP_COLUMN].iloc[0] == FIRST_CANDLE
    assert frame[TIMESTAMP_COLUMN].iloc[1] == FIRST_CANDLE + timedelta(minutes=5)
    assert frame[TIMESTAMP_COLUMN].iloc[2] == FIRST_CANDLE + timedelta(minutes=10)


def test_millisecond_and_microsecond_archives_normalize_identically():
    """Candle open times must not reveal which epoch unit the archive used."""
    from_ms = normalize_klines(make_parsed_archive(make_raw_rows("ms", count=6)))
    from_us = normalize_klines(make_parsed_archive(make_raw_rows("us", count=6)))

    pd.testing.assert_series_equal(
        from_ms[TIMESTAMP_COLUMN], from_us[TIMESTAMP_COLUMN]
    )
    pd.testing.assert_frame_equal(
        from_ms.drop(columns=["close_time"]), from_us.drop(columns=["close_time"])
    )


def test_close_time_keeps_the_precision_the_archive_published():
    """A microsecond archive's close time is finer than a millisecond one's.

    Binance closes each candle one epoch tick before the next opens, so a
    millisecond archive ends at .999 and a microsecond archive at .999999.
    Both describe the same candle, and neither is rounded away.
    """
    from_ms = normalize_klines(make_parsed_archive(make_raw_rows("ms", count=3)))
    from_us = normalize_klines(make_parsed_archive(make_raw_rows("us", count=3)))

    assert from_ms["close_time"].iloc[0] == pd.Timestamp(
        "2024-01-01 00:04:59.999", tz="UTC"
    )
    assert from_us["close_time"].iloc[0] == pd.Timestamp(
        "2024-01-01 00:04:59.999999", tz="UTC"
    )
    widest_candle = (from_us["close_time"] - from_us[TIMESTAMP_COLUMN]).max()
    assert widest_candle.total_seconds() < 300


def test_timestamps_are_timezone_aware_utc():
    frame = normalize_klines(make_parsed_archive(make_raw_rows("ms")))

    for column in (TIMESTAMP_COLUMN, "close_time"):
        assert str(frame[column].dtype) == "datetime64[ns, UTC]"
        assert str(frame[column].dt.tz) == "UTC"
        assert frame[column].iloc[0].tzinfo is not None


def test_close_time_stays_inside_its_own_candle():
    frame = normalize_klines(make_parsed_archive(make_raw_rows("us", count=4)))

    delta = frame["close_time"] - frame[TIMESTAMP_COLUMN]
    assert (delta < timedelta(minutes=5)).all()
    assert (delta > timedelta(minutes=4)).all()


# --- Period cross-check -----------------------------------------------------


def test_period_cross_check_accepts_matching_archive():
    parsed = make_parsed_archive(make_raw_rows("ms", count=3))

    frame = normalize_klines(
        parsed, expected_start=date(2024, 1, 1), expected_end=date(2024, 1, 31)
    )

    assert len(frame) == 3


def test_period_cross_check_rejects_timestamps_outside_advertised_month():
    """A misread epoch unit must fail loudly instead of producing nonsense."""
    parsed = make_parsed_archive(make_raw_rows("ms", count=3))

    with pytest.raises(NormalizerError, match="epoch unit was misread"):
        normalize_klines(
            parsed, expected_start=date(2025, 6, 1), expected_end=date(2025, 6, 30)
        )


# --- Schema and dtypes ------------------------------------------------------


def test_processed_schema_uses_explicit_numeric_dtypes():
    frame = normalize_klines(make_parsed_archive(make_raw_rows("ms", count=5)))

    assert list(frame.columns) == list(PROCESSED_COLUMNS)
    for column, dtype in PROCESSED_DTYPES.items():
        assert str(frame[column].dtype) == dtype
    assert frame["number_of_trades"].dtype == "int64"
    assert not any(frame[column].dtype == object for column in frame.columns)


def test_row_order_is_preserved_so_ordering_faults_stay_visible():
    rows = make_raw_rows("ms", count=4)
    rows[1], rows[2] = rows[2], rows[1]
    parsed = make_parsed_archive(rows)

    frame = normalize_klines(parsed)

    assert not frame[TIMESTAMP_COLUMN].is_monotonic_increasing


def test_unused_ignore_field_is_not_discarded_silently():
    rows = make_raw_rows("ms", count=3)
    rows[1][-1] = "12345"
    parsed = make_parsed_archive(rows)

    with pytest.raises(NormalizerError, match="refusing to discard it silently"):
        normalize_klines(parsed)


def test_non_numeric_price_is_rejected():
    rows = make_raw_rows("ms", count=3)
    rows[2][1] = "not-a-price"
    parsed = make_parsed_archive(rows)

    with pytest.raises(NormalizerError, match="non-numeric"):
        normalize_klines(parsed)


# --- Expected index ---------------------------------------------------------


def test_expected_index_covers_the_inclusive_period():
    index = expected_timestamp_index(date(2024, 1, 1), date(2024, 1, 1), "5m")

    assert len(index) == 288
    assert index[0] == FIRST_CANDLE
    assert index[-1] == pd.Timestamp("2024-01-01 23:55:00", tz="UTC")


def test_expected_index_matches_the_configured_two_year_range():
    index = expected_timestamp_index(date(2024, 1, 1), date(2025, 12, 31), "5m")

    assert len(index) == 731 * 288
    assert index[0] == datetime(2024, 1, 1, tzinfo=UTC)
    assert index[-1] == datetime(2025, 12, 31, 23, 55, tzinfo=UTC)
