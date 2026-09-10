"""Validator tests: severity model, gaps, duplicates, OHLC and timestamp checks."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.normalizer import TIMESTAMP_COLUMN
from src.data.validator import Severity, validate_klines

from .conftest import DEFAULT_DAY, make_klines

CANDLES_PER_DAY = 288


def validate(frame: pd.DataFrame, day: date = DEFAULT_DAY):
    return validate_klines(
        frame,
        symbol="BTCUSDT",
        market="spot",
        interval="5m",
        start_date=day,
        end_date=day,
    )


def codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


# --- Clean data -------------------------------------------------------------


def test_valid_ohlc_dataset_passes(valid_klines):
    report = validate(valid_klines)

    assert report.status == "PASS"
    assert report.passed is True
    assert report.findings == []
    assert report.expected_candles == CANDLES_PER_DAY
    assert report.actual_candles == CANDLES_PER_DAY
    assert report.missing_candles == 0
    assert report.duplicate_candles == 0
    assert report.invalid_ohlc_rows == 0
    assert report.invalid_price_rows == 0
    assert report.invalid_volume_rows == 0
    assert report.timestamp_errors == 0


def test_expected_and_actual_counts_are_both_reported(valid_klines):
    report = validate(valid_klines)

    assert report.expected_candles == report.actual_candles
    assert report.unique_candles == CANDLES_PER_DAY


def test_boundary_candles_are_checked_explicitly(valid_klines):
    report = validate(valid_klines)

    assert report.first_timestamp == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert report.last_timestamp == pd.Timestamp("2024-01-01 23:55:00", tz="UTC")
    assert report.first_timestamp == report.expected_first_timestamp
    assert report.last_timestamp == report.expected_last_timestamp
    assert report.boundary_ok is True


# --- OHLC relations ---------------------------------------------------------


def test_high_below_the_candle_body_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[3, "high"] = frame.loc[3, "close"] - 0.5

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_ohlc_rows == 1
    assert "ohlc_relations_violated" in codes(report)


def test_low_above_the_candle_body_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[3, "low"] = frame.loc[3, "open"] + 0.5

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_ohlc_rows == 1
    assert "ohlc_relations_violated" in codes(report)


def test_high_below_low_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[7, "high"] = 10.0
    frame.loc[7, "low"] = 20.0

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_ohlc_rows == 1


def test_ohlc_finding_includes_offending_rows(valid_klines):
    frame = valid_klines.copy()
    frame.loc[3, "high"] = frame.loc[3, "close"] - 0.5

    report = validate(frame)
    finding = next(f for f in report.findings if f.code == "ohlc_relations_violated")

    assert finding.severity is Severity.FAIL
    assert len(finding.samples) == 1
    assert "2024-01-01 00:15:00 UTC" in finding.samples[0]


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_nan_in_ohlc_is_detected(valid_klines, column):
    """A NaN comparison is always False, so relation checks alone would miss
    this. It must be caught explicitly, not left as "no violation found".
    """
    frame = valid_klines.copy()
    frame.loc[3, column] = float("nan")

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_ohlc_rows == 1
    assert "ohlc_relations_violated" in codes(report)


def test_non_numeric_value_in_ohlc_is_detected_not_crashed(valid_klines):
    """The normalizer guarantees numeric OHLC, but the validator must not
    crash with a TypeError if a non-numeric value ever reaches it anyway; it
    must be treated as invalid data instead.
    """
    frame = valid_klines.copy()
    frame["close"] = frame["close"].astype(object)
    frame.loc[3, "close"] = "not-a-number"

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_ohlc_rows == 1
    assert "ohlc_relations_violated" in codes(report)


# --- Prices and volume ------------------------------------------------------


def test_negative_volume_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[5, "volume"] = -1.0

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_volume_rows == 1
    assert "volumes_negative" in codes(report)


def test_zero_volume_is_accepted(valid_klines):
    """Zero volume is a legitimate quiet candle, not an error."""
    frame = valid_klines.copy()
    frame.loc[5, "volume"] = 0.0

    report = validate(frame)

    assert report.status == "PASS"
    assert report.invalid_volume_rows == 0


def test_zero_is_accepted_in_every_volume_and_count_column(valid_klines):
    frame = valid_klines.copy()
    for column in ("volume", "quote_asset_volume", "taker_buy_base_asset_volume",
                   "taker_buy_quote_asset_volume", "number_of_trades"):
        frame.loc[5, column] = 0

    report = validate(frame)

    assert report.status == "PASS"
    assert report.invalid_volume_rows == 0


def test_nan_in_volume_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[5, "volume"] = float("nan")

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_volume_rows == 1
    assert "volumes_negative" in codes(report)


@pytest.mark.parametrize(
    "column",
    [
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "number_of_trades",
    ],
)
def test_nan_in_any_volume_or_count_column_is_detected(valid_klines, column):
    frame = valid_klines.copy()
    frame.loc[5, column] = float("nan")

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_volume_rows == 1


def test_non_numeric_value_in_volume_is_detected_not_crashed(valid_klines):
    """Mirrors the OHLC case: a non-numeric volume must FAIL, not crash."""
    frame = valid_klines.copy()
    frame["volume"] = frame["volume"].astype(object)
    frame.loc[5, "volume"] = "not-a-number"

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_volume_rows == 1
    assert "volumes_negative" in codes(report)


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_non_positive_prices_are_detected(valid_klines, price):
    frame = valid_klines.copy()
    for column in ("open", "high", "low", "close"):
        frame.loc[9, column] = price

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_price_rows == 1
    assert "prices_not_positive" in codes(report)


def test_nan_price_is_detected_as_invalid_price(valid_klines):
    frame = valid_klines.copy()
    frame.loc[9, "open"] = float("nan")

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_price_rows == 1
    assert "prices_not_positive" in codes(report)


def test_non_numeric_value_in_price_is_detected_not_crashed(valid_klines):
    frame = valid_klines.copy()
    frame["open"] = frame["open"].astype(object)
    frame.loc[9, "open"] = "not-a-number"

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.invalid_price_rows == 1
    assert "prices_not_positive" in codes(report)


# --- Duplicates -------------------------------------------------------------


def test_duplicate_timestamp_is_detected_and_not_removed(valid_klines):
    frame = pd.concat(
        [valid_klines.iloc[:6], valid_klines.iloc[[5]], valid_klines.iloc[6:]],
        ignore_index=True,
    )

    report = validate(frame)

    assert report.status == "FAIL"
    assert report.duplicate_candles == 1
    assert report.actual_candles == CANDLES_PER_DAY + 1
    assert report.unique_candles == CANDLES_PER_DAY
    assert report.missing_candles == 0
    assert "timestamp_duplicated" in codes(report)


def test_duplicate_finding_shows_count_and_examples(valid_klines):
    frame = pd.concat(
        [valid_klines, valid_klines.iloc[[5]], valid_klines.iloc[[5]]],
        ignore_index=True,
    )

    report = validate(frame)
    finding = next(f for f in report.findings if f.code == "timestamp_duplicated")

    assert report.duplicate_candles == 2
    assert finding.severity is Severity.FAIL
    assert "2024-01-01 00:25:00 UTC x3" in finding.samples


# --- Missing candles (WARNING, not FAIL) ------------------------------------


def test_missing_candles_warn_but_do_not_fail(valid_klines):
    frame = valid_klines.drop(index=[3, 4]).reset_index(drop=True)

    report = validate(frame)

    assert report.status == "PASS"
    assert report.passed is True
    assert report.missing_candles == 2
    assert len(report.warnings) == 1
    assert report.failures == []
    assert codes(report) == {"candles_missing"}
    assert report.warnings[0].severity is Severity.WARNING


def test_gap_reports_surrounding_timestamps_duration_and_count(valid_klines):
    frame = valid_klines.drop(index=[3, 4]).reset_index(drop=True)

    report = validate(frame)

    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.before == pd.Timestamp("2024-01-01 00:10:00", tz="UTC")
    assert gap.after == pd.Timestamp("2024-01-01 00:25:00", tz="UTC")
    assert gap.first_missing == pd.Timestamp("2024-01-01 00:15:00", tz="UTC")
    assert gap.last_missing == pd.Timestamp("2024-01-01 00:20:00", tz="UTC")
    assert gap.missing_candles == 2
    assert gap.duration == timedelta(minutes=15)


def test_multiple_gaps_are_enumerated_separately(valid_klines):
    frame = valid_klines.drop(index=[10, 50, 51, 52]).reset_index(drop=True)

    report = validate(frame)

    assert report.status == "PASS"
    assert report.missing_candles == 4
    assert [gap.missing_candles for gap in report.gaps] == [1, 3]


def test_missing_candles_are_never_synthesized(valid_klines):
    frame = valid_klines.drop(index=[3, 4]).reset_index(drop=True)

    report = validate(frame)

    assert report.actual_candles == CANDLES_PER_DAY - 2
    assert len(frame) == CANDLES_PER_DAY - 2


# --- Timestamp integrity ----------------------------------------------------


def test_sorted_timestamps_pass_the_ordering_check(valid_klines):
    report = validate(valid_klines)

    assert "timestamp_not_sorted" not in codes(report)
    assert valid_klines[TIMESTAMP_COLUMN].is_monotonic_increasing


def test_unsorted_timestamps_are_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[5, TIMESTAMP_COLUMN], frame.loc[6, TIMESTAMP_COLUMN] = (
        valid_klines.loc[6, TIMESTAMP_COLUMN],
        valid_klines.loc[5, TIMESTAMP_COLUMN],
    )

    report = validate(frame)

    assert report.status == "FAIL"
    assert "timestamp_not_sorted" in codes(report)
    assert report.timestamp_errors >= 1
    assert report.missing_candles == 0


def test_missing_timestamp_value_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[4, TIMESTAMP_COLUMN] = pd.NaT

    report = validate(frame)

    assert report.status == "FAIL"
    assert "timestamp_missing" in codes(report)


def test_timestamp_off_the_interval_grid_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[5, TIMESTAMP_COLUMN] = pd.Timestamp("2024-01-01 00:27:00", tz="UTC")

    report = validate(frame)

    assert report.status == "FAIL"
    assert "timestamp_off_grid" in codes(report)


def test_timestamp_outside_the_configured_range_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame.loc[287, TIMESTAMP_COLUMN] = pd.Timestamp("2024-01-02 04:00:00", tz="UTC")

    report = validate(frame)

    assert report.status == "FAIL"
    assert "timestamp_off_grid" in codes(report)


# --- Timezone ---------------------------------------------------------------


def test_utc_timezone_is_accepted(valid_klines):
    report = validate(valid_klines)

    assert str(valid_klines[TIMESTAMP_COLUMN].dt.tz) == "UTC"
    assert report.status == "PASS"


def test_timezone_naive_timestamps_are_detected(valid_klines):
    frame = valid_klines.copy()
    frame[TIMESTAMP_COLUMN] = frame[TIMESTAMP_COLUMN].dt.tz_localize(None)

    report = validate(frame)

    assert report.status == "FAIL"
    assert "timestamp_not_timezone_aware" in codes(report)


def test_non_utc_timezone_is_detected(valid_klines):
    frame = valid_klines.copy()
    frame[TIMESTAMP_COLUMN] = frame[TIMESTAMP_COLUMN].dt.tz_convert("Europe/Berlin")

    report = validate(frame)

    assert report.status == "FAIL"
    assert "timestamp_timezone_not_utc" in codes(report)


# --- Boundary candles -------------------------------------------------------


def test_absent_first_expected_candle_fails(valid_klines):
    frame = valid_klines.drop(index=[0]).reset_index(drop=True)

    report = validate(frame)

    assert report.status == "FAIL"
    assert "first_candle_mismatch" in codes(report)
    assert report.boundary_ok is False
    assert report.first_timestamp == pd.Timestamp("2024-01-01 00:05:00", tz="UTC")


def test_absent_last_expected_candle_fails(valid_klines):
    frame = valid_klines.drop(index=[287]).reset_index(drop=True)

    report = validate(frame)

    assert report.status == "FAIL"
    assert "last_candle_mismatch" in codes(report)
    assert report.boundary_ok is False


def test_full_configured_range_boundaries(valid_klines):
    """The production range expects 2024-01-01 00:00 to 2025-12-31 23:55 UTC."""
    report = validate_klines(
        make_klines(),
        symbol="BTCUSDT",
        market="spot",
        interval="5m",
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
    )

    assert report.expected_candles == 731 * 288
    assert report.expected_first_timestamp == pd.Timestamp(
        "2024-01-01 00:00:00", tz="UTC"
    )
    assert report.expected_last_timestamp == pd.Timestamp(
        "2025-12-31 23:55:00", tz="UTC"
    )
    assert report.status == "FAIL"
    assert "last_candle_mismatch" in codes(report)


# --- Schema -----------------------------------------------------------------


def test_missing_schema_column_fails(valid_klines):
    frame = valid_klines.drop(columns=["quote_asset_volume"])

    report = validate(frame)

    assert report.status == "FAIL"
    assert "schema_columns_missing" in codes(report)


def test_empty_dataset_fails(valid_klines):
    report = validate(valid_klines.iloc[0:0])

    assert report.status == "FAIL"
    assert "empty_dataset" in codes(report)
    assert report.missing_candles == report.expected_candles
