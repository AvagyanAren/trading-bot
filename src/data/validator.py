"""Validates a normalized kline dataset. Reports problems; never repairs them.

Severity model
--------------
Missing candles are a ``WARNING``. Genuine Binance exchange downtime leaves
real holes in the archive, so a gap is something the pipeline must surface, not
something that invalidates the dataset.

Everything else is a ``FAIL``: duplicate timestamps, invalid OHLC relations
(including a missing, NaN or non-numeric open/high/low/close), non-positive,
missing or non-numeric prices, negative, missing or non-numeric volumes,
timestamps off the interval grid or outside the configured range, unsorted
timestamps, a missing or non-UTC timezone, and an absent first or last expected
candle. A row where any OHLC or volume value is unusable can never be silently
treated as valid: it is coerced to a comparison that is guaranteed to fail
rather than one that is skipped.

Overall status is ``PASS`` when no ``FAIL`` finding exists, even if warnings
were raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from .normalizer import (
    COUNT_COLUMNS,
    PRICE_COLUMNS,
    PROCESSED_COLUMNS,
    TIMESTAMP_COLUMN,
    VOLUME_COLUMNS,
    expected_timestamp_index,
    interval_to_timedelta,
)

MAX_SAMPLES = 5


class Severity(str, Enum):
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass(frozen=True)
class Finding:
    """A single validation problem."""

    severity: Severity
    code: str
    message: str
    samples: tuple[str, ...] = ()


@dataclass(frozen=True)
class Gap:
    """A run of consecutive expected candles that the dataset does not contain."""

    before: pd.Timestamp | None
    after: pd.Timestamp | None
    first_missing: pd.Timestamp
    last_missing: pd.Timestamp
    missing_candles: int
    duration: timedelta


@dataclass
class ValidationReport:
    """Everything the pipeline needs to render a verdict."""

    symbol: str
    market: str
    interval: str
    start_date: date
    end_date: date

    expected_candles: int = 0
    actual_candles: int = 0
    unique_candles: int = 0
    missing_candles: int = 0
    duplicate_candles: int = 0
    invalid_ohlc_rows: int = 0
    invalid_price_rows: int = 0
    invalid_volume_rows: int = 0
    timestamp_errors: int = 0

    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    expected_first_timestamp: pd.Timestamp | None = None
    expected_last_timestamp: pd.Timestamp | None = None

    gaps: list[Gap] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        samples: tuple[str, ...] = (),
    ) -> None:
        self.findings.append(Finding(severity, code, message, samples))

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.FAIL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    @property
    def boundary_ok(self) -> bool:
        return (
            self.first_timestamp is not None
            and self.last_timestamp is not None
            and self.first_timestamp == self.expected_first_timestamp
            and self.last_timestamp == self.expected_last_timestamp
        )


def validate_klines(
    frame: pd.DataFrame,
    symbol: str,
    market: str,
    interval: str,
    start_date: date,
    end_date: date,
) -> ValidationReport:
    """Check a normalized kline dataset against the configured period."""
    report = ValidationReport(
        symbol=symbol,
        market=market,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        actual_candles=len(frame),
    )

    expected_index = expected_timestamp_index(start_date, end_date, interval)
    report.expected_candles = len(expected_index)
    report.expected_first_timestamp = expected_index[0]
    report.expected_last_timestamp = expected_index[-1]

    if not _has_required_columns(frame, report):
        report.missing_candles = report.expected_candles
        return report

    if frame.empty:
        report.add(
            Severity.FAIL, "empty_dataset", "Dataset contains no candles at all"
        )
        report.missing_candles = report.expected_candles
        return report

    timestamps = frame[TIMESTAMP_COLUMN]

    timezone_ok = _check_timezone(timestamps, report)
    _check_missing_values(timestamps, report)
    _check_duplicates(timestamps, report)
    _check_ordering(timestamps, report)

    valid = timestamps.dropna()
    if valid.empty:
        report.missing_candles = report.expected_candles
        return report

    report.first_timestamp = valid.min()
    report.last_timestamp = valid.max()
    report.unique_candles = int(valid.nunique())

    if timezone_ok:
        _check_grid_alignment(valid, expected_index, interval, report)
        _check_gaps(valid, expected_index, interval, report)
        _check_boundaries(report)
    else:
        # Grid alignment, gaps and boundaries are all comparisons against a UTC
        # index. Without a valid UTC timezone those comparisons would be
        # meaningless, so they are skipped rather than guessed at.
        report.add(
            Severity.FAIL,
            "grid_checks_skipped",
            "Gap, grid-alignment and boundary checks were skipped because the "
            "timestamp column is not timezone-aware UTC",
        )

    _check_ohlc(frame, report)
    _check_prices(frame, report)
    _check_volumes(frame, report)

    return report


# --- Column and dtype checks -----------------------------------------------


def _has_required_columns(frame: pd.DataFrame, report: ValidationReport) -> bool:
    missing = [column for column in PROCESSED_COLUMNS if column not in frame.columns]
    if missing:
        report.add(
            Severity.FAIL,
            "schema_columns_missing",
            f"Dataset is missing {len(missing)} required column(s)",
            tuple(missing),
        )
        return False
    return True


# --- Timestamp checks -------------------------------------------------------


def _check_timezone(timestamps: pd.Series, report: ValidationReport) -> bool:
    """Confirm the timestamp column is timezone-aware UTC."""
    tz = getattr(timestamps.dtype, "tz", None)
    if tz is None:
        report.timestamp_errors += 1
        report.add(
            Severity.FAIL,
            "timestamp_not_timezone_aware",
            f"Timestamp column must be timezone-aware UTC, got dtype "
            f"{timestamps.dtype}",
        )
        return False
    if str(tz) != "UTC":
        report.timestamp_errors += 1
        report.add(
            Severity.FAIL,
            "timestamp_timezone_not_utc",
            f"Timestamp timezone must be UTC, got {tz}",
        )
        return False
    return True


def _check_missing_values(timestamps: pd.Series, report: ValidationReport) -> None:
    nat_count = int(timestamps.isna().sum())
    if nat_count:
        report.timestamp_errors += nat_count
        report.add(
            Severity.FAIL,
            "timestamp_missing",
            f"{nat_count} row(s) have no timestamp",
            tuple(str(i) for i in timestamps.index[timestamps.isna()][:MAX_SAMPLES]),
        )


def _check_duplicates(timestamps: pd.Series, report: ValidationReport) -> None:
    duplicated = timestamps.duplicated(keep="first")
    report.duplicate_candles = int(duplicated.sum())
    if report.duplicate_candles == 0:
        return

    offenders = timestamps[timestamps.duplicated(keep=False)]
    counts = offenders.value_counts().sort_index()
    samples = tuple(
        f"{_fmt(timestamp)} x{count}"
        for timestamp, count in counts.head(MAX_SAMPLES).items()
    )
    report.add(
        Severity.FAIL,
        "timestamp_duplicated",
        f"{report.duplicate_candles} duplicate row(s) across "
        f"{len(counts)} repeated timestamp(s). Duplicates are reported, not removed.",
        samples,
    )


def _check_ordering(timestamps: pd.Series, report: ValidationReport) -> None:
    if timestamps.is_monotonic_increasing:
        return
    values = timestamps.to_numpy()
    regressions = np.flatnonzero(values[1:] < values[:-1])
    report.timestamp_errors += 1
    report.add(
        Severity.FAIL,
        "timestamp_not_sorted",
        f"Timestamps are not sorted ascending; {len(regressions)} backward step(s)",
        tuple(
            f"{_fmt(values[i])} -> {_fmt(values[i + 1])}"
            for i in regressions[:MAX_SAMPLES]
        ),
    )


def _check_grid_alignment(
    timestamps: pd.Series,
    expected_index: pd.DatetimeIndex,
    interval: str,
    report: ValidationReport,
) -> None:
    """Flag timestamps outside the configured range or off the interval grid."""
    off_grid = ~timestamps.isin(expected_index)
    count = int(off_grid.sum())
    if count == 0:
        return

    report.timestamp_errors += count
    offenders = timestamps[off_grid]
    out_of_range = offenders[
        (offenders < expected_index[0]) | (offenders > expected_index[-1])
    ]
    report.add(
        Severity.FAIL,
        "timestamp_off_grid",
        f"{count} timestamp(s) are not valid {interval} candles for "
        f"{report.start_date}..{report.end_date} "
        f"({len(out_of_range)} outside the range, "
        f"{count - len(out_of_range)} misaligned to the grid)",
        tuple(_fmt(t) for t in offenders.head(MAX_SAMPLES)),
    )


def _check_boundaries(report: ValidationReport) -> None:
    """The exact first and last expected candles must be present."""
    if report.first_timestamp != report.expected_first_timestamp:
        report.add(
            Severity.FAIL,
            "first_candle_mismatch",
            f"First candle should be {_fmt(report.expected_first_timestamp)} "
            f"but dataset starts at {_fmt(report.first_timestamp)}",
        )
    if report.last_timestamp != report.expected_last_timestamp:
        report.add(
            Severity.FAIL,
            "last_candle_mismatch",
            f"Last candle should be {_fmt(report.expected_last_timestamp)} "
            f"but dataset ends at {_fmt(report.last_timestamp)}",
        )


def _check_gaps(
    timestamps: pd.Series,
    expected_index: pd.DatetimeIndex,
    interval: str,
    report: ValidationReport,
) -> None:
    """Enumerate every run of expected candles the dataset does not contain."""
    step = interval_to_timedelta(interval)
    present = expected_index.isin(pd.Index(timestamps.unique()))
    missing_positions = np.flatnonzero(~present)

    report.missing_candles = int(missing_positions.size)
    if report.missing_candles == 0:
        return

    breaks = np.flatnonzero(np.diff(missing_positions) != 1) + 1
    for run in np.split(missing_positions, breaks):
        start, end = int(run[0]), int(run[-1])
        before = expected_index[start - 1] if start > 0 else None
        after = expected_index[end + 1] if end + 1 < len(expected_index) else None
        count = end - start + 1
        duration = (after - before) if (before is not None and after is not None) else step * count
        report.gaps.append(
            Gap(
                before=before,
                after=after,
                first_missing=expected_index[start],
                last_missing=expected_index[end],
                missing_candles=count,
                duration=pd.Timedelta(duration).to_pytimedelta(),
            )
        )

    report.add(
        Severity.WARNING,
        "candles_missing",
        f"{report.missing_candles} expected candle(s) absent across "
        f"{len(report.gaps)} gap(s). Reported only; no candle is synthesized.",
        tuple(
            f"{_fmt(gap.before)} -> {_fmt(gap.after)} "
            f"({gap.missing_candles} missing, {gap.duration})"
            for gap in report.gaps[:MAX_SAMPLES]
        ),
    )


# --- Value checks -----------------------------------------------------------


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Read a column as numeric, coercing anything unparseable to NaN.

    The normalizer guarantees ``open``/``high``/``low``/``close``/volume
    columns are already numeric before the validator ever sees them, so this
    is a defensive second line, not a substitute for that guarantee. If a
    non-numeric value ever did reach here (for example a validator called
    directly on hand-built or corrupted data), a plain comparison such as
    ``frame[column] > 0`` would raise ``TypeError`` on an ``object`` dtype
    column. Coercing first turns that unusable value into NaN instead, which
    every check below already treats as invalid data, not as a crash.
    """
    return pd.to_numeric(frame[column], errors="coerce")


def _check_ohlc(frame: pd.DataFrame, report: ValidationReport) -> None:
    ohlc = pd.DataFrame(
        {column: _numeric(frame, column) for column in PRICE_COLUMNS}
    )

    # A comparison against NaN is always False in pandas/NumPy, so a row with a
    # missing or non-numeric open/high/low/close would otherwise slip through
    # every relational check below undetected. Flagging it explicitly closes
    # that gap: unusable OHLC data is invalid OHLC data, not "no violation
    # found".
    unusable = ohlc.isna().any(axis=1)

    body_high = ohlc[["open", "close"]].max(axis=1)
    body_low = ohlc[["open", "close"]].min(axis=1)

    broken = (
        unusable
        | (ohlc["high"] < body_high)
        | (ohlc["low"] > body_low)
        | (ohlc["high"] < ohlc["low"])
    )
    report.invalid_ohlc_rows = int(broken.sum())
    if report.invalid_ohlc_rows == 0:
        return

    report.add(
        Severity.FAIL,
        "ohlc_relations_violated",
        f"{report.invalid_ohlc_rows} row(s) violate high >= max(open, close), "
        "low <= min(open, close), high >= low, or contain a missing/non-numeric "
        "OHLC value",
        _row_samples(frame, broken),
    )


def _check_prices(frame: pd.DataFrame, report: ValidationReport) -> None:
    non_positive = pd.Series(False, index=frame.index)
    for column in PRICE_COLUMNS:
        values = _numeric(frame, column)
        # NaN > 0 is False, so ~(values > 0) is True for NaN: missing or
        # non-numeric prices are already caught by this negation, not just
        # zero or negative ones.
        non_positive |= ~(values > 0)

    report.invalid_price_rows = int(non_positive.sum())
    if report.invalid_price_rows == 0:
        return

    report.add(
        Severity.FAIL,
        "prices_not_positive",
        f"{report.invalid_price_rows} row(s) have a non-positive, missing or "
        f"non-numeric value in {', '.join(PRICE_COLUMNS)}",
        _row_samples(frame, non_positive),
    )


def _check_volumes(frame: pd.DataFrame, report: ValidationReport) -> None:
    negative = pd.Series(False, index=frame.index)
    for column in VOLUME_COLUMNS + COUNT_COLUMNS:
        values = _numeric(frame, column)
        # Same NaN-safe negation as prices: zero is >= 0 and stays valid, but
        # a missing or non-numeric count/volume is caught, not silently passed.
        negative |= ~(values >= 0)

    report.invalid_volume_rows = int(negative.sum())
    if report.invalid_volume_rows == 0:
        return

    report.add(
        Severity.FAIL,
        "volumes_negative",
        f"{report.invalid_volume_rows} row(s) have a negative, missing or "
        f"non-numeric value in {', '.join(VOLUME_COLUMNS + COUNT_COLUMNS)}",
        _row_samples(frame, negative),
    )


# --- Formatting helpers -----------------------------------------------------


def _row_samples(frame: pd.DataFrame, mask: pd.Series) -> tuple[str, ...]:
    offenders = frame.loc[mask].head(MAX_SAMPLES)
    return tuple(
        f"{_fmt(row[TIMESTAMP_COLUMN])} "
        f"O={row['open']} H={row['high']} L={row['low']} C={row['close']} "
        f"V={row['volume']}"
        for _, row in offenders.iterrows()
    )


def _fmt(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None or pd.isna(timestamp):
        return "n/a"
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
