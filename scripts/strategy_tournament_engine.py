"""Strategy tournament: look-ahead-free signals on the v0.4 backtest engine.

Research-only. Not imported by v0.3–v0.9 production pipelines.
Does not download data, call Binance, or write outside the caller-chosen directory.
Does not modify strategy, backtest, or indicator implementations.
Does not search or optimize parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.config import BacktestConfig
from src.backtest.engine import run_backtest
from src.backtest.ledger import STATUS_CLOSED
from src.indicators.engine import IndicatorConfig, calculate_indicators
from src.research.aggregation import monthly_table
from src.research.equity import reconstruct_equity
from src.research.metrics import ExperimentMetrics, compute_metrics
from src.research.periods import (
    align_signals_to_candles,
    assert_development_window,
    assert_trades_in_window,
    prepare_experiment_frames,
)
from src.strategy.engine import StrategyConfig, generate_signals
from src.strategy.signals import SignalType

YEAR_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
YEAR_END = pd.Timestamp("2024-12-31 23:55:00", tz="UTC")
EXPECTED_2024_5M = 105_408
EXPECTED_2024_15M = 35_136
EXPECTED_2024_1H = 8_784

DEV_START = date(2024, 1, 1)
DEV_END = date(2024, 8, 31)
VAL_START = date(2024, 9, 1)
VAL_END = date(2024, 12, 31)
DEV_START_TS = pd.Timestamp("2024-01-01", tz="UTC")
VAL_START_TS = pd.Timestamp("2024-09-01", tz="UTC")
YEAR_END_EXCL = pd.Timestamp("2025-01-01", tz="UTC")

TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "1h")
BARS_PER_15M = 3
BARS_PER_1H = 12

STARTING_CAPITAL = 20.0
RANDOM_SEED = 20240101
RANDOM_SIGNAL_P = 1.0 / 12.0

# Frozen from market audit 01_time_of_day, 12h race delta vs ALL.
# Highest P(+1R before -1R) delta: bucket 08-12 UTC (+2.945 pp).
# Not re-estimated on this tournament's development window.
G04_AUDIT_BUCKET = "08-12"
HOUR_BUCKETS: tuple[str, ...] = ("00-04", "04-08", "08-12", "12-16", "16-20", "20-24")

NEAR_BOTTOM = 0.25
NEAR_TOP = 0.75
EMA_DISCOUNT = 0.01
RET6_MOMENTUM = 0.005
RET12_MOMENTUM = 0.01
RET6_LARGE_NEGATIVE = -0.01
RSI_C01 = 35.0
RSI_C02 = 30.0
RSI_C03 = 40.0
RSI_C04 = 35.0

MIN_DEV_TRADES = 30
MIN_VAL_TRADES = 8
MAX_DD_PCT = 0.60
ONE_BIG_WIN = 0.40
ONE_MONTH_SHARE = 0.50
CATASTROPHIC_R_RATIO = 0.25
CATASTROPHIC_PF_RATIO = 0.50
PF_SCORE_CAP = 3.0

SYMBOL = "BTCUSDT"
NAN = float("nan")


class TournamentError(RuntimeError):
    """The tournament could not complete safely."""


@dataclass(frozen=True)
class FrozenThresholds:
    range12_median: float
    range12_q1: float
    range12_q3: float
    range24_median: float
    range24_q1: float
    range24_q3: float
    n_dev_valid: int


@dataclass
class WindowArtifacts:
    period_role: str
    period_start: date
    period_end: date
    metrics: ExperimentMetrics
    monthly: pd.DataFrame
    trades: pd.DataFrame
    signals: int


@dataclass
class RankedRow:
    strategy_id: str
    family: str
    timeframe: str
    definition: str
    development: WindowArtifacts
    validation: WindowArtifacts
    thresholds: FrozenThresholds | None
    group: str
    score: float | None
    gate_failures: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    beat_baseline_avg_r: bool | None = None
    beat_baseline_net: bool | None = None
    beat_random_avg_r: bool | None = None
    beat_random_net: bool | None = None
    r_ratio: float | None = None
    pf_ratio: float | None = None
    wr_ratio: float | None = None
    freq_ratio: float | None = None
    dd_ratio: float | None = None


STRATEGY_ORDER: tuple[str, ...] = (
    "A01",
    "A02",
    "A03",
    "A04",
    "B01",
    "B02",
    "B03",
    "B04",
    "C01",
    "C02",
    "C03",
    "C04",
    "D01",
    "D02",
    "D03",
    "D04",
    "E01",
    "E02",
    "E03",
    "E04",
    "F01",
    "F02",
    "F03",
    "F04",
    "G01",
    "G02",
    "G03",
    "G04",
    "BASELINE_A",
    "RANDOM_BASELINE",
)

FAMILY_BY_ID: dict[str, str] = {
    "A01": "TREND FOLLOWING",
    "A02": "TREND FOLLOWING",
    "A03": "TREND FOLLOWING",
    "A04": "TREND FOLLOWING",
    "B01": "BREAKOUT",
    "B02": "BREAKOUT",
    "B03": "BREAKOUT",
    "B04": "BREAKOUT",
    "C01": "MEAN REVERSION",
    "C02": "MEAN REVERSION",
    "C03": "MEAN REVERSION",
    "C04": "MEAN REVERSION",
    "D01": "MOMENTUM",
    "D02": "MOMENTUM",
    "D03": "MOMENTUM",
    "D04": "MOMENTUM",
    "E01": "VOLATILITY / REGIME",
    "E02": "VOLATILITY / REGIME",
    "E03": "VOLATILITY / REGIME",
    "E04": "VOLATILITY / REGIME",
    "F01": "RANGE / MARKET STRUCTURE",
    "F02": "RANGE / MARKET STRUCTURE",
    "F03": "RANGE / MARKET STRUCTURE",
    "F04": "RANGE / MARKET STRUCTURE",
    "G01": "TIME / REGIME",
    "G02": "TIME / REGIME",
    "G03": "TIME / REGIME",
    "G04": "TIME / REGIME",
    "BASELINE_A": "BASELINE",
    "RANDOM_BASELINE": "RANDOM",
}

DEFINITIONS: dict[str, str] = {
    "A01": (
        "EMA20>EMA50; prior close above EMA20; current low touches EMA20; "
        "bullish close (close>open) resumes above previous candle high."
    ),
    "A02": "EMA20>EMA50; close > previous 20-bar high; volume > volume_ma20.",
    "A03": "EMA20>EMA50; previous 6-bar return > 0; current candle closes bullish.",
    "A04": (
        "EMA20>EMA50; close crosses back above EMA20 after prior close <= EMA20; "
        "close > previous candle high."
    ),
    "B01": "20-bar close breakout; EMA20>EMA50; volume > volume_ma20.",
    "B02": "50-bar close breakout; EMA20>EMA50.",
    "B03": (
        "20-bar close breakout; previous 12-bar range below DEVELOPMENT median; "
        "EMA20>EMA50."
    ),
    "B04": (
        "20-bar close breakout; close in the upper half of the candle range; "
        "EMA20>EMA50."
    ),
    "C01": (
        "Close <= EMA20*(1-1%); RSI14 < 35; bullish reversal "
        "(prior bearish, current bullish)."
    ),
    "C02": "Close <= EMA20*(1-1%); RSI14 < 30; close > previous candle high.",
    "C03": (
        "Position in previous 24-bar range <= 0.25; RSI14 < 40; bullish candle."
    ),
    "C04": (
        "6-bar return <= -1%; RSI14 < 35; bullish reversal "
        "(prior bearish, current bullish)."
    ),
    "D01": "6-bar return > 0.5%; EMA20>EMA50; bullish candle.",
    "D02": "12-bar return > 1%; EMA20>EMA50.",
    "D03": "Three consecutive bullish candles; EMA20>EMA50.",
    "D04": "12-bar return > 0; close > previous candle high; EMA20>EMA50.",
    "E01": (
        "Previous 12-bar range < DEVELOPMENT median; EMA20>EMA50; "
        "close > previous 12-bar high."
    ),
    "E02": (
        "Previous 24-bar range < DEVELOPMENT median; EMA20>EMA50; "
        "close > previous 24-bar high."
    ),
    "E03": (
        "Current candle range > median of previous 20 candle ranges; "
        "EMA20>EMA50; bullish candle."
    ),
    "E04": (
        "Previous 12-bar range >= DEVELOPMENT Q3; EMA20>EMA50; "
        "bullish continuation (current and prior candle bullish)."
    ),
    "F01": (
        "Position in previous 24-bar range <= 0.25; bullish reversal; EMA20>EMA50."
    ),
    "F02": (
        "Position in previous 24-bar range >= 0.75; close > previous 24-bar high; "
        "EMA20>EMA50."
    ),
    "F03": "Higher low (low>prev low); higher close (close>prev close); EMA20>EMA50.",
    "F04": (
        "Close breaks previous confirmed 3-bar pivot swing high; EMA20>EMA50. "
        "Pivot at i confirmed at i+1 when high[i]>high[i-1] and high[i]>high[i+1]."
    ),
    "G01": "EMA20>EMA50; candle open hour in 00-08 UTC.",
    "G02": "EMA20>EMA50; candle open hour in 08-16 UTC.",
    "G03": "EMA20>EMA50; candle open hour in 16-24 UTC.",
    "G04": (
        "EMA20>EMA50; previous 12-bar range < DEVELOPMENT median; "
        f"market-audit 4-hour bucket {G04_AUDIT_BUCKET} UTC."
    ),
    "BASELINE_A": (
        "Existing v0.5 / v0.3 generate_signals: 20-bar close breakout, "
        "volume > 1.5*volume_ma20, EMA20>EMA50, 50<RSI14<70."
    ),
    "RANDOM_BASELINE": (
        "Existing occupancy process: independent Bernoulli(p=1/12) on each bar, "
        f"seed={RANDOM_SEED}, first sample of the investigation-11 generator; "
        "replayed through the v0.4 engine."
    ),
}


def refuse_2025(frame: pd.DataFrame, label: str) -> None:
    if "timestamp" not in frame.columns:
        raise TournamentError(f"{label} missing timestamp")
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    if ts.isna().any():
        raise TournamentError(f"{label} has NaT timestamps")
    if (ts.dt.year != 2024).any() or (ts >= YEAR_END_EXCL).any():
        raise TournamentError(f"{label} contains non-2024 timestamps; refusing")


def load_2024_enriched(enriched_dir: Path) -> pd.DataFrame:
    files = sorted(enriched_dir.glob("BTCUSDT-5m-2024-*.parquet"))
    if len(files) != 12:
        raise TournamentError(
            f"expected 12 2024 enriched parquet files in {enriched_dir}, found {len(files)}"
        )
    for path in files:
        if "2025" in path.name:
            raise TournamentError(f"refused 2025 file: {path}")
    frame = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in files],
        ignore_index=True,
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    required = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema20",
        "ema50",
        "rsi14",
        "volume_ma20",
    )
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise TournamentError(f"enriched store missing columns: {missing}")
    if frame["timestamp"].duplicated().any():
        raise TournamentError("duplicate timestamps in 2024 enriched store")
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    frame["timestamp"] = ts
    refuse_2025(frame, "5m enriched")
    if (ts < YEAR_START).any() or (ts > YEAR_END).any():
        raise TournamentError("5m enriched timestamps outside 2024 UTC bounds")
    if len(frame) != EXPECTED_2024_5M:
        raise TournamentError(
            f"unexpected 2024 5m row count {len(frame)}; expected {EXPECTED_2024_5M}"
        )
    return frame


def resample_ohlcv(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Left-labeled OHLCV resample. Incomplete buckets are dropped."""
    if interval not in {"15m", "1h"}:
        raise TournamentError(f"unsupported resample interval {interval}")
    rule = "15min" if interval == "15m" else "1h"
    expected_bars = BARS_PER_15M if interval == "15m" else BARS_PER_1H
    indexed = frame.set_index("timestamp").sort_index()
    ohlcv = indexed[["open", "high", "low", "close", "volume"]].copy()
    for column in ("open", "high", "low", "close", "volume"):
        ohlcv[column] = ohlcv[column].astype("float64")
    grouped = ohlcv.resample(rule, label="left", closed="left")
    out = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    counts = grouped["close"].count()
    out = out.loc[counts == expected_bars].copy()
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = out[column].astype("float64")
    refuse_2025(out, f"resampled {interval}")
    if out["timestamp"].duplicated().any():
        raise TournamentError(f"duplicate timestamps after {interval} resample")
    expected = EXPECTED_2024_15M if interval == "15m" else EXPECTED_2024_1H
    if len(out) != expected:
        raise TournamentError(
            f"unexpected {interval} row count {len(out)}; expected {expected}"
        )
    bad = (out["high"] + 1e-12 < out[["open", "close"]].max(axis=1)) | (
        out["low"] - 1e-12 > out[["open", "close"]].min(axis=1)
    )
    if bool(bad.any()):
        raise TournamentError(f"{interval} resample violated OHLC bounds")
    return out


def build_timeframes(
    candles_5m: pd.DataFrame, indicator_config: IndicatorConfig
) -> dict[str, pd.DataFrame]:
    frames = {"5m": candles_5m.copy(deep=True)}
    ohlcv_5m = candles_5m[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ].copy()
    for interval in ("15m", "1h"):
        resampled = resample_ohlcv(ohlcv_5m, interval)
        frames[interval] = calculate_indicators(resampled, indicator_config)
        refuse_2025(frames[interval], f"{interval} indicators")
    return frames


def _shift(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = NAN
    out[1:] = values[:-1]
    return out


def previous_n_max(values: np.ndarray, period: int) -> np.ndarray:
    return (
        pd.Series(values, dtype=np.float64)
        .shift(1)
        .rolling(period, min_periods=period)
        .max()
        .to_numpy(dtype=np.float64)
    )


def previous_n_min(values: np.ndarray, period: int) -> np.ndarray:
    return (
        pd.Series(values, dtype=np.float64)
        .shift(1)
        .rolling(period, min_periods=period)
        .min()
        .to_numpy(dtype=np.float64)
    )


def previous_confirmed_swing_high(high: np.ndarray) -> np.ndarray:
    """Most recent 3-bar pivot high confirmed without using future bars.

    Pivot at i iff high[i] > high[i-1] and high[i] > high[i+1].
    Confirmation is known at bar i+1. Forward-filled thereafter.
    """
    n = len(high)
    confirmed = np.full(n, NAN, dtype=np.float64)
    if n >= 3:
        mid = high[1:-1]
        is_pivot = (mid > high[:-2]) & (mid > high[2:])
        pivot_i = np.flatnonzero(is_pivot) + 1
        confirm_at = pivot_i + 1
        valid = confirm_at < n
        confirmed[confirm_at[valid]] = high[pivot_i[valid]]
    return pd.Series(confirmed).ffill().to_numpy(dtype=np.float64)


def build_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    volume = frame["volume"].to_numpy(dtype=np.float64)
    ema20 = frame["ema20"].to_numpy(dtype=np.float64)
    ema50 = frame["ema50"].to_numpy(dtype=np.float64)
    rsi = frame["rsi14"].to_numpy(dtype=np.float64)
    vol_ma = frame["volume_ma20"].to_numpy(dtype=np.float64)
    n = len(frame)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    hour = ts.dt.hour.to_numpy()
    bucket = np.array([HOUR_BUCKETS[int(h) // 4] for h in hour], dtype=object)

    prev_open = _shift(open_px)
    prev_high = _shift(high)
    prev_low = _shift(low)
    prev_close = _shift(close)
    prev_ema20 = _shift(ema20)

    prev_high_12 = previous_n_max(high, 12)
    prev_low_12 = previous_n_min(low, 12)
    prev_high_20 = previous_n_max(high, 20)
    prev_high_24 = previous_n_max(high, 24)
    prev_low_24 = previous_n_min(low, 24)
    prev_high_50 = previous_n_max(high, 50)

    span12 = prev_high_12 - prev_low_12
    span24 = prev_high_24 - prev_low_24
    range12 = np.where(close > 0, span12 / close, NAN)
    range24 = np.where(close > 0, span24 / close, NAN)
    pos24 = np.where(span24 > 0, (close - prev_low_24) / span24, NAN)

    ret6 = np.full(n, NAN)
    ret12 = np.full(n, NAN)
    ret6[6:] = close[6:] / close[:-6] - 1.0
    ret12[12:] = close[12:] / close[:-12] - 1.0

    bullish = close > open_px
    prev_bullish = prev_close > prev_open
    prev_bearish = prev_close < prev_open
    reversal = prev_bearish & bullish

    candle_range = high - low
    prev_range_med20 = (
        pd.Series(candle_range, dtype=np.float64)
        .shift(1)
        .rolling(20, min_periods=20)
        .median()
        .to_numpy(dtype=np.float64)
    )
    three_bull = np.zeros(n, dtype=bool)
    if n >= 3:
        three_bull[2:] = bullish[2:] & bullish[1:-1] & bullish[:-2]

    swing_high = previous_confirmed_swing_high(high)

    warmup = (
        np.isfinite(ema20)
        & np.isfinite(ema50)
        & np.isfinite(rsi)
        & np.isfinite(vol_ma)
        & np.isfinite(close)
        & (close > 0)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(open_px)
    )
    return {
        "open": open_px,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi,
        "vol_ma": vol_ma,
        "hour": hour.astype(np.int64),
        "bucket": bucket,
        "prev_open": prev_open,
        "prev_high": prev_high,
        "prev_low": prev_low,
        "prev_close": prev_close,
        "prev_ema20": prev_ema20,
        "prev_high_12": prev_high_12,
        "prev_low_12": prev_low_12,
        "prev_high_20": prev_high_20,
        "prev_high_24": prev_high_24,
        "prev_low_24": prev_low_24,
        "prev_high_50": prev_high_50,
        "range12": range12,
        "range24": range24,
        "pos24": pos24,
        "ret6": ret6,
        "ret12": ret12,
        "bullish": bullish,
        "prev_bullish": prev_bullish,
        "reversal": reversal,
        "candle_range": candle_range,
        "prev_range_med20": prev_range_med20,
        "three_bull": three_bull,
        "swing_high": swing_high,
        "warmup": warmup,
    }


def assert_feature_no_lookahead(feat: dict[str, np.ndarray], high: np.ndarray) -> None:
    """Spot-check that previous-N highs exclude the current bar."""
    n = len(high)
    if n < 60:
        return
    for index in (50, n // 2, n - 1):
        expected20 = float(np.max(high[index - 20 : index]))
        got20 = float(feat["prev_high_20"][index])
        if not np.isclose(expected20, got20, rtol=0.0, atol=1e-12):
            raise TournamentError("prev_high_20 used the current or future bar")
        expected50 = float(np.max(high[index - 50 : index]))
        got50 = float(feat["prev_high_50"][index])
        if not np.isclose(expected50, got50, rtol=0.0, atol=1e-12):
            raise TournamentError("prev_high_50 used the current or future bar")


def freeze_thresholds(feat: dict[str, np.ndarray], timestamps: pd.Series) -> FrozenThresholds:
    ts = pd.to_datetime(timestamps, utc=True)
    if (ts.dt.year != 2024).any():
        raise TournamentError("threshold frame leaked non-2024 timestamps")
    dev = (ts >= DEV_START_TS) & (ts < VAL_START_TS)
    valid = (
        dev.to_numpy()
        & feat["warmup"]
        & np.isfinite(feat["range12"])
        & np.isfinite(feat["range24"])
    )
    r12 = feat["range12"][valid]
    r24 = feat["range24"][valid]
    if r12.size < 100 or r24.size < 100:
        raise TournamentError("insufficient development rows to freeze range thresholds")
    return FrozenThresholds(
        range12_median=float(np.median(r12)),
        range12_q1=float(np.quantile(r12, 0.25)),
        range12_q3=float(np.quantile(r12, 0.75)),
        range24_median=float(np.median(r24)),
        range24_q1=float(np.quantile(r24, 0.25)),
        range24_q3=float(np.quantile(r24, 0.75)),
        n_dev_valid=int(valid.sum()),
    )


def _finite_gt(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right) & (left > right)


def _finite_ge(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right) & (left >= right)


def _finite_lt(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right) & (left < right)


def _finite_le(left: np.ndarray, right: np.ndarray | float) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right) & (left <= right)


def trend_up(feat: dict[str, np.ndarray]) -> np.ndarray:
    return feat["warmup"] & _finite_gt(feat["ema20"], feat["ema50"])


def signal_mask(
    strategy_id: str, feat: dict[str, np.ndarray], thresh: FrozenThresholds
) -> np.ndarray:
    up = trend_up(feat)
    close = feat["close"]
    ema20 = feat["ema20"]
    breakout20 = _finite_gt(close, feat["prev_high_20"])
    vol_ok = _finite_gt(feat["volume"], feat["vol_ma"])
    above_prev_high = _finite_gt(close, feat["prev_high"])
    below_ema_1pct = _finite_le(close, ema20 * (1.0 - EMA_DISCOUNT))

    if strategy_id == "A01":
        pullback = (
            _finite_gt(feat["prev_close"], feat["prev_ema20"])
            & _finite_le(feat["low"], ema20)
            & _finite_ge(close, ema20)
        )
        return up & pullback & feat["bullish"] & above_prev_high
    if strategy_id == "A02":
        return up & breakout20 & vol_ok
    if strategy_id == "A03":
        return up & _finite_gt(feat["ret6"], 0.0) & feat["bullish"]
    if strategy_id == "A04":
        recross = _finite_le(feat["prev_close"], feat["prev_ema20"]) & _finite_gt(
            close, ema20
        )
        return up & recross & above_prev_high
    if strategy_id == "B01":
        return up & breakout20 & vol_ok
    if strategy_id == "B02":
        return up & _finite_gt(close, feat["prev_high_50"])
    if strategy_id == "B03":
        tight = _finite_lt(feat["range12"], thresh.range12_median)
        return up & breakout20 & tight
    if strategy_id == "B04":
        upper_half = _finite_ge(close, (feat["high"] + feat["low"]) / 2.0)
        return up & breakout20 & upper_half
    if strategy_id == "C01":
        return (
            feat["warmup"]
            & below_ema_1pct
            & _finite_lt(feat["rsi"], RSI_C01)
            & feat["reversal"]
        )
    if strategy_id == "C02":
        return (
            feat["warmup"]
            & below_ema_1pct
            & _finite_lt(feat["rsi"], RSI_C02)
            & above_prev_high
        )
    if strategy_id == "C03":
        return (
            feat["warmup"]
            & _finite_le(feat["pos24"], NEAR_BOTTOM)
            & _finite_lt(feat["rsi"], RSI_C03)
            & feat["bullish"]
        )
    if strategy_id == "C04":
        return (
            feat["warmup"]
            & _finite_le(feat["ret6"], RET6_LARGE_NEGATIVE)
            & _finite_lt(feat["rsi"], RSI_C04)
            & feat["reversal"]
        )
    if strategy_id == "D01":
        return up & _finite_gt(feat["ret6"], RET6_MOMENTUM) & feat["bullish"]
    if strategy_id == "D02":
        return up & _finite_gt(feat["ret12"], RET12_MOMENTUM)
    if strategy_id == "D03":
        return up & feat["three_bull"]
    if strategy_id == "D04":
        return up & _finite_gt(feat["ret12"], 0.0) & above_prev_high
    if strategy_id == "E01":
        return (
            up
            & _finite_lt(feat["range12"], thresh.range12_median)
            & _finite_gt(close, feat["prev_high_12"])
        )
    if strategy_id == "E02":
        return (
            up
            & _finite_lt(feat["range24"], thresh.range24_median)
            & _finite_gt(close, feat["prev_high_24"])
        )
    if strategy_id == "E03":
        expand = _finite_gt(feat["candle_range"], feat["prev_range_med20"])
        return up & expand & feat["bullish"]
    if strategy_id == "E04":
        high_vol = _finite_ge(feat["range12"], thresh.range12_q3)
        continuation = feat["bullish"] & feat["prev_bullish"]
        return up & high_vol & continuation
    if strategy_id == "F01":
        return (
            up
            & _finite_le(feat["pos24"], NEAR_BOTTOM)
            & feat["reversal"]
        )
    if strategy_id == "F02":
        return (
            up
            & _finite_ge(feat["pos24"], NEAR_TOP)
            & _finite_gt(close, feat["prev_high_24"])
        )
    if strategy_id == "F03":
        return (
            up
            & _finite_gt(feat["low"], feat["prev_low"])
            & _finite_gt(close, feat["prev_close"])
        )
    if strategy_id == "F04":
        return up & _finite_gt(close, feat["swing_high"])
    if strategy_id == "G01":
        return up & (feat["hour"] < 8)
    if strategy_id == "G02":
        return up & (feat["hour"] >= 8) & (feat["hour"] < 16)
    if strategy_id == "G03":
        return up & (feat["hour"] >= 16)
    if strategy_id == "G04":
        return (
            up
            & _finite_lt(feat["range12"], thresh.range12_median)
            & (feat["bucket"] == G04_AUDIT_BUCKET)
        )
    raise TournamentError(f"unknown strategy {strategy_id}")


def mask_to_signals(timestamps: pd.Series, mask: np.ndarray) -> pd.DataFrame:
    if len(timestamps) != len(mask):
        raise TournamentError("signal mask length does not match timestamps")
    values = np.where(mask, SignalType.LONG_ENTRY.value, SignalType.NONE.value)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True).reset_index(drop=True),
            "signal": values,
        }
    )


def random_signal_mask(n: int, *, seed: int = RANDOM_SEED) -> np.ndarray:
    """First sample of the investigation-11 generator (unchanged occupancy process)."""
    rng = np.random.default_rng(seed)
    child = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
    return child.random(n) < RANDOM_SIGNAL_P


def build_all_signal_frames(
    frames: dict[str, pd.DataFrame],
    features: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, FrozenThresholds],
    strategy_config: StrategyConfig,
    indicator_config: IndicatorConfig,
) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for interval, candles in frames.items():
        refuse_2025(candles, f"{interval} signal candles")
        feat = features[interval]
        thresh = thresholds[interval]
        for strategy_id in STRATEGY_ORDER:
            if strategy_id == "BASELINE_A":
                signals = generate_signals(
                    candles,
                    strategy_config,
                    indicator_config,
                    interval=interval,
                    symbol=SYMBOL,
                )
            elif strategy_id == "RANDOM_BASELINE":
                mask = random_signal_mask(len(candles), seed=RANDOM_SEED)
                signals = mask_to_signals(candles["timestamp"], mask)
            else:
                mask = signal_mask(strategy_id, feat, thresh)
                signals = mask_to_signals(candles["timestamp"], mask)
            if len(signals) != len(candles):
                raise TournamentError(
                    f"{strategy_id} {interval} signals length mismatch"
                )
            left_ts = pd.to_datetime(candles["timestamp"], utc=True).reset_index(drop=True)
            right_ts = pd.to_datetime(signals["timestamp"], utc=True).reset_index(drop=True)
            if not left_ts.equals(right_ts):
                raise TournamentError(
                    f"{strategy_id} {interval} signal timestamps drifted"
                )
            out[(strategy_id, interval)] = signals
    return out


def run_window(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    backtest_config: BacktestConfig,
    period_start: date,
    period_end: date,
    period_role: str,
    interval: str,
) -> WindowArtifacts:
    refuse_2025(candles, f"{period_role} {interval} candles")
    original_len = len(candles)
    _, trade_candles = prepare_experiment_frames(
        candles, period_start, period_end, lookback_bars=0
    )
    if len(candles) != original_len:
        raise TournamentError("prepare_experiment_frames mutated candles")
    refuse_2025(trade_candles, f"{period_role} {interval} trade candles")
    assert_development_window(trade_candles)
    trade_signals = align_signals_to_candles(signals, trade_candles)
    needed = trade_candles[["timestamp", "open", "high", "low", "close"]].copy()
    result = run_backtest(
        needed,
        trade_signals[["timestamp", "signal"]],
        backtest_config,
        symbol=SYMBOL,
        interval=interval,
    )
    assert_trades_in_window(result.trades, period_start, period_end)
    for trade in result.trades:
        entry = pd.Timestamp(trade.entry_timestamp)
        if entry.tzinfo is None:
            entry = entry.tz_localize("UTC")
        if entry.year != 2024:
            raise TournamentError("trade entry leaked out of 2024")
        if trade.exit_timestamp is None:
            continue
        exit_ts = pd.Timestamp(trade.exit_timestamp)
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.tz_localize("UTC")
        if exit_ts.year != 2024:
            raise TournamentError("trade exit leaked out of 2024")
    equity = reconstruct_equity(
        trade_candles, result.trades, backtest_config.starting_capital
    )
    monthly = monthly_table(result.trades, equity)
    metrics = compute_metrics(result, equity, monthly=monthly)
    trades = result.trade_frame.copy()
    return WindowArtifacts(
        period_role=period_role,
        period_start=period_start,
        period_end=period_end,
        metrics=metrics,
        monthly=monthly,
        trades=trades,
        signals=int(result.signals_received),
    )


def _num(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return float(value)


def _ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    if right == 0:
        return None
    return float(left) / float(right)


def concentration_flags(metrics: ExperimentMetrics) -> list[str]:
    flags: list[str] = []
    win_sum = metrics.sum_of_winning_net_pnl
    largest = metrics.largest_win_net_pnl
    if (
        win_sum is not None
        and largest is not None
        and win_sum > 0
        and largest > ONE_BIG_WIN * win_sum
    ):
        flags.append("ONE_BIG_WIN")
    share = metrics.max_month_share_of_net
    if share is not None and share > ONE_MONTH_SHARE:
        flags.append("ONE_MONTH_CONCENTRATION")
    return flags


def is_catastrophic(dev: ExperimentMetrics, val: ExperimentMetrics) -> bool:
    dev_r = _num(dev.average_R_per_closed_trade)
    val_r = _num(val.average_R_per_closed_trade)
    if (
        dev.closed_trades >= MIN_DEV_TRADES
        and dev_r is not None
        and dev_r > 0
        and _num(dev.total_net_pnl) is not None
        and float(dev.total_net_pnl) > 0
    ):
        if val_r is not None and val_r < 0 and dev_r >= 0.10:
            return True
        ratio = _ratio(val_r, dev_r)
        if ratio is not None and ratio < CATASTROPHIC_R_RATIO:
            return True
        dev_pf = _num(dev.profit_factor)
        val_pf = _num(val.profit_factor)
        if (
            dev_pf is not None
            and val_pf is not None
            and dev_pf > 1.0
            and (val_pf / dev_pf) < CATASTROPHIC_PF_RATIO
        ):
            return True
    return False


def robustness_score(val: ExperimentMetrics) -> float | None:
    if val.closed_trades < 1:
        return None
    r_term = _num(val.average_R_per_closed_trade) or 0.0
    pnl_term = float(val.total_net_pnl) / STARTING_CAPITAL
    pf = _num(val.profit_factor)
    if pf is None:
        pf_term = 1.0 if val.loss_count == 0 and val.win_count > 0 else 0.0
    else:
        pf_term = min(pf, PF_SCORE_CAP) / PF_SCORE_CAP
    dd = _num(val.max_drawdown_pct)
    dd_term = 1.0 - min(dd if dd is not None else 1.0, 1.0)
    return 0.4 * r_term + 0.3 * pnl_term + 0.2 * pf_term + 0.1 * dd_term


def evaluate_pair(
    strategy_id: str,
    timeframe: str,
    definition: str,
    development: WindowArtifacts,
    validation: WindowArtifacts,
    thresholds: FrozenThresholds | None,
) -> RankedRow:
    family = FAMILY_BY_ID[strategy_id]
    dev = development.metrics
    val = validation.metrics
    failures: list[str] = []
    flags: list[str] = []
    notes: list[str] = []

    if dev.closed_trades < MIN_DEV_TRADES:
        failures.append(f"dev_closed_trades<{MIN_DEV_TRADES}")
    if val.closed_trades < MIN_VAL_TRADES:
        failures.append(f"val_closed_trades<{MIN_VAL_TRADES}")
    dev_r = _num(dev.average_R_per_closed_trade)
    val_r = _num(val.average_R_per_closed_trade)
    if dev_r is None or dev_r <= 0:
        failures.append("dev_avg_R<=0")
    if _num(dev.total_net_pnl) is None or float(dev.total_net_pnl) <= 0:
        failures.append("dev_net_pnl<=0")
    dev_dd = _num(dev.max_drawdown_pct)
    val_dd = _num(val.max_drawdown_pct)
    if dev_dd is not None and dev_dd > MAX_DD_PCT:
        failures.append("dev_max_drawdown>60%")
    if val_dd is not None and val_dd > MAX_DD_PCT:
        failures.append("val_max_drawdown>60%")
    if val_r is None or val_r <= 0:
        failures.append("val_avg_R<=0")
    if _num(val.total_net_pnl) is None or float(val.total_net_pnl) <= 0:
        failures.append("val_net_pnl<=0")
    if is_catastrophic(dev, val):
        failures.append("validation_catastrophically_worse")
        flags.append("OVERFIT / UNSTABLE")

    flags.extend(f"DEV_{item}" for item in concentration_flags(dev))
    flags.extend(f"VAL_{item}" for item in concentration_flags(val))

    r_ratio = _ratio(val_r, dev_r)
    pf_ratio = _ratio(_num(val.profit_factor), _num(dev.profit_factor))
    wr_ratio = _ratio(_num(val.win_rate), _num(dev.win_rate))
    freq_ratio = _ratio(float(val.closed_trades), float(dev.closed_trades) or None)
    dd_ratio = _ratio(val_dd, dev_dd)

    insufficient = any(
        item.startswith("dev_closed_trades") or item.startswith("val_closed_trades")
        for item in failures
    )
    if insufficient:
        group = "GROUP 4"
    elif failures:
        group = "GROUP 3"
    else:
        group = "SURVIVED"

    if group == "SURVIVED" and "OVERFIT / UNSTABLE" in flags:
        group = "GROUP 3"
        notes.append("survived numeric gates but labeled OVERFIT / UNSTABLE")

    score = robustness_score(val) if group in {"SURVIVED", "GROUP 1", "GROUP 2"} else None
    if group == "SURVIVED":
        score = robustness_score(val)

    return RankedRow(
        strategy_id=strategy_id,
        family=family,
        timeframe=timeframe,
        definition=definition,
        development=development,
        validation=validation,
        thresholds=thresholds,
        group=group,
        score=score,
        gate_failures=failures,
        flags=flags,
        notes=notes,
        r_ratio=r_ratio,
        pf_ratio=pf_ratio,
        wr_ratio=wr_ratio,
        freq_ratio=freq_ratio,
        dd_ratio=dd_ratio,
    )


def assign_groups(rows: list[RankedRow]) -> None:
    random_r = {
        row.timeframe: _num(row.validation.metrics.average_R_per_closed_trade)
        for row in rows
        if row.strategy_id == "RANDOM_BASELINE"
    }
    random_net = {
        row.timeframe: _num(row.validation.metrics.total_net_pnl)
        for row in rows
        if row.strategy_id == "RANDOM_BASELINE"
    }
    base_r = {
        row.timeframe: _num(row.validation.metrics.average_R_per_closed_trade)
        for row in rows
        if row.strategy_id == "BASELINE_A"
    }
    base_net = {
        row.timeframe: _num(row.validation.metrics.total_net_pnl)
        for row in rows
        if row.strategy_id == "BASELINE_A"
    }
    for row in rows:
        val_r = _num(row.validation.metrics.average_R_per_closed_trade)
        val_net = _num(row.validation.metrics.total_net_pnl)
        br = base_r.get(row.timeframe)
        bn = base_net.get(row.timeframe)
        rr = random_r.get(row.timeframe)
        rn = random_net.get(row.timeframe)
        row.beat_baseline_avg_r = (
            val_r is not None and br is not None and val_r > br
        )
        row.beat_baseline_net = (
            val_net is not None and bn is not None and val_net > bn
        )
        row.beat_random_avg_r = val_r is not None and rr is not None and val_r > rr
        row.beat_random_net = val_net is not None and rn is not None and val_net > rn
        if row.group != "SURVIVED":
            continue
        concentrated = (
            "VAL_ONE_BIG_WIN" in row.flags and "VAL_ONE_MONTH_CONCENTRATION" in row.flags
        )
        if concentrated:
            row.group = "GROUP 2"
            row.notes.append("survived gates but validation concentration on win and month")
            continue
        if row.beat_random_avg_r:
            row.group = "GROUP 1"
        else:
            row.group = "GROUP 2"
            row.notes.append("survived gates but did not beat RANDOM_BASELINE on validation avg R")


def near_duplicate(left: RankedRow, right: RankedRow) -> bool:
    if left.strategy_id == right.strategy_id:
        return True
    if left.family == right.family and left.timeframe == right.timeframe:
        return True
    same_trades = (
        left.development.metrics.closed_trades == right.development.metrics.closed_trades
        and left.validation.metrics.closed_trades == right.validation.metrics.closed_trades
    )
    same_pnl = np.isclose(
        float(left.validation.metrics.total_net_pnl),
        float(right.validation.metrics.total_net_pnl),
        rtol=0.0,
        atol=1e-9,
    )
    return bool(same_trades and same_pnl and left.timeframe == right.timeframe)


def select_oos_candidates(rows: list[RankedRow], *, limit: int = 5) -> list[RankedRow]:
    pool = [
        row
        for row in rows
        if row.strategy_id not in {"BASELINE_A", "RANDOM_BASELINE"}
        and row.group in {"GROUP 1", "GROUP 2"}
        and "OVERFIT / UNSTABLE" not in row.flags
        and _num(row.development.metrics.total_net_pnl) is not None
        and float(row.development.metrics.total_net_pnl) > 0
        and _num(row.validation.metrics.total_net_pnl) is not None
        and float(row.validation.metrics.total_net_pnl) > 0
    ]
    pool.sort(
        key=lambda row: (
            0 if row.group == "GROUP 1" else 1,
            -(row.score if row.score is not None else -999.0),
        )
    )
    picked: list[RankedRow] = []
    used_families: set[str] = set()
    for row in pool:
        if len(picked) >= limit:
            break
        if row.family in used_families:
            continue
        if any(near_duplicate(row, item) for item in picked):
            continue
        if "VAL_ONE_BIG_WIN" in row.flags and "VAL_ONE_MONTH_CONCENTRATION" in row.flags:
            continue
        picked.append(row)
        used_families.add(row.family)
    if len(picked) < limit:
        for row in pool:
            if len(picked) >= limit:
                break
            if any(item.strategy_id == row.strategy_id for item in picked):
                continue
            if any(near_duplicate(row, item) for item in picked):
                continue
            picked.append(row)
    return picked[:limit]


def metrics_to_rows(
    metrics: ExperimentMetrics, *, timeframe: str, period_role: str
) -> list[dict[str, object]]:
    rows = []
    for key, value in metrics.as_pairs():
        rows.append(
            {
                "timeframe": timeframe,
                "period_role": period_role,
                "metric": key,
                "value": value,
            }
        )
    return rows


def monthly_consistency_label(monthly: pd.DataFrame) -> str:
    if monthly.empty:
        return "no months"
    traded = monthly.loc[monthly["trades_closed"] > 0]
    if traded.empty:
        return "no months with trades"
    n = len(traded)
    positive = int((traded["net_pnl"] > 0).sum())
    return f"{positive}/{n} months with trades were net-positive"
