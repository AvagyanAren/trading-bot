"""Look-ahead-free path metrics and random-baseline simulator for the 2024 market audit.

Research-only. Not imported by v0.3–v0.5 production pipelines.
Does not download data, call Binance, or write outside the caller-chosen directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

YEAR_END = pd.Timestamp("2024-12-31 23:55:00", tz="UTC")
YEAR_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
EXPECTED_2024_ROWS = 105_408
BAR_MINUTES = 5
THIN_EVERY = 144  # 12 hours
R = 0.01
COST_R = 0.30

HORIZON_BARS: dict[str, int] = {
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "2h": 24,
    "4h": 48,
    "8h": 96,
    "12h": 144,
    "24h": 288,
}
COMMON_HORIZONS: tuple[str, ...] = ("15m", "30m", "1h", "2h", "4h", "12h")
HOLDING_HORIZONS: tuple[str, ...] = (
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "8h",
    "12h",
    "24h",
)
HOUR_BUCKETS: tuple[str, ...] = ("00-04", "04-08", "08-12", "12-16", "16-20", "20-24")
WEEKDAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
QUANTILES: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")

STARTING_CAPITAL = 20.0
RISK_PER_TRADE = 0.01
STOP_LOSS = 0.01
TAKE_PROFIT = 0.02
SLIPPAGE = 0.0005
ENTRY_RATE = 0.001
EXIT_RATE = 0.001
RANDOM_SEED = 20240101
RANDOM_SAMPLES = 100
RANDOM_SIGNAL_P = 1.0 / 12.0  # fixed; not calibrated to strategy A

NAN = float("nan")


def _finite_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return NAN
    return float(np.mean(x))


def _finite_median(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return NAN
    return float(np.median(x))


def _finite_quantile(values: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return NAN
    return float(np.quantile(x, q))


def qcut4(values: np.ndarray) -> np.ndarray:
    """Equal-count quartiles via rank; Q1 is the lowest values."""
    out = np.array(["NA"] * len(values), dtype=object)
    finite = np.isfinite(values)
    if int(finite.sum()) < 4:
        return out
    ranks = pd.Series(values).rank(method="first")
    labeled = pd.qcut(ranks[finite], 4, labels=list(QUANTILES))
    out[finite] = np.asarray(labeled.astype(str))
    return out


def cluster_stats(mask: np.ndarray) -> dict[str, float | int]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    n = int(idx.size)
    if n <= 1:
        return {
            "n_events": n,
            "median_gap_bars": NAN,
            "median_gap_minutes": NAN,
            "pct_within_1h": NAN,
            "pct_within_3h": NAN,
            "pct_within_12h": NAN,
        }
    gaps = np.diff(idx).astype(np.float64)
    return {
        "n_events": n,
        "median_gap_bars": float(np.median(gaps)),
        "median_gap_minutes": float(np.median(gaps) * BAR_MINUTES),
        "pct_within_1h": float(np.mean(gaps <= 12) * 100.0),
        "pct_within_3h": float(np.mean(gaps <= 36) * 100.0),
        "pct_within_12h": float(np.mean(gaps <= 144) * 100.0),
    }


def thin_regular(n: int, every: int = THIN_EVERY) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[::every] = True
    return mask


def month_consistency(
    months: np.ndarray,
    regime_mask: np.ndarray,
    baseline_mask: np.ndarray,
    race_plus: np.ndarray,
    valid: np.ndarray,
    median_fwd: np.ndarray,
) -> dict[str, float | int | str]:
    """Count months where regime race and median forward beat that month's baseline."""
    unique = [str(m) for m in pd.unique(months) if str(m) != "nan"]
    unique.sort()
    race_wins = 0
    fwd_wins = 0
    usable = 0
    notes: list[str] = []
    for month in unique:
        in_month = months == month
        reg = regime_mask & in_month & valid
        base = baseline_mask & in_month & valid
        n_reg = int(reg.sum())
        n_base = int(base.sum())
        if n_reg < 30 or n_base < 30:
            notes.append(f"{month}:small")
            continue
        usable += 1
        p_reg = float(np.mean(race_plus[reg]))
        p_base = float(np.mean(race_plus[base]))
        med_reg = _finite_median(median_fwd[reg])
        med_base = _finite_median(median_fwd[base])
        if p_reg > p_base:
            race_wins += 1
        if med_reg > med_base:
            fwd_wins += 1
    if usable == 0:
        flag = "insufficient monthly sample"
    elif race_wins <= 2:
        flag = "unstable (race edge concentrated in <=2 months)"
    elif race_wins < max(8, int(np.ceil(0.67 * usable))):
        flag = f"mixed ({race_wins}/{usable} months race>baseline)"
    else:
        flag = f"broad ({race_wins}/{usable} months race>baseline)"
    return {
        "months_usable": usable,
        "months_race_above_baseline": race_wins,
        "months_fwd_above_baseline": fwd_wins,
        "consistency_flag": flag,
        "small_month_notes": ";".join(notes),
    }


@dataclass
class PathBook:
    valid: dict[str, np.ndarray]
    fwd: dict[str, np.ndarray]
    mfe: dict[str, np.ndarray]
    mae: dict[str, np.ndarray]
    fut_vol: dict[str, np.ndarray]
    hit_05: dict[str, np.ndarray]
    hit_10: dict[str, np.ndarray]
    hit_20: dict[str, np.ndarray]
    race_plus: dict[str, np.ndarray]
    race_minus: dict[str, np.ndarray]
    race_neither: dict[str, np.ndarray]
    t05: dict[str, np.ndarray]
    t10: dict[str, np.ndarray]
    t20: dict[str, np.ndarray]


@dataclass(frozen=True)
class Agg:
    n: int
    mean_fwd: float
    median_fwd: float
    mfe: float
    mae: float
    fut_vol: float
    p05: float
    p10: float
    p20: float
    p_plus: float
    p_minus: float
    p_neither: float
    t05: float
    t10: float
    t20: float


def load_2024_enriched(enriched_dir: Path) -> pd.DataFrame:
    files = sorted(enriched_dir.glob("BTCUSDT-5m-2024-*.parquet"))
    if len(files) != 12:
        raise SystemExit(
            f"expected 12 2024 enriched parquet files in {enriched_dir}, found {len(files)}"
        )
    for path in files:
        if "2025" in path.name:
            raise SystemExit(f"refused 2025 file: {path}")
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
        raise SystemExit(f"enriched store missing columns: {missing}")
    if frame["timestamp"].duplicated().any():
        raise SystemExit("duplicate timestamps in 2024 enriched store")
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    frame["timestamp"] = ts
    if (ts < YEAR_START).any() or (ts > YEAR_END).any() or (ts.dt.year != 2024).any():
        raise SystemExit("non-2024 rows present; refusing to continue")
    if len(frame) != EXPECTED_2024_ROWS:
        raise SystemExit(
            f"unexpected 2024 row count {len(frame)}; expected {EXPECTED_2024_ROWS}"
        )
    return frame


def build_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    volume = frame["volume"].to_numpy(dtype=np.float64)
    ema20 = frame["ema20"].to_numpy(dtype=np.float64)
    ema50 = frame["ema50"].to_numpy(dtype=np.float64)
    vol_ma = frame["volume_ma20"].to_numpy(dtype=np.float64)
    n = len(frame)
    ts = frame["timestamp"]

    hour = ts.dt.hour.to_numpy()
    bucket = np.array([HOUR_BUCKETS[int(h) // 4] for h in hour], dtype=object)
    dow = np.array([WEEKDAYS[int(d)] for d in ts.dt.dayofweek.to_numpy()], dtype=object)
    month = ts.dt.strftime("%Y-%m").to_numpy()

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    prev_high_12 = high_s.shift(1).rolling(12, min_periods=12).max().to_numpy()
    prev_low_12 = low_s.shift(1).rolling(12, min_periods=12).min().to_numpy()
    prev_high_24 = high_s.shift(1).rolling(24, min_periods=24).max().to_numpy()
    prev_low_24 = low_s.shift(1).rolling(24, min_periods=24).min().to_numpy()
    range12 = (prev_high_12 - prev_low_12) / close
    range24 = (prev_high_24 - prev_low_24) / close
    span12 = prev_high_12 - prev_low_12
    span24 = prev_high_24 - prev_low_24
    pos12 = np.where(span12 > 0, (close - prev_low_12) / span12, NAN)
    pos24 = np.where(span24 > 0, (close - prev_low_24) / span24, NAN)

    ret = {k: np.full(n, NAN) for k in (3, 6, 12, 24)}
    for k in ret:
        ret[k][k:] = close[k:] / close[:-k] - 1.0

    abs_ret = np.empty(n, dtype=np.float64)
    abs_ret[0] = NAN
    abs_ret[1:] = np.abs(close[1:] / close[:-1] - 1.0)
    realized12 = pd.Series(abs_ret).shift(1).rolling(12, min_periods=12).sum().to_numpy()
    realized24 = pd.Series(abs_ret).shift(1).rolling(24, min_periods=24).sum().to_numpy()

    vol_ratio = np.where((vol_ma > 0) & np.isfinite(vol_ma), volume / vol_ma, NAN)
    spread = np.where(close > 0, (ema20 - ema50) / close, NAN)
    candle_ret = np.full(n, NAN)
    candle_ret[1:] = close[1:] / close[:-1] - 1.0

    warmup = (
        np.isfinite(ema20)
        & np.isfinite(ema50)
        & np.isfinite(vol_ma)
        & np.isfinite(range24)
        & np.isfinite(ret[24])
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
        "vol_ma": vol_ma,
        "hour_bucket": bucket,
        "dow": dow,
        "month": month,
        "range12": range12,
        "range24": range24,
        "pos12": pos12,
        "pos24": pos24,
        "ret3": ret[3],
        "ret6": ret[6],
        "ret12": ret[12],
        "ret24": ret[24],
        "realized12": realized12,
        "realized24": realized24,
        "vol_ratio": vol_ratio,
        "ema_spread": spread,
        "candle_ret": candle_ret,
        "bull": (ema20 > ema50) & warmup,
        "bear": (ema20 <= ema50) & warmup,
        "warmup_ok": warmup,
        "q_spread": qcut4(np.where(warmup, spread, NAN)),
        "q_range12": qcut4(np.where(warmup, range12, NAN)),
        "q_range24": qcut4(np.where(warmup, range24, NAN)),
        "q_ret3": qcut4(np.where(warmup, ret[3], NAN)),
        "q_ret6": qcut4(np.where(warmup, ret[6], NAN)),
        "q_ret12": qcut4(np.where(warmup, ret[12], NAN)),
        "q_ret24": qcut4(np.where(warmup, ret[24], NAN)),
        "q_pos12": qcut4(np.where(warmup, pos12, NAN)),
        "q_pos24": qcut4(np.where(warmup, pos24, NAN)),
        "q_vol": qcut4(np.where(warmup, vol_ratio, NAN)),
    }


def build_path_book(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    horizons: Iterable[str],
) -> PathBook:
    """Forward path from the next completed candle. Event reference is close[t]."""
    wanted = {name: HORIZON_BARS[name] for name in horizons}
    max_h = max(wanted.values())
    n = len(close)
    entry = close
    thr_p05 = entry * (1.0 + 0.5 * R)
    thr_p10 = entry * (1.0 + 1.0 * R)
    thr_p20 = entry * (1.0 + 2.0 * R)
    thr_m10 = entry * (1.0 - 1.0 * R)

    run_max = np.full(n, -np.inf)
    run_min = np.full(n, np.inf)
    first_p05 = np.full(n, NAN)
    first_p10 = np.full(n, NAN)
    first_p20 = np.full(n, NAN)
    first_m10 = np.full(n, NAN)

    valid: dict[str, np.ndarray] = {}
    fwd: dict[str, np.ndarray] = {}
    mfe: dict[str, np.ndarray] = {}
    mae: dict[str, np.ndarray] = {}
    fut_vol: dict[str, np.ndarray] = {}
    hit_05: dict[str, np.ndarray] = {}
    hit_10: dict[str, np.ndarray] = {}
    hit_20: dict[str, np.ndarray] = {}
    race_plus: dict[str, np.ndarray] = {}
    race_minus: dict[str, np.ndarray] = {}
    race_neither: dict[str, np.ndarray] = {}
    t05: dict[str, np.ndarray] = {}
    t10: dict[str, np.ndarray] = {}
    t20: dict[str, np.ndarray] = {}
    name_by_h = {bars: name for name, bars in wanted.items()}

    for k in range(1, max_h + 1):
        h_k = np.full(n, NAN)
        l_k = np.full(n, NAN)
        c_k = np.full(n, NAN)
        stop = n - k
        if stop > 0:
            h_k[:stop] = high[k:]
            l_k[:stop] = low[k:]
            c_k[:stop] = close[k:]
        run_max = np.fmax(run_max, h_k)
        run_min = np.fmin(run_min, l_k)
        present = np.isfinite(h_k) & np.isfinite(l_k)
        unset_p05 = present & np.isnan(first_p05) & (h_k >= thr_p05)
        unset_p10 = present & np.isnan(first_p10) & (h_k >= thr_p10)
        unset_p20 = present & np.isnan(first_p20) & (h_k >= thr_p20)
        unset_m10 = present & np.isnan(first_m10) & (l_k <= thr_m10)
        first_p05[unset_p05] = k
        first_p10[unset_p10] = k
        first_p20[unset_p20] = k
        first_m10[unset_m10] = k
        if k not in name_by_h:
            continue
        name = name_by_h[k]
        ok = np.isfinite(c_k) & np.isfinite(entry) & (entry > 0)
        valid[name] = ok
        fwd[name] = np.where(ok, c_k / entry - 1.0, NAN)
        mfe[name] = np.where(ok, run_max / entry - 1.0, NAN)
        mae[name] = np.where(ok, 1.0 - run_min / entry, NAN)
        fut_vol[name] = np.where(ok, (run_max - run_min) / entry, NAN)
        hp = ok & np.isfinite(first_p10) & (first_p10 <= k)
        hm = ok & np.isfinite(first_m10) & (first_m10 <= k)
        # Same-candle: treat adverse as first (first_m10 <= first_p10).
        race_plus[name] = hp & (~hm | (first_p10 < first_m10))
        race_minus[name] = hm & (~hp | (first_m10 <= first_p10))
        race_neither[name] = ok & ~hp & ~hm
        hit_05[name] = ok & np.isfinite(first_p05) & (first_p05 <= k)
        hit_10[name] = hp
        hit_20[name] = ok & np.isfinite(first_p20) & (first_p20 <= k)
        t05[name] = np.where(hit_05[name], first_p05 * BAR_MINUTES, NAN)
        t10[name] = np.where(hit_10[name], first_p10 * BAR_MINUTES, NAN)
        t20[name] = np.where(hit_20[name], first_p20 * BAR_MINUTES, NAN)
    return PathBook(
        valid=valid,
        fwd=fwd,
        mfe=mfe,
        mae=mae,
        fut_vol=fut_vol,
        hit_05=hit_05,
        hit_10=hit_10,
        hit_20=hit_20,
        race_plus=race_plus,
        race_minus=race_minus,
        race_neither=race_neither,
        t05=t05,
        t10=t10,
        t20=t20,
    )


def first_hit_close(close: np.ndarray, target: np.ndarray, max_bars: int) -> np.ndarray:
    """Bars until close[t+k] >= target[t]. NaN if not reached."""
    n = len(close)
    out = np.full(n, NAN)
    for k in range(1, max_bars + 1):
        unset = np.isnan(out)
        hit = np.zeros(n, dtype=bool)
        if n > k:
            hit[: n - k] = close[k:] >= target[: n - k]
        take = unset & hit & np.isfinite(target)
        out[take] = k
    return out


def aggregate(book: PathBook, mask: np.ndarray, horizon: str) -> Agg:
    valid = book.valid[horizon] & np.asarray(mask, dtype=bool)
    n = int(valid.sum())
    if n == 0:
        return Agg(
            n=0,
            mean_fwd=NAN,
            median_fwd=NAN,
            mfe=NAN,
            mae=NAN,
            fut_vol=NAN,
            p05=NAN,
            p10=NAN,
            p20=NAN,
            p_plus=NAN,
            p_minus=NAN,
            p_neither=NAN,
            t05=NAN,
            t10=NAN,
            t20=NAN,
        )
    return Agg(
        n=n,
        mean_fwd=_finite_mean(book.fwd[horizon][valid]),
        median_fwd=_finite_median(book.fwd[horizon][valid]),
        mfe=_finite_median(book.mfe[horizon][valid]),
        mae=_finite_median(book.mae[horizon][valid]),
        fut_vol=_finite_median(book.fut_vol[horizon][valid]),
        p05=float(np.mean(book.hit_05[horizon][valid])),
        p10=float(np.mean(book.hit_10[horizon][valid])),
        p20=float(np.mean(book.hit_20[horizon][valid])),
        p_plus=float(np.mean(book.race_plus[horizon][valid])),
        p_minus=float(np.mean(book.race_minus[horizon][valid])),
        p_neither=float(np.mean(book.race_neither[horizon][valid])),
        t05=_finite_median(book.t05[horizon][valid]),
        t10=_finite_median(book.t10[horizon][valid]),
        t20=_finite_median(book.t20[horizon][valid]),
    )


def agg_row(
    investigation: str,
    regime: str,
    sample: str,
    horizon: str,
    agg: Agg,
    baseline: Agg | None = None,
) -> dict[str, object]:
    row = {
        "investigation": investigation,
        "regime": regime,
        "sample": sample,
        "horizon": horizon,
        "n": agg.n,
        "mean_forward_return": agg.mean_fwd,
        "median_forward_return": agg.median_fwd,
        "mfe_median": agg.mfe,
        "mae_median": agg.mae,
        "future_vol_median": agg.fut_vol,
        "p_plus_0_5R": agg.p05,
        "p_plus_1R": agg.p10,
        "p_plus_2R": agg.p20,
        "p_plus_1R_before_minus_1R": agg.p_plus,
        "p_minus_1R_before_plus_1R": agg.p_minus,
        "p_neither": agg.p_neither,
        "time_to_0_5R_median_min": agg.t05,
        "time_to_1R_median_min": agg.t10,
        "time_to_2R_median_min": agg.t20,
        "delta_median_fwd_vs_baseline": NAN,
        "delta_p_race_vs_baseline": NAN,
    }
    if baseline is not None and baseline.n > 0 and agg.n > 0:
        row["delta_median_fwd_vs_baseline"] = agg.median_fwd - baseline.median_fwd
        row["delta_p_race_vs_baseline"] = agg.p_plus - baseline.p_plus
    return row


def dist_rows(
    investigation: str,
    regime: str,
    book: PathBook,
    mask: np.ndarray,
    horizon: str = "12h",
) -> list[dict[str, object]]:
    valid = book.valid[horizon] & np.asarray(mask, dtype=bool)
    rows = []
    for metric, arr in (
        ("forward_return", book.fwd[horizon]),
        ("mfe", book.mfe[horizon]),
        ("mae", book.mae[horizon]),
        ("future_vol", book.fut_vol[horizon]),
    ):
        x = arr[valid]
        rows.append(
            {
                "investigation": investigation,
                "regime": regime,
                "horizon": horizon,
                "metric": metric,
                "n": int(np.isfinite(x).sum()),
                "mean": _finite_mean(x),
                "p05": _finite_quantile(x, 0.05),
                "p25": _finite_quantile(x, 0.25),
                "p50": _finite_median(x),
                "p75": _finite_quantile(x, 0.75),
                "p95": _finite_quantile(x, 0.95),
            }
        )
    return rows


def monthly_rows(
    investigation: str,
    regime: str,
    months: np.ndarray,
    mask: np.ndarray,
    book: PathBook,
    horizons: tuple[str, ...] = ("1h", "4h", "12h"),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    unique = [str(m) for m in pd.unique(months) if str(m) != "nan"]
    unique.sort()
    for month in unique:
        in_month = (months == month) & np.asarray(mask, dtype=bool)
        for horizon in horizons:
            agg = aggregate(book, in_month, horizon)
            rows.append(
                {
                    "investigation": investigation,
                    "regime": regime,
                    "month": month,
                    "horizon": horizon,
                    "n": agg.n,
                    "median_forward_return": agg.median_fwd,
                    "mean_forward_return": agg.mean_fwd,
                    "mfe_median": agg.mfe,
                    "mae_median": agg.mae,
                    "p_plus_1R_before_minus_1R": agg.p_plus,
                    "p_minus_1R_before_plus_1R": agg.p_minus,
                    "p_neither": agg.p_neither,
                    "future_vol_median": agg.fut_vol,
                }
            )
    return rows


def simulate_random_long(
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    signal_mask: np.ndarray,
    *,
    starting_capital: float = STARTING_CAPITAL,
    risk_per_trade: float = RISK_PER_TRADE,
    stop_loss: float = STOP_LOSS,
    take_profit: float = TAKE_PROFIT,
    slippage: float = SLIPPAGE,
    entry_rate: float = ENTRY_RATE,
    exit_rate: float = EXIT_RATE,
) -> dict[str, float]:
    """v0.4-semantic LONG replay: next-open, SL-first, 1% risk, fees and slippage."""
    n = len(close)
    cash = float(starting_capital)
    realized = 0.0
    pending = False
    qty = 0.0
    entry_px = 0.0
    entry_ref = 0.0
    sl = 0.0
    tp = 0.0
    entry_fee = 0.0
    entry_slip = 0.0
    risk_amt = 0.0
    in_pos = False
    peak = float(starting_capital)
    max_dd = 0.0
    max_dd_pct = 0.0
    signals = 0
    fills = 0
    closed = 0
    wins = 0
    losses = 0
    tp_count = 0
    sl_count = 0
    ignored_open = 0
    ignored_end = 0
    ignored_cash = 0
    win_pnl = 0.0
    loss_pnl = 0.0
    sum_r = 0.0
    open_end = 0

    for t in range(n):
        if pending:
            pending = False
            entry_ref = float(open_px[t])
            entry_px = entry_ref * (1.0 + slippage)
            sl = entry_px * (1.0 - stop_loss)
            tp = entry_px * (1.0 + take_profit)
            if cash <= 0 or entry_px <= 0:
                ignored_cash += 1
            else:
                qty_risk = (cash * risk_per_trade) / (entry_px * stop_loss)
                qty_cash = cash / (entry_px * (1.0 + entry_rate))
                qty = min(qty_risk, qty_cash)
                pos_val = qty * entry_px
                entry_fee = pos_val * entry_rate
                needed = pos_val + entry_fee
                if qty <= 0 or needed - cash > 1e-9:
                    ignored_cash += 1
                else:
                    cash -= needed
                    entry_slip = qty * (entry_px - entry_ref)
                    risk_amt = qty * entry_px * stop_loss
                    in_pos = True
                    fills += 1

        if in_pos:
            sl_hit = low[t] <= sl
            tp_hit = high[t] >= tp
            hit = None
            xref = 0.0
            if sl_hit:
                hit = "SL"
                xref = sl
            elif tp_hit:
                hit = "TP"
                xref = tp
            if hit is not None:
                exit_px = xref * (1.0 - slippage)
                exit_fee = qty * exit_px * exit_rate
                net = qty * (exit_px - entry_px) - entry_fee - exit_fee
                cash += qty * exit_px - exit_fee
                realized += net
                r_mult = net / risk_amt if risk_amt else NAN
                sum_r += r_mult
                closed += 1
                if net > 0:
                    wins += 1
                    win_pnl += net
                elif net < 0:
                    losses += 1
                    loss_pnl += net
                if hit == "TP":
                    tp_count += 1
                else:
                    sl_count += 1
                in_pos = False
                qty = 0.0

        mark = cash + (qty * close[t] if in_pos else 0.0)
        if mark > peak:
            peak = mark
        dd = peak - mark
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd / peak if peak > 0 else 0.0

        if signal_mask[t]:
            signals += 1
            if in_pos:
                ignored_open += 1
            elif t == n - 1:
                ignored_end += 1
            else:
                pending = True

    if in_pos:
        open_end = 1
    ending_realized = float(starting_capital) + realized
    ending_mtm = cash + (qty * close[-1] if in_pos else 0.0)
    profit_factor = (win_pnl / abs(loss_pnl)) if loss_pnl < 0 else NAN
    win_rate = (wins / closed) if closed else NAN
    avg_r = (sum_r / closed) if closed else NAN
    return {
        "ending_cash": cash,
        "ending_realized_equity": ending_realized,
        "ending_mtm_equity": ending_mtm,
        "total_net_pnl": realized,
        "signals": float(signals),
        "filled_trades": float(fills),
        "closed_trades": float(closed),
        "open_end_of_data": float(open_end),
        "ignored_position_open": float(ignored_open),
        "tp_count": float(tp_count),
        "sl_count": float(sl_count),
        "win_rate": win_rate,
        "average_R": avg_r,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "entry_slippage_last": entry_slip,
    }


def run_random_baselines(
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    n_samples: int = RANDOM_SAMPLES,
    seed: int = RANDOM_SEED,
    signal_p: float = RANDOM_SIGNAL_P,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(close)
    for i in range(n_samples):
        sample_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        signals = sample_rng.random(n) < signal_p
        stats = simulate_random_long(open_px, high, low, close, signals)
        stats["sample_id"] = i
        rows.append(stats)
    return pd.DataFrame(rows)


def percentile_rank(value: float, series: np.ndarray) -> float:
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or not np.isfinite(value):
        return NAN
    return float(np.mean(x <= value) * 100.0)
