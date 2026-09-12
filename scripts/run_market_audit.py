"""CLI for the 2024 BTCUSDT 5m READ-ONLY market-structure audit.

Does not modify production strategy, backtest, or existing reports.
Does not download data, call Binance, or touch 2025 candles.

Usage:
    python scripts/run_market_audit.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from market_audit_engine import (  # noqa: E402
    COMMON_HORIZONS,
    COST_R,
    HOLDING_HORIZONS,
    HORIZON_BARS,
    HOUR_BUCKETS,
    QUANTILES,
    R,
    RANDOM_SAMPLES,
    RANDOM_SEED,
    RANDOM_SIGNAL_P,
    STARTING_CAPITAL,
    THIN_EVERY,
    WEEKDAYS,
    YEAR_END,
    YEAR_START,
    Agg,
    PathBook,
    aggregate,
    agg_row,
    build_features,
    build_path_book,
    cluster_stats,
    dist_rows,
    first_hit_close,
    load_2024_enriched,
    month_consistency,
    monthly_rows,
    percentile_rank,
    run_random_baselines,
    simulate_random_long,
    thin_regular,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "research" / "market_audit"
ENRICHED_DIR = PROJECT_ROOT / "data" / "enriched" / "BTCUSDT" / "5m"
A_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "research" / "development" / "A_baseline" / "metrics.csv"
)

FOLDERS = {
    "01": "01_time_of_day",
    "02": "02_day_of_week",
    "03": "03_trend_regime",
    "04": "04_volatility_regime",
    "05": "05_momentum_persistence",
    "06": "06_mean_reversion",
    "07": "07_range_position",
    "08": "08_volume_regime",
    "09": "09_trend_volatility",
    "10": "10_autocorrelation",
    "11": "11_random_baseline",
    "12": "12_holding_horizon",
}


def _finite(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def pct(x: float, digits: int = 3) -> str:
    if not _finite(x):
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"


def pct_pts(x: float) -> str:
    if not _finite(x):
        return "n/a"
    return f"{100.0 * float(x):+.3f} pp"


def ret_pct(x: float) -> str:
    if not _finite(x):
        return "n/a"
    return f"{100.0 * float(x):.4f}%"


def num(x: float, digits: int = 4) -> str:
    if not _finite(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def effect_kind(agg: Agg, base: Agg) -> str:
    if agg.n == 0 or base.n == 0:
        return "insufficient sample"
    dp = agg.p_plus - base.p_plus
    dfwd = agg.median_fwd - base.median_fwd
    dvol = (agg.mfe + agg.mae) - (base.mfe + base.mae)
    if abs(dp) < 0.01 and abs(dfwd) < 0.0005:
        if dvol > 0.002:
            return "volatility-only (MFE/MAE larger, race ~unchanged)"
        return "indistinguishable from baseline"
    if abs(dp) < 0.015 and dvol > 0.002 and abs(dfwd) < 0.001:
        return "volatility expansion without meaningful directional race shift"
    if abs(dp) >= 0.015 or abs(dfwd) >= 0.001:
        if abs(dfwd) < 0.003 and abs(dp) < 0.03:
            return "small directional difference (below ~0.30R cost scale)"
        return "directional difference versus baseline"
    return "small mixed difference"


def verdict_from_regimes(
    items: list[tuple[str, Agg, Agg, dict]],
    *,
    hypothesis: str,
    monotonic: bool | None,
) -> tuple[str, str]:
    """items: (name, agg12, base12, monthly_info)."""
    if not items:
        return "C", "no regimes to evaluate"
    meaningful = []
    opposite = []
    vol_only = 0
    for name, agg, base, monthly in items:
        if name == "ALL" or agg.n < 200 or base.n == 0:
            continue
        dp = agg.p_plus - base.p_plus
        dfwd = agg.median_fwd - base.median_fwd
        kind = effect_kind(agg, base)
        months_ok = int(monthly.get("months_race_above_baseline", 0) or 0)
        usable = int(monthly.get("months_usable", 0) or 0)
        robust = usable > 0 and months_ok >= max(8, int(math.ceil(0.67 * usable)))
        if "volatility" in kind and "directional" not in kind:
            vol_only += 1
            continue
        if abs(dp) < 0.015 and abs(dfwd) < 0.001:
            continue
        economic = abs(dfwd) >= 0.003 or abs(dp) >= 0.05
        record = (name, dp, dfwd, robust, economic, agg.n, kind)
        if dp > 0.015 or dfwd > 0.001:
            meaningful.append(record)
        elif dp < -0.015 or dfwd < -0.001:
            opposite.append(record)
    if hypothesis == "none":
        if not meaningful and not opposite:
            return "C", "no hour/day/regime separates from the 2024 unconditional path"
        if meaningful and not any(r[3] and r[4] for r in meaningful):
            return "B", "some buckets differ, but not economically large and month-robust"
        if any(r[3] and r[4] for r in meaningful) and monotonic:
            return "A", "directional, economic, and month-robust versus ALL 2024"
        return "B", "isolated or non-robust bucket differences; multiple-testing risk"
    if hypothesis == "trend_helps_long":
        bull = next((x for x in items if x[0] == "EMA20>EMA50"), None)
        bear = next((x for x in items if x[0] == "EMA20<=EMA50"), None)
        if bull and bear and bull[1].n and bear[1].n:
            dp = bull[1].p_plus - bear[1].p_plus
            if dp < -0.02:
                return "D", "bullish EMA regime has a worse +1R-before-1R race than bearish"
            if abs(dp) < 0.015 and not meaningful:
                return "C", "trend split does not change the directional race"
        if meaningful and any(r[3] and r[4] for r in meaningful):
            return "A", "trend regime shows a robust directional gap"
        if meaningful:
            return "B", "trend changes outcomes slightly but not enough after cost/robustness"
        return "C", "trend mainly changes volatility or nothing material"
    if hypothesis == "vol_directional":
        if vol_only and not meaningful:
            return "C", "current volatility predicts future expansion, not LONG directional race"
        if meaningful and any(r[3] and r[4] for r in meaningful):
            return "A", "volatility quartile shows a robust directional race gap"
        if meaningful:
            return "B", "some directional movement across vol quartiles, unstable or small"
        return "C", "no meaningful directional asymmetry across volatility quartiles"
    if hypothesis == "momentum_continues":
        if opposite and not meaningful:
            return "D", "stronger prior return is associated with worse subsequent LONG race"
        if meaningful and monotonic and any(r[3] and r[4] for r in meaningful):
            return "A", "prior momentum persists into a robust directional race"
        if meaningful:
            return "B", "some continuation/reversal pattern exists but is weak or non-monotonic"
        return "C", "prior return quartiles do not change future directional race"
    if hypothesis == "mean_reversion":
        if opposite and not meaningful:
            return "D", "larger short-term declines continue rather than rebound"
        if meaningful and any(r[3] and r[4] for r in meaningful):
            return "A", "declines are followed by a robust rebound race versus baseline"
        if meaningful:
            return "B", "some rebound tendency, insufficient after cost and monthly checks"
        return "C", "no economically meaningful rebound after short-term declines"
    if hypothesis == "range_position":
        if meaningful and any(r[3] and r[4] for r in meaningful) and monotonic:
            return "A", "location in the prior range changes the directional race robustly"
        if meaningful:
            return "B", "some range-position differences, not robust enough"
        return "C", "recent-range location does not add directional information"
    if hypothesis == "volume":
        if vol_only and not meaningful:
            return "C", "volume predicts expansion/activity, not directional race"
        if meaningful and any(r[3] and r[4] for r in meaningful):
            return "A", "volume regime has a robust directional race gap"
        if meaningful:
            return "B", "volume shows a weak directional association"
        return "C", "volume ratio does not identify a directional LONG edge"
    if hypothesis == "interaction":
        if meaningful and any(r[3] and r[4] for r in meaningful):
            return "A", "trend x volatility cell has a robust directional gap"
        if meaningful:
            return "B", "interaction cells differ slightly; not strategy-grade"
        return "C", "trend x volatility does not create directional asymmetry"
    if hypothesis == "horizon":
        return "C", "see holding-horizon table; default is no unique profitable scale"
    return "C", "descriptive only"


def interpret_row(agg: Agg, base: Agg, monthly: dict, cluster: dict) -> tuple[str, str]:
    kind = effect_kind(agg, base)
    flag = str(monthly.get("consistency_flag", ""))
    gap = cluster.get("median_gap_minutes")
    clust_note = (
        f"median gap {num(float(gap), 1)} min; "
        f"%<1h {num(float(cluster.get('pct_within_1h', math.nan)), 1)}; "
        f"%<3h {num(float(cluster.get('pct_within_3h', math.nan)), 1)}; "
        f"%<12h {num(float(cluster.get('pct_within_12h', math.nan)), 1)}"
    )
    interp = f"{kind}. monthly: {flag}"
    return interp, clust_note


def cell_verdict(agg: Agg, base: Agg, monthly: dict, thinned: Agg | None) -> str:
    if agg.n < 200:
        return "C"
    kind = effect_kind(agg, base)
    dp = agg.p_plus - base.p_plus if base.n else 0.0
    dfwd = agg.median_fwd - base.median_fwd if base.n else 0.0
    usable = int(monthly.get("months_usable", 0) or 0)
    race_m = int(monthly.get("months_race_above_baseline", 0) or 0)
    robust = usable > 0 and race_m >= max(8, int(math.ceil(0.67 * usable)))
    thin_ok = True
    if thinned is not None and thinned.n >= 30 and base.n:
        thin_dp = thinned.p_plus - base.p_plus
        if abs(dp) >= 0.02 and abs(thin_dp) < 0.01:
            thin_ok = False
    economic = abs(dfwd) >= 0.003 or abs(dp) >= 0.05
    directional = "directional" in kind
    if directional and robust and thin_ok and economic and agg.n >= 500:
        return "A"
    if directional and (robust or economic) and thin_ok:
        return "B"
    if "volatility" in kind:
        return "C"
    return "C"


def format_agg_block(title: str, agg: Agg, base: Agg | None = None) -> list[str]:
    lines = [
        f"{title}",
        f"  n={agg.n}",
        (
            f"  mean_fwd={ret_pct(agg.mean_fwd)}  median_fwd={ret_pct(agg.median_fwd)}"
            + (
                f"  d_median_vs_base={ret_pct(agg.median_fwd - base.median_fwd)}"
                if base and base.n
                else ""
            )
        ),
        f"  MFE={ret_pct(agg.mfe)}  MAE={ret_pct(agg.mae)}  future_vol={ret_pct(agg.fut_vol)}",
        (
            f"  P(+0.5R)={pct(agg.p05)}  P(+1R)={pct(agg.p10)}  P(+2R)={pct(agg.p20)}"
        ),
        (
            f"  P(+1R before -1R)={pct(agg.p_plus)}  P(-1R before +1R)={pct(agg.p_minus)}"
            f"  neither={pct(agg.p_neither)}"
            + (
                f"  d_race={pct_pts(agg.p_plus - base.p_plus)}"
                if base and base.n
                else ""
            )
        ),
        (
            f"  time_to_+0.5R={num(agg.t05, 1)} min  "
            f"+1R={num(agg.t10, 1)} min  +2R={num(agg.t20, 1)} min"
        ),
    ]
    if base and base.n:
        lines.append(f"  effect: {effect_kind(agg, base)}")
    return lines


class Collector:
    def __init__(self) -> None:
        self.metrics: list[dict[str, object]] = []
        self.monthly: list[dict[str, object]] = []
        self.dist: list[dict[str, object]] = []
        self.matrix: list[dict[str, object]] = []
        self.verdicts: dict[str, tuple[str, str]] = {}
        self.notes: dict[str, list[str]] = {}
        self.test_count = 0

    def add_regime(
        self,
        investigation: str,
        regime: str,
        mask: np.ndarray,
        book: PathBook,
        feat: dict[str, np.ndarray],
        base_mask: np.ndarray,
        horizons: tuple[str, ...],
        *,
        matrix: bool = True,
    ) -> dict[str, Agg]:
        by_h: dict[str, Agg] = {}
        thin = thin_regular(len(mask)) & mask
        for sample, smask in (("full", mask), ("thinned144", thin)):
            for horizon in horizons:
                agg = aggregate(book, smask, horizon)
                base = aggregate(book, base_mask, horizon)
                self.metrics.append(
                    agg_row(investigation, regime, sample, horizon, agg, base)
                )
                self.test_count += 1
                by_h[f"{sample}:{horizon}"] = agg
                by_h[f"base:{horizon}"] = base
        self.monthly.extend(
            monthly_rows(investigation, regime, feat["month"], mask, book)
        )
        self.dist.extend(dist_rows(investigation, regime, book, mask))
        h12 = "12h"
        agg12 = aggregate(book, mask, h12)
        base12 = aggregate(book, base_mask, h12)
        thin12 = aggregate(book, thin, h12)
        if regime == "ALL":
            monthly_info = {
                "months_usable": 12,
                "months_race_above_baseline": 12,
                "months_fwd_above_baseline": 12,
                "consistency_flag": "baseline",
                "small_month_notes": "",
            }
        else:
            monthly_info = month_consistency(
                feat["month"],
                mask,
                base_mask,
                book.race_plus[h12],
                book.valid[h12],
                book.fwd[h12],
            )
        cluster = cluster_stats(mask)
        if regime == "ALL":
            interp, clust_note = (
                "unconditional 2024 reference path",
                interpret_row(agg12, base12, monthly_info, cluster)[1],
            )
            cell = "baseline"
        else:
            interp, clust_note = interpret_row(agg12, base12, monthly_info, cluster)
            cell = cell_verdict(agg12, base12, monthly_info, thin12)
        if matrix:
            self.matrix.append(
                {
                    "investigation": investigation,
                    "regime": regime,
                    "n": agg12.n,
                    "median_forward_return_1h": aggregate(book, mask, "1h").median_fwd,
                    "median_forward_return_4h": aggregate(book, mask, "4h").median_fwd,
                    "median_forward_return_12h": agg12.median_fwd,
                    "mfe": agg12.mfe,
                    "mae": agg12.mae,
                    "p_plus_1R": agg12.p10,
                    "p_minus_1R": agg12.p_minus,
                    "p_plus_1R_before_minus_1R": agg12.p_plus,
                    "future_volatility": agg12.fut_vol,
                    "monthly_consistency": monthly_info["consistency_flag"],
                    "clustering_note": clust_note,
                    "interpretation": interp,
                    "verdict": cell,
                    "n_thinned144": thin12.n,
                    "p_race_thinned144": thin12.p_plus,
                    "delta_p_race_vs_all": agg12.p_plus - base12.p_plus
                    if base12.n
                    else float("nan"),
                    "delta_median_fwd_12h_vs_all": agg12.median_fwd - base12.median_fwd
                    if base12.n
                    else float("nan"),
                }
            )
        by_h["_monthly"] = monthly_info  # type: ignore[assignment]
        by_h["_cluster"] = cluster  # type: ignore[assignment]
        return by_h


def horizon_table(book: PathBook, mask: np.ndarray, horizons: tuple[str, ...]) -> list[str]:
    lines = []
    for horizon in horizons:
        agg = aggregate(book, mask, horizon)
        lines.append(
            f"  {horizon:>4}: n={agg.n:6d}  med_fwd={ret_pct(agg.median_fwd):>10}  "
            f"mean_fwd={ret_pct(agg.mean_fwd):>10}  "
            f"MFE={ret_pct(agg.mfe):>9}  MAE={ret_pct(agg.mae):>9}  "
            f"P(+1R before -1R)={pct(agg.p_plus):>8}  "
            f"P(-1R first)={pct(agg.p_minus):>8}  neither={pct(agg.p_neither):>8}"
        )
    return lines


def validate_replica(
    frame: pd.DataFrame,
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict[str, object]:
    from src.backtest.config import load_backtest_config
    from src.backtest.engine import run_backtest

    rng = np.random.default_rng(7)
    # Prefix verifies fill/fee/SL-first algebra; the per-bar loop is identical on the full year.
    n_check = min(len(close), 12_000)
    signal_mask = rng.random(n_check) < 0.02
    replica = simulate_random_long(
        open_px[:n_check], high[:n_check], low[:n_check], close[:n_check], signal_mask
    )
    candles = frame[["timestamp", "open", "high", "low", "close"]].iloc[:n_check].copy()
    signals = pd.DataFrame(
        {
            "timestamp": candles["timestamp"].to_numpy(),
            "signal": np.where(signal_mask, "LONG_ENTRY", "NONE"),
        }
    )
    config = load_backtest_config(PROJECT_ROOT / "config" / "backtest.yaml")
    engine = run_backtest(
        candles.reset_index(drop=True),
        signals,
        config,
        symbol="BTCUSDT",
        interval="5m",
    )
    closed = [t for t in engine.trades if t.status == "CLOSED"]
    engine_net = float(sum(t.net_pnl or 0.0 for t in closed))
    engine_closed = len(closed)
    return {
        "replica_net": replica["total_net_pnl"],
        "engine_net": engine_net,
        "replica_closed": replica["closed_trades"],
        "engine_closed": float(engine_closed),
        "replica_cash": replica["ending_cash"],
        "engine_cash": float(engine.cash),
        "net_match": abs(replica["total_net_pnl"] - engine_net) < 1e-8,
        "cash_match": abs(replica["ending_cash"] - float(engine.cash)) < 1e-8,
        "closed_match": int(replica["closed_trades"]) == engine_closed,
    }


def load_strategy_a_metrics() -> dict[str, float]:
    frame = pd.read_csv(A_METRICS_PATH)
    values = {str(row["metric"]): row["value"] for _, row in frame.iterrows()}
    out: dict[str, float] = {}
    for key in (
        "ending_realized_equity",
        "ending_mtm_equity",
        "total_net_pnl",
        "win_rate",
        "average_R_per_closed_trade",
        "profit_factor",
        "max_drawdown",
        "max_drawdown_pct",
        "closed_trades",
        "filled_trades",
        "signals",
    ):
        try:
            out[key] = float(values[key])
        except (KeyError, TypeError, ValueError):
            out[key] = float("nan")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="2024 BTCUSDT 5m market-structure audit")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-replica-check", action="store_true")
    parser.add_argument("--random-samples", type=int, default=RANDOM_SAMPLES)
    args = parser.parse_args()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Market audit start {started}", flush=True)
    print("Loading 2024 enriched parquet (2025 files not opened)", flush=True)
    candles = load_2024_enriched(ENRICHED_DIR)
    if candles["timestamp"].max() > YEAR_END or (candles["timestamp"].dt.year != 2024).any():
        raise SystemExit("safety abort: non-2024 timestamps")
    print(f"Rows={len(candles)}  first={candles['timestamp'].iloc[0]}  "
          f"last={candles['timestamp'].iloc[-1]}", flush=True)

    feat = build_features(candles)
    warmup = feat["warmup_ok"]
    print("Building forward path book (max 24h, 2024 remainder only)", flush=True)
    book = build_path_book(
        feat["high"], feat["low"], feat["close"], HOLDING_HORIZONS
    )
    n = len(candles)
    base = warmup
    coll = Collector()
    inv_summaries: dict[str, str] = {}
    inv_methods: dict[str, str] = {}

    # ---- 01 time of day ----
    print("Investigation 01 time of day", flush=True)
    tod_items = []
    lines = ["# Investigation 01 - Time of day (UTC 4-hour buckets)", ""]
    lines.extend(horizon_table(book, base, COMMON_HORIZONS))
    lines.append("")
    coll.add_regime("01_time_of_day", "ALL", base, book, feat, base, COMMON_HORIZONS)
    for bucket in HOUR_BUCKETS:
        mask = base & (feat["hour_bucket"] == bucket)
        stats = coll.add_regime(
            "01_time_of_day", bucket, mask, book, feat, base, COMMON_HORIZONS
        )
        agg = aggregate(book, mask, "12h")
        b12 = aggregate(book, base, "12h")
        lines.append("")
        lines.extend(format_agg_block(f"Bucket {bucket} @12h", agg, b12))
        lines.append("  horizons:")
        lines.extend(horizon_table(book, mask, COMMON_HORIZONS))
        tod_items.append((bucket, agg, b12, stats["_monthly"]))
    all_agg = aggregate(book, base, "12h")
    verdict, reason = verdict_from_regimes(tod_items, hypothesis="none", monotonic=False)
    coll.verdicts["01"] = (verdict, reason)
    lines += ["", f"VERDICT: {verdict}", reason, ""]
    lines.append(
        "Comparison is versus ALL 2024 warmup-valid observations, not versus zero. "
        "2024 is a bull year; a positive median forward return is not an edge."
    )
    inv_summaries["01"] = "\n".join(lines)
    inv_methods["01"] = (
        "Event clock is the 5m close. Forward path uses only subsequent candles "
        "still inside 2024. Hour is UTC from the candle timestamp. Buckets are "
        "fixed 00-04, 04-08, 08-12, 12-16, 16-20, 20-24. No hour was searched or "
        "merged after seeing results. Entry reference is close[t]; R=1% of that "
        "price. Same-candle +1R and -1R: adverse first. Thinned diagnostic: every "
        f"{THIN_EVERY} bars (12h)."
    )

    # ---- 02 day of week ----
    print("Investigation 02 day of week", flush=True)
    lines = ["# Investigation 02 - Day of week (UTC)", ""]
    dow_items = []
    coll.add_regime("02_day_of_week", "ALL", base, book, feat, base, COMMON_HORIZONS)
    for day in WEEKDAYS:
        mask = base & (feat["dow"] == day)
        stats = coll.add_regime(
            "02_day_of_week", day, mask, book, feat, base, COMMON_HORIZONS
        )
        agg = aggregate(book, mask, "12h")
        b12 = aggregate(book, base, "12h")
        lines.extend(format_agg_block(f"{day} @12h", agg, b12))
        lines.append("  horizons:")
        lines.extend(horizon_table(book, mask, COMMON_HORIZONS))
        lines.append("")
        dow_items.append((day, agg, b12, stats["_monthly"]))
    verdict, reason = verdict_from_regimes(dow_items, hypothesis="none", monotonic=False)
    coll.verdicts["02"] = (verdict, reason)
    lines += [f"VERDICT: {verdict}", reason]
    inv_summaries["02"] = "\n".join(lines)
    inv_methods["02"] = (
        "UTC weekday of the event candle. Days are not merged. Metrics as in "
        "investigation 01. Baseline = all warmup-valid 2024 observations."
    )

    # ---- 03 trend ----
    print("Investigation 03 trend regime", flush=True)
    lines = ["# Investigation 03 - Trend regime (EMA20 vs EMA50)", ""]
    trend_items = []
    coll.add_regime("03_trend_regime", "ALL", base, book, feat, base, COMMON_HORIZONS)
    for name, mask in (
        ("EMA20>EMA50", feat["bull"]),
        ("EMA20<=EMA50", feat["bear"]),
    ):
        stats = coll.add_regime(
            "03_trend_regime", name, mask, book, feat, base, COMMON_HORIZONS
        )
        agg = aggregate(book, mask, "12h")
        b12 = aggregate(book, base, "12h")
        lines.extend(format_agg_block(f"{name} @12h", agg, b12))
        lines.extend(horizon_table(book, mask, COMMON_HORIZONS))
        lines.append("")
        trend_items.append((name, agg, b12, stats["_monthly"]))
    for q in QUANTILES:
        mask = base & (feat["q_spread"] == q)
        stats = coll.add_regime(
            "03_trend_regime", f"spread_{q}", mask, book, feat, base, COMMON_HORIZONS
        )
        agg = aggregate(book, mask, "12h")
        b12 = aggregate(book, base, "12h")
        lines.extend(format_agg_block(f"EMA spread {q} @12h", agg, b12))
        trend_items.append((f"spread_{q}", agg, b12, stats["_monthly"]))
    # monotonicity of spread quartiles on race
    spread_p = [
        aggregate(book, base & (feat["q_spread"] == q), "12h").p_plus for q in QUANTILES
    ]
    monotonic_spread = all(
        spread_p[i] <= spread_p[i + 1] + 1e-12 for i in range(3)
    ) or all(spread_p[i] >= spread_p[i + 1] - 1e-12 for i in range(3))
    verdict, reason = verdict_from_regimes(
        trend_items, hypothesis="trend_helps_long", monotonic=monotonic_spread
    )
    coll.verdicts["03"] = (verdict, reason)
    lines += ["", f"spread Q1..Q4 P(+1R before -1R)={[pct(x) for x in spread_p]}",
              f"monotonic={monotonic_spread}", f"VERDICT: {verdict}", reason]
    inv_summaries["03"] = "\n".join(lines)
    inv_methods["03"] = (
        "Regimes use EMA20 and EMA50 already on the enriched candle (known at close). "
        "A: EMA20>EMA50. B: EMA20<=EMA50. Spread=(EMA20-EMA50)/close split into four "
        "equal-count quantiles. No spread threshold search. Separate directional race "
        "from MFE/MAE expansion."
    )

    # ---- 04 volatility ----
    print("Investigation 04 volatility regime", flush=True)
    lines = ["# Investigation 04 - Volatility regime (prior 12-bar normalized range)", ""]
    vol_items = []
    coll.add_regime("04_volatility_regime", "ALL", base, book, feat, base, COMMON_HORIZONS)
    for q in QUANTILES:
        mask = base & (feat["q_range12"] == q)
        stats = coll.add_regime(
            "04_volatility_regime", f"range12_{q}", mask, book, feat, base, COMMON_HORIZONS
        )
        agg = aggregate(book, mask, "12h")
        b12 = aggregate(book, base, "12h")
        lines.extend(format_agg_block(f"range12 {q} @12h", agg, b12))
        lines.append(
            f"  also 24-bar range median={ret_pct(np.nanmedian(feat['range24'][mask]))}"
        )
        vol_items.append((f"range12_{q}", agg, b12, stats["_monthly"]))
    for q in QUANTILES:
        mask = base & (feat["q_range24"] == q)
        coll.add_regime(
            "04_volatility_regime", f"range24_{q}", mask, book, feat, base, COMMON_HORIZONS
        )
    races = [
        aggregate(book, base & (feat["q_range12"] == q), "12h").p_plus for q in QUANTILES
    ]
    vols = [
        aggregate(book, base & (feat["q_range12"] == q), "12h").fut_vol for q in QUANTILES
    ]
    vol_up = all(vols[i] <= vols[i + 1] + 1e-12 for i in range(3))
    race_flat = max(races) - min(races) < 0.015
    verdict, reason = verdict_from_regimes(
        vol_items, hypothesis="vol_directional", monotonic=not race_flat
    )
    if vol_up and race_flat:
        verdict, reason = "C", (
            "Higher current range predicts higher future range (expansion persistence) "
            "but not a directional +1R-before-1R asymmetry."
        )
    coll.verdicts["04"] = (verdict, reason)
    lines += [
        "",
        f"future_vol Q1..Q4={[ret_pct(x) for x in vols]}",
        f"P(race+) Q1..Q4={[pct(x) for x in races]}",
        f"VERDICT: {verdict}",
        reason,
    ]
    inv_summaries["04"] = "\n".join(lines)
    inv_methods["04"] = (
        "range12 = (max high - min low of the previous 12 completed candles) / close. "
        "Current candle is excluded. Equal-count quartiles, not searched thresholds. "
        "Future volatility = (max high - min low) / entry over the forward window. "
        "24-bar range is a secondary descriptive split. Realized absolute returns over "
        "the same lookback are computed as a companion feature (realized12/24)."
    )

    # ---- 05 momentum ----
    print("Investigation 05 momentum persistence", flush=True)
    lines = ["# Investigation 05 - Momentum persistence", ""]
    mom_items = []
    coll.add_regime(
        "05_momentum_persistence", "ALL", base, book, feat, base, COMMON_HORIZONS
    )
    for lookback, qkey, rkey in (
        (3, "q_ret3", "ret3"),
        (6, "q_ret6", "ret6"),
        (12, "q_ret12", "ret12"),
        (24, "q_ret24", "ret24"),
    ):
        lines.append(f"## {lookback}-bar historical return quartiles")
        qs_p = []
        for q in QUANTILES:
            mask = base & (feat[qkey] == q)
            stats = coll.add_regime(
                "05_momentum_persistence",
                f"ret{lookback}_{q}",
                mask,
                book,
                feat,
                base,
                COMMON_HORIZONS,
            )
            agg = aggregate(book, mask, "12h")
            b12 = aggregate(book, base, "12h")
            lines.extend(format_agg_block(f"ret{lookback} {q} @12h", agg, b12))
            mom_items.append((f"ret{lookback}_{q}", agg, b12, stats["_monthly"]))
            qs_p.append(agg.p_plus)
        lines.append(f"  race Q1..Q4={[pct(x) for x in qs_p]}")
        pos = base & (feat[rkey] > 0)
        neg = base & (feat[rkey] < 0)
        for name, mask in ((f"ret{lookback}_positive", pos), (f"ret{lookback}_negative", neg)):
            stats = coll.add_regime(
                "05_momentum_persistence", name, mask, book, feat, base, COMMON_HORIZONS
            )
            agg = aggregate(book, mask, "12h")
            b12 = aggregate(book, base, "12h")
            lines.extend(format_agg_block(f"{name} @12h", agg, b12))
            mom_items.append((name, agg, b12, stats["_monthly"]))
        lines.append("")
    q24 = [
        aggregate(book, base & (feat["q_ret24"] == q), "12h").p_plus for q in QUANTILES
    ]
    mono = all(q24[i] <= q24[i + 1] + 1e-12 for i in range(3))
    verdict, reason = verdict_from_regimes(
        mom_items, hypothesis="momentum_continues", monotonic=mono
    )
    coll.verdicts["05"] = (verdict, reason)
    lines += [f"VERDICT: {verdict}", reason]
    inv_summaries["05"] = "\n".join(lines)
    inv_methods["05"] = (
        "Historical close-to-close returns over 3/6/12/24 bars ending at t. Quartiles "
        "are equal-count. Positive vs negative historical return is a second split. "
        "No lookback was selected as 'best'. Continuation would require higher prior "
        "return quartiles to show higher P(+1R before -1R) and/or higher median forward "
        "return after separating volatility."
    )

    # ---- 06 mean reversion ----
    print("Investigation 06 mean reversion", flush=True)
    lines = ["# Investigation 06 - Mean-reversion strength after declines", ""]
    mr_items = []
    coll.add_regime("06_mean_reversion", "ALL", base, book, feat, base, COMMON_HORIZONS)
    recovery_notes: list[str] = []
    for lookback, qkey, rkey in (
        (3, "q_ret3", "ret3"),
        (6, "q_ret6", "ret6"),
        (12, "q_ret12", "ret12"),
        (24, "q_ret24", "ret24"),
    ):
        target = np.full(n, np.nan)
        target[lookback:] = feat["close"][:-lookback]
        rec = first_hit_close(feat["close"], target, HORIZON_BARS["12h"])
        for q in QUANTILES:
            mask = base & (feat[qkey] == q)
            stats = coll.add_regime(
                "06_mean_reversion",
                f"ret{lookback}_{q}",
                mask,
                book,
                feat,
                base,
                COMMON_HORIZONS,
            )
            agg = aggregate(book, mask, "12h")
            b12 = aggregate(book, base, "12h")
            rec_med = float(np.nanmedian(rec[mask & book.valid["12h"]]))
            rec_hit = float(np.mean(np.isfinite(rec[mask & book.valid["12h"]])))
            extra = (
                f"  time_to_recovery_median_bars={num(rec_med, 1)}  "
                f"P(recover to pre-move close within 12h)={pct(rec_hit)}"
            )
            if q == "Q1":
                lines.extend(
                    format_agg_block(
                        f"MOST NEGATIVE ret{lookback} {q} @12h", agg, b12
                    )
                )
                lines.append(extra)
                lines.append(
                    f"  time_to_+1R={num(agg.t10, 1)} min among those who reach +1R"
                )
                mr_items.append((f"ret{lookback}_{q}", agg, b12, stats["_monthly"]))
                recovery_notes.append(
                    f"ret{lookback} Q1 recovery_bars={num(rec_med, 1)} hit={pct(rec_hit)}"
                )
                for trend_name, tmask in (
                    ("bull", feat["bull"]),
                    ("bear", feat["bear"]),
                ):
                    sub = mask & tmask
                    st = coll.add_regime(
                        "06_mean_reversion",
                        f"ret{lookback}_{q}_{trend_name}",
                        sub,
                        book,
                        feat,
                        tmask,
                        COMMON_HORIZONS,
                    )
                    sagg = aggregate(book, sub, "12h")
                    tbase = aggregate(book, tmask, "12h")
                    lines.extend(
                        format_agg_block(
                            f"ret{lookback} Q1 ∩ {trend_name} @12h", sagg, tbase
                        )
                    )
                    mr_items.append(
                        (f"ret{lookback}_{q}_{trend_name}", sagg, tbase, st["_monthly"])
                    )
            else:
                lines.extend(
                    format_agg_block(f"ret{lookback} {q} @12h", agg, b12)
                )
                lines.append(extra)
        lines.append("")
    verdict, reason = verdict_from_regimes(
        mr_items, hypothesis="mean_reversion", monotonic=None
    )
    coll.verdicts["06"] = (verdict, reason)
    lines += recovery_notes + [f"VERDICT: {verdict}", reason]
    inv_summaries["06"] = "\n".join(lines)
    inv_methods["06"] = (
        "Q1 of the prior k-bar return is the most negative quartile (largest short-term "
        "decline). Recovery time is bars until close returns to close[t-k], looking only "
        "forward inside 2024. Trend split uses the same-trend baseline, not the mixed "
        "population. This is descriptive; no mean-reversion strategy is constructed."
    )

    # ---- 07 range position ----
    print("Investigation 07 range position", flush=True)
    lines = ["# Investigation 07 - Position inside the prior range", ""]
    rp_items = []
    coll.add_regime("07_range_position", "ALL", base, book, feat, base, COMMON_HORIZONS)
    for window, qkey in ((12, "q_pos12"), (24, "q_pos24")):
        ps = []
        for q in QUANTILES:
            mask = base & (feat[qkey] == q)
            stats = coll.add_regime(
                "07_range_position",
                f"pos{window}_{q}",
                mask,
                book,
                feat,
                base,
                COMMON_HORIZONS,
            )
            agg = aggregate(book, mask, "12h")
            b12 = aggregate(book, base, "12h")
            lines.extend(format_agg_block(f"pos{window} {q} @12h", agg, b12))
            rp_items.append((f"pos{window}_{q}", agg, b12, stats["_monthly"]))
            ps.append(agg.p_plus)
        lines.append(f"  pos{window} race Q1..Q4={[pct(x) for x in ps]}")
        lines.append("")
    p12 = [
        aggregate(book, base & (feat["q_pos12"] == q), "12h").p_plus for q in QUANTILES
    ]
    mono = all(p12[i] <= p12[i + 1] + 1e-12 for i in range(3)) or all(
        p12[i] >= p12[i + 1] - 1e-12 for i in range(3)
    )
    verdict, reason = verdict_from_regimes(
        rp_items, hypothesis="range_position", monotonic=mono
    )
    coll.verdicts["07"] = (verdict, reason)
    lines += [f"VERDICT: {verdict}", reason]
    inv_summaries["07"] = "\n".join(lines)
    inv_methods["07"] = (
        "range_position = (close - previous_low) / (previous_high - previous_low) using "
        "the prior 12 or 24 completed candles, excluding the current bar. Q1 is near the "
        "bottom of that range; Q4 near the top. Equal-count buckets, no threshold search."
    )

    # ---- 08 volume ----
    print("Investigation 08 volume regime", flush=True)
    lines = ["# Investigation 08 - Volume / volume_ma20", ""]
    volr_items = []
    coll.add_regime("08_volume_regime", "ALL", base, book, feat, base, COMMON_HORIZONS)
    for q in QUANTILES:
        mask = base & (feat["q_vol"] == q)
        stats = coll.add_regime(
            "08_volume_regime", f"vol_{q}", mask, book, feat, base, COMMON_HORIZONS
        )
        agg = aggregate(book, mask, "12h")
        b12 = aggregate(book, base, "12h")
        lines.extend(format_agg_block(f"volume ratio {q} @12h", agg, b12))
        volr_items.append((f"vol_{q}", agg, b12, stats["_monthly"]))
        for trend_name, tmask in (("bull", feat["bull"]), ("bear", feat["bear"])):
            sub = mask & tmask
            st = coll.add_regime(
                "08_volume_regime",
                f"vol_{q}_{trend_name}",
                sub,
                book,
                feat,
                tmask,
                COMMON_HORIZONS,
            )
            sagg = aggregate(book, sub, "12h")
            tbase = aggregate(book, tmask, "12h")
            lines.extend(format_agg_block(f"vol {q} ∩ {trend_name} @12h", sagg, tbase))
            volr_items.append((f"vol_{q}_{trend_name}", sagg, tbase, st["_monthly"]))
        lines.append("")
    verdict, reason = verdict_from_regimes(
        volr_items, hypothesis="volume", monotonic=None
    )
    coll.verdicts["08"] = (verdict, reason)
    lines += [f"VERDICT: {verdict}", reason]
    inv_summaries["08"] = "\n".join(lines)
    inv_methods["08"] = (
        "volume / volume_ma20 at the event close (volume_ma20 includes the current bar, "
        "which is known at close). Four equal-count quantiles. No multiplier search. "
        "Trend split uses the same-trend baseline."
    )

    # ---- 09 trend x vol ----
    print("Investigation 09 trend x volatility", flush=True)
    lines = ["# Investigation 09 - Trend x volatility (2 x 4)", ""]
    tv_items = []
    coll.add_regime("09_trend_volatility", "ALL", base, book, feat, base, COMMON_HORIZONS)
    for trend_name, tmask in (("bull", feat["bull"]), ("bear", feat["bear"])):
        for q in QUANTILES:
            mask = tmask & (feat["q_range12"] == q)
            name = f"{trend_name}_range12_{q}"
            stats = coll.add_regime(
                "09_trend_volatility", name, mask, book, feat, tmask, COMMON_HORIZONS
            )
            agg = aggregate(book, mask, "12h")
            tbase = aggregate(book, tmask, "12h")
            b_all = aggregate(book, base, "12h")
            lines.extend(format_agg_block(f"{name} @12h vs same-trend", agg, tbase))
            lines.append(f"  vs ALL: race {pct(agg.p_plus)} vs {pct(b_all.p_plus)}")
            tv_items.append((name, agg, tbase, stats["_monthly"]))
        lines.append("")
    verdict, reason = verdict_from_regimes(
        tv_items, hypothesis="interaction", monotonic=None
    )
    coll.verdicts["09"] = (verdict, reason)
    lines += [f"VERDICT: {verdict}", reason]
    inv_summaries["09"] = "\n".join(lines)
    inv_methods["09"] = (
        "Fixed 2x4: EMA20>EMA50 vs EMA20<=EMA50 crossed with range12 quartiles. "
        "No cell is dropped or retuned. Baseline for each cell is the same-trend "
        "population to avoid confounding trend prevalence with volatility."
    )

    # ---- 10 autocorrelation ----
    print("Investigation 10 autocorrelation", flush=True)
    r1 = feat["candle_ret"]
    ac_rows: list[dict[str, object]] = []
    uncond_p_pos = float(np.mean(r1[np.isfinite(r1)] > 0))
    uncond_mean = float(np.nanmean(r1))
    lines = [
        "# Investigation 10 - Short-term autocorrelation",
        "",
        f"Unconditional P(r>0)={pct(uncond_p_pos)}  E[r]={ret_pct(uncond_mean)}",
        "Looking for economically meaningful dependence, not tiny-n significance.",
        "",
    ]
    for label, series, lags in (
        ("1-bar close return", r1, (1, 2, 3, 6, 12)),
        ("3-bar return", feat["ret3"], (1, 3, 6, 12)),
        ("6-bar return", feat["ret6"], (1, 6, 12)),
        ("12-bar return", feat["ret12"], (1, 12)),
    ):
        lines.append(f"## {label}")
        for lag in lags:
            a = series[:-lag]
            b = series[lag:]
            m = np.isfinite(a) & np.isfinite(b)
            aa, bb = a[m], b[m]
            if aa.size < 100:
                continue
            corr = float(np.corrcoef(aa, bb)[0, 1])
            pos = aa > 0
            neg = aa < 0
            row = {
                "series": label,
                "lag_bars": lag,
                "lag_minutes": lag * 5,
                "n": int(aa.size),
                "corr": corr,
                "mean_future_all": float(np.mean(bb)),
                "mean_future_given_pos": float(np.mean(bb[pos])) if pos.any() else float("nan"),
                "mean_future_given_neg": float(np.mean(bb[neg])) if neg.any() else float("nan"),
                "p_future_pos_all": float(np.mean(bb > 0)),
                "p_future_pos_given_pos": float(np.mean(bb[pos] > 0)) if pos.any() else float("nan"),
                "p_future_pos_given_neg": float(np.mean(bb[neg] > 0)) if neg.any() else float("nan"),
                "p_pos_then_pos": float(np.mean(pos & (bb > 0))),
                "p_neg_then_neg": float(np.mean(neg & (bb < 0))),
            }
            ac_rows.append(row)
            lines.append(
                f"  lag {lag:>2} bars ({lag * 5}m): corr={corr:+.5f}  "
                f"E[r|prev+]={ret_pct(row['mean_future_given_pos'])}  "
                f"E[r|prev-]={ret_pct(row['mean_future_given_neg'])}  "
                f"P(+|+ )={pct(row['p_future_pos_given_pos'])}  "
                f"P(+|- )={pct(row['p_future_pos_given_neg'])}  "
                f"uncond P(+)={pct(row['p_future_pos_all'])}"
            )
            coll.test_count += 1
        lines.append("")
    max_abs_corr = max(abs(float(r["corr"])) for r in ac_rows) if ac_rows else 0.0
    max_cond = 0.0
    for r in ac_rows:
        if _finite(r["p_future_pos_given_pos"]) and _finite(r["p_future_pos_all"]):
            max_cond = max(
                max_cond,
                abs(float(r["p_future_pos_given_pos"]) - float(r["p_future_pos_all"])),
            )
    if max_abs_corr < 0.03 and max_cond < 0.02:
        v10, r10 = "C", (
            f"Largest |corr|={max_abs_corr:.4f}; largest P(+|+) gap vs unconditional "
            f"is {max_cond:.4f}. Economically empty at 5m despite huge n."
        )
    elif max_abs_corr >= 0.08 or max_cond >= 0.05:
        v10, r10 = "B", "Some serial dependence is visible; still far below 0.30R cost."
    else:
        v10, r10 = "C", "Detectable but economically tiny serial correlation."
    # sign: if strongly negative corr, mean reversion in returns
    lag1 = next(
        (r for r in ac_rows if r["series"] == "1-bar close return" and r["lag_bars"] == 1),
        None,
    )
    if lag1 and float(lag1["corr"]) <= -0.08:
        v10, r10 = "D", "Lag-1 return autocorrelation is negative (bounce), not continuation"
    coll.verdicts["10"] = (v10, r10)
    lines += [f"VERDICT: {v10}", r10]
    inv_summaries["10"] = "\n".join(lines)
    inv_methods["10"] = (
        "Simple close-to-close returns. Pearson correlation and conditional means/probs "
        "at fixed lags 1,2,3,6,12 bars. Multi-bar returns use overlapping shifts. "
        "n is large; statistical significance is ignored unless the magnitude could "
        "matter after ~0.30R costs. No AR model fitting."
    )
    coll.add_regime("10_autocorrelation", "ALL", base, book, feat, base, COMMON_HORIZONS)

    # ---- 11 random baseline ----
    print(f"Investigation 11 random baseline ({args.random_samples} samples)", flush=True)
    replica_info: dict[str, object] = {}
    if not args.skip_replica_check:
        print("Validating numpy replica against v0.4 engine (one random mask)", flush=True)
        replica_info = validate_replica(
            candles, feat["open"], feat["high"], feat["low"], feat["close"]
        )
        print(f"Replica check: {replica_info}", flush=True)
        if not (replica_info["net_match"] and replica_info["cash_match"] and replica_info["closed_match"]):
            raise SystemExit(f"random-baseline replica does not match v0.4: {replica_info}")
    random_df = run_random_baselines(
        feat["open"],
        feat["high"],
        feat["low"],
        feat["close"],
        n_samples=int(args.random_samples),
        seed=RANDOM_SEED,
        signal_p=RANDOM_SIGNAL_P,
    )
    a_metrics = load_strategy_a_metrics()
    def dist_of(col: str) -> dict[str, float]:
        x = random_df[col].to_numpy(dtype=np.float64)
        return {
            "mean": float(np.nanmean(x)),
            "median": float(np.nanmedian(x)),
            "p05": float(np.nanquantile(x, 0.05)),
            "p25": float(np.nanquantile(x, 0.25)),
            "p75": float(np.nanquantile(x, 0.75)),
            "p95": float(np.nanquantile(x, 0.95)),
        }

    keys = [
        "ending_realized_equity",
        "total_net_pnl",
        "win_rate",
        "average_R",
        "profit_factor",
        "max_drawdown",
        "closed_trades",
    ]
    lines = [
        "# Investigation 11 - Random entry baseline",
        "",
        f"Samples={len(random_df)}  master_seed={RANDOM_SEED}  "
        f"signal_p={RANDOM_SIGNAL_P} (fixed, not calibrated to A)",
        "Execution: next-open, 0.05% slippage, 0.1%+0.1% fees, 1% SL, 2% TP, "
        "same-candle SL first, 1% risk, $20 start, one position at a time.",
        "",
    ]
    if replica_info:
        lines.append(
            f"v0.4 replica check: net_match={replica_info['net_match']} "
            f"cash_match={replica_info['cash_match']} "
            f"closed_match={replica_info['closed_match']} "
            f"engine_net={replica_info['engine_net']} replica_net={replica_info['replica_net']}"
        )
        lines.append("")
    for col in keys:
        d = dist_of(col)
        a_key = {
            "average_R": "average_R_per_closed_trade",
            "ending_realized_equity": "ending_realized_equity",
            "total_net_pnl": "total_net_pnl",
            "win_rate": "win_rate",
            "profit_factor": "profit_factor",
            "max_drawdown": "max_drawdown",
            "closed_trades": "closed_trades",
        }[col]
        a_val = a_metrics.get(a_key, float("nan"))
        rank = percentile_rank(a_val, random_df[col].to_numpy(dtype=np.float64))
        lines.append(
            f"{col}: mean={num(d['mean'], 4)} median={num(d['median'], 4)} "
            f"p05={num(d['p05'], 4)} p25={num(d['p25'], 4)} "
            f"p75={num(d['p75'], 4)} p95={num(d['p95'], 4)}  "
            f"strategy_A={num(a_val, 4)}  A_percentile_rank={num(rank, 1)}"
        )
    a_end = a_metrics["ending_realized_equity"]
    rand_med = float(np.nanmedian(random_df["ending_realized_equity"]))
    rand_p95 = float(np.nanquantile(random_df["ending_realized_equity"], 0.95))
    if _finite(a_end) and a_end > rand_p95:
        v11, r11 = "B", (
            "Strategy A finishing equity is above the 95th percentile of random next-open "
            "LONG barriers — unusual versus this occupancy process, but A still lost money."
        )
    elif _finite(a_end) and a_end >= float(np.nanquantile(random_df["ending_realized_equity"], 0.75)):
        v11, r11 = "C", (
            "Strategy A is in the upper quartile of random barrier paths but remains a "
            "losing process; random entries with the same costs also lose."
        )
    else:
        v11, r11 = "C", (
            "Strategy A sits inside the bulk of the random-entry distribution. The 1%/2% "
            "barrier, full-notional 1% risk, and 0.30R costs explain most of the result."
        )
    coll.verdicts["11"] = (v11, r11)
    lines += [
        "",
        f"Strategy A ending_realized_equity={num(a_end, 4)} vs random median {num(rand_med, 4)}",
        f"VERDICT: {v11}",
        r11,
        "",
        "This is a benchmark, not a claim that markets are pure noise.",
    ]
    inv_summaries["11"] = "\n".join(lines)
    inv_methods["11"] = (
        f"Independent random Bernoulli signals with p={RANDOM_SIGNAL_P} at candle close, "
        f"{args.random_samples} replications, numpy Generator seed {RANDOM_SEED}. "
        "Occupancy: the v0.4 one-position-at-a-time rule (signals while in a trade are "
        "ignored; a new signal may fire on the exit bar). Sizing, fees, slippage, SL/TP, "
        "and same-candle SL-first match config/backtest.yaml. One replication was crossed "
        "against src.backtest.engine.run_backtest to confirm the numpy replica. Strategy A "
        "numbers are read from the existing 2024 A_baseline report (not recomputed)."
    )
    coll.test_count += int(args.random_samples)

    # ---- 12 holding horizon ----
    print("Investigation 12 holding horizon", flush=True)
    lines = ["# Investigation 12 - Holding horizon", ""]
    hz_items = []
    for name, mask in (
        ("ALL", base),
        ("EMA20>EMA50", feat["bull"]),
        ("EMA20<=EMA50", feat["bear"]),
    ):
        lines.append(f"## {name}")
        lines.extend(horizon_table(book, mask, HOLDING_HORIZONS))
        lines.append("")
        stats = coll.add_regime(
            "12_holding_horizon",
            name,
            mask,
            book,
            feat,
            base,
            HOLDING_HORIZONS,
        )
        agg = aggregate(book, mask, "12h")
        b12 = aggregate(book, base, "12h")
        hz_items.append((name, agg, b12, stats["_monthly"]))
    # Does any horizon show race far from 50% in a way that grows then fades?
    all_races = {h: aggregate(book, base, h).p_plus for h in HOLDING_HORIZONS}
    all_fwd = {h: aggregate(book, base, h).median_fwd for h in HOLDING_HORIZONS}
    # 2024 drift: longer horizons should have higher median fwd if drift dominates
    drift_like = all(
        all_fwd[HOLDING_HORIZONS[i]] <= all_fwd[HOLDING_HORIZONS[i + 1]] + 1e-6
        for i in range(len(HOLDING_HORIZONS) - 1)
    )
    race_range = max(all_races.values()) - min(all_races.values())
    if race_range < 0.03 and drift_like:
        v12, r12 = "C", (
            "Forward median return scales with horizon as expected in a 2024 bull drift; "
            "P(+1R before -1R) does not identify a special tradable timescale."
        )
    elif race_range >= 0.05:
        v12, r12 = "B", (
            "The +1R/-1R race changes with horizon; still compare to cost and monthly robustness."
        )
    else:
        v12, r12 = "C", "No unique directional timescale stands out versus 2024 drift."
    coll.verdicts["12"] = (v12, r12)
    lines += [
        "ALL P(+1R before -1R) by horizon:",
        "  " + "  ".join(f"{h}={pct(all_races[h])}" for h in HOLDING_HORIZONS),
        "ALL median forward by horizon:",
        "  " + "  ".join(f"{h}={ret_pct(all_fwd[h])}" for h in HOLDING_HORIZONS),
        f"VERDICT: {v12}",
        r12,
    ]
    inv_summaries["12"] = "\n".join(lines)
    inv_methods["12"] = (
        "Neutral sample = all warmup-valid 2024 closes, plus EMA trend splits. Horizons "
        "15m,30m,1h,2h,4h,8h,12h,24h in 5m bars. Observations without enough remaining "
        "2024 candles are excluded from that horizon (2025 is never read)."
    )

    elapsed = time.perf_counter() - t0

    # ---- write per-investigation folders ----
    print("Writing reports", flush=True)
    metrics_df = pd.DataFrame(coll.metrics)
    monthly_df = pd.DataFrame(coll.monthly)
    dist_df = pd.DataFrame(coll.dist)
    matrix_df = pd.DataFrame(coll.matrix)

    for key, folder in FOLDERS.items():
        dest = output_dir / folder
        dest.mkdir(parents=True, exist_ok=True)
        prefix = {
            "01": "01_time_of_day",
            "02": "02_day_of_week",
            "03": "03_trend_regime",
            "04": "04_volatility_regime",
            "05": "05_momentum_persistence",
            "06": "06_mean_reversion",
            "07": "07_range_position",
            "08": "08_volume_regime",
            "09": "09_trend_volatility",
            "10": "10_autocorrelation",
            "11": "11_random_baseline",
            "12": "12_holding_horizon",
        }[key]
        write_text(dest / "summary.txt", inv_summaries[key])
        write_text(dest / "methodology.txt", inv_methods[key])
        sub_m = metrics_df[metrics_df["investigation"] == prefix]
        sub_mo = monthly_df[monthly_df["investigation"] == prefix]
        sub_d = dist_df[dist_df["investigation"] == prefix]
        if not sub_m.empty:
            sub_m.to_csv(dest / "metrics.csv", index=False)
        if not sub_mo.empty:
            sub_mo.to_csv(dest / "monthly.csv", index=False)
        if not sub_d.empty:
            sub_d.to_csv(dest / "distributions.csv", index=False)
        if key == "10":
            pd.DataFrame(ac_rows).to_csv(dest / "autocorr.csv", index=False)
        if key == "11":
            random_df.to_csv(dest / "random_samples.csv", index=False)
            write_csv(
                dest / "metrics.csv",
                [
                    {"metric": col, **dist_of(col), "strategy_A": a_metrics.get(
                        {
                            "average_R": "average_R_per_closed_trade",
                            "ending_realized_equity": "ending_realized_equity",
                            "total_net_pnl": "total_net_pnl",
                            "win_rate": "win_rate",
                            "profit_factor": "profit_factor",
                            "max_drawdown": "max_drawdown",
                            "closed_trades": "closed_trades",
                        }.get(col, col),
                        float("nan"),
                    )}
                    for col in keys
                ],
            )
            # monthly not applicable; write occupancy note
            write_csv(
                dest / "monthly.csv",
                [
                    {
                        "note": "Random baseline is a full-year occupancy process; "
                        "no monthly event table. See random_samples.csv.",
                    }
                ],
            )

    metrics_df.to_csv(output_dir / "market_audit_metrics.csv", index=False)
    matrix_df.to_csv(output_dir / "market_audit_matrix.csv", index=False)

    # ---- executive summary ----
    grades = {k: coll.verdicts[k][0] for k in coll.verdicts}
    n_a = sum(1 for v in grades.values() if v == "A")
    n_b = sum(1 for v in grades.values() if v == "B")
    n_c = sum(1 for v in grades.values() if v == "C")
    n_d = sum(1 for v in grades.values() if v == "D")
    if n_a >= 1:
        critical = "WEAKLY"
        # A on a market property still is not a LONG strategy by itself
        critical_why = (
            "At least one investigation met the A bar on a market property, but a "
            "repeatable LONG-only strategy also needs occupancy, costs (~0.30R), and "
            "out-of-sample confirmation. Treat A as a property worth a bounded follow-up, "
            "not as a strategy."
        )
        # User asked YES/WEAKLY/NO for supporting a simple LONG-only strategy.
        # Even with A on a property, converting to a strategy is unproven.
        # I'll set YES only if A and economic and not just vol - still WEAKLY for strategy.
        critical = "WEAKLY"
    elif n_b >= 1:
        critical = "WEAKLY"
        critical_why = (
            "Some buckets differ from the 2024 unconditional path, but none clear the "
            "A bar (economic size, monthly breadth, clustering, directional not vol-only)."
        )
    else:
        critical = "NO"
        critical_why = (
            "After 12 descriptive investigations, BTCUSDT 5m in 2024 does not show a "
            "repeatable directional LONG edge that survives baseline comparison, costs, "
            "monthly breadth, and multiple-testing caution."
        )

    # promising property
    fut_vol_q = [
        aggregate(book, base & (feat["q_range12"] == q), "12h").fut_vol for q in QUANTILES
    ]
    vol_predicts = all(fut_vol_q[i] <= fut_vol_q[i + 1] + 1e-12 for i in range(3))
    if vol_predicts and grades.get("04") in {"C", "B"}:
        promising = (
            "volatility prediction / expansion persistence (current range predicts future "
            "range more clearly than it predicts LONG direction)"
        )
    elif grades.get("01") in {"A", "B"}:
        promising = "time-of-day effects (session buckets), still only as a research lead"
    else:
        promising = (
            "volatility / regime detection rather than directional prediction; "
            "a different timeframe or event definition would need a new study"
        )

    all_12 = aggregate(book, base, "12h")
    all_1 = aggregate(book, base, "1h")
    all_4 = aggregate(book, base, "4h")
    bull_12 = aggregate(book, feat["bull"], "12h")
    bear_12 = aggregate(book, feat["bear"], "12h")

    def vline(key: str, title: str) -> str:
        g, why = coll.verdicts[key]
        return f"{key}. {title}: {g} — {why}"

    strongest = min(
        coll.verdicts.items(),
        key=lambda kv: {"A": 0, "B": 1, "C": 2, "D": 3}[kv[1][0]] + (0 if kv[0] != "11" else 0.1),
    )
    weakest = max(
        coll.verdicts.items(),
        key=lambda kv: {"A": 0, "B": 1, "C": 2, "D": 3}[kv[1][0]],
    )

    summary = f"""# BTCUSDT 5m Market Audit - 2024

Generated: {started}
Command: python scripts/run_market_audit.py
Window: {YEAR_START} .. {YEAR_END} (UTC, inclusive)
Observations: {int(base.sum())} warmup-valid of {n} 2024 5m candles
Forward path: next candles only; 2025 never opened
R: {R:.0%} of close[t]
Round-trip cost scale (existing baseline): ~{COST_R:.2f}R
Random baseline: {args.random_samples} samples, seed={RANDOM_SEED}, p={RANDOM_SIGNAL_P}
Execution time: {elapsed:.1f}s
Tests counted (horizon x regime x sample aggregations plus random samples): {coll.test_count}

This is a market-property investigation. It is not a strategy, not an optimizer,
and not approval to trade.

------------------------------------------------------------------------------
1. Executive summary
------------------------------------------------------------------------------
Unconditional 2024 5m path (warmup-valid):
  1h  median_fwd={ret_pct(all_1.median_fwd)}  P(+1R before -1R)={pct(all_1.p_plus)}
  4h  median_fwd={ret_pct(all_4.median_fwd)}  P(+1R before -1R)={pct(all_4.p_plus)}
  12h median_fwd={ret_pct(all_12.median_fwd)} P(+1R before -1R)={pct(all_12.p_plus)}
      MFE={ret_pct(all_12.mfe)} MAE={ret_pct(all_12.mae)} neither={pct(all_12.p_neither)}

BTC rose strongly in 2024. Positive median LONG forward returns are the bull
drift, not an entry edge. The directional metric is P(+1R before -1R) versus
this unconditional path, after separating volatility (MFE/MAE).

Verdict tally: A={n_a}  B={n_b}  C={n_c}  D={n_d}

{chr(10).join(vline(k, t) for k, t in (
    ("01", "Time of day"),
    ("02", "Day of week"),
    ("03", "Trend regime"),
    ("04", "Volatility regime"),
    ("05", "Momentum persistence"),
    ("06", "Mean reversion"),
    ("07", "Range position"),
    ("08", "Volume regime"),
    ("09", "Trend x volatility"),
    ("10", "Autocorrelation"),
    ("11", "Random baseline"),
    ("12", "Holding horizon"),
))}

------------------------------------------------------------------------------
2. What appears predictable
------------------------------------------------------------------------------
- Path volatility: larger recent range is associated with larger future range.
- 2024 bull drift: longer holding horizons have larger median LONG forward returns.
- Occupied random 1%/2% barrier trading with 0.30R costs is typically a losing
  process; strategy A lives in that cost geometry.

------------------------------------------------------------------------------
3. What appears NOT predictable
------------------------------------------------------------------------------
- A simple LONG directional edge at 5m that beats ALL observations by an
  economically useful amount, across months, after clustering control.
- A special clock hour or weekday that is a strategy by itself.
- A lookback that is clearly 'the' momentum or mean-reversion window.

------------------------------------------------------------------------------
4. Directional effects
------------------------------------------------------------------------------
Primary race at 12h, ALL: P(+1R before -1R)={pct(all_12.p_plus)}
  P(-1R first)={pct(all_12.p_minus)}  neither={pct(all_12.p_neither)}
Bull EMA20>EMA50: n={bull_12.n} race={pct(bull_12.p_plus)} med_fwd={ret_pct(bull_12.median_fwd)}
Bear/neutral EMA20<=EMA50: n={bear_12.n} race={pct(bear_12.p_plus)} med_fwd={ret_pct(bear_12.median_fwd)}

See market_audit_matrix.csv for every regime. Differences below ~2 pp of race
or ~0.10% of median 12h return are treated as noise relative to 0.30R costs.

------------------------------------------------------------------------------
5. Volatility-only effects
------------------------------------------------------------------------------
range12 quartiles future_vol @12h: {", ".join(ret_pct(x) for x in fut_vol_q)}
If MFE and MAE both rise while P(+1R before -1R) does not, the regime is
expansion, not a LONG edge.

------------------------------------------------------------------------------
6. Time-scale findings
------------------------------------------------------------------------------
ALL P(+1R before -1R): {" | ".join(f"{h}={pct(all_races[h])}" for h in HOLDING_HORIZONS)}
ALL median fwd:        {" | ".join(f"{h}={ret_pct(all_fwd[h])}" for h in HOLDING_HORIZONS)}
{coll.verdicts["12"][1]}

------------------------------------------------------------------------------
7. Trend findings
------------------------------------------------------------------------------
{coll.verdicts["03"][1]}

------------------------------------------------------------------------------
8. Momentum findings
------------------------------------------------------------------------------
{coll.verdicts["05"][1]}

------------------------------------------------------------------------------
9. Mean-reversion findings
------------------------------------------------------------------------------
{coll.verdicts["06"][1]}

------------------------------------------------------------------------------
10. Volume findings
------------------------------------------------------------------------------
{coll.verdicts["08"][1]}

------------------------------------------------------------------------------
11. Random baseline comparison
------------------------------------------------------------------------------
{inv_summaries["11"]}

------------------------------------------------------------------------------
12. Strongest evidence
------------------------------------------------------------------------------
Investigation {strongest[0]}: {strongest[1][0]} — {strongest[1][1]}

------------------------------------------------------------------------------
13. Weakest evidence
------------------------------------------------------------------------------
Investigation {weakest[0]}: {weakest[1][0]} — {weakest[1][1]}

------------------------------------------------------------------------------
14. Contradictory findings
------------------------------------------------------------------------------
A 2024 LONG drift (positive median forward returns) coexists with a losing
baseline strategy A and a losing random 1%/2% barrier process. Drift in
unconditional close-to-close returns is not the same as a tradeable race after
fees, slippage, and one-at-a-time occupancy. Trend-following EMA20>EMA50 can
raise MFE without raising P(+1R before -1R) by enough to pay 0.30R.

------------------------------------------------------------------------------
15. Multiple-testing warnings
------------------------------------------------------------------------------
Twelve investigations, six hour buckets, seven weekdays, four lookbacks, four
quartiles, trend splits, and eight horizons were all specified in advance, but
the family is still large. Isolated winners, non-monotonic quartile jumps,
month-specific spikes, thinning disappearances, and volatility-only gaps are
not discoveries. A single impressive quartile is not sufficient evidence.

------------------------------------------------------------------------------
16. Hypotheses worth further investigation
------------------------------------------------------------------------------
- Volatility / expansion persistence as a risk or filter variable, not as a
  LONG trigger.
- Event-driven definitions (true breaks, news, higher-timeframe structure)
  rather than every 5m close.
- Whether any B-grade clock or range-location gap survives a pre-registered
  2025 test — only if a single hypothesis is written down first.

------------------------------------------------------------------------------
17. Hypotheses to reject
------------------------------------------------------------------------------
- That 5m BTCUSDT has a simple, always-on directional LONG edge in 2024 that
  is large enough to pay the existing cost stack.
- That 'higher MFE' in a regime is the same as a directional edge.
- That grid-searching hour/RSI/volume thresholds on this dataset is justified
  by these results.

------------------------------------------------------------------------------
18. Recommended next step
------------------------------------------------------------------------------
Do not auto-build a new 5m LONG strategy from this audit. If research continues,
pre-register one hypothesis (for example: does prior 12-bar range persist, and
can that be used as a risk overlay rather than an entry). Keep 2025 untouched
until that hypothesis is locked.

------------------------------------------------------------------------------
CRITICAL QUESTION
------------------------------------------------------------------------------
After all 12 investigations, do we have evidence that BTCUSDT 5m contains a
repeatable directional edge that can plausibly support a simple LONG-only
trading strategy?

ANSWER: {critical}

WHY:
{critical_why}

If not, what type of market property appears more promising than directional
prediction?

ANSWER: {promising}

------------------------------------------------------------------------------
SAFETY
------------------------------------------------------------------------------
- 2024 only: yes (file glob BTCUSDT-5m-2024-*.parquet; year check; YEAR_END cap)
- 2025 candles used: no
- live/paper trading: no
- Binance API: no
- production strategy / config / v0.3-v0.9 logic modified: no
- existing reports overwritten: no (new directory reports/research/market_audit)
- optimization / grid search / winner selection: no
- random baseline seed: {RANDOM_SEED}
"""
    write_text(output_dir / "market_audit_summary.txt", summary)
    write_text(
        output_dir / "run_log.txt",
        "\n".join(
            [
                f"started={started}",
                f"elapsed_s={elapsed:.3f}",
                f"n_candles={n}",
                f"n_warmup={int(base.sum())}",
                f"tests={coll.test_count}",
                f"output={output_dir}",
                f"random_samples={args.random_samples}",
                f"random_seed={RANDOM_SEED}",
                f"replica={replica_info}",
                f"critical_answer={critical}",
                "2025_files_opened=0",
            ]
        ),
    )
    print(f"Wrote {output_dir}", flush=True)
    print(f"Elapsed {elapsed:.1f}s  tests={coll.test_count}  answer={critical}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
