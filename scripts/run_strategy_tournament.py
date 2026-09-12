"""CLI for the 2024 BTCUSDT strategy tournament (research only).

Reuses v0.3 generate_signals for BASELINE_A and the v0.4 backtest engine for
every candidate. Does not download data, call Binance, optimize parameters,
touch 2025 candles, or overwrite existing reports outside the output folder.

Usage:
    python scripts/run_strategy_tournament.py
"""

from __future__ import annotations

import argparse
import subprocess
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

from src.backtest.config import load_backtest_config  # noqa: E402
from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.ledger import TRADE_FRAME_COLUMNS  # noqa: E402
from src.indicators.engine import load_indicator_config  # noqa: E402
from src.indicators.pipeline import enriched_dataset_dir  # noqa: E402
from src.research.aggregation import monthly_table  # noqa: E402
from src.research.equity import reconstruct_equity  # noqa: E402
from src.research.metrics import compute_metrics  # noqa: E402
from src.strategy.engine import generate_signals, load_strategy_config  # noqa: E402
from src.data.pipeline import load_config  # noqa: E402

from strategy_tournament_engine import (  # noqa: E402
    DEFINITIONS,
    DEV_END,
    DEV_START,
    G04_AUDIT_BUCKET,
    RANDOM_SEED,
    RANDOM_SIGNAL_P,
    STARTING_CAPITAL,
    STRATEGY_ORDER,
    SYMBOL,
    TIMEFRAMES,
    VAL_END,
    VAL_START,
    TournamentError,
    _num,
    assign_groups,
    build_all_signal_frames,
    build_features,
    assert_feature_no_lookahead,
    build_timeframes,
    evaluate_pair,
    freeze_thresholds,
    load_2024_enriched,
    metrics_to_rows,
    monthly_consistency_label,
    refuse_2025,
    select_oos_candidates,
    RankedRow,
    run_window,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "research" / "strategy_tournament"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "config" / "data.yaml"
DEFAULT_INDICATOR_CONFIG = PROJECT_ROOT / "config" / "indicators.yaml"
DEFAULT_STRATEGY_CONFIG = PROJECT_ROOT / "config" / "strategy.yaml"
DEFAULT_BACKTEST_CONFIG = PROJECT_ROOT / "config" / "backtest.yaml"
A_METRICS_PATH = (
    PROJECT_ROOT / "reports" / "research" / "development" / "A_baseline" / "metrics.csv"
)
PROTECTED_REPORT_DIRS = (
    PROJECT_ROOT / "reports" / "research" / "development",
    PROJECT_ROOT / "reports" / "research" / "market_audit",
    PROJECT_ROOT / "reports" / "research" / "test",
    PROJECT_ROOT / "reports" / "research" / "integrity",
)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2
_RULE = "-" * 78
NA = "n/a"


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fmt(value: object) -> str:
    if value is None:
        return NA
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return NA
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    return str(value)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def assert_safe_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    for protected in PROTECTED_REPORT_DIRS:
        if resolved == protected.resolve():
            raise TournamentError(f"refusing to write into protected report dir {protected}")
        try:
            resolved.relative_to(protected.resolve())
        except ValueError:
            continue
        raise TournamentError(f"refusing to overwrite protected reports under {protected}")


def load_baseline_a_full_year_net() -> tuple[float, int] | None:
    if not A_METRICS_PATH.is_file():
        return None
    frame = pd.read_csv(A_METRICS_PATH)
    values = {str(row["metric"]): row["value"] for _, row in frame.iterrows()}
    try:
        net = float(values["total_net_pnl"])
        closed = int(float(values["closed_trades"]))
    except (KeyError, TypeError, ValueError):
        return None
    return net, closed


def integrity_baseline_full_year(
    candles_5m: pd.DataFrame,
    backtest_config,
    strategy_config,
    indicator_config,
) -> str:
    expected = load_baseline_a_full_year_net()
    signals = generate_signals(
        candles_5m,
        strategy_config,
        indicator_config,
        interval="5m",
        symbol=SYMBOL,
    )
    needed = candles_5m[["timestamp", "open", "high", "low", "close"]]
    result = run_backtest(
        needed,
        signals[["timestamp", "signal"]],
        backtest_config,
        symbol=SYMBOL,
        interval="5m",
    )
    equity = reconstruct_equity(candles_5m, result.trades, backtest_config.starting_capital)
    monthly = monthly_table(result.trades, equity)
    metrics = compute_metrics(result, equity, monthly=monthly)
    if expected is None:
        return (
            f"BASELINE_A full-2024 replay net={metrics.total_net_pnl:.10g} "
            f"closed={metrics.closed_trades}; no frozen A_baseline metrics.csv to compare"
        )
    exp_net, exp_closed = expected
    net_ok = np.isclose(metrics.total_net_pnl, exp_net, rtol=0.0, atol=1e-8)
    closed_ok = metrics.closed_trades == exp_closed
    status = "MATCH" if net_ok and closed_ok else "MISMATCH"
    return (
        f"BASELINE_A full-2024 integrity {status}: "
        f"replay net={metrics.total_net_pnl:.10g} closed={metrics.closed_trades} vs "
        f"frozen net={exp_net:.10g} closed={exp_closed}"
    )


def flatten_metrics(row: RankedRow, which: str) -> dict[str, object]:
    artifacts = row.development if which == "development" else row.validation
    m = artifacts.metrics
    return {
        "closed_trades": m.closed_trades,
        "wins": m.win_count,
        "losses": m.loss_count,
        "win_rate": m.win_rate,
        "gross_pnl": m.total_gross_pnl,
        "fees": m.total_fees,
        "slippage": m.total_slippage,
        "net_pnl": m.total_net_pnl,
        "avg_net_pnl": m.average_net_pnl_per_closed_trade,
        "avg_R": m.average_R_per_closed_trade,
        "profit_factor": m.profit_factor,
        "max_drawdown": m.max_drawdown,
        "max_drawdown_pct": m.max_drawdown_pct,
        "final_equity": m.ending_realized_equity,
        "median_holding_time": m.median_holding_time,
        "average_holding_time": m.average_holding_time,
        "longest_winning_streak": m.max_consecutive_wins,
        "longest_losing_streak": m.max_consecutive_losses,
        "signals": m.signals,
        "filled_trades": m.filled_trades,
        "tp_count": m.tp_count,
        "sl_count": m.sl_count,
        "months_with_trades": m.months_with_closed_trades,
        "largest_win_net_pnl": m.largest_win_net_pnl,
        "sum_of_winning_net_pnl": m.sum_of_winning_net_pnl,
        "max_month_share_of_net": m.max_month_share_of_net,
    }


def matrix_row(row: RankedRow) -> dict[str, object]:
    dev = flatten_metrics(row, "development")
    val = flatten_metrics(row, "validation")
    out: dict[str, object] = {
        "strategy_id": row.strategy_id,
        "family": row.family,
        "timeframe": row.timeframe,
        "group": row.group,
        "robustness_score": row.score,
        "definition": row.definition,
        "gate_failures": "|".join(row.gate_failures) if row.gate_failures else "",
        "flags": "|".join(row.flags) if row.flags else "",
        "val_dev_avg_R_ratio": row.r_ratio,
        "val_dev_pf_ratio": row.pf_ratio,
        "val_dev_win_rate_ratio": row.wr_ratio,
        "val_dev_trade_count_ratio": row.freq_ratio,
        "val_dev_dd_pct_ratio": row.dd_ratio,
        "beat_BASELINE_A_val_avg_R": row.beat_baseline_avg_r,
        "beat_BASELINE_A_val_net": row.beat_baseline_net,
        "beat_RANDOM_val_avg_R": row.beat_random_avg_r,
        "beat_RANDOM_val_net": row.beat_random_net,
        "monthly_consistency_dev": monthly_consistency_label(row.development.monthly),
        "monthly_consistency_val": monthly_consistency_label(row.validation.monthly),
    }
    if row.thresholds is not None:
        out.update(
            {
                "dev_range12_median": row.thresholds.range12_median,
                "dev_range12_q1": row.thresholds.range12_q1,
                "dev_range12_q3": row.thresholds.range12_q3,
                "dev_range24_median": row.thresholds.range24_median,
            }
        )
    for key, value in dev.items():
        out[f"dev_{key}"] = value
    for key, value in val.items():
        out[f"val_{key}"] = value
    return out


def write_candidate_folder(dest: Path, rows: list[RankedRow]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    val_metric_rows: list[dict[str, object]] = []
    monthly_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    lines = [
        f"# {rows[0].strategy_id} - {rows[0].family}",
        "",
        f"Definition: {rows[0].definition}",
        "Execution: v0.4 engine, next-open, 1% SL, 2% TP, 0.05% slippage, 0.1%+0.1% fees,",
        f"1% risk, ${STARTING_CAPITAL:.0f} start, LONG-only, one position, no leverage.",
        "Development and validation are independent $20 replays. Parameters frozen.",
        "",
    ]
    for row in rows:
        metric_rows.extend(
            metrics_to_rows(
                row.development.metrics,
                timeframe=row.timeframe,
                period_role="DEVELOPMENT",
            )
        )
        val_metric_rows.extend(
            metrics_to_rows(
                row.validation.metrics,
                timeframe=row.timeframe,
                period_role="VALIDATION",
            )
        )
        for artifacts, role in (
            (row.development, "DEVELOPMENT"),
            (row.validation, "VALIDATION"),
        ):
            monthly = artifacts.monthly.copy()
            monthly.insert(0, "period_role", role)
            monthly.insert(0, "timeframe", row.timeframe)
            monthly_parts.append(monthly)
            trades = artifacts.trades.copy()
            trades.insert(0, "period_role", role)
            trades.insert(0, "timeframe", row.timeframe)
            trade_parts.append(trades)
        lines.extend(
            [
                f"## {row.timeframe}",
                _RULE,
                f"group:               {row.group}",
                f"robustness_score:    {fmt(row.score)}",
                f"flags:               {', '.join(row.flags) if row.flags else '(none)'}",
                f"gate_failures:       {', '.join(row.gate_failures) if row.gate_failures else '(none)'}",
                f"beat BASELINE_A val avg R / net: {row.beat_baseline_avg_r} / {row.beat_baseline_net}",
                f"beat RANDOM val avg R / net:     {row.beat_random_avg_r} / {row.beat_random_net}",
                f"val/dev avg R ratio: {fmt(row.r_ratio)}",
                f"val/dev PF ratio:    {fmt(row.pf_ratio)}",
                "",
                "DEVELOPMENT 2024-01-01 .. 2024-08-31",
                f"  closed={row.development.metrics.closed_trades}  "
                f"wins={row.development.metrics.win_count}  "
                f"losses={row.development.metrics.loss_count}  "
                f"win_rate={fmt(row.development.metrics.win_rate)}",
                f"  gross={fmt(row.development.metrics.total_gross_pnl)}  "
                f"fees={fmt(row.development.metrics.total_fees)}  "
                f"slip={fmt(row.development.metrics.total_slippage)}  "
                f"net={fmt(row.development.metrics.total_net_pnl)}",
                f"  avg_net={fmt(row.development.metrics.average_net_pnl_per_closed_trade)}  "
                f"avg_R={fmt(row.development.metrics.average_R_per_closed_trade)}  "
                f"PF={fmt(row.development.metrics.profit_factor)}",
                f"  DD$={fmt(row.development.metrics.max_drawdown)}  "
                f"DD%={fmt(row.development.metrics.max_drawdown_pct)}  "
                f"final_equity={fmt(row.development.metrics.ending_realized_equity)}",
                f"  median_hold={fmt(row.development.metrics.median_holding_time)}  "
                f"avg_hold={fmt(row.development.metrics.average_holding_time)}",
                f"  win_streak={row.development.metrics.max_consecutive_wins}  "
                f"loss_streak={row.development.metrics.max_consecutive_losses}",
                f"  {monthly_consistency_label(row.development.monthly)}",
                "",
                "VALIDATION 2024-09-01 .. 2024-12-31",
                f"  closed={row.validation.metrics.closed_trades}  "
                f"wins={row.validation.metrics.win_count}  "
                f"losses={row.validation.metrics.loss_count}  "
                f"win_rate={fmt(row.validation.metrics.win_rate)}",
                f"  gross={fmt(row.validation.metrics.total_gross_pnl)}  "
                f"fees={fmt(row.validation.metrics.total_fees)}  "
                f"slip={fmt(row.validation.metrics.total_slippage)}  "
                f"net={fmt(row.validation.metrics.total_net_pnl)}",
                f"  avg_net={fmt(row.validation.metrics.average_net_pnl_per_closed_trade)}  "
                f"avg_R={fmt(row.validation.metrics.average_R_per_closed_trade)}  "
                f"PF={fmt(row.validation.metrics.profit_factor)}",
                f"  DD$={fmt(row.validation.metrics.max_drawdown)}  "
                f"DD%={fmt(row.validation.metrics.max_drawdown_pct)}  "
                f"final_equity={fmt(row.validation.metrics.ending_realized_equity)}",
                f"  median_hold={fmt(row.validation.metrics.median_holding_time)}  "
                f"avg_hold={fmt(row.validation.metrics.average_holding_time)}",
                f"  win_streak={row.validation.metrics.max_consecutive_wins}  "
                f"loss_streak={row.validation.metrics.max_consecutive_losses}",
                f"  {monthly_consistency_label(row.validation.monthly)}",
                "",
            ]
        )
        if row.notes:
            lines.append("Notes: " + "; ".join(row.notes))
            lines.append("")
    pd.DataFrame(metric_rows).to_csv(dest / "development_metrics.csv", index=False)
    pd.DataFrame(val_metric_rows).to_csv(dest / "validation_metrics.csv", index=False)
    if monthly_parts:
        pd.concat(monthly_parts, ignore_index=True).to_csv(dest / "monthly.csv", index=False)
    else:
        pd.DataFrame().to_csv(dest / "monthly.csv", index=False)
    if trade_parts:
        trades = pd.concat(trade_parts, ignore_index=True)
        extra = [col for col in ("timeframe", "period_role") if col in trades.columns]
        ordered = extra + [col for col in TRADE_FRAME_COLUMNS if col in trades.columns]
        if "holding_time" in trades.columns:
            trades["holding_time"] = trades["holding_time"].map(
                lambda value: "" if value is None or pd.isna(value) else str(value)
            )
        trades.to_csv(dest / "trades.csv", index=False, columns=ordered)
    else:
        pd.DataFrame().to_csv(dest / "trades.csv", index=False)
    write_text(dest / "summary.txt", "\n".join(lines))


def family_stats(rows: list[RankedRow]) -> list[tuple[str, int, int, float | None]]:
    families = []
    seen = []
    for row in rows:
        if row.family not in seen:
            seen.append(row.family)
    for family in seen:
        items = [row for row in rows if row.family == family]
        survived = [row for row in items if row.group in {"GROUP 1", "GROUP 2"}]
        val_r = [
            _num(row.validation.metrics.average_R_per_closed_trade)
            for row in items
        ]
        finite = [value for value in val_r if value is not None]
        mean_r = float(np.mean(finite)) if finite else None
        families.append((family, len(survived), len(items), mean_r))
    families.sort(key=lambda item: (-item[1], -(item[3] or -999.0)))
    return families


def timeframe_stats(rows: list[RankedRow]) -> list[tuple[str, int, int, float | None]]:
    out = []
    for timeframe in TIMEFRAMES:
        items = [row for row in rows if row.timeframe == timeframe]
        survived = [row for row in items if row.group in {"GROUP 1", "GROUP 2"}]
        val_r = [
            _num(row.validation.metrics.average_R_per_closed_trade)
            for row in items
            if row.strategy_id not in {"RANDOM_BASELINE"}
        ]
        finite = [value for value in val_r if value is not None]
        mean_r = float(np.mean(finite)) if finite else None
        out.append((timeframe, len(survived), len(items), mean_r))
    return out


def render_summary(
    rows: list[RankedRow],
    *,
    command: str,
    elapsed_s: float,
    commit: str,
    n_backtests: int,
    n_strategies: int,
    n_timeframes: int,
    integrity: str,
    warnings: list[str],
    oos: list[RankedRow],
) -> str:
    group1 = [row for row in rows if row.group == "GROUP 1"]
    group2 = [row for row in rows if row.group == "GROUP 2"]
    group3 = [row for row in rows if row.group == "GROUP 3"]
    group4 = [row for row in rows if row.group == "GROUP 4"]
    survived_dev = [
        row
        for row in rows
        if row.development.metrics.closed_trades >= 30
        and (_num(row.development.metrics.average_R_per_closed_trade) or 0) > 0
        and float(row.development.metrics.total_net_pnl) > 0
        and (
            _num(row.development.metrics.max_drawdown_pct) is None
            or float(row.development.metrics.max_drawdown_pct) <= 0.60
        )
    ]
    survived_val = [
        row
        for row in survived_dev
        if row.validation.metrics.closed_trades >= 8
        and (_num(row.validation.metrics.average_R_per_closed_trade) or 0) > 0
        and float(row.validation.metrics.total_net_pnl) > 0
        and (
            _num(row.validation.metrics.max_drawdown_pct) is None
            or float(row.validation.metrics.max_drawdown_pct) <= 0.60
        )
        and "validation_catastrophically_worse" not in row.gate_failures
    ]
    beat_base = [
        row
        for row in rows
        if row.strategy_id not in {"BASELINE_A", "RANDOM_BASELINE"}
        and row.beat_baseline_avg_r
        and row.beat_baseline_net
    ]
    beat_rand = [
        row
        for row in rows
        if row.strategy_id not in {"BASELINE_A", "RANDOM_BASELINE"}
        and row.beat_random_avg_r
        and row.beat_random_net
    ]
    suspicious = [
        row
        for row in rows
        if "OVERFIT / UNSTABLE" in row.flags
        or "VAL_ONE_BIG_WIN" in row.flags
        or "VAL_ONE_MONTH_CONCENTRATION" in row.flags
        or "DEV_ONE_BIG_WIN" in row.flags
        or "DEV_ONE_MONTH_CONCENTRATION" in row.flags
    ]
    families = family_stats(rows)
    tfs = timeframe_stats(rows)
    best_tf = max(tfs, key=lambda item: (item[1], item[3] or -999.0))
    abandoned = [
        family
        for family, survived, _total, _mean in families
        if survived == 0 and family not in {"BASELINE", "RANDOM"}
    ]
    both_periods = [
        row
        for row in rows
        if row.group in {"GROUP 1", "GROUP 2"}
    ]

    viable = "NO"
    if group1:
        strong = [
            row
            for row in group1
            if row.beat_baseline_avg_r
            and row.beat_random_avg_r
            and (_num(row.validation.metrics.profit_factor) or 0) > 1.05
            and "VAL_ONE_BIG_WIN" not in row.flags
        ]
        viable = "PROMISING BUT UNPROVEN"
        if strong:
            top = strong[0]
            val_r = _num(top.validation.metrics.average_R_per_closed_trade) or 0.0
            if (
                val_r >= 0.05
                and (_num(top.validation.metrics.profit_factor) or 0) > 1.10
                and top.validation.metrics.closed_trades >= 20
            ):
                viable = "YES"

    directional = (
        "Some candidates posted positive development AND validation average R "
        "on this split, but 2024 is a bull year and 2025 is untouched. "
        "A repeatable directional edge is not established."
        if both_periods
        else (
            "No candidate produced positive average R and net PnL in both "
            "development and validation after costs. This does not show a "
            "tradeable directional edge on this split."
        )
    )
    if viable == "YES":
        directional = (
            "At least one structurally simple LONG rule was positive in both "
            "2024 windows after the v0.4 cost stack and beat both comparators. "
            "That is still only a 2024 split result. 2025 has not been opened."
        )
    elif viable == "PROMISING BUT UNPROVEN":
        directional = (
            "A small number of families survived both 2024 windows with positive "
            "average R after costs. The edge is small relative to occupancy and "
            "fees, and 2025 has not been used. Treat as a hypothesis, not an edge."
        )

    tf5 = next(item for item in tfs if item[0] == "5m")
    five_still_best = best_tf[0] == "5m"

    lines = [
        "# BTCUSDT strategy tournament - 2024 development / validation",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Command: {command}",
        f"Git commit: {commit}",
        f"Execution time: {elapsed_s:.1f}s",
        f"Dataset: BTCUSDT spot 5m 2024-01-01 .. 2024-12-31 UTC (2025 files never opened)",
        f"Resampled: 15m and 1h from 2024 5m only",
        f"Strategies: {n_strategies}  Timeframes: {n_timeframes}  Backtests: {n_backtests}",
        f"Random seed: {RANDOM_SEED}  random p={RANDOM_SIGNAL_P}",
        f"Starting capital: ${STARTING_CAPITAL:.0f}  risk 1%  slip 0.05%  fees 0.1%+0.1%",
        f"G04 time bucket (frozen market audit): {G04_AUDIT_BUCKET} UTC",
        "",
        "This is research. It is not live trading, paper trading, or approval to trade.",
        "Parameters were predetermined. No grid search. No 2025 selection.",
        "",
        _RULE,
        "Reproducibility / safety",
        _RULE,
        integrity,
        "2025 data used for selection: NO",
        "Live / paper / Binance API: NO",
        "Production / executor / v0.3-v0.9 code modified: NO",
        "Existing reports overwritten: NO (wrote only reports/research/strategy_tournament/)",
        "Parameter grid search / optimization: NO",
        "Deterministic definitions: YES (plus one seeded random occupancy path)",
        "",
        _RULE,
        "Predetermined definitions (not optimized)",
        _RULE,
        "bullish candle: close > open",
        "bullish reversal: prior close < prior open AND current close > current open",
        "N-bar breakout: close > max(high of previous N bars); current high excluded",
        "pullback to EMA20: prior close > prior EMA20, current low <= EMA20, close >= EMA20",
        "near bottom / top of 24-bar range: position <= 0.25 / >= 0.75",
        "large negative 6-bar return: <= -1.0%",
        "low range: previous 12- or 24-bar range < DEVELOPMENT median (frozen per timeframe)",
        "high volatility: previous 12-bar range >= DEVELOPMENT Q3 (frozen per timeframe)",
        "E03 expansion: current high-low > median of previous 20 candle ranges",
        "F04 swing: previous confirmed 3-bar pivot high, known at confirmation bar",
        f"G04 bucket: {G04_AUDIT_BUCKET} from market audit 01_time_of_day (highest 12h race delta vs ALL)",
        "Independent $20 windows: development Jan-Aug; validation Sep-Dec; no equity carry",
        "",
        _RULE,
        "Counts",
        _RULE,
        f"Survived development filters: {len(survived_dev)}",
        f"Survived development AND validation filters: {len(survived_val)}",
        f"GROUP 1 strong: {len(group1)}",
        f"GROUP 2 interesting but weak: {len(group2)}",
        f"GROUP 3 failed validation / development filters: {len(group3)}",
        f"GROUP 4 insufficient sample: {len(group4)}",
        "",
        _RULE,
        "1. Which strategy families worked best?",
        _RULE,
    ]
    if not families:
        lines.append("No families produced results.")
    else:
        for family, survived, total, mean_r in families:
            lines.append(
                f"{family}: {survived}/{total} (strategy x timeframe) in GROUP 1-2; "
                f"mean validation avg R={fmt(mean_r)}"
            )
    lines.extend(["", _RULE, "2. Which timeframes worked best?", _RULE])
    for timeframe, survived, total, mean_r in tfs:
        lines.append(
            f"{timeframe}: {survived}/{total} in GROUP 1-2; "
            f"mean validation avg R (ex-random)={fmt(mean_r)}"
        )
    lines.append(f"Best by survivor count then mean val avg R: {best_tf[0]}")
    lines.extend(
        [
            "",
            _RULE,
            "3. Did any candidate beat BASELINE_A?",
            _RULE,
        ]
    )
    if beat_base:
        lines.append(
            "YES, on validation average R AND net PnL (same timeframe): "
            + ", ".join(f"{row.strategy_id}/{row.timeframe}" for row in beat_base[:20])
        )
        if len(beat_base) > 20:
            lines.append(f"... {len(beat_base)} total")
    else:
        lines.append(
            "NO candidate beat BASELINE_A on both validation average R and validation net PnL."
        )
        any_r = [
            row
            for row in rows
            if row.strategy_id not in {"BASELINE_A", "RANDOM_BASELINE"}
            and row.beat_baseline_avg_r
        ]
        if any_r:
            lines.append(
                "Higher validation avg R only: "
                + ", ".join(f"{row.strategy_id}/{row.timeframe}" for row in any_r[:15])
            )
    lines.extend(
        [
            "",
            _RULE,
            "4. Did any candidate beat RANDOM_BASELINE?",
            _RULE,
        ]
    )
    if beat_rand:
        lines.append(
            "YES, on validation average R AND net PnL (same timeframe): "
            + ", ".join(f"{row.strategy_id}/{row.timeframe}" for row in beat_rand[:20])
        )
        if len(beat_rand) > 20:
            lines.append(f"... {len(beat_rand)} total")
    else:
        lines.append(
            "NO candidate beat RANDOM_BASELINE on both validation average R and net PnL."
        )
    lines.extend(
        [
            "",
            _RULE,
            "5. Which candidates survived both development and validation?",
            _RULE,
        ]
    )
    if both_periods:
        for row in both_periods:
            lines.append(
                f"{row.group} {row.strategy_id}/{row.timeframe}: "
                f"dev net={fmt(row.development.metrics.total_net_pnl)} "
                f"avgR={fmt(row.development.metrics.average_R_per_closed_trade)} "
                f"n={row.development.metrics.closed_trades}; "
                f"val net={fmt(row.validation.metrics.total_net_pnl)} "
                f"avgR={fmt(row.validation.metrics.average_R_per_closed_trade)} "
                f"PF={fmt(row.validation.metrics.profit_factor)} "
                f"DD%={fmt(row.validation.metrics.max_drawdown_pct)} "
                f"n={row.validation.metrics.closed_trades}"
            )
    else:
        lines.append("None.")
    lines.extend(["", _RULE, "6. Which results look suspicious?", _RULE])
    if suspicious:
        for row in suspicious:
            lines.append(
                f"{row.strategy_id}/{row.timeframe}: {', '.join(row.flags) or '(flags)'} "
                f"failures={', '.join(row.gate_failures) or '(none)'}"
            )
    else:
        lines.append("No concentration or OVERFIT flags fired.")
    lines.extend(
        [
            "",
            _RULE,
            "7. Is there evidence of a real directional edge?",
            _RULE,
            directional,
            "",
            _RULE,
            "8. Is 5m still the best timeframe?",
            _RULE,
            (
                f"YES - 5m had the most GROUP 1-2 survivors ({tf5[1]}) "
                f"and led the timeframe ranking."
                if five_still_best
                else (
                    f"NO - {best_tf[0]} ranked ahead of 5m on survivor count / "
                    f"mean validation avg R. 5m survivors={tf5[1]}."
                )
            ),
            "",
            _RULE,
            "9. Which candidates deserve 2025 OOS?",
            _RULE,
        ]
    )
    if oos:
        for row in oos:
            lines.append(
                f"{row.strategy_id}/{row.timeframe} ({row.family}, {row.group}): "
                f"dev net={fmt(row.development.metrics.total_net_pnl)} "
                f"val net={fmt(row.validation.metrics.total_net_pnl)} "
                f"dev avgR={fmt(row.development.metrics.average_R_per_closed_trade)} "
                f"val avgR={fmt(row.validation.metrics.average_R_per_closed_trade)} "
                f"val PF={fmt(row.validation.metrics.profit_factor)} "
                f"val DD%={fmt(row.validation.metrics.max_drawdown_pct)} "
                f"trades dev/val={row.development.metrics.closed_trades}/"
                f"{row.validation.metrics.closed_trades} "
                f"monthly val: {monthly_consistency_label(row.validation.monthly)}"
            )
        lines.append("2025 was not opened. These are locked candidates only.")
    else:
        lines.append("None. Fewer than one candidate met the OOS selection bar.")
    lines.extend(
        [
            "",
            _RULE,
            "10. Which strategy families should now be abandoned?",
            _RULE,
        ]
    )
    if abandoned:
        lines.append(
            "No (strategy x timeframe) member reached GROUP 1-2: " + ", ".join(abandoned)
        )
    else:
        lines.append(
            "Every non-baseline family had at least one timeframe in GROUP 1-2, "
            "or else no family fully failed. See the family table above."
        )
        zero = [
            family
            for family, survived, total, _mean in families
            if survived == 0 and family not in {"BASELINE", "RANDOM"}
        ]
        if not zero:
            lines.append(
                "None fully abandoned on this split; still drop families with "
                "only GROUP 3/4 rows if a later cut is needed."
            )

    lines.extend(["", _RULE, "GROUP 1 detail", _RULE])
    if group1:
        for row in sorted(
            group1,
            key=lambda item: -(item.score if item.score is not None else -999.0),
        ):
            lines.append(
                f"{row.strategy_id}/{row.timeframe}: "
                f"dev net={fmt(row.development.metrics.total_net_pnl)} "
                f"val net={fmt(row.validation.metrics.total_net_pnl)} "
                f"dev avgR={fmt(row.development.metrics.average_R_per_closed_trade)} "
                f"val avgR={fmt(row.validation.metrics.average_R_per_closed_trade)} "
                f"val PF={fmt(row.validation.metrics.profit_factor)} "
                f"val DD%={fmt(row.validation.metrics.max_drawdown_pct)} "
                f"trades={row.development.metrics.closed_trades}/"
                f"{row.validation.metrics.closed_trades} "
                f"monthly val: {monthly_consistency_label(row.validation.monthly)} "
                f"score={fmt(row.score)}"
            )
    else:
        lines.append("Empty.")

    lines.extend(
        [
            "",
            _RULE,
            'Did we find a potentially viable small-edge strategy?',
            _RULE,
            viable,
            "",
            "YES would still be 2024-split only. 2025 OOS has not been run.",
            "",
        ]
    )
    if warnings:
        lines.extend([_RULE, "Warnings", _RULE])
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def render_rankings(rows: list[RankedRow]) -> pd.DataFrame:
    ranked = sorted(
        rows,
        key=lambda row: (
            {"GROUP 1": 0, "GROUP 2": 1, "GROUP 3": 2, "GROUP 4": 3}.get(row.group, 9),
            -(row.score if row.score is not None else -999.0),
            row.strategy_id,
            row.timeframe,
        ),
    )
    payload = []
    for index, row in enumerate(ranked, start=1):
        payload.append(
            {
                "rank": index,
                "group": row.group,
                "strategy_id": row.strategy_id,
                "family": row.family,
                "timeframe": row.timeframe,
                "robustness_score": row.score,
                "dev_net_pnl": row.development.metrics.total_net_pnl,
                "val_net_pnl": row.validation.metrics.total_net_pnl,
                "dev_avg_R": row.development.metrics.average_R_per_closed_trade,
                "val_avg_R": row.validation.metrics.average_R_per_closed_trade,
                "val_profit_factor": row.validation.metrics.profit_factor,
                "val_max_drawdown_pct": row.validation.metrics.max_drawdown_pct,
                "dev_closed_trades": row.development.metrics.closed_trades,
                "val_closed_trades": row.validation.metrics.closed_trades,
                "val_dev_avg_R_ratio": row.r_ratio,
                "val_dev_pf_ratio": row.pf_ratio,
                "flags": "|".join(row.flags),
                "gate_failures": "|".join(row.gate_failures),
                "beat_BASELINE_A": bool(row.beat_baseline_avg_r and row.beat_baseline_net),
                "beat_RANDOM": bool(row.beat_random_avg_r and row.beat_random_net),
            }
        )
    return pd.DataFrame(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_strategy_tournament",
        description=(
            "Run the 2024 development/validation strategy tournament on BTCUSDT "
            "without live trading, API calls, or 2025 selection."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory (default: reports/research/strategy_tournament)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
    )
    parser.add_argument(
        "--indicators-config",
        type=Path,
        default=DEFAULT_INDICATOR_CONFIG,
    )
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=DEFAULT_STRATEGY_CONFIG,
    )
    parser.add_argument(
        "--backtest-config",
        type=Path,
        default=DEFAULT_BACKTEST_CONFIG,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = r".\.venv\Scripts\python.exe scripts\run_strategy_tournament.py"
    warnings: list[str] = []
    started = time.perf_counter()
    try:
        output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        assert_safe_output(output_dir)
        data_config = load_config(args.config)
        indicator_config = load_indicator_config(args.indicators_config)
        strategy_config = load_strategy_config(args.strategy_config)
        backtest_config = load_backtest_config(args.backtest_config)
        if backtest_config.starting_capital != STARTING_CAPITAL:
            raise TournamentError("starting_capital must remain 20")
        if abs(backtest_config.risk_per_trade - 0.01) > 1e-12:
            raise TournamentError("risk_per_trade must remain 1%")
        if abs(backtest_config.slippage - 0.0005) > 1e-12:
            raise TournamentError("slippage must remain 0.05%")
        if abs(backtest_config.entry_rate - 0.001) > 1e-12:
            raise TournamentError("entry_rate must remain 0.1%")
        if abs(backtest_config.exit_rate - 0.001) > 1e-12:
            raise TournamentError("exit_rate must remain 0.1%")

        def log(message: str) -> None:
            if not args.quiet:
                print(message, flush=True)

        enriched_dir = enriched_dataset_dir(data_config, indicator_config)
        log("Loading 2024 enriched 5m parquet (2025 files not opened)")
        candles_5m = load_2024_enriched(enriched_dir)
        log("Integrity replay of BASELINE_A on full 2024 5m")
        integrity = integrity_baseline_full_year(
            candles_5m, backtest_config, strategy_config, indicator_config
        )
        log(integrity)
        if "MISMATCH" in integrity:
            warnings.append(integrity)

        log("Resampling 5m -> 15m and 1h; recalculating indicators on resampled series")
        frames = build_timeframes(candles_5m, indicator_config)
        features: dict[str, dict] = {}
        thresholds = {}
        for interval, frame in frames.items():
            refuse_2025(frame, interval)
            feat = build_features(frame)
            assert_feature_no_lookahead(feat, frame["high"].to_numpy(dtype=np.float64))
            features[interval] = feat
            thresholds[interval] = freeze_thresholds(feat, frame["timestamp"])
            log(
                f"  {interval}: rows={len(frame)} "
                f"dev range12 median={thresholds[interval].range12_median:.6g} "
                f"q3={thresholds[interval].range12_q3:.6g} "
                f"n={thresholds[interval].n_dev_valid}"
            )

        log("Building frozen signal frames")
        signal_frames = build_all_signal_frames(
            frames,
            features,
            thresholds,
            strategy_config,
            indicator_config,
        )
        a02_5m = signal_frames[("A02", "5m")]["signal"].to_numpy()
        b01_5m = signal_frames[("B01", "5m")]["signal"].to_numpy()
        if np.array_equal(a02_5m, b01_5m):
            warnings.append(
                "A02 and B01 produced identical 5m signals by construction "
                "(20-bar breakout + EMA trend + volume > MA)."
            )

        jobs = [
            (strategy_id, interval)
            for strategy_id in STRATEGY_ORDER
            for interval in TIMEFRAMES
        ]
        n_backtests = len(jobs) * 2
        log(f"Running {len(jobs)} candidates x 2 windows = {n_backtests} backtests")
        rows: list[RankedRow] = []
        by_strategy: dict[str, list[RankedRow]] = {key: [] for key in STRATEGY_ORDER}
        for index, (strategy_id, interval) in enumerate(jobs, start=1):
            log(f"[{index}/{len(jobs)}] {strategy_id} {interval}")
            candles = frames[interval]
            signals = signal_frames[(strategy_id, interval)]
            development = run_window(
                candles,
                signals,
                backtest_config=backtest_config,
                period_start=DEV_START,
                period_end=DEV_END,
                period_role="DEVELOPMENT",
                interval=interval,
            )
            validation = run_window(
                candles,
                signals,
                backtest_config=backtest_config,
                period_start=VAL_START,
                period_end=VAL_END,
                period_role="VALIDATION",
                interval=interval,
            )
            row = evaluate_pair(
                strategy_id,
                interval,
                DEFINITIONS[strategy_id],
                development,
                validation,
                thresholds[interval],
            )
            rows.append(row)
            by_strategy[strategy_id].append(row)

        assign_groups(rows)
        oos = select_oos_candidates(rows)
        elapsed = time.perf_counter() - started
        commit = git_commit()

        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = pd.DataFrame([matrix_row(row) for row in rows])
        matrix.to_csv(output_dir / "tournament_matrix.csv", index=False)
        render_rankings(rows).to_csv(output_dir / "tournament_rankings.csv", index=False)
        for strategy_id, strategy_rows in by_strategy.items():
            write_candidate_folder(output_dir / strategy_id, strategy_rows)
        summary = render_summary(
            rows,
            command=command,
            elapsed_s=elapsed,
            commit=commit,
            n_backtests=n_backtests,
            n_strategies=len(STRATEGY_ORDER),
            n_timeframes=len(TIMEFRAMES),
            integrity=integrity,
            warnings=warnings,
            oos=oos,
        )
        write_text(output_dir / "tournament_summary.txt", summary)

        group1 = sum(1 for row in rows if row.group == "GROUP 1")
        survived_dev = sum(
            1
            for row in rows
            if row.development.metrics.closed_trades >= 30
            and (_num(row.development.metrics.average_R_per_closed_trade) or 0) > 0
            and float(row.development.metrics.total_net_pnl) > 0
        )
        survived_val = sum(
            1
            for row in rows
            if row.group in {"GROUP 1", "GROUP 2"}
        )
        print()
        print("1. exact command:", command)
        print(f"2. execution time: {elapsed:.1f}s")
        print(f"3. number of candidates: {len(STRATEGY_ORDER)} strategies x {len(TIMEFRAMES)} timeframes = {len(rows)}")
        print(f"4. number of backtests: {n_backtests}")
        print(f"5. number surviving development: {survived_dev}")
        print(f"6. number surviving validation: {survived_val}")
        if oos:
            print(
                "7. top 5 candidates:",
                ", ".join(f"{row.strategy_id}/{row.timeframe}" for row in oos),
            )
        else:
            print("7. top 5 candidates: (none)")
        print(f"8. output directory: {output_dir}")
        if warnings:
            print("9. any errors/warnings:")
            for item in warnings:
                print(f"   - {item}")
        else:
            print("9. any errors/warnings: none")
        print(f"(GROUP 1 count={group1}; commit={commit})")
        return EXIT_PASS
    except TournamentError as error:
        print(f"\nTournament aborted: {error}", file=sys.stderr)
        return EXIT_FAIL
    except Exception as error:  # noqa: BLE001 - surface unexpected research failures
        print(f"\nTournament error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
