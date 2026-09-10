"""Text and CSV writers for v0.5 research experiments.

``datetime.now`` is metadata only. Numeric files do not include a clock.
Reports never use the labels ``ending capital`` or ``final capital``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from src.backtest.ledger import TRADE_FRAME_COLUMNS

from .config import VARIANT_ORDER, VariantSpec
from .metrics import ExperimentMetrics
from .selection import GATE_DISCLAIMER, SelectionReport

NA = "n/a"
_RULE = "-" * 78
FORBIDDEN_LABELS = ("ending capital", "final capital")

COMPARISON_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "label",
    "closed_trades",
    "win_rate",
    "average_winning_trade",
    "average_losing_trade",
    "average_net_pnl_per_closed_trade",
    "average_R_per_closed_trade",
    "expectancy_per_trade",
    "expectancy_R",
    "profit_factor",
    "total_gross_pnl",
    "total_fees",
    "total_slippage",
    "total_net_pnl",
    "max_drawdown",
    "max_drawdown_pct",
    "total_R",
    "max_consecutive_losses",
    "ending_cash",
    "ending_realized_equity",
    "ending_mtm_equity",
)


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return NA
    return str(value)


def _assert_no_forbidden(text: str) -> None:
    lowered = text.lower()
    for phrase in FORBIDDEN_LABELS:
        if phrase in lowered:
            raise ValueError(f"Report text contains forbidden label {phrase!r}")


def write_metrics_csv(metrics: ExperimentMetrics, path: Path) -> None:
    rows = [{"metric": key, "value": _fmt(value)} for key, value in metrics.as_pairs()]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_frame_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep=NA)


def write_trades_csv(trades: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = trades.copy()
    if "holding_time" in frame.columns:
        frame["holding_time"] = frame["holding_time"].map(
            lambda value: "" if value is None or pd.isna(value) else str(value)
        )
    expected = [column for column in TRADE_FRAME_COLUMNS if column in frame.columns]
    frame.to_csv(path, index=False, columns=expected)


def write_snapshot(snapshot: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(snapshot, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def render_summary(
    *,
    period_role: str,
    period_start: object,
    period_end: object,
    symbol: str,
    interval: str,
    variant: VariantSpec,
    metrics: ExperimentMetrics,
    strategy_lines: list[str],
    backtest_lines: list[str],
    generated: str | None = None,
) -> str:
    stamp = generated or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "=" * 78,
        "v0.5 research experiment report",
        "=" * 78,
        "",
        f"Period role:         {period_role}",
        f"Period:              {period_start} .. {period_end} (UTC, end date inclusive)",
        f"Generated:           {stamp}",
        "",
        "This report is research analytics on top of the v0.4 simulator.",
        "It does NOT mean the strategy is profitable, statistically validated,",
        "or ready for live trading. STATUS PASS from v0.4 is integrity only.",
        "A profitable DEVELOPMENT result is not validated until the untouched",
        "out-of-sample test.",
        "",
        f"Symbol:              {symbol}",
        f"Interval:            {interval}",
        f"Experiment:          {variant.experiment_id} ({variant.label})",
        "",
        "Strategy",
        _RULE,
        *strategy_lines,
        "",
        "Backtest (v0.4 engine, unchanged semantics)",
        _RULE,
        *backtest_lines,
        "",
        "Account endings (do not collapse these into one number)",
        _RULE,
        f"ending_cash:         {_fmt(metrics.ending_cash)}",
        f"ending_realized_equity: {_fmt(metrics.ending_realized_equity)}",
        f"ending_mtm_equity:    {_fmt(metrics.ending_mtm_equity)}",
        "",
        "Cost separation (CLOSED trades only)",
        _RULE,
        f"total_gross_pnl:     {_fmt(metrics.total_gross_pnl)}",
        f"total_fees:          {_fmt(metrics.total_fees)}",
        f"total_slippage:      {_fmt(metrics.total_slippage)}",
        f"total_net_pnl:       {_fmt(metrics.total_net_pnl)}",
        "",
        "Counts",
        _RULE,
        f"signals:             {metrics.signals}",
        f"filled_trades:       {metrics.filled_trades}",
        f"closed_trades:       {metrics.closed_trades}",
        f"open_end_of_data:    {metrics.open_end_of_data}",
        f"ignored_signals:     {metrics.ignored_signals}",
        f"tp_count:            {metrics.tp_count}",
        f"sl_count:            {metrics.sl_count}",
        "",
        "Closed-trade analytics",
        _RULE,
        f"win_rate:            {_fmt(metrics.win_rate)}",
        f"loss_rate:           {_fmt(metrics.loss_rate)}",
        f"average_winning_trade: {_fmt(metrics.average_winning_trade)}",
        f"average_losing_trade:  {_fmt(metrics.average_losing_trade)}",
        f"average_net_pnl_per_closed_trade: {_fmt(metrics.average_net_pnl_per_closed_trade)}",
        f"average_R_per_closed_trade: {_fmt(metrics.average_R_per_closed_trade)}",
        f"expectancy_per_trade: {_fmt(metrics.expectancy_per_trade)}",
        f"expectancy_R:        {_fmt(metrics.expectancy_R)}",
        f"profit_factor:       {_fmt(metrics.profit_factor)}",
        f"total_R:             {_fmt(metrics.total_R)}",
        f"max_consecutive_losses: {metrics.max_consecutive_losses}",
        f"max_drawdown:        {_fmt(metrics.max_drawdown)}  (global MTM)",
        f"max_drawdown_pct:    {_fmt(metrics.max_drawdown_pct)}",
        "",
        "OPEN / END_OF_DATA mark-to-market is informational and is not realized P&L.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    _assert_no_forbidden(text)
    return text


def render_comparison(
    rows: list[dict[str, object]],
    *,
    generated: str | None = None,
) -> str:
    stamp = generated or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "=" * 78,
        "v0.5 A/B/C/D comparison  (DEVELOPMENT only)",
        "=" * 78,
        "",
        f"Generated:           {stamp}",
        "",
        "This comparison is confined to the 2024 development period.",
        "It does not declare a winner. It does not use 2025 data.",
        GATE_DISCLAIMER,
        "",
    ]
    header = "  ".join(f"{name:>24}" for name in ("metric", *VARIANT_ORDER))
    lines.append(header)
    lines.append(_RULE)
    by_id = {row["experiment_id"]: row for row in rows}
    metric_names = [name for name in COMPARISON_FIELDS if name not in {"experiment_id", "label"}]
    for metric in metric_names:
        cells = [_fmt(by_id[experiment_id].get(metric)) for experiment_id in VARIANT_ORDER]
        lines.append(
            f"{metric:>24}  " + "  ".join(f"{cell:>24}" for cell in cells)
        )
    lines.extend(
        [
            "",
            "Trade-offs: higher trade count can increase fee and slippage drag;",
            "higher TP can lower win rate while raising average win; a stronger",
            "breakout can starve the sample. Read the four columns together.",
            "",
        ]
    )
    text = "\n".join(lines) + "\n"
    _assert_no_forbidden(text)
    return text


def render_checklist(report: SelectionReport) -> str:
    lines = [
        "=" * 78,
        "v0.5 selection checklist  (DEVELOPMENT only)",
        "=" * 78,
        "",
        GATE_DISCLAIMER,
        "",
        "No variant is selected automatically. Pass --candidate explicitly.",
        "Do not use 2025 to choose or modify a candidate.",
        "",
    ]
    for item in report.variants:
        lines.append(f"Variant {item.experiment_id}")
        lines.append(_RULE)
        lines.append(f"eligible:            {'PASS' if item.eligible else 'FAIL'}")
        lines.append(
            f"advisory_rank:       {_fmt(item.advisory_rank)} "
            "(eligible variants only; not a winner)"
        )
        for gate in item.gates:
            mark = "PASS" if gate.passed else "FAIL"
            lines.append(f"  {gate.name}: {mark}  {gate.detail}")
        lines.append("")
    if report.eligible_ids:
        lines.append(
            "Eligible (advisory order): " + ", ".join(report.eligible_ids)
        )
    else:
        lines.append(
            "No variant met the 2024 research eligibility gates. "
            "Out-of-sample testing is still allowed only via explicit "
            "--candidate as a negative confirmation, not as validation."
        )
    lines.append("")
    text = "\n".join(lines) + "\n"
    _assert_no_forbidden(text)
    return text


def comparison_rows(
    metrics_by_id: dict[str, ExperimentMetrics],
    labels: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for experiment_id in VARIANT_ORDER:
        metrics = metrics_by_id[experiment_id]
        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "label": labels[experiment_id],
        }
        for name in COMPARISON_FIELDS:
            if name in {"experiment_id", "label"}:
                continue
            row[name] = getattr(metrics, name)
        rows.append(row)
    return rows
