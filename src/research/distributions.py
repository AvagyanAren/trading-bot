"""Quantiles and fixed-bin histograms for R, net P&L, and holding time."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.backtest.ledger import Trade

from .metrics import closed_trades

QUANTILE_NAMES: tuple[str, ...] = ("min", "p10", "p25", "p50", "p75", "p90", "max")
QUANTILE_PROBS: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)

R_EDGES = np.array([-np.inf, -2.0, -1.0, 0.0, 1.0, 2.0, np.inf])
NET_PNL_EDGES = np.array([-np.inf, -0.5, -0.2, 0.0, 0.2, 0.5, np.inf])
HOLDING_EDGES_MINUTES = np.array([0.0, 15.0, 60.0, 240.0, 1440.0, np.inf])

DISTRIBUTION_COLUMNS: tuple[str, ...] = (
    "series",
    "kind",
    "label",
    "value",
)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in QUANTILE_NAMES}
    array = np.asarray(values, dtype="float64")
    quantiles = np.quantile(array, QUANTILE_PROBS, method="linear")
    return {
        name: float(quantiles[index])
        for index, name in enumerate(QUANTILE_NAMES)
    }


def _bin_label(left: float, right: float) -> str:
    left_s = "-inf" if not np.isfinite(left) else str(left)
    right_s = "inf" if not np.isfinite(right) else str(right)
    return f"({left_s}, {right_s}]"


def _histogram(values: list[float], edges: np.ndarray) -> list[tuple[str, int]]:
    if not values:
        return [
            (_bin_label(edges[index], edges[index + 1]), 0)
            for index in range(len(edges) - 1)
        ]
    counts, _ = np.histogram(np.asarray(values, dtype="float64"), bins=edges)
    return [
        (_bin_label(edges[index], edges[index + 1]), int(counts[index]))
        for index in range(len(edges) - 1)
    ]


def distribution_table(trades: Sequence[Trade]) -> pd.DataFrame:
    closed = closed_trades(trades)
    r_values = [float(trade.R) for trade in closed if trade.R is not None]
    pnl_values = [float(trade.net_pnl) for trade in closed if trade.net_pnl is not None]
    holding_minutes = [
        float(trade.holding_time.total_seconds()) / 60.0
        for trade in closed
        if trade.holding_time is not None and not pd.isna(trade.holding_time)
    ]
    rows: list[dict[str, object]] = []
    for series, values, edges in (
        ("R", r_values, R_EDGES),
        ("net_pnl", pnl_values, NET_PNL_EDGES),
        ("holding_time_minutes", holding_minutes, HOLDING_EDGES_MINUTES),
    ):
        for name, value in _quantiles(values).items():
            rows.append(
                {
                    "series": series,
                    "kind": "quantile",
                    "label": name,
                    "value": value,
                }
            )
        for label, count in _histogram(values, edges):
            rows.append(
                {
                    "series": series,
                    "kind": "histogram",
                    "label": label,
                    "value": count,
                }
            )
    return pd.DataFrame(rows, columns=list(DISTRIBUTION_COLUMNS))
